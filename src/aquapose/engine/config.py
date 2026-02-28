"""Frozen dataclass config hierarchy for the AquaPose pipeline.

Loading precedence: defaults -> YAML file -> CLI overrides -> freeze.

The frozen guarantee prevents accidental mutation during execution. Full
serialized config is written as the first artifact of every run to ensure
reproducibility.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Stage-specific config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionConfig:
    """Config for the detection stage.

    Attributes:
        detector_kind: Detector backend to use (e.g. "yolo", "mog2").
        model_path: Path to detector weights file (required for YOLO).
            ``None`` means no path configured (caller must supply via
            ``detector_kwargs`` or construct the stage directly).
        device: Device to run inference on (e.g. ``"cuda"``, ``"cpu"``).
        stop_frame: If set, stop processing after this frame index.
        extra: Catch-all dict for detector-specific kwargs not covered above.
    """

    detector_kind: str = "yolo"
    model_path: str | None = None
    device: str = "cuda"
    stop_frame: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclasses forbid normal assignment in __post_init__, but
        # dict is mutable so we just ensure we got a dict (not None).
        if not isinstance(self.extra, dict):
            object.__setattr__(self, "extra", dict(self.extra))


@dataclass(frozen=True)
class MidlineConfig:
    """Config for the Midline stage (Stage 2).

    The Midline stage uses a segment-then-extract backend: it runs U-Net
    segmentation internally to produce binary masks, then extracts 15-point
    2D midlines with half-widths from those masks. Both segmentation and
    midline extraction are configured here.

    Attributes:
        confidence_threshold: Minimum confidence for mask acceptance by the
            segmentation backend.
        weights_path: Path to U-Net model weights file (None = use default/pretrained).
        backend: Midline backend to use. Currently only "segment_then_extract"
            is implemented; "direct_pose" is a planned stub.
        n_points: Number of midline points to produce per detection.
        min_area: Minimum mask area (pixels) required to attempt midline extraction.
        detection_tolerance: Maximum pixel distance for matching a tracklet
            centroid to a detection. OC-SORT Kalman predictions can drift
            10-30px from raw centroids during coast/recovery. Default 50.0.
        speed_threshold: Minimum pixel-per-frame speed for velocity signal in
            head-tail orientation. Below this, velocity is ignored. Default 2.0.
        orientation_weight_geometric: Weight for cross-camera geometric vote
            in orientation resolution. Default 1.0.
        orientation_weight_velocity: Weight for velocity alignment signal.
            Default 0.5.
        orientation_weight_temporal: Weight for temporal prior signal.
            Default 0.3.
    """

    confidence_threshold: float = 0.5
    weights_path: str | None = None
    backend: str = "segment_then_extract"
    n_points: int = 15
    min_area: int = 300
    detection_tolerance: float = 50.0
    speed_threshold: float = 2.0
    orientation_weight_geometric: float = 1.0
    orientation_weight_velocity: float = 0.5
    orientation_weight_temporal: float = 0.3


@dataclass(frozen=True)
class AssociationConfig:
    """Config for the Association stage (Stage 3).

    Controls pairwise cross-camera tracklet scoring and Leiden clustering
    for fish identity assignment. All thresholds are YAML-tunable.

    Attributes:
        ray_distance_threshold: Maximum ray-ray closest-point distance (metres)
            to classify a frame as an inlier. Default 0.03 (3 cm -- fish are
            ~10 cm long, ~2 cm wide; 3 cm accommodates centroid jitter).
        score_min: Minimum affinity score to create a graph edge. Default 0.3.
        t_min: Minimum shared frames for a tracklet pair to be scored. Default 10.
        t_saturate: Frame count at which overlap reliability saturates. Default 100.
        early_k: Number of initial frames for early termination check. Default 10.
        expected_fish_count: Number of fish in the tank (fixed). Default 9.
            Auto-populated from top-level ``n_animals`` when not explicitly set.
        ghost_pixel_threshold: Max pixel distance for a detection to count as
            "supporting" in ghost penalty. Default 50.0.
        min_shared_voxels: Minimum shared voxels for camera pair adjacency.
            Default 100.
        leiden_resolution: Resolution parameter for Leiden clustering. Default 1.0.
        max_merge_gap: Maximum frame gap for fragment merging. Default 30.
        eviction_reproj_threshold: Maximum median ray-to-consensus-point distance
            (metres) for a tracklet to remain in its cluster. Default 0.025
            (2.5 cm -- fish are ~10 cm long, ~2 cm wide).
        min_cameras_refine: Minimum cameras in a cluster to attempt 3D
            refinement. Clusters with fewer cameras skip refinement. Default 3.
        refinement_enabled: Toggle to skip refinement entirely. Default True.
    """

    ray_distance_threshold: float = 0.03
    score_min: float = 0.3
    t_min: int = 10
    t_saturate: int = 100
    early_k: int = 10
    expected_fish_count: int = 9
    ghost_pixel_threshold: float = 50.0
    min_shared_voxels: int = 100
    leiden_resolution: float = 1.0
    max_merge_gap: int = 30
    eviction_reproj_threshold: float = 0.025
    min_cameras_refine: int = 3
    refinement_enabled: bool = True


@dataclass(frozen=True)
class TrackingConfig:
    """Config for the 2D Tracking stage (Stage 2).

    Controls the OC-SORT tracker used for per-camera 2D fish tracking.

    Attributes:
        tracker_kind: Tracker backend to use. Currently only ``"ocsort"`` is
            implemented; ``"bytetrack"`` and ``"sort"`` are reserved for
            future use.
        max_coast_frames: Maximum frames to coast (Kalman predict with no
            observation) before dropping a track. Maps to boxmot ``max_age``.
        n_init: Minimum number of matched detection frames before a track is
            confirmed and included in stage output. Maps to boxmot
            ``min_hits``.
        iou_threshold: IoU threshold for matching detections to existing tracks.
        det_thresh: Minimum detection confidence forwarded to the tracker.
    """

    tracker_kind: str = "ocsort"
    max_coast_frames: int = 30
    n_init: int = 3
    iou_threshold: float = 0.3
    det_thresh: float = 0.3


@dataclass(frozen=True)
class LutConfig:
    """Config for refractive lookup table generation.

    Attributes:
        tank_diameter: Cylindrical tank diameter in metres.
        tank_height: Tank depth (water column height) in metres.
        voxel_resolution_m: Voxel grid spacing in metres (default 0.02 = 2 cm).
        margin_fraction: Fractional margin beyond tank dimensions for LUT coverage
            (default 0.1 = 10%).
        forward_grid_step: Pixel step size for forward LUT grid (default 1 = every
            pixel).
    """

    tank_diameter: float = 2.0
    tank_height: float = 1.0
    voxel_resolution_m: float = 0.02
    margin_fraction: float = 0.1
    forward_grid_step: int = 1


@dataclass(frozen=True)
class SyntheticConfig:
    """Config for synthetic data generation in synthetic mode.

    Attributes:
        fish_count: Number of synthetic fish to generate.
            Auto-populated from top-level ``n_animals`` when not explicitly set.
        frame_count: Number of frames to simulate.
        noise_std: Standard deviation of Gaussian noise added to 2D
            projections (pixels). 0 = no noise.
        seed: Random seed for reproducible generation.
    """

    fish_count: int = 3
    frame_count: int = 30
    noise_std: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class ReconstructionConfig:
    """Config for the Reconstruction stage (Stage 5).

    Attributes:
        backend: Reconstruction backend to use. ``"triangulation"`` (default)
            uses RANSAC multi-view triangulation + B-spline fitting.
            ``"curve_optimizer"`` uses correspondence-free 3D B-spline
            optimization via chamfer distance.
        inlier_threshold: Maximum reprojection error (pixels) for RANSAC
            inlier classification during triangulation.
        snap_threshold: Maximum pixel distance from the epipolar curve to
            accept a correspondence during epipolar refinement.
        max_depth: Maximum allowed fish depth below the water surface (metres).
            When None (default), no upper depth bound is enforced. Set to the
            physical tank depth to catch above-water outliers.
        min_cameras: Minimum cameras observing a fish in a frame to attempt
            triangulation. Frames with fewer cameras are dropped. Default 3
            (well-supported by rig geometry — 2-camera triangulation is
            ill-conditioned with refractive geometry).
        max_interp_gap: Maximum consecutive dropped frames to interpolate.
            Gaps longer than this are left as missing data. Default 5
            (~167ms at 30fps, within fish trajectory smoothness).
        n_control_points: Fixed B-spline control point count per fish per
            frame. Must match the triangulation module's SPLINE_N_CTRL.
            Default 7.
    """

    backend: str = "triangulation"
    inlier_threshold: float = 50.0
    snap_threshold: float = 20.0
    max_depth: float | None = None
    min_cameras: int = 3
    max_interp_gap: int = 5
    n_control_points: int = 7


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level frozen config for a full pipeline run.

    Attributes:
        run_id: Unique run identifier (timestamp-based by default).
        output_dir: Root directory for run artifacts.
        video_dir: Directory containing input video files.
        calibration_path: Path to the AquaCal calibration file.
        mode: Execution mode preset (production, diagnostic, synthetic, benchmark).
        n_animals: Number of animals in the scene. Propagates to
            ``association.expected_fish_count`` and ``synthetic.fish_count``
            when those fields are not explicitly overridden.
        detection: Detection stage config (Stage 1).
        midline: Midline stage config (Stage 2) — configures segment-then-extract backend.
        association: Association stage config (Stage 3).
        tracking: Tracking stage config (Stage 4).
        reconstruction: Reconstruction stage config (Stage 5).
        synthetic: Synthetic data generation config (synthetic mode only).
        lut: Refractive lookup table config (tank geometry and grid resolution).
    """

    run_id: str = dataclasses.field(default="")
    output_dir: str = dataclasses.field(default="")
    video_dir: str = ""
    calibration_path: str = ""
    mode: str = "production"
    n_animals: int = 9
    detection: DetectionConfig = dataclasses.field(default_factory=DetectionConfig)
    midline: MidlineConfig = dataclasses.field(default_factory=MidlineConfig)
    association: AssociationConfig = dataclasses.field(
        default_factory=AssociationConfig
    )
    tracking: TrackingConfig = dataclasses.field(default_factory=TrackingConfig)
    reconstruction: ReconstructionConfig = dataclasses.field(
        default_factory=ReconstructionConfig
    )
    synthetic: SyntheticConfig = dataclasses.field(default_factory=SyntheticConfig)
    lut: LutConfig = dataclasses.field(default_factory=LutConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_run_id() -> str:
    """Generate a timestamp-based run identifier.

    Returns:
        Run ID string of the form "run_YYYYMMDD_HHMMSS".
    """
    return f"run_{datetime.now():%Y%m%d_%H%M%S}"


def _default_output_dir(run_id: str) -> str:
    """Return the default artifact output directory for a run.

    Args:
        run_id: The run identifier.

    Returns:
        Expanded absolute path string.
    """
    return str(Path(f"~/aquapose/runs/{run_id}").expanduser())


def _merge_stage_config(
    defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Shallow-merge *overrides* onto *defaults*, returning a new dict.

    Args:
        defaults: Base key-value pairs.
        overrides: Values that take precedence over defaults.

    Returns:
        Merged dict with overrides applied.
    """
    merged = dict(defaults)
    merged.update(overrides)
    return merged


def _apply_nested_overrides(
    flat: dict[str, Any], nested: dict[str, Any]
) -> dict[str, Any]:
    """Apply nested dict overrides onto a flat key->value mapping.

    CLI overrides may arrive as dot-notation keys ("detection.detector_kind")
    or as nested dicts ({"detection": {"detector_kind": "mog2"}}). This
    function flattens nested dicts to dot-notation before merging.

    Args:
        flat: Existing flat override dict (dot-notation keys).
        nested: Override source; may be nested or already flat.

    Returns:
        New flat dict combining both sources, nested taking precedence.
    """
    result = dict(flat)
    for key, value in nested.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                result[f"{key}.{subkey}"] = subvalue
        else:
            result[key] = value
    return result


def _build_stage_dict_from_dotted(
    flat: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Convert dot-notation override keys to a nested stage->field mapping.

    For example, {"detection.detector_kind": "mog2"} becomes
    {"detection": {"detector_kind": "mog2"}}.

    Top-level keys (no dot) remain in a special "__top__" bucket.

    Args:
        flat: Flat dict with dot-notation keys.

    Returns:
        Nested dict: stage_name -> {field: value}, plus "__top__" for
        top-level overrides.
    """
    nested: dict[str, Any] = {"__top__": {}}
    for key, value in flat.items():
        if "." in key:
            stage, _, field_name = key.partition(".")
            nested.setdefault(stage, {})[field_name] = value
        else:
            nested["__top__"][key] = value
    return nested


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_config(
    yaml_path: str | Path | None = None,
    *,
    cli_overrides: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> PipelineConfig:
    """Construct a frozen :class:`PipelineConfig` using layered overrides.

    Loading precedence (lowest → highest priority):

    1. Dataclass field defaults
    2. YAML file (*yaml_path*)
    3. CLI overrides (*cli_overrides*)
    4. Freeze

    CLI overrides may use dot-notation keys ("detection.detector_kind") or
    nested dicts ({"detection": {"detector_kind": "mog2"}}) — both forms
    work.  YAML files use nested dicts.

    Args:
        yaml_path: Optional path to a YAML config file.
        cli_overrides: Optional dict of CLI overrides (highest precedence).
        run_id: Explicit run identifier. Auto-generated if not provided.

    Returns:
        Frozen :class:`PipelineConfig` with all overrides applied.
    """
    # --- layer 1: defaults ------------------------------------------------
    det_kwargs: dict[str, Any] = {}
    mid_kwargs: dict[str, Any] = {}
    assoc_kwargs: dict[str, Any] = {}
    trk_kwargs: dict[str, Any] = {}
    rec_kwargs: dict[str, Any] = {}
    syn_kwargs: dict[str, Any] = {}
    lut_kwargs: dict[str, Any] = {}
    top_kwargs: dict[str, Any] = {}

    # --- layer 2: YAML overrides ------------------------------------------
    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        with yaml_path.open() as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        # Flatten YAML nested dicts to dot-notation then re-bucket
        flat_yaml = _apply_nested_overrides({}, raw)
        yaml_nested = _build_stage_dict_from_dotted(flat_yaml)

        det_kwargs = _merge_stage_config(det_kwargs, yaml_nested.get("detection", {}))
        # Accept old name ("segmentation") for backward compat; new name ("midline") takes precedence
        mid_kwargs = _merge_stage_config(
            mid_kwargs, yaml_nested.get("segmentation", {})
        )
        mid_kwargs = _merge_stage_config(mid_kwargs, yaml_nested.get("midline", {}))
        assoc_kwargs = _merge_stage_config(
            assoc_kwargs, yaml_nested.get("association", {})
        )
        trk_kwargs = _merge_stage_config(trk_kwargs, yaml_nested.get("tracking", {}))
        # Accept old name ("triangulation") for backward compat; new name ("reconstruction") takes precedence
        rec_kwargs = _merge_stage_config(
            rec_kwargs, yaml_nested.get("triangulation", {})
        )
        rec_kwargs = _merge_stage_config(
            rec_kwargs, yaml_nested.get("reconstruction", {})
        )
        syn_kwargs = _merge_stage_config(syn_kwargs, yaml_nested.get("synthetic", {}))
        lut_kwargs = _merge_stage_config(lut_kwargs, yaml_nested.get("lut", {}))
        top_kwargs = _merge_stage_config(top_kwargs, yaml_nested.get("__top__", {}))

    # --- layer 3: CLI overrides -------------------------------------------
    if cli_overrides is not None:
        flat_cli = _apply_nested_overrides({}, cli_overrides)
        cli_nested = _build_stage_dict_from_dotted(flat_cli)

        det_kwargs = _merge_stage_config(det_kwargs, cli_nested.get("detection", {}))
        # Accept old name ("segmentation") for backward compat; new name ("midline") takes precedence
        mid_kwargs = _merge_stage_config(mid_kwargs, cli_nested.get("segmentation", {}))
        mid_kwargs = _merge_stage_config(mid_kwargs, cli_nested.get("midline", {}))
        assoc_kwargs = _merge_stage_config(
            assoc_kwargs, cli_nested.get("association", {})
        )
        trk_kwargs = _merge_stage_config(trk_kwargs, cli_nested.get("tracking", {}))
        # Accept old name ("triangulation") for backward compat; new name ("reconstruction") takes precedence
        rec_kwargs = _merge_stage_config(
            rec_kwargs, cli_nested.get("triangulation", {})
        )
        rec_kwargs = _merge_stage_config(
            rec_kwargs, cli_nested.get("reconstruction", {})
        )
        syn_kwargs = _merge_stage_config(syn_kwargs, cli_nested.get("synthetic", {}))
        lut_kwargs = _merge_stage_config(lut_kwargs, cli_nested.get("lut", {}))
        top_kwargs = _merge_stage_config(top_kwargs, cli_nested.get("__top__", {}))

    # --- layer 4: resolve run_id and output_dir ---------------------------
    resolved_run_id = run_id or top_kwargs.pop("run_id", None) or _generate_run_id()
    resolved_output_dir = top_kwargs.pop(
        "output_dir", _default_output_dir(resolved_run_id)
    )

    # --- propagate n_animals to sub-configs --------------------------------
    n_animals = top_kwargs.get("n_animals", PipelineConfig.n_animals)
    if "expected_fish_count" not in assoc_kwargs:
        assoc_kwargs["expected_fish_count"] = n_animals
    if "fish_count" not in syn_kwargs:
        syn_kwargs["fish_count"] = n_animals

    # --- filter kwargs to only include fields present on each dataclass ----
    # AssociationConfig and TrackingConfig are stripped-down placeholders in
    # v2.1 Phase 22. Filter out any old YAML/CLI keys that no longer exist as
    # dataclass fields, so stale config files don't raise TypeError.
    def _filter_fields(dc_type: type, kwargs: dict[str, Any]) -> dict[str, Any]:
        valid = {f.name for f in dataclasses.fields(dc_type)}
        return {k: v for k, v in kwargs.items() if k in valid}

    # --- construct & freeze -----------------------------------------------
    return PipelineConfig(
        run_id=resolved_run_id,
        output_dir=resolved_output_dir,
        detection=DetectionConfig(**det_kwargs),
        midline=MidlineConfig(**mid_kwargs),
        association=AssociationConfig(
            **_filter_fields(AssociationConfig, assoc_kwargs)
        ),
        tracking=TrackingConfig(**_filter_fields(TrackingConfig, trk_kwargs)),
        reconstruction=ReconstructionConfig(**rec_kwargs),
        synthetic=SyntheticConfig(**syn_kwargs),
        lut=LutConfig(**_filter_fields(LutConfig, lut_kwargs)),
        **top_kwargs,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_config(config: PipelineConfig) -> str:
    """Serialize *config* to a YAML string.

    Uses :func:`dataclasses.asdict` to convert the frozen hierarchy to a
    plain dict, then :func:`yaml.dump` to produce a human-readable YAML
    string.  This YAML is written as the first run artifact by the
    orchestrator.

    Args:
        config: Frozen pipeline config to serialize.

    Returns:
        YAML string representation of the config.
    """
    return yaml.dump(
        dataclasses.asdict(config), default_flow_style=False, sort_keys=True
    )
