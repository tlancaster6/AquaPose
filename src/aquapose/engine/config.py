"""Frozen dataclass config hierarchy for the AquaPose pipeline.

Loading precedence: defaults -> YAML file -> CLI overrides -> freeze.

The frozen guarantee prevents accidental mutation during execution. Full
serialized config is written as the first artifact of every run to ensure
reproducibility.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage-specific config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionConfig:
    """Config for the detection stage.

    Attributes:
        detector_kind: Detector backend to use (e.g. "yolo", "yolo_obb").
        conf_threshold: Minimum detection confidence passed to the YOLO backend.
            Detections below this score are discarded at inference time.
        weights_path: Path to model weights for the active detection backend.
            ``None`` means no path configured (caller must supply via
            ``detector_kwargs`` or construct the stage directly).
        crop_size: Output size ``[width, height]`` in pixels for affine crops
            produced by :func:`~aquapose.core.pose.crop.extract_affine_crop`.
            Used by downstream stages when ``detector_kind`` is ``"yolo_obb"``.
            Defaults to ``[256, 128]`` — wide enough for a typical elongated fish
            body at standard frame resolutions. Stored as a list so that YAML
            round-trips cleanly (Python tuples serialize as ``!!python/tuple``
            which ``yaml.safe_load`` cannot parse).
        iou_threshold: IoU threshold for geometric polygon NMS. Detections
            with polygon IoU above this value are suppressed in favor of the
            higher-confidence detection. Default 0.45.
        detection_batch_frames: Maximum number of frames per YOLO detection
            batch.  ``0`` means no limit (batch all frames in the chunk).
        extra: Catch-all dict for detector-specific kwargs not covered above.

    Note:
        ``device`` has been promoted to a top-level :class:`PipelineConfig`
        field. ``stop_frame`` has been removed — use ``max_frames`` on the
        frame source instead. Passing either in ``detection:`` YAML raises a
        :exc:`ValueError` with a "did you mean?" hint.
    """

    detector_kind: str = "yolo"
    conf_threshold: float = 0.2
    iou_threshold: float = 0.45
    weights_path: str | None = None
    crop_size: list[int] = field(default_factory=lambda: [128, 64])
    detection_batch_frames: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclasses forbid normal assignment in __post_init__, but
        # dict is mutable so we just ensure we got a dict (not None).
        if not isinstance(self.extra, dict):
            object.__setattr__(self, "extra", dict(self.extra))
        # Validate detector_kind — only "yolo" and "yolo_obb" are supported.
        _valid_detector_kinds = {"yolo", "yolo_obb"}
        if self.detector_kind not in _valid_detector_kinds:
            raise ValueError(
                f"Unknown detector_kind: {self.detector_kind!r}. "
                f"Available: {sorted(_valid_detector_kinds)}"
            )


@dataclass(frozen=True)
class PoseConfig:
    """Config for the Pose stage (Stage 2).

    The Pose stage runs YOLO-pose inference on all detections and writes raw
    anatomical keypoints directly onto Detection objects before tracking.

    Attributes:
        confidence_threshold: YOLO detection confidence threshold for
            model.predict(). Detections below this score are discarded at
            inference time. Default 0.5.
        weights_path: Path to YOLO-pose model weights file.
        detection_tolerance: Maximum pixel distance for matching a tracklet
            centroid to a detection. Default 50.0.
        n_keypoints: Number of anatomical keypoints. Default 6.
        keypoint_t_values: Per-keypoint arc-fraction values in [0, 1] from nose
            (0.0) to tail (1.0). If ``None``, defaults to uniform spacing.
        keypoint_confidence_floor: Minimum per-keypoint confidence to treat as
            visible. Default 0.3.
        min_observed_keypoints: Minimum number of visible keypoints required for
            a valid pose result. Default 3.
        pose_batch_crops: Maximum number of crops per YOLO pose batch.
            ``0`` means no limit (batch all crops in the chunk).
    """

    confidence_threshold: float = 0.5
    weights_path: str | None = None
    detection_tolerance: float = 50.0
    # pose_estimation backend fields
    n_keypoints: int = 6
    keypoint_t_values: list[float] | None = None
    keypoint_confidence_floor: float = 0.3
    min_observed_keypoints: int = 3
    pose_batch_crops: int = 0


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
        t_min: Minimum shared frames for a tracklet pair to be scored. Default 3
            (matches ``n_init`` so tracklets confirmed by tracking can always be
            scored; raise for longer clips to reduce noise).
        t_saturate: Frame count at which overlap reliability saturates. Default 100.
        early_k: Number of initial frames for early termination check. Default 10.
        expected_fish_count: Number of fish in the tank (fixed). Default 9.
            Auto-populated from top-level ``n_animals`` when not explicitly set.
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
        centroid_keypoint_index: Index into Detection.keypoints for tracklet
            centroid. 0=nose, 1=head, 2=spine1 (default), 3=spine2, 4=spine3,
            5=tail. Falls back to OBB centroid when keypoint is absent or below
            ``centroid_confidence_floor``.
        centroid_confidence_floor: Minimum keypoint confidence to use keypoint
            as centroid. Below threshold falls back to OBB centroid. Default 0.3
            (matches the pose backend confidence floor).
    """

    ray_distance_threshold: float = 0.03
    score_min: float = 0.3
    t_min: int = 3
    t_saturate: int = 100
    early_k: int = 10
    expected_fish_count: int = 9
    min_shared_voxels: int = 100
    leiden_resolution: float = 1.0
    max_merge_gap: int = 30
    eviction_reproj_threshold: float = 0.025
    min_cameras_refine: int = 3
    refinement_enabled: bool = True
    centroid_keypoint_index: int = 2
    centroid_confidence_floor: float = 0.3


@dataclass(frozen=True)
class TrackingConfig:
    """Config for the 2D Tracking stage (Stage 2).

    Controls the tracker used for per-camera 2D fish tracking.

    Attributes:
        tracker_kind: Tracker backend. ``"ocsort"`` uses OC-SORT (boxmot).
            ``"keypoint_bidi"`` uses the custom bidirectional keypoint tracker.
        max_coast_frames: Maximum frames to coast (Kalman predict with no
            observation) before dropping a track. Maps to boxmot ``max_age``
            for OC-SORT and ``max_age`` for ``keypoint_bidi``.
        n_init: Minimum number of matched detection frames before a track is
            confirmed and included in stage output. Maps to boxmot
            ``min_hits`` for OC-SORT.
        iou_threshold: IoU threshold for matching detections to existing tracks.
            Used by ``"ocsort"`` only; ignored by ``"keypoint_bidi"``.
        det_thresh: Minimum detection confidence forwarded to the tracker.
        base_r: KF base measurement noise variance for ``"keypoint_bidi"``.
            Ignored when ``tracker_kind="ocsort"``. Default 10.0.
        lambda_ocm: OCM weight in the cost matrix for ``"keypoint_bidi"``.
            Ignored when ``tracker_kind="ocsort"``. Default 0.2.
        max_gap_frames: Maximum gap size (frames) for spline interpolation in
            ``"keypoint_bidi"``. Ignored when ``tracker_kind="ocsort"``.
            Default 5.

    Note:
        ``oks_sigmas`` for the keypoint tracker are loaded from
        ``keypoint_sigmas.DEFAULT_SIGMAS`` and not stored in config, to avoid
        coupling config serialization to the sigma array format.
    """

    tracker_kind: str = "ocsort"
    max_coast_frames: int = 30
    n_init: int = 3
    iou_threshold: float = 0.3
    det_thresh: float = 0.5
    # --- keypoint_bidi fields (ignored when tracker_kind="ocsort") ---
    base_r: float = 10.0
    lambda_ocm: float = 0.2
    max_gap_frames: int = 5

    def __post_init__(self) -> None:
        """Validate tracker_kind on construction."""
        valid_kinds = {"ocsort", "keypoint_bidi"}
        if self.tracker_kind not in valid_kinds:
            raise ValueError(
                f"Unknown tracker_kind: {self.tracker_kind!r}. "
                f"Available: {sorted(valid_kinds)}"
            )


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
class ZDenoisingConfig:
    """Config for z-denoising during reconstruction.

    When enabled, all triangulated body points are flattened to their
    centroid z before spline fitting.  This eliminates z-axis noise that
    the camera geometry cannot resolve.  The raw per-point z-offsets are
    preserved in the output for potential future use.

    Attributes:
        enabled: Whether to apply z-flattening during reconstruction.
            Default True.
    """

    enabled: bool = True


@dataclass(frozen=True)
class ReconstructionConfig:
    """Config for the Reconstruction stage (Stage 5).

    Attributes:
        backend: Reconstruction backend to use. Only ``"dlt"`` is supported.
            Uses confidence-weighted DLT triangulation with single-pass
            outlier rejection.
        outlier_threshold: Maximum reprojection error (pixels) for DLT
            backend outlier rejection during triangulation. Empirically
            tuned via ``aquapose tune``.
        min_cameras: Minimum cameras observing a fish in a frame to attempt
            triangulation. Frames with fewer cameras are dropped. Default 3
            (well-supported by rig geometry — 2-camera triangulation is
            ill-conditioned with refractive geometry).
        max_interp_gap: Maximum consecutive dropped frames to interpolate.
            Gaps longer than this are left as missing data. Default 5
            (~167ms at 30fps, within fish trajectory smoothness).
        n_control_points: Fixed B-spline control point count per fish per
            frame. Default 7.
        n_sample_points: Number of sample points along each midline for
            triangulation output. Default 15. Propagated from top-level
            n_sample_points when not explicitly overridden.
        z_denoising: Z-denoising config (flatten body points to centroid z).
    """

    backend: str = "dlt"
    outlier_threshold: float = 10.0
    min_cameras: int = 3
    max_interp_gap: int = 5
    n_control_points: int = 7
    n_sample_points: int = 15
    z_denoising: ZDenoisingConfig = dataclasses.field(default_factory=ZDenoisingConfig)


# ---------------------------------------------------------------------------
# Top-level config helpers
# ---------------------------------------------------------------------------


def _default_device() -> str:
    """Return the best available compute device.

    Returns:
        ``"cuda:0"`` when CUDA is available, otherwise ``"cpu"``.
    """
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


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
        n_animals: Number of animals in the scene. Sentinel value of 0 means
            not set; :func:`load_config` raises :exc:`ValueError` when this is
            0 or negative. Propagates to ``association.expected_fish_count`` and
            ``synthetic.fish_count`` when those fields are not explicitly overridden.
        device: Compute device for all stages (e.g. ``"cuda:0"``, ``"cpu"``).
            Auto-detected from :func:`_default_device` when absent from YAML.
            Propagates to DetectionStage and PoseStage via :func:`build_stages`.
        n_sample_points: Number of 2D midline points produced per detection and
            used throughout the reconstruction pipeline. Default is 15. Propagates
            to ``reconstruction.n_sample_points`` when that field is not explicitly
            overridden.
        project_dir: Optional project root directory for resolving relative paths.
            Empty string means no resolution — paths are used as-is.
        detection: Detection stage config (Stage 1).
        pose: Pose stage config (Stage 2) — configures YOLO-pose keypoint backend.
        association: Association stage config (Stage 4).
        tracking: Tracking stage config (Stage 3).
        reconstruction: Reconstruction stage config (Stage 5).
        synthetic: Synthetic data generation config (synthetic mode only).
        lut: Refractive lookup table config (tank geometry and grid resolution).
        stop_after: If set, truncate the stage list after the named stage.
            Valid values: ``"detection"``, ``"pose"``, ``"tracking"``,
            ``"association"``, or ``None`` (run all stages).
        chunk_size: Number of frames per processing chunk. None or 0 means
            process the entire video as a single chunk. Callers should check
            ``config.chunk_size or None`` to treat 0 as None.
    """

    run_id: str = dataclasses.field(default="")
    output_dir: str = dataclasses.field(default="")
    video_dir: str = ""
    calibration_path: str = ""
    mode: str = "production"
    n_animals: int = 0
    device: str = dataclasses.field(default_factory=_default_device)
    n_sample_points: int = 15
    project_dir: str = ""
    detection: DetectionConfig = dataclasses.field(default_factory=DetectionConfig)
    pose: PoseConfig = dataclasses.field(default_factory=PoseConfig)
    association: AssociationConfig = dataclasses.field(
        default_factory=AssociationConfig
    )
    tracking: TrackingConfig = dataclasses.field(default_factory=TrackingConfig)
    reconstruction: ReconstructionConfig = dataclasses.field(
        default_factory=ReconstructionConfig
    )
    synthetic: SyntheticConfig = dataclasses.field(default_factory=SyntheticConfig)
    lut: LutConfig = dataclasses.field(default_factory=LutConfig)
    stop_after: str | None = None
    chunk_size: int | None = None


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
# Field validation helpers
# ---------------------------------------------------------------------------

#: Maps obsolete/renamed YAML field names to their current equivalents.
#: Used by _filter_fields() to produce actionable error messages.
_RENAME_HINTS: dict[str, str] = {
    "expect_fish_count": "n_animals (top-level)",
    "device": "device (top-level)",
    "stop_frame": "max_frames on frame source (set via CLI --set or orchestrator)",
    "model_path": "weights_path",
    "n_points": "n_sample_points (top-level)",
}


def _filter_fields(dc_type: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate and filter kwargs to only fields present on *dc_type*.

    Raises ValueError for unknown fields (strict reject). Fields in
    _RENAME_HINTS produce a "did you mean X?" message.

    Args:
        dc_type: Target frozen dataclass type.
        kwargs: Key-value pairs from YAML/CLI.

    Returns:
        Filtered dict containing only valid fields.

    Raises:
        ValueError: If any keys are not valid fields of *dc_type*.
    """
    valid = {f.name for f in dataclasses.fields(dc_type)}
    unknown = [k for k in kwargs if k not in valid]
    if unknown:
        msgs = []
        for k in unknown:
            if k in _RENAME_HINTS:
                msgs.append(f"unknown field {k!r} — did you mean {_RENAME_HINTS[k]!r}?")
            else:
                msgs.append(f"unknown field {k!r}")
        raise ValueError(f"{dc_type.__name__}: " + "; ".join(msgs))
    return {k: v for k, v in kwargs.items() if k in valid}


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
    pose_kwargs: dict[str, Any] = {}
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
        # Accept old names ("segmentation", "midline") for backward compat; new name ("pose") takes precedence
        pose_kwargs = _merge_stage_config(
            pose_kwargs, yaml_nested.get("segmentation", {})
        )
        pose_kwargs = _merge_stage_config(pose_kwargs, yaml_nested.get("midline", {}))
        pose_kwargs = _merge_stage_config(pose_kwargs, yaml_nested.get("pose", {}))
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
        # Accept old names ("segmentation", "midline") for backward compat; new name ("pose") takes precedence
        pose_kwargs = _merge_stage_config(
            pose_kwargs, cli_nested.get("segmentation", {})
        )
        pose_kwargs = _merge_stage_config(pose_kwargs, cli_nested.get("midline", {}))
        pose_kwargs = _merge_stage_config(pose_kwargs, cli_nested.get("pose", {}))
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

    # --- layer 3.5: resolve project_dir-relative paths -------------------
    # Resolve relative path fields relative to project_dir when set.
    # This runs BEFORE run_id/output_dir resolution so output_dir gets
    # resolved relative to project_dir when it's a relative path.
    project_dir_str = top_kwargs.get("project_dir", "")
    if project_dir_str:
        project_dir = Path(project_dir_str).expanduser().resolve()
        # Resolve top-level path fields
        for key in ("video_dir", "calibration_path", "output_dir"):
            val = top_kwargs.get(key, "")
            if val and not Path(val).is_absolute():
                top_kwargs[key] = str(project_dir / val)
        # Resolve sub-config path fields
        for sub_kwargs, field_names in [
            (det_kwargs, ["weights_path"]),
            (pose_kwargs, ["weights_path"]),
        ]:
            for fname in field_names:
                val = sub_kwargs.get(fname, "")
                if val and not Path(val).is_absolute():
                    sub_kwargs[fname] = str(project_dir / val)

    # --- layer 4: resolve run_id and output_dir ---------------------------
    resolved_run_id = run_id or top_kwargs.pop("run_id", None) or _generate_run_id()
    explicit_output_dir = top_kwargs.pop("output_dir", "")
    if explicit_output_dir:
        resolved_output_dir = str(Path(explicit_output_dir) / resolved_run_id)
    elif project_dir_str:
        # Project has a project_dir — put runs inside it.
        resolved_output_dir = str(
            Path(project_dir_str).expanduser().resolve() / "runs" / resolved_run_id
        )
    else:
        resolved_output_dir = _default_output_dir(resolved_run_id)

    # --- validate n_animals (sentinel 0 means not set) --------------------
    resolved_n_animals = top_kwargs.get("n_animals", 0)
    if resolved_n_animals <= 0:
        raise ValueError("n_animals is required and must be > 0")

    # --- propagate n_animals to sub-configs --------------------------------
    n_animals = resolved_n_animals
    if "expected_fish_count" not in assoc_kwargs:
        assoc_kwargs["expected_fish_count"] = n_animals
    if "fish_count" not in syn_kwargs:
        syn_kwargs["fish_count"] = n_animals

    # --- propagate n_sample_points to reconstruction.n_sample_points -------
    if "n_sample_points" not in rec_kwargs:
        rec_kwargs["n_sample_points"] = top_kwargs.get("n_sample_points", 15)

    # --- resolve nested sub-configs in reconstruction -----------------------
    # z_denoising may arrive as a plain dict from YAML; convert to
    # the frozen dataclass before constructing ReconstructionConfig.
    if "z_denoising" in rec_kwargs and isinstance(rec_kwargs["z_denoising"], dict):
        rec_kwargs["z_denoising"] = ZDenoisingConfig(
            **_filter_fields(ZDenoisingConfig, rec_kwargs["z_denoising"])
        )

    # --- construct & freeze -----------------------------------------------
    # Apply _filter_fields() to all stage configs and the top-level config.
    # This provides strict reject: unknown fields raise ValueError, and
    # known renamed fields produce a "did you mean?" hint.
    # top_kwargs is filtered before passing to PipelineConfig since run_id
    # and output_dir have already been popped out above.
    config = PipelineConfig(
        run_id=resolved_run_id,
        output_dir=resolved_output_dir,
        detection=DetectionConfig(**_filter_fields(DetectionConfig, det_kwargs)),
        pose=PoseConfig(**_filter_fields(PoseConfig, pose_kwargs)),
        association=AssociationConfig(
            **_filter_fields(AssociationConfig, assoc_kwargs)
        ),
        tracking=TrackingConfig(**_filter_fields(TrackingConfig, trk_kwargs)),
        reconstruction=ReconstructionConfig(
            **_filter_fields(ReconstructionConfig, rec_kwargs)
        ),
        synthetic=SyntheticConfig(**_filter_fields(SyntheticConfig, syn_kwargs)),
        lut=LutConfig(**_filter_fields(LutConfig, lut_kwargs)),
        **_filter_fields(PipelineConfig, top_kwargs),
    )

    if config.chunk_size is not None and 0 < config.chunk_size < 100:
        logger.warning(
            "chunk_size=%d is less than 100 — insufficient temporal evidence for reliable association scoring",
            config.chunk_size,
        )

    return config


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
