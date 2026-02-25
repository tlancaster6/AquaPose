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
        stop_frame: If set, stop processing after this frame index.
        extra: Catch-all dict for detector-specific kwargs not covered above.
    """

    detector_kind: str = "yolo"
    stop_frame: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclasses forbid normal assignment in __post_init__, but
        # dict is mutable so we just ensure we got a dict (not None).
        if not isinstance(self.extra, dict):
            object.__setattr__(self, "extra", dict(self.extra))


@dataclass(frozen=True)
class SegmentationConfig:
    """Config for the segmentation stage.

    Attributes:
        confidence_threshold: Minimum confidence for mask acceptance.
        weights_path: Path to model weights file (None = use default/pretrained).
    """

    confidence_threshold: float = 0.5
    weights_path: str | None = None


@dataclass(frozen=True)
class TrackingConfig:
    """Config for the tracking stage.

    Attributes:
        max_fish: Maximum number of fish identities to track simultaneously.
    """

    max_fish: int = 9


@dataclass(frozen=True)
class TriangulationConfig:
    """Config for the triangulation stage.

    Placeholder — extended in future plans as triangulation parameters are
    determined.
    """


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
        detection: Detection stage config.
        segmentation: Segmentation stage config.
        tracking: Tracking stage config.
        triangulation: Triangulation stage config.
    """

    run_id: str = dataclasses.field(default="")
    output_dir: str = dataclasses.field(default="")
    video_dir: str = ""
    calibration_path: str = ""
    mode: str = "production"
    detection: DetectionConfig = dataclasses.field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = dataclasses.field(
        default_factory=SegmentationConfig
    )
    tracking: TrackingConfig = dataclasses.field(default_factory=TrackingConfig)
    triangulation: TriangulationConfig = dataclasses.field(
        default_factory=TriangulationConfig
    )


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
    seg_kwargs: dict[str, Any] = {}
    trk_kwargs: dict[str, Any] = {}
    tri_kwargs: dict[str, Any] = {}
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
        seg_kwargs = _merge_stage_config(
            seg_kwargs, yaml_nested.get("segmentation", {})
        )
        trk_kwargs = _merge_stage_config(trk_kwargs, yaml_nested.get("tracking", {}))
        tri_kwargs = _merge_stage_config(
            tri_kwargs, yaml_nested.get("triangulation", {})
        )
        top_kwargs = _merge_stage_config(top_kwargs, yaml_nested.get("__top__", {}))

    # --- layer 3: CLI overrides -------------------------------------------
    if cli_overrides is not None:
        flat_cli = _apply_nested_overrides({}, cli_overrides)
        cli_nested = _build_stage_dict_from_dotted(flat_cli)

        det_kwargs = _merge_stage_config(det_kwargs, cli_nested.get("detection", {}))
        seg_kwargs = _merge_stage_config(seg_kwargs, cli_nested.get("segmentation", {}))
        trk_kwargs = _merge_stage_config(trk_kwargs, cli_nested.get("tracking", {}))
        tri_kwargs = _merge_stage_config(
            tri_kwargs, cli_nested.get("triangulation", {})
        )
        top_kwargs = _merge_stage_config(top_kwargs, cli_nested.get("__top__", {}))

    # --- layer 4: resolve run_id and output_dir ---------------------------
    resolved_run_id = run_id or top_kwargs.pop("run_id", None) or _generate_run_id()
    resolved_output_dir = top_kwargs.pop(
        "output_dir", _default_output_dir(resolved_run_id)
    )

    # --- construct & freeze -----------------------------------------------
    return PipelineConfig(
        run_id=resolved_run_id,
        output_dir=resolved_output_dir,
        detection=DetectionConfig(**det_kwargs),
        segmentation=SegmentationConfig(**seg_kwargs),
        tracking=TrackingConfig(**trk_kwargs),
        triangulation=TriangulationConfig(**tri_kwargs),
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
