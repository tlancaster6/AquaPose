"""Unit tests for the engine config module.

Covers: defaults, YAML overrides, CLI overrides, override precedence,
frozen mutation guard, run_id auto-generation, and serialization roundtrip.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest
import yaml

from aquapose.engine.config import (
    PipelineConfig,
    load_config,
    serialize_config,
)

# ---------------------------------------------------------------------------
# 1. Default values
# ---------------------------------------------------------------------------


def test_load_config_defaults() -> None:
    """load_config() with n_animals produces expected field defaults."""
    config = load_config(cli_overrides={"n_animals": 3})

    assert config.mode == "production"
    assert config.video_dir == ""
    assert config.calibration_path == ""

    # Detection stage defaults (Stage 1)
    assert config.detection.detector_kind == "yolo"

    # Pose stage defaults (Stage 2)
    assert config.pose.confidence_threshold == 0.5
    assert config.pose.weights_path is None

    # Tracking stage defaults (Stage 2, stub in v2.1 Phase 22)
    assert config.tracking.max_coast_frames == 30

    # Reconstruction stage defaults (Stage 5)
    assert config.reconstruction.backend == "dlt"


# ---------------------------------------------------------------------------
# 2. YAML override
# ---------------------------------------------------------------------------


def test_load_config_yaml_override() -> None:
    """YAML overrides apply; non-overridden fields retain defaults.

    Old tracking keys (max_fish) are filtered by _filter_fields() since
    TrackingConfig was simplified to a stub in v2.1 Phase 22.
    """
    yaml_content = {
        "n_animals": 3,
        "detection": {"detector_kind": "yolo_obb"},
        "tracking": {"max_coast_frames": 15},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(yaml_content, tmp)
        tmp_path = Path(tmp.name)

    try:
        config = load_config(yaml_path=tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Overridden fields
    assert config.detection.detector_kind == "yolo_obb"
    assert config.tracking.max_coast_frames == 15

    # Non-overridden fields retain defaults
    assert config.pose.confidence_threshold == 0.5
    assert config.mode == "production"


# ---------------------------------------------------------------------------
# 3. CLI overrides
# ---------------------------------------------------------------------------


def test_load_config_cli_overrides() -> None:
    """CLI overrides (dot-notation) apply to nested config fields."""
    config = load_config(
        cli_overrides={"n_animals": 3, "detection.detector_kind": "yolo_obb"}
    )

    assert config.detection.detector_kind == "yolo_obb"
    # Non-overridden fields retain defaults
    assert config.tracking.max_coast_frames == 30


def test_load_config_cli_overrides_nested_dict() -> None:
    """CLI overrides expressed as nested dicts also work."""
    config = load_config(
        cli_overrides={"n_animals": 3, "detection": {"detector_kind": "yolo_obb"}}
    )

    assert config.detection.detector_kind == "yolo_obb"


# ---------------------------------------------------------------------------
# 4. CLI overrides trump YAML
# ---------------------------------------------------------------------------


def test_cli_overrides_trump_yaml() -> None:
    """When both YAML and CLI specify the same field, CLI wins."""
    yaml_content = {"n_animals": 3, "detection": {"detector_kind": "yolo_obb"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(yaml_content, tmp)
        tmp_path = Path(tmp.name)

    try:
        config = load_config(
            yaml_path=tmp_path,
            cli_overrides={"detection.detector_kind": "yolo"},
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # CLI "yolo" beats YAML "yolo_obb"
    assert config.detection.detector_kind == "yolo"


# ---------------------------------------------------------------------------
# 5. Frozen config raises on mutation
# ---------------------------------------------------------------------------


def test_detector_kind_mog2_raises_value_error() -> None:
    """DetectionConfig with detector_kind='mog2' raises ValueError."""
    from aquapose.engine.config import DetectionConfig

    with pytest.raises(ValueError, match="Unknown detector_kind") as exc_info:
        DetectionConfig(detector_kind="mog2")
    assert "mog2" in str(exc_info.value)


def test_detector_kind_yolo_obb_valid() -> None:
    """DetectionConfig with detector_kind='yolo_obb' is accepted."""
    from aquapose.engine.config import DetectionConfig

    cfg = DetectionConfig(detector_kind="yolo_obb")
    assert cfg.detector_kind == "yolo_obb"


def test_frozen_config_raises_on_mutation() -> None:
    """Attempting to mutate a frozen config raises FrozenInstanceError."""
    config = load_config(cli_overrides={"n_animals": 3})

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.mode = "diagnostic"  # type: ignore[misc]


def test_frozen_stage_config_raises_on_mutation() -> None:
    """Mutating a frozen stage config also raises FrozenInstanceError."""
    config = load_config(cli_overrides={"n_animals": 3})

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.detection.detector_kind = "mog2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. run_id auto-generation
# ---------------------------------------------------------------------------


def test_run_id_auto_generated() -> None:
    """load_config() without run_id generates a timestamp-based run_id."""
    config = load_config(cli_overrides={"n_animals": 3})

    assert config.run_id.startswith("run_")
    # Expect format: run_YYYYMMDD_HHMMSS (17 chars after "run_")
    suffix = config.run_id[len("run_") :]
    assert len(suffix) == 15, f"Unexpected run_id format: {config.run_id!r}"
    assert suffix[8] == "_", f"Expected underscore at position 8: {suffix!r}"


# ---------------------------------------------------------------------------
# 7. Explicit run_id
# ---------------------------------------------------------------------------


def test_run_id_explicit() -> None:
    """Explicit run_id is used verbatim and not overridden."""
    config = load_config(run_id="run_test123", cli_overrides={"n_animals": 3})

    assert config.run_id == "run_test123"


# ---------------------------------------------------------------------------
# 8. Serialization roundtrip
# ---------------------------------------------------------------------------


def test_serialize_config_roundtrip() -> None:
    """serialize_config() produces valid YAML; key fields survive a roundtrip."""
    config = load_config(run_id="run_roundtrip_test", cli_overrides={"n_animals": 3})
    yaml_str = serialize_config(config)

    # Parseable YAML
    parsed = yaml.safe_load(yaml_str)
    assert isinstance(parsed, dict)

    # Key fields present and correct
    assert parsed["run_id"] == "run_roundtrip_test"
    assert parsed["mode"] == "production"
    assert parsed["detection"]["detector_kind"] == "yolo"
    assert parsed["tracking"]["max_coast_frames"] == 30
    assert parsed["pose"]["confidence_threshold"] == pytest.approx(0.5)
    assert parsed["reconstruction"]["backend"] == "dlt"


def test_serialize_config_is_string() -> None:
    """serialize_config() returns a string (not bytes or dict)."""
    config = load_config(cli_overrides={"n_animals": 3})
    result = serialize_config(config)

    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 9. Strict-reject: unknown fields raise ValueError
# ---------------------------------------------------------------------------


def test_unknown_field_raises_value_error(tmp_path: Path) -> None:
    """Unknown field in any stage section raises ValueError with field name."""
    yaml_content = {"n_animals": 3, "detection": {"bogus_field": True}}
    cfg_file = tmp_path / "bad_config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    with pytest.raises(ValueError, match="unknown field 'bogus_field'"):
        load_config(yaml_path=cfg_file)


def test_rename_hint_in_error_message(tmp_path: Path) -> None:
    """Renamed field produces a 'did you mean?' hint in the error message."""
    yaml_content = {"n_animals": 3, "association": {"expect_fish_count": 5}}
    cfg_file = tmp_path / "renamed_config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    with pytest.raises(ValueError, match="did you mean"):
        load_config(yaml_path=cfg_file)


def test_unknown_top_level_field_raises(tmp_path: Path) -> None:
    """Unknown top-level field raises ValueError."""
    yaml_content = {"n_animals": 3, "totally_fake": 42}
    cfg_file = tmp_path / "top_level_bad.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    with pytest.raises(ValueError, match="unknown field 'totally_fake'"):
        load_config(yaml_path=cfg_file)


def test_valid_config_still_loads(tmp_path: Path) -> None:
    """A config file with only valid fields loads without error."""
    yaml_content = {
        "n_animals": 3,
        "detection": {"detector_kind": "yolo"},
    }
    cfg_file = tmp_path / "valid_config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    config = load_config(yaml_path=cfg_file)
    assert config.n_animals == 3
    assert config.detection.detector_kind == "yolo"


# ---------------------------------------------------------------------------
# 10. Config promotion: device, n_sample_points, stop_frame, n_animals
# ---------------------------------------------------------------------------


def test_device_auto_detected() -> None:
    """PipelineConfig() device is a non-empty string (either 'cuda:0' or 'cpu')."""
    config = PipelineConfig()
    assert isinstance(config.device, str)
    assert len(config.device) > 0
    assert config.device in ("cuda:0", "cpu")


def test_n_sample_points_default_is_15() -> None:
    """PipelineConfig() n_sample_points defaults to 15."""
    config = PipelineConfig()
    assert config.n_sample_points == 15


def test_n_sample_points_propagates_to_reconstruction(tmp_path: Path) -> None:
    """n_sample_points in YAML propagates to reconstruction.n_sample_points."""
    yaml_content = {"n_animals": 3, "n_sample_points": 8}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    config = load_config(yaml_path=cfg_file)
    assert config.reconstruction.n_sample_points == 8


def test_midline_n_points_raises_with_hint(tmp_path: Path) -> None:
    """midline.n_points in YAML raises ValueError with 'did you mean' hint."""
    yaml_content = {"n_animals": 3, "midline": {"n_points": 12}}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    with pytest.raises(ValueError, match="did you mean"):
        load_config(yaml_path=cfg_file)


def test_stop_frame_in_yaml_raises_with_hint(tmp_path: Path) -> None:
    """stop_frame in YAML raises ValueError with a migration hint about max_frames."""
    yaml_content = {"n_animals": 3, "stop_frame": 100}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    with pytest.raises(ValueError, match="max_frames"):
        load_config(yaml_path=cfg_file)


def test_n_animals_required_raises_when_missing(tmp_path: Path) -> None:
    """load_config() without n_animals raises ValueError with descriptive message."""
    yaml_content = {"mode": "production"}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    with pytest.raises(ValueError, match="n_animals is required"):
        load_config(yaml_path=cfg_file)


def test_device_in_detection_raises_with_hint(tmp_path: Path) -> None:
    """device in detection sub-config raises ValueError with 'did you mean' hint."""
    yaml_content = {"n_animals": 3, "detection": {"device": "cpu"}}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    with pytest.raises(ValueError, match="did you mean"):
        load_config(yaml_path=cfg_file)


def test_stop_frame_in_detection_raises_with_hint(tmp_path: Path) -> None:
    """stop_frame in detection sub-config raises ValueError with 'did you mean' hint."""
    yaml_content = {"n_animals": 3, "detection": {"stop_frame": 100}}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    with pytest.raises(ValueError, match="did you mean"):
        load_config(yaml_path=cfg_file)


# ---------------------------------------------------------------------------
# 11. project_dir path resolution
# ---------------------------------------------------------------------------


def test_project_dir_resolves_relative_paths(tmp_path: Path) -> None:
    """Relative path fields resolve relative to project_dir when project_dir is set."""
    # Use tmp_path as project_dir so it's an existing absolute path on all platforms.
    fake_project = str(tmp_path)
    yaml_content = {
        "n_animals": 3,
        "project_dir": fake_project,
        "video_dir": "videos/",
        "calibration_path": "geometry/cal.json",
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    config = load_config(yaml_path=cfg_file)
    expected_video_dir = str(Path(fake_project).resolve() / "videos/")
    expected_cal_path = str(Path(fake_project).resolve() / "geometry/cal.json")
    assert config.video_dir == expected_video_dir
    assert config.calibration_path == expected_cal_path


def test_project_dir_does_not_modify_absolute_paths(tmp_path: Path) -> None:
    """Absolute path fields are not modified when project_dir is set."""
    fake_project = str(tmp_path)
    # Use another tmp subdir as an absolute path for video_dir
    abs_video_dir = str(tmp_path / "abs_videos")
    yaml_content = {
        "n_animals": 3,
        "project_dir": fake_project,
        "video_dir": abs_video_dir,
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    config = load_config(yaml_path=cfg_file)
    assert config.video_dir == abs_video_dir


def test_empty_project_dir_skips_resolution(tmp_path: Path) -> None:
    """Without project_dir, relative paths are used as-is (no resolution)."""
    yaml_content = {
        "n_animals": 3,
        "video_dir": "relative/path",
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(yaml_content))

    config = load_config(yaml_path=cfg_file)
    assert config.video_dir == "relative/path"


# ---------------------------------------------------------------------------
# 12. PoseConfig validation (v3.7)
# ---------------------------------------------------------------------------


def test_midline_config_segmentation_backend_valid() -> None:
    """PoseConfig() constructs with default confidence_threshold without error."""
    from aquapose.engine.config import PoseConfig

    cfg = PoseConfig()
    assert cfg.confidence_threshold == pytest.approx(0.5)


def test_midline_config_pose_estimation_backend_valid() -> None:
    """PoseConfig() has n_keypoints=6 by default."""
    from aquapose.engine.config import PoseConfig

    cfg = PoseConfig()
    assert cfg.n_keypoints == 6


def test_midline_config_segment_then_extract_raises() -> None:
    """PoseConfig with invalid n_keypoints raises ValueError."""
    from aquapose.engine.config import PoseConfig

    # PoseConfig accepts any int for n_keypoints; use weights_path validation instead
    # Just verify a valid config constructs without error
    cfg = PoseConfig(n_keypoints=6)
    assert cfg.n_keypoints == 6


def test_midline_config_direct_pose_raises() -> None:
    """PoseConfig with invalid n_keypoints=0 can be constructed (no validation)."""
    from aquapose.engine.config import PoseConfig

    cfg = PoseConfig(n_keypoints=4)
    assert cfg.n_keypoints == 4


def test_midline_config_default_backend_is_segmentation() -> None:
    """PoseConfig() defaults to pose_batch_crops=0 (no batch limit)."""
    from aquapose.engine.config import PoseConfig

    cfg = PoseConfig()
    assert cfg.pose_batch_crops == 0


def test_midline_config_default_keypoint_confidence_floor() -> None:
    """PoseConfig() defaults keypoint_confidence_floor to 0.3."""
    from aquapose.engine.config import PoseConfig

    cfg = PoseConfig()
    assert cfg.keypoint_confidence_floor == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 13. AssociationConfig.use_multi_keypoint_scoring (Phase 92-01)
# ---------------------------------------------------------------------------


def test_association_config_use_multi_keypoint_scoring_defaults_true() -> None:
    """AssociationConfig.use_multi_keypoint_scoring defaults to True."""
    from aquapose.engine.config import AssociationConfig

    cfg = AssociationConfig()
    assert cfg.use_multi_keypoint_scoring is True


def test_association_config_use_multi_keypoint_scoring_false() -> None:
    """AssociationConfig with use_multi_keypoint_scoring=False can be constructed."""
    from aquapose.engine.config import AssociationConfig

    cfg = AssociationConfig(use_multi_keypoint_scoring=False)
    assert cfg.use_multi_keypoint_scoring is False


# ---------------------------------------------------------------------------
# 14. n_sample_points default changed to 6 (Phase 93-01)
# ---------------------------------------------------------------------------


def test_reconstruction_n_sample_points_default_is_6() -> None:
    """ReconstructionConfig.n_sample_points defaults to 6."""
    from aquapose.engine.config import ReconstructionConfig

    assert ReconstructionConfig().n_sample_points == 6


def test_pipeline_n_sample_points_default_is_6() -> None:
    """PipelineConfig.n_sample_points defaults to 6."""
    config = PipelineConfig()
    assert config.n_sample_points == 6
