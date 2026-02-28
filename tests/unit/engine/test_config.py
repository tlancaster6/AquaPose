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

    # Midline stage defaults (Stage 2)
    assert config.midline.confidence_threshold == 0.5
    assert config.midline.weights_path is None

    # Tracking stage defaults (Stage 2, stub in v2.1 Phase 22)
    assert config.tracking.max_coast_frames == 30

    # Reconstruction stage defaults (Stage 5)
    assert config.reconstruction.backend == "triangulation"


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
        "detection": {"detector_kind": "mog2"},
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
    assert config.detection.detector_kind == "mog2"
    assert config.tracking.max_coast_frames == 15

    # Non-overridden fields retain defaults
    assert config.midline.confidence_threshold == 0.5
    assert config.mode == "production"


# ---------------------------------------------------------------------------
# 3. CLI overrides
# ---------------------------------------------------------------------------


def test_load_config_cli_overrides() -> None:
    """CLI overrides (dot-notation) apply to nested config fields."""
    config = load_config(
        cli_overrides={"n_animals": 3, "detection.detector_kind": "mog2"}
    )

    assert config.detection.detector_kind == "mog2"
    # Non-overridden fields retain defaults
    assert config.tracking.max_coast_frames == 30


def test_load_config_cli_overrides_nested_dict() -> None:
    """CLI overrides expressed as nested dicts also work."""
    config = load_config(
        cli_overrides={"n_animals": 3, "detection": {"detector_kind": "mog2"}}
    )

    assert config.detection.detector_kind == "mog2"


# ---------------------------------------------------------------------------
# 4. CLI overrides trump YAML
# ---------------------------------------------------------------------------


def test_cli_overrides_trump_yaml() -> None:
    """When both YAML and CLI specify the same field, CLI wins."""
    yaml_content = {"n_animals": 3, "detection": {"detector_kind": "mog2"}}

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

    # CLI "yolo" beats YAML "mog2"
    assert config.detection.detector_kind == "yolo"


# ---------------------------------------------------------------------------
# 5. Frozen config raises on mutation
# ---------------------------------------------------------------------------


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
    assert parsed["midline"]["confidence_threshold"] == pytest.approx(0.5)
    assert parsed["reconstruction"]["backend"] == "triangulation"


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
