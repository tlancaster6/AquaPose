"""Integration tests for --resume-from CLI flag and stage-cache round-trip."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import click.testing
import pytest

from aquapose.cli import cli
from aquapose.core.context import PipelineContext, StaleCacheError, load_stage_cache
from aquapose.engine.diagnostic_observer import DiagnosticObserver
from aquapose.engine.events import PipelineStart, StageComplete

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(frame_count: int = 3) -> PipelineContext:
    """Build a minimal PipelineContext with detection outputs populated."""
    ctx = PipelineContext()
    ctx.frame_count = frame_count
    ctx.camera_ids = ["cam1"]
    ctx.detections = [{} for _ in range(frame_count)]
    return ctx


def _write_envelope(path: Path, ctx: PipelineContext) -> None:
    """Write a valid cache envelope pickle to *path*."""
    envelope = {
        "run_id": "test_run",
        "timestamp": "2026-01-01T00:00:00Z",
        "stage_name": "DetectionStage",
        "version_fingerprint": "abc123",
        "context": ctx,
    }
    path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))


def _mock_pipeline_context(tmp_path: Path):
    """Context manager that patches load_config, build_stages, and PosePipeline."""
    mock_config = MagicMock()
    mock_config.output_dir = str(tmp_path / "output")
    mock_config.video_dir = str(tmp_path / "videos")
    mock_config.calibration_path = str(tmp_path / "cal.json")
    mock_config.mode = "production"

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run.return_value = MagicMock()

    return (
        patch("aquapose.cli.load_config", return_value=mock_config),
        patch("aquapose.cli.build_stages", return_value=[MagicMock()]),
        patch("aquapose.cli.PosePipeline", return_value=mock_pipeline_instance),
        mock_pipeline_instance,
    )


# ---------------------------------------------------------------------------
# Task 1: load_stage_cache loads valid cache and returns context
# ---------------------------------------------------------------------------


def test_resume_from_loads_and_returns_context(tmp_path: Path) -> None:
    """load_stage_cache() returns a PipelineContext from a valid envelope pickle."""
    ctx = _make_context(frame_count=3)
    cache_path = tmp_path / "test_cache.pkl"
    _write_envelope(cache_path, ctx)

    loaded = load_stage_cache(cache_path)

    assert loaded.frame_count == 3


# ---------------------------------------------------------------------------
# Task 2: --resume-from with a corrupt file gives ClickException
# ---------------------------------------------------------------------------


def test_resume_from_stale_cache_gives_click_exception(tmp_path: Path) -> None:
    """--resume-from with an unloadable pickle emits a non-zero exit and error message."""
    runner = click.testing.CliRunner()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("mode: production\n")

    corrupt_file = tmp_path / "corrupt_cache.pkl"
    corrupt_file.write_bytes(b"this is not a valid pickle file at all")

    # Mock load_config and build_stages so the corrupt-pickle code path is reached
    p_lc, p_bs, p_pp, _mock_inst = _mock_pipeline_context(tmp_path)
    with p_lc, p_bs, p_pp:
        result = runner.invoke(
            cli,
            ["run", "--config", str(config_file), "--resume-from", str(corrupt_file)],
        )

    assert result.exit_code != 0
    # The error message should reference "incompatible" or "re-run"
    assert "incompatible" in result.output or "re-run" in result.output, (
        f"Expected incompatible/re-run in output, got: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# Task 3: --resume-from with nonexistent file gives non-zero exit
# ---------------------------------------------------------------------------


def test_resume_from_nonexistent_file_gives_click_exception(tmp_path: Path) -> None:
    """--resume-from with a path that does not exist causes Click to fail (exists=True)."""
    runner = click.testing.CliRunner()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("mode: production\n")

    nonexistent = tmp_path / "does_not_exist.pkl"
    # nonexistent does NOT exist on disk

    result = runner.invoke(
        cli,
        ["run", "--config", str(config_file), "--resume-from", str(nonexistent)],
    )

    # Click's exists=True on the Path option handles the missing-file check automatically
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Task 4: End-to-end — DiagnosticObserver writes cache, load_stage_cache loads it
# ---------------------------------------------------------------------------


def test_end_to_end_cache_write_and_reload(tmp_path: Path) -> None:
    """DiagnosticObserver writes a cache; load_stage_cache loads it; context is intact."""
    observer = DiagnosticObserver(output_dir=tmp_path)

    # Fire PipelineStart so observer captures run_id
    observer.on_event(PipelineStart(run_id="e2e_run"))

    # Build a context with detections populated
    ctx = _make_context(frame_count=5)

    # Fire StageComplete for DetectionStage to trigger cache write
    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.2,
            context=ctx,
        )
    )

    # Verify cache file was written
    cache_path = tmp_path / "diagnostics" / "detection_cache.pkl"
    assert cache_path.exists(), f"Cache file should exist at {cache_path}"

    # Load the cache and verify context
    loaded = load_stage_cache(cache_path)
    assert loaded.frame_count == 5
    assert loaded.detections is not None
    assert len(loaded.detections) == 5

    # Verify that a pipeline using loaded context would skip DetectionStage
    # (by checking that its output fields are all populated in the context)
    from aquapose.engine.pipeline import _STAGE_OUTPUT_FIELDS

    detection_fields = _STAGE_OUTPUT_FIELDS.get("DetectionStage", ())
    all_populated = all(getattr(loaded, f, None) is not None for f in detection_fields)
    assert all_populated, (
        f"All DetectionStage output fields should be populated. "
        f"Fields: {detection_fields!r}"
    )


# ---------------------------------------------------------------------------
# Additional: verify StaleCacheError import from aquapose.core.context
# ---------------------------------------------------------------------------


def test_stale_cache_error_importable() -> None:
    """StaleCacheError and load_stage_cache are importable from aquapose.core.context."""
    # These imports are done at the top of this file; if they fail, the test module
    # would not even load. Assert the types are accessible.
    assert StaleCacheError is not None
    assert load_stage_cache is not None


def test_load_stage_cache_raises_stale_cache_error_on_invalid_envelope(
    tmp_path: Path,
) -> None:
    """load_stage_cache raises StaleCacheError when envelope dict lacks 'context' key."""
    bad_envelope = {"run_id": "x", "timestamp": "y"}  # missing 'context' key
    bad_path = tmp_path / "bad_envelope.pkl"
    bad_path.write_bytes(pickle.dumps(bad_envelope))

    with pytest.raises(StaleCacheError, match="valid envelope"):
        load_stage_cache(bad_path)
