"""Tests for the stage-cache programmatic API (load_stage_cache, StaleCacheError).

The --resume-from CLI flag was removed in Phase 53. These tests cover the
programmatic stage-cache functions that remain in aquapose.core.context for
use by evaluation sweeps and other tooling.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from aquapose.core.context import (
    PipelineContext,
    StaleCacheError,
    load_stage_cache,
)
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


# ---------------------------------------------------------------------------
# Task 1: load_stage_cache loads valid cache and returns context
# ---------------------------------------------------------------------------


def test_load_stage_cache_returns_context(tmp_path: Path) -> None:
    """load_stage_cache() returns a PipelineContext from a valid envelope pickle."""
    ctx = _make_context(frame_count=3)
    cache_path = tmp_path / "test_cache.pkl"
    _write_envelope(cache_path, ctx)

    loaded = load_stage_cache(cache_path)

    assert loaded.frame_count == 3


# ---------------------------------------------------------------------------
# Task 2: End-to-end — DiagnosticObserver writes cache, load_stage_cache loads it
# ---------------------------------------------------------------------------


def test_end_to_end_cache_write_and_reload(tmp_path: Path) -> None:
    """DiagnosticObserver writes a chunk cache; load_stage_cache loads it; context intact."""
    from aquapose.engine.events import PipelineComplete

    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)

    # Fire PipelineStart so observer captures run_id
    observer.on_event(PipelineStart(run_id="e2e_run"))

    # Build a context with detections populated
    ctx = _make_context(frame_count=5)

    # Fire StageComplete for DetectionStage (in-memory snapshot)
    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.2,
            context=ctx,
        )
    )

    # Fire PipelineComplete (deferred — no disk writes yet)
    observer.on_event(PipelineComplete(context=ctx))

    # Flush the cache (orchestrator does this after identity stitching)
    observer.flush_cache()

    # Verify cache file was written at the new chunk-aware path
    cache_path = tmp_path / "diagnostics" / "chunk_000" / "cache.pkl"
    assert cache_path.exists(), f"Cache file should exist at {cache_path}"

    # load_stage_cache still works with the new format
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
