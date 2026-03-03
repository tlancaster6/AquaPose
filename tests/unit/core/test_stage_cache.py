"""Unit tests for StaleCacheError, load_stage_cache, and context_fingerprint."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from aquapose.core.context import (
    CarryForward,
    PipelineContext,
    StaleCacheError,
    context_fingerprint,
    load_stage_cache,
)


def _make_envelope(ctx: PipelineContext, **kwargs: object) -> dict:
    """Build a minimal valid envelope dict wrapping ctx."""
    return {
        "run_id": "test-run",
        "timestamp": "2026-01-01T00:00:00Z",
        "stage_name": "detection",
        "version_fingerprint": context_fingerprint(ctx),
        "context": ctx,
        **kwargs,
    }


def test_load_stage_cache_round_trip(tmp_path: Path) -> None:
    """Round-trip: write envelope pickle, load it, check field values."""
    ctx = PipelineContext(frame_count=5, detections=[{} for _ in range(5)])
    envelope = _make_envelope(ctx)
    cache_file = tmp_path / "detection_cache.pkl"
    cache_file.write_bytes(pickle.dumps(envelope))

    loaded = load_stage_cache(cache_file)

    assert loaded.frame_count == 5
    assert len(loaded.detections) == 5  # type: ignore[arg-type]


def test_load_stage_cache_stale_cache_error(tmp_path: Path) -> None:
    """Non-dict pickle content raises StaleCacheError."""
    cache_file = tmp_path / "stale_cache.pkl"
    # Write a valid pickle of a plain string — not a dict
    cache_file.write_bytes(pickle.dumps("this is not an envelope"))

    with pytest.raises(StaleCacheError):
        load_stage_cache(cache_file)


def test_load_stage_cache_invalid_envelope(tmp_path: Path) -> None:
    """A raw PipelineContext (not in envelope) raises StaleCacheError mentioning 'envelope'."""
    ctx = PipelineContext(frame_count=3)
    cache_file = tmp_path / "raw_ctx.pkl"
    cache_file.write_bytes(pickle.dumps(ctx))

    with pytest.raises(StaleCacheError, match="envelope"):
        load_stage_cache(cache_file)


def test_load_stage_cache_shape_mismatch(tmp_path: Path) -> None:
    """frame_count vs len(detections) mismatch raises StaleCacheError."""
    ctx = PipelineContext(frame_count=5, detections=[{} for _ in range(3)])
    envelope = _make_envelope(ctx)
    cache_file = tmp_path / "mismatch_cache.pkl"
    cache_file.write_bytes(pickle.dumps(envelope))

    with pytest.raises(StaleCacheError, match="frame_count") as exc_info:
        load_stage_cache(cache_file)

    assert "len(detections)" in str(exc_info.value)


def test_load_stage_cache_file_not_found() -> None:
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_stage_cache("nonexistent_path_that_does_not_exist.pkl")


def test_context_fingerprint_stable() -> None:
    """context_fingerprint returns the same value on repeated calls."""
    ctx = PipelineContext()
    fp1 = context_fingerprint(ctx)
    fp2 = context_fingerprint(ctx)
    assert fp1 == fp2
    assert len(fp1) == 12


def test_carry_forward_field_exists() -> None:
    """PipelineContext.carry_forward defaults to None; can be set."""
    ctx_default = PipelineContext()
    assert ctx_default.carry_forward is None

    ctx_with_carry = PipelineContext(carry_forward=CarryForward())
    assert ctx_with_carry.carry_forward is not None
