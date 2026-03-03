"""Unit tests for DiagnosticObserver pickle cache writing."""

from __future__ import annotations

import pickle

import pytest

from aquapose.core.context import PipelineContext, load_stage_cache
from aquapose.engine.diagnostic_observer import DiagnosticObserver
from aquapose.engine.events import PipelineStart, StageComplete


def _make_stage_complete(stage_name: str, context: PipelineContext) -> StageComplete:
    """Helper: build a StageComplete event."""
    return StageComplete(
        stage_name=stage_name,
        stage_index=0,
        elapsed_seconds=1.0,
        summary={},
        context=context,
    )


def test_diagnostic_observer_writes_cache_on_stage_complete(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """DiagnosticObserver writes diagnostics/<stage>_cache.pkl on StageComplete."""
    observer = DiagnosticObserver(output_dir=tmp_path)
    observer.on_event(PipelineStart(run_id="test_run"))

    ctx = PipelineContext()
    ctx.frame_count = 5
    ctx.camera_ids = ["cam1"]
    ctx.detections = [{} for _ in range(5)]

    observer.on_event(_make_stage_complete("DetectionStage", ctx))

    cache_path = tmp_path / "diagnostics" / "detection_cache.pkl"
    assert cache_path.exists(), f"Expected cache file at {cache_path}"

    envelope = pickle.loads(cache_path.read_bytes())
    assert "run_id" in envelope
    assert "timestamp" in envelope
    assert "stage_name" in envelope
    assert "version_fingerprint" in envelope
    assert "context" in envelope

    assert envelope["run_id"] == "test_run"
    assert envelope["stage_name"] == "DetectionStage"


def test_diagnostic_observer_no_cache_without_output_dir() -> None:
    """DiagnosticObserver does NOT write any files when output_dir is None."""
    observer = DiagnosticObserver()  # no output_dir
    observer.on_event(PipelineStart(run_id="no_output_run"))

    ctx = PipelineContext()
    ctx.frame_count = 3
    ctx.camera_ids = ["cam1"]
    ctx.detections = [{} for _ in range(3)]

    # Should not raise and should not write any files
    observer.on_event(_make_stage_complete("DetectionStage", ctx))

    # Verify observer still captured the snapshot in memory
    assert "DetectionStage" in observer.stages


def test_diagnostic_observer_cache_round_trips_with_load_stage_cache(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Cache written by DiagnosticObserver can be loaded by load_stage_cache()."""
    observer = DiagnosticObserver(output_dir=tmp_path)
    observer.on_event(PipelineStart(run_id="round_trip_run"))

    ctx = PipelineContext()
    ctx.frame_count = 2
    ctx.camera_ids = ["cam1", "cam2"]
    ctx.detections = [{} for _ in range(2)]
    ctx.tracks_2d = {"cam1": [], "cam2": []}

    observer.on_event(_make_stage_complete("TrackingStage", ctx))

    cache_path = tmp_path / "diagnostics" / "tracking_cache.pkl"
    assert cache_path.exists()

    loaded_ctx = load_stage_cache(cache_path)
    assert loaded_ctx.frame_count == ctx.frame_count
    assert loaded_ctx.camera_ids == ctx.camera_ids
    assert loaded_ctx.tracks_2d == ctx.tracks_2d


def test_diagnostic_observer_captures_run_id_from_pipeline_start(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """run_id captured from PipelineStart appears in the cache envelope."""
    observer = DiagnosticObserver(output_dir=tmp_path)
    observer.on_event(PipelineStart(run_id="abc"))

    ctx = PipelineContext()
    ctx.tracklet_groups = []

    observer.on_event(_make_stage_complete("AssociationStage", ctx))

    cache_path = tmp_path / "diagnostics" / "association_cache.pkl"
    assert cache_path.exists()

    envelope = pickle.loads(cache_path.read_bytes())
    assert envelope["run_id"] == "abc"
