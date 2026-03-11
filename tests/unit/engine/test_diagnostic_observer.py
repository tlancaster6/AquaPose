"""Unit tests for DiagnosticObserver."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from aquapose.core.context import PipelineContext, load_stage_cache
from aquapose.engine.diagnostic_observer import DiagnosticObserver
from aquapose.engine.events import PipelineComplete, PipelineStart, StageComplete
from aquapose.engine.observers import Observer

# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_diagnostic_observer_satisfies_protocol() -> None:
    """DiagnosticObserver satisfies the Observer protocol via isinstance check."""
    observer = DiagnosticObserver()
    assert isinstance(observer, Observer)


def test_captures_stage_output() -> None:
    """StageComplete with context populates observer.stages with a snapshot."""
    observer = DiagnosticObserver()
    ctx = PipelineContext()
    ctx.detections = [[{"cam1": [1, 2]}]]
    ctx.frame_count = 1

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.5,
            context=ctx,
        )
    )

    assert "DetectionStage" in observer.stages
    snapshot = observer.stages["DetectionStage"]
    assert snapshot.detections is ctx.detections


def test_captures_multiple_stages() -> None:
    """Multiple StageComplete events populate separate snapshot entries."""
    observer = DiagnosticObserver()

    ctx1 = PipelineContext()
    ctx1.detections = [[{"cam1": [1]}]]
    ctx1.frame_count = 1

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.3,
            context=ctx1,
        )
    )

    ctx2 = PipelineContext()
    ctx2.detections = [[{"cam1": [1]}]]
    ctx2.midlines_3d = [[{}]]
    ctx2.frame_count = 1

    observer.on_event(
        StageComplete(
            stage_name="PoseStage",
            stage_index=1,
            elapsed_seconds=0.7,
            context=ctx2,
        )
    )

    assert len(observer.stages) == 2
    assert "DetectionStage" in observer.stages
    assert "PoseStage" in observer.stages
    assert observer.stages["PoseStage"].detections is not None


def test_snapshot_getitem() -> None:
    """StageSnapshot[frame_idx] returns dict of per-frame fields."""
    observer = DiagnosticObserver()
    ctx = PipelineContext()
    ctx.detections = [{"cam1": [1]}, {"cam1": [2]}]
    ctx.frame_count = 2

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.1,
            context=ctx,
        )
    )

    snapshot = observer.stages["DetectionStage"]
    frame_0 = snapshot[0]
    assert "detections" in frame_0
    assert frame_0["detections"] == {"cam1": [1]}


def test_stores_references_not_copies() -> None:
    """Snapshot fields are the same objects as PipelineContext (identity check)."""
    observer = DiagnosticObserver()
    ctx = PipelineContext()
    ctx.detections = [[{"cam1": [1]}]]
    ctx.frame_count = 1

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.2,
            context=ctx,
        )
    )

    assert observer.stages["DetectionStage"].detections is ctx.detections


def test_skips_if_no_context() -> None:
    """StageComplete without context field does not create a snapshot."""
    observer = DiagnosticObserver()

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.1,
        )
    )

    assert len(observer.stages) == 0


def test_stage_timing_captured() -> None:
    """Snapshot preserves elapsed_seconds from the StageComplete event."""
    observer = DiagnosticObserver()
    ctx = PipelineContext()
    ctx.frame_count = 0

    observer.on_event(
        StageComplete(
            stage_name="TrackingStubStage",
            stage_index=1,
            elapsed_seconds=1.23,
            context=ctx,
        )
    )

    assert observer.stages["TrackingStubStage"].elapsed_seconds == 1.23


def test_all_stages_captured_in_full_sequence() -> None:
    """Five StageComplete events produce five snapshot entries (v3.7 stage names)."""
    observer = DiagnosticObserver()
    stage_names = [
        "DetectionStage",
        "PoseStage",
        "TrackingStage",
        "AssociationStage",
        "ReconstructionStage",
    ]

    ctx = PipelineContext()
    ctx.frame_count = 1
    ctx.detections = [[{}]]

    for i, name in enumerate(stage_names):
        observer.on_event(
            StageComplete(
                stage_name=name,
                stage_index=i,
                elapsed_seconds=float(i) * 0.5,
                context=ctx,
            )
        )

    assert len(observer.stages) == 5
    for name in stage_names:
        assert name in observer.stages


def test_snapshot_has_tracks_2d_and_tracklet_groups_fields() -> None:
    """StageSnapshot has tracks_2d and tracklet_groups fields (v2.1)."""
    observer = DiagnosticObserver()
    ctx = PipelineContext()
    ctx.frame_count = 1
    ctx.tracks_2d = {"cam1": []}
    ctx.tracklet_groups = []

    observer.on_event(
        StageComplete(
            stage_name="TrackingStubStage",
            stage_index=1,
            elapsed_seconds=0.1,
            context=ctx,
        )
    )

    snapshot = observer.stages["TrackingStubStage"]
    assert snapshot.tracks_2d is ctx.tracks_2d
    assert snapshot.tracklet_groups is ctx.tracklet_groups


# ---------------------------------------------------------------------------
# New tests for per-chunk single-cache layout
# ---------------------------------------------------------------------------


def _fire_pipeline(
    observer: DiagnosticObserver, run_id: str = "run_test"
) -> PipelineContext:
    """Helper: fire PipelineStart, one StageComplete, then PipelineComplete."""
    observer.on_event(PipelineStart(run_id=run_id))

    ctx = PipelineContext()
    ctx.frame_count = 5
    ctx.detections = [[{"cam1": []}] for _ in range(5)]

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.1,
            context=ctx,
        )
    )
    observer.on_event(PipelineComplete(context=ctx))
    return ctx


def test_chunk_000_cache_written(tmp_path: Path) -> None:
    """DiagnosticObserver with chunk_idx=0 writes cache to diagnostics/chunk_000/cache.pkl."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)
    _fire_pipeline(observer)

    cache_path = tmp_path / "diagnostics" / "chunk_000" / "cache.pkl"
    assert cache_path.exists(), f"Expected cache at {cache_path}"


def test_chunk_002_cache_written(tmp_path: Path) -> None:
    """DiagnosticObserver with chunk_idx=2 writes cache to diagnostics/chunk_002/cache.pkl."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=2)
    _fire_pipeline(observer)

    cache_path = tmp_path / "diagnostics" / "chunk_002" / "cache.pkl"
    assert cache_path.exists(), f"Expected cache at {cache_path}"


def test_chunk_cache_contains_full_context(tmp_path: Path) -> None:
    """Cache envelope contains full PipelineContext (not per-stage snapshots)."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)
    original_ctx = _fire_pipeline(observer)

    cache_path = tmp_path / "diagnostics" / "chunk_000" / "cache.pkl"
    envelope = pickle.loads(cache_path.read_bytes())

    assert isinstance(envelope, dict)
    assert "context" in envelope
    assert isinstance(envelope["context"], PipelineContext)
    # Should be the same frame_count
    assert envelope["context"].frame_count == original_ctx.frame_count


def test_chunk_cache_envelope_schema(tmp_path: Path) -> None:
    """Cache envelope has run_id, timestamp, version_fingerprint, context keys."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)
    _fire_pipeline(observer, run_id="run_xyz")

    cache_path = tmp_path / "diagnostics" / "chunk_000" / "cache.pkl"
    envelope = pickle.loads(cache_path.read_bytes())

    assert envelope["run_id"] == "run_xyz"
    assert "timestamp" in envelope
    assert "version_fingerprint" in envelope
    assert "context" in envelope


def test_only_one_cache_per_chunk_written_on_pipeline_complete(tmp_path: Path) -> None:
    """Only one cache.pkl per chunk is written (on PipelineComplete, not per-stage)."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)

    observer.on_event(PipelineStart(run_id="run_test"))
    ctx = PipelineContext()
    ctx.frame_count = 3

    # Fire multiple StageComplete events
    for i, name in enumerate(["DetectionStage", "TrackingStage", "AssociationStage"]):
        observer.on_event(
            StageComplete(
                stage_name=name, stage_index=i, elapsed_seconds=0.1, context=ctx
            )
        )

    # Before PipelineComplete: no cache file yet
    chunk_dir = tmp_path / "diagnostics" / "chunk_000"
    pkl_files_before = list(chunk_dir.glob("*.pkl")) if chunk_dir.exists() else []
    assert len(pkl_files_before) == 0, (
        "Cache should not be written before PipelineComplete"
    )

    observer.on_event(PipelineComplete(context=ctx))

    # After PipelineComplete: exactly one cache.pkl
    pkl_files_after = list(chunk_dir.glob("*.pkl"))
    assert len(pkl_files_after) == 1
    assert pkl_files_after[0].name == "cache.pkl"


def test_no_per_stage_cache_files_written(tmp_path: Path) -> None:
    """No per-stage cache files (detection_cache.pkl etc.) are written."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)
    _fire_pipeline(observer)

    diagnostics_dir = tmp_path / "diagnostics"
    # Ensure no old-style *_cache.pkl files exist at the top level
    old_style = list(diagnostics_dir.glob("*_cache.pkl"))
    assert old_style == [], f"Old-style per-stage cache files found: {old_style}"


def test_inmemory_stages_dict_still_works(tmp_path: Path) -> None:
    """In-memory StageSnapshot dict still works for Jupyter exploration."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)
    _fire_pipeline(observer)

    assert "DetectionStage" in observer.stages
    snapshot = observer.stages["DetectionStage"]
    assert snapshot.frame_count == 5


def test_manifest_written_on_pipeline_complete(tmp_path: Path) -> None:
    """manifest.json is written at diagnostics/manifest.json after PipelineComplete."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)
    _fire_pipeline(observer, run_id="run_abc")

    manifest_path = tmp_path / "diagnostics" / "manifest.json"
    assert manifest_path.exists(), "manifest.json should exist after PipelineComplete"


def test_manifest_schema(tmp_path: Path) -> None:
    """manifest.json has run_id, total_frames, chunk_size, chunks, version_fingerprint."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)
    _fire_pipeline(observer, run_id="run_schema")

    manifest_path = tmp_path / "diagnostics" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert "run_id" in manifest
    assert "total_frames" in manifest
    assert "chunk_size" in manifest
    assert "version_fingerprint" in manifest
    assert "chunks" in manifest
    assert isinstance(manifest["chunks"], list)
    assert len(manifest["chunks"]) >= 1


def test_manifest_chunk_entry_schema(tmp_path: Path) -> None:
    """Each chunk entry in manifest.json has index, start_frame, end_frame, stages_cached."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)
    _fire_pipeline(observer, run_id="run_entry")

    manifest_path = tmp_path / "diagnostics" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    chunk_entry = manifest["chunks"][0]
    assert "index" in chunk_entry
    assert "start_frame" in chunk_entry
    assert "end_frame" in chunk_entry
    assert "stages_cached" in chunk_entry
    assert isinstance(chunk_entry["stages_cached"], list)


def test_manifest_appended_for_multiple_chunks(tmp_path: Path) -> None:
    """manifest.json accumulates entries across multiple chunks."""
    for chunk_idx in range(3):
        observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=chunk_idx)
        _fire_pipeline(observer, run_id="run_multi")

    manifest_path = tmp_path / "diagnostics" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert len(manifest["chunks"]) == 3
    indices = [e["index"] for e in manifest["chunks"]]
    assert sorted(indices) == [0, 1, 2]


def test_load_chunk_cache_loads_new_format(tmp_path: Path) -> None:
    """load_chunk_cache loads from the new chunk cache layout."""
    from aquapose.core.context import load_chunk_cache

    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)
    original_ctx = _fire_pipeline(observer)

    cache_path = tmp_path / "diagnostics" / "chunk_000" / "cache.pkl"
    loaded_ctx = load_chunk_cache(cache_path)

    assert isinstance(loaded_ctx, PipelineContext)
    assert loaded_ctx.frame_count == original_ctx.frame_count


def test_load_chunk_cache_warns_on_fingerprint_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """load_chunk_cache logs warning on version fingerprint mismatch but still loads."""
    import logging

    from aquapose.core.context import load_chunk_cache

    # Build a cache with a wrong fingerprint
    ctx = PipelineContext()
    ctx.frame_count = 2
    envelope = {
        "run_id": "run_fp",
        "timestamp": "2026-01-01T00:00:00",
        "version_fingerprint": "000000000000",  # intentionally wrong
        "context": ctx,
    }
    cache_dir = tmp_path / "diagnostics" / "chunk_000"
    cache_dir.mkdir(parents=True)
    (cache_dir / "cache.pkl").write_bytes(pickle.dumps(envelope))

    with caplog.at_level(logging.WARNING, logger="aquapose.core.context"):
        loaded = load_chunk_cache(cache_dir / "cache.pkl")

    assert isinstance(loaded, PipelineContext)
    assert any("fingerprint" in rec.message.lower() for rec in caplog.records)


def test_load_stage_cache_still_works_with_new_format(tmp_path: Path) -> None:
    """load_stage_cache works with the new chunk cache format (backward compat)."""
    observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0)
    original_ctx = _fire_pipeline(observer)

    cache_path = tmp_path / "diagnostics" / "chunk_000" / "cache.pkl"
    loaded_ctx = load_stage_cache(cache_path)

    assert isinstance(loaded_ctx, PipelineContext)
    assert loaded_ctx.frame_count == original_ctx.frame_count
