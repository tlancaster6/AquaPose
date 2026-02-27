"""Unit tests for DiagnosticObserver."""

from __future__ import annotations

from aquapose.core.context import PipelineContext
from aquapose.engine.diagnostic_observer import DiagnosticObserver
from aquapose.engine.events import StageComplete
from aquapose.engine.observers import Observer


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
    ctx2.annotated_detections = [[{"cam1": [{"midline": [1, 2]}]}]]
    ctx2.frame_count = 1

    observer.on_event(
        StageComplete(
            stage_name="MidlineStage",
            stage_index=1,
            elapsed_seconds=0.7,
            context=ctx2,
        )
    )

    assert len(observer.stages) == 2
    assert "DetectionStage" in observer.stages
    assert "MidlineStage" in observer.stages
    assert observer.stages["MidlineStage"].annotated_detections is not None


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
    """Five StageComplete events produce five snapshot entries (v2.1 stage names)."""
    observer = DiagnosticObserver()
    stage_names = [
        "DetectionStage",
        "TrackingStubStage",
        "AssociationStubStage",
        "MidlineStage",
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
