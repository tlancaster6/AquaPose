"""Unit tests for PosePipeline stage-skip logic with initial_context."""

from __future__ import annotations

from pathlib import Path

from aquapose.core.context import ChunkHandoff, PipelineContext
from aquapose.engine import (
    Event,
    PosePipeline,
    StageComplete,
    load_config,
)

# ---------------------------------------------------------------------------
# Stub stages that satisfy the Stage protocol.
# Class names MUST match keys in _STAGE_OUTPUT_FIELDS in pipeline.py so that
# the skip logic can identify them correctly.
# ---------------------------------------------------------------------------


class DetectionStage:
    """Stub detection stage whose class name matches the pipeline stage map."""

    def __init__(self) -> None:
        self.called = False

    def run(self, context: PipelineContext) -> PipelineContext:
        """Populate detection fields and return context."""
        self.called = True
        context.frame_count = 5
        context.camera_ids = ["cam1"]
        context.detections = [{} for _ in range(5)]
        return context


class TrackingStageStub:
    """Stub tracking stage (NOT named TrackingStage to avoid isinstance check)."""

    def __init__(self) -> None:
        self.called = False

    def run(self, context: PipelineContext) -> PipelineContext:
        """Populate tracks_2d and return context."""
        self.called = True
        context.tracks_2d = {"cam1": []}
        return context


class RecordingObserver:
    """Observer that records all received events."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def on_event(self, event: Event) -> None:
        """Append event to internal list."""
        self.events.append(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> object:
    return load_config(
        run_id="test_skip", cli_overrides={"n_animals": 1, "output_dir": str(tmp_path)}
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_skips_populated_stages(tmp_path: Path) -> None:
    """Stages whose output fields are pre-populated are not called."""
    config = _make_config(tmp_path)

    # DetectionStage class name matches _STAGE_OUTPUT_FIELDS key, so skip logic applies.
    stub_detection = DetectionStage()
    # TrackingStageStub class name does NOT match any key, so it always runs.
    stub_tracking = TrackingStageStub()

    # Create an initial context with detection outputs already filled
    initial_ctx = PipelineContext()
    initial_ctx.frame_count = 5
    initial_ctx.camera_ids = ["cam1"]
    initial_ctx.detections = [{} for _ in range(5)]

    pipeline = PosePipeline(
        stages=[stub_detection, stub_tracking],
        config=config,
    )

    pipeline.run(initial_context=initial_ctx)

    # Detection was already populated, so DetectionStage.run() must NOT have been called
    assert not stub_detection.called, "DetectionStage should have been skipped"
    # Tracking was not populated, so stub_tracking.run() must have been called
    assert stub_tracking.called, "TrackingStageStub should have run"


def test_pipeline_emits_skipped_stage_complete(tmp_path: Path) -> None:
    """Skipped stage emits StageComplete with summary={'skipped': True} and elapsed_seconds==0."""
    config = _make_config(tmp_path)
    observer = RecordingObserver()

    # DetectionStage class name matches _STAGE_OUTPUT_FIELDS so skip logic fires.
    stub_a = DetectionStage()

    # Pre-populate detection outputs
    initial_ctx = PipelineContext()
    initial_ctx.frame_count = 3
    initial_ctx.camera_ids = ["cam1"]
    initial_ctx.detections = [{} for _ in range(3)]

    pipeline = PosePipeline(stages=[stub_a], config=config, observers=[observer])
    pipeline.run(initial_context=initial_ctx)

    stage_completes = [e for e in observer.events if isinstance(e, StageComplete)]
    assert len(stage_completes) == 1

    sc = stage_completes[0]
    assert sc.summary == {"skipped": True}, (
        f"Expected skipped summary, got {sc.summary}"
    )
    assert sc.elapsed_seconds == 0.0, f"Expected 0.0 elapsed, got {sc.elapsed_seconds}"
    assert sc.stage_name == "DetectionStage"


def test_pipeline_no_skip_without_initial_context(tmp_path: Path) -> None:
    """Without initial_context, all stages execute normally."""
    config = _make_config(tmp_path)

    stub = DetectionStage()

    pipeline = PosePipeline(stages=[stub], config=config)
    pipeline.run()  # no initial_context

    assert stub.called, "Stage should run when no initial_context is provided"


def test_pipeline_extracts_carry_from_initial_context(tmp_path: Path) -> None:
    """carry_forward from initial_context is extracted and available to the pipeline."""
    config = _make_config(tmp_path)

    carry_state = {"cam1": {"some": "state"}}
    initial_ctx = PipelineContext()
    initial_ctx.carry_forward = ChunkHandoff(
        tracks_2d_state=carry_state,
        identity_map={},
        fish_tracklet_sets={},
        next_global_id=0,
    )

    # Use a stage that records the carry passed to it
    received_carry: list[ChunkHandoff | None] = []

    class CarryCapture:
        def run(self, ctx: PipelineContext) -> PipelineContext:
            # At this point, pipeline should have extracted carry from initial_ctx
            # We can verify by checking that context.carry_forward is set
            received_carry.append(ctx.carry_forward)
            return ctx

    pipeline = PosePipeline(stages=[CarryCapture()], config=config)
    pipeline.run(initial_context=initial_ctx)

    # The carry extracted from initial_context is stored as context.carry_forward
    # before any stage runs (since no tracking stage ran to reset it)
    assert len(received_carry) == 1
    # carry_forward is the one from initial_ctx
    assert received_carry[0] is initial_ctx.carry_forward


def test_carry_forward_injected_after_tracking_stage(tmp_path: Path) -> None:
    """context.carry_forward is set after a real TrackingStage-like stub runs."""
    config = _make_config(tmp_path)

    # We can't import TrackingStage (GPU-dependent), so we verify the carry injection
    # logic by checking that context.carry_forward is set when the stub returns it.
    # The pipeline injects carry into context after isinstance(stage, TrackingStage).
    # Since our stub is NOT a TrackingStage instance, carry injection won't happen here
    # — instead we test it indirectly via the initial_context carry extraction path.

    # Verify: if a stage sets context.carry_forward directly, it persists.
    class CarrySettingStage:
        def run(self, ctx: PipelineContext) -> PipelineContext:
            ctx.carry_forward = ChunkHandoff(
                tracks_2d_state={"cam1": {}},
                identity_map={},
                fish_tracklet_sets={},
                next_global_id=0,
            )
            return ctx

    pipeline = PosePipeline(stages=[CarrySettingStage()], config=config)
    ctx = pipeline.run()

    assert ctx.carry_forward is not None
    assert ctx.carry_forward.tracks_2d_state == {"cam1": {}}
