"""Unit tests for PosePipeline orchestration, event emission, and config artifact."""

from __future__ import annotations

from pathlib import Path

import yaml

from aquapose.core.context import PipelineContext
from aquapose.engine import (
    Event,
    PipelineComplete,
    PipelineFailed,
    PipelineStart,
    PosePipeline,
    StageComplete,
    StageStart,
    load_config,
)

# ---------------------------------------------------------------------------
# Test helpers / fixtures
# ---------------------------------------------------------------------------


class MockStage:
    """Simple stage conforming to the Stage protocol via structural typing.

    Records its own name to a list stored in ``context.stage_timing`` under the
    special key "_order" (a list of names) so tests can inspect execution order
    without importing domain types.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, context: PipelineContext) -> PipelineContext:
        """Append self.name to context.stage_timing['_order'], return context."""
        order = context.stage_timing.setdefault("_order", [])
        order.append(self.name)  # type: ignore[arg-type]
        return context


class FailingStage:
    """Stage that always raises RuntimeError."""

    def run(self, context: PipelineContext) -> PipelineContext:
        """Raise RuntimeError immediately."""
        raise RuntimeError("boom")


class RecordingObserver:
    """Observer that records every received event in order."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def on_event(self, event: Event) -> None:
        """Append *event* to internal event list."""
        self.events.append(event)


class ConfigCheckStage:
    """Stage that asserts the config.yaml artifact already exists when it runs."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.config_existed: bool = False

    def run(self, context: PipelineContext) -> PipelineContext:
        """Check config.yaml existence and record result."""
        self.config_existed = (self.output_dir / "config.yaml").exists()
        return context


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_runs_stages_in_order(tmp_path: Path) -> None:
    """Stages execute in the order A, B, C."""
    config = load_config(run_id="test_run", cli_overrides={"output_dir": str(tmp_path)})
    stages = [MockStage("A"), MockStage("B"), MockStage("C")]
    pipeline = PosePipeline(stages=stages, config=config)
    context = pipeline.run()

    assert context.stage_timing["_order"] == ["A", "B", "C"]


def test_pipeline_emits_lifecycle_events(tmp_path: Path) -> None:
    """Observer receives PipelineStart, StageStart, StageComplete, PipelineComplete."""
    config = load_config(run_id="test_run", cli_overrides={"output_dir": str(tmp_path)})
    observer = RecordingObserver()
    pipeline = PosePipeline(
        stages=[MockStage("only")],
        config=config,
        observers=[observer],
    )
    pipeline.run()

    types = [type(e) for e in observer.events]
    assert types == [PipelineStart, StageStart, StageComplete, PipelineComplete]


def test_pipeline_writes_config_artifact(tmp_path: Path) -> None:
    """config.yaml written to output_dir with expected run_id and mode fields."""
    config = load_config(run_id="test_run", cli_overrides={"output_dir": str(tmp_path)})
    pipeline = PosePipeline(stages=[MockStage("x")], config=config)
    pipeline.run()

    config_file = tmp_path / "config.yaml"
    assert config_file.exists(), "config.yaml must be written to output_dir"

    parsed = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert parsed["run_id"] == "test_run"
    assert "mode" in parsed


def test_config_artifact_written_before_stages(tmp_path: Path) -> None:
    """config.yaml exists on disk when the first stage executes."""
    config = load_config(run_id="test_run", cli_overrides={"output_dir": str(tmp_path)})
    checker = ConfigCheckStage(output_dir=tmp_path)
    pipeline = PosePipeline(stages=[checker], config=config)
    pipeline.run()

    assert checker.config_existed, "config.yaml must exist before any stage runs"


def test_pipeline_emits_failed_on_error(tmp_path: Path) -> None:
    """PipelineFailed emitted with correct error message when stage raises."""
    config = load_config(run_id="test_run", cli_overrides={"output_dir": str(tmp_path)})
    observer = RecordingObserver()
    pipeline = PosePipeline(
        stages=[FailingStage()],
        config=config,
        observers=[observer],
    )

    try:
        pipeline.run()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError to be re-raised")

    failed_events = [e for e in observer.events if isinstance(e, PipelineFailed)]
    assert len(failed_events) == 1
    assert "boom" in failed_events[0].error


def test_pipeline_records_stage_timing(tmp_path: Path) -> None:
    """stage_timing dict has positive elapsed entries keyed by stage class name."""

    class FirstStage:
        def run(self, context: PipelineContext) -> PipelineContext:
            return context

    class SecondStage:
        def run(self, context: PipelineContext) -> PipelineContext:
            return context

    config = load_config(run_id="test_run", cli_overrides={"output_dir": str(tmp_path)})
    pipeline = PosePipeline(
        stages=[FirstStage(), SecondStage()],
        config=config,
    )
    context = pipeline.run()

    assert "FirstStage" in context.stage_timing
    assert "SecondStage" in context.stage_timing
    assert context.stage_timing["FirstStage"] >= 0.0
    assert context.stage_timing["SecondStage"] >= 0.0


def test_pipeline_no_observers_still_works(tmp_path: Path) -> None:
    """Pipeline completes normally with no observers attached."""
    config = load_config(run_id="test_run", cli_overrides={"output_dir": str(tmp_path)})
    pipeline = PosePipeline(stages=[MockStage("solo")], config=config)
    context = pipeline.run()

    assert context is not None
    assert "solo" in context.stage_timing["_order"]


def test_pipeline_context_passed_between_stages(tmp_path: Path) -> None:
    """Context accumulates across stages — second stage can read first stage's output."""

    class WriterStage:
        def run(self, context: PipelineContext) -> PipelineContext:
            context.detections = [{"cam1": []}]
            return context

    class ReaderStage:
        def __init__(self) -> None:
            self.saw_detections: bool = False

        def run(self, context: PipelineContext) -> PipelineContext:
            self.saw_detections = context.detections is not None
            return context

    config = load_config(run_id="test_run", cli_overrides={"output_dir": str(tmp_path)})
    reader = ReaderStage()
    pipeline = PosePipeline(stages=[WriterStage(), reader], config=config)
    pipeline.run()

    assert reader.saw_detections, "Second stage must see context set by first stage"
