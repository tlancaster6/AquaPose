"""Unit tests for TimingObserver."""

from __future__ import annotations

from aquapose.engine.events import (
    PipelineComplete,
    PipelineFailed,
    PipelineStart,
    StageComplete,
    StageStart,
)
from aquapose.engine.observers import Observer
from aquapose.engine.timing import TimingObserver


def test_timing_observer_satisfies_protocol() -> None:
    """TimingObserver satisfies the Observer protocol via isinstance check."""
    observer = TimingObserver()
    assert isinstance(observer, Observer)


def test_timing_observer_captures_stage_times() -> None:
    """StageComplete events populate stage_times dict."""
    observer = TimingObserver()
    observer.on_event(
        StageComplete(stage_name="DetectionStage", stage_index=0, elapsed_seconds=1.5)
    )
    observer.on_event(
        StageComplete(stage_name="MidlineStage", stage_index=1, elapsed_seconds=2.3)
    )
    observer.on_event(
        StageComplete(stage_name="TrackingStage", stage_index=2, elapsed_seconds=0.8)
    )

    assert observer.stage_times == {
        "DetectionStage": 1.5,
        "MidlineStage": 2.3,
        "TrackingStage": 0.8,
    }


def test_timing_observer_captures_total_time() -> None:
    """PipelineComplete sets total_time."""
    observer = TimingObserver()
    observer.on_event(PipelineComplete(run_id="test_run", elapsed_seconds=10.5))
    assert observer.total_time == 10.5


def test_timing_report_format() -> None:
    """Full event sequence produces a report with all stage names and total."""
    observer = TimingObserver()
    observer.on_event(PipelineStart(run_id="run_001"))
    observer.on_event(StageStart(stage_name="DetectionStage", stage_index=0))
    observer.on_event(
        StageComplete(stage_name="DetectionStage", stage_index=0, elapsed_seconds=2.0)
    )
    observer.on_event(StageStart(stage_name="MidlineStage", stage_index=1))
    observer.on_event(
        StageComplete(stage_name="MidlineStage", stage_index=1, elapsed_seconds=3.0)
    )
    observer.on_event(StageStart(stage_name="TrackingStage", stage_index=2))
    observer.on_event(
        StageComplete(stage_name="TrackingStage", stage_index=2, elapsed_seconds=5.0)
    )
    observer.on_event(PipelineComplete(run_id="run_001", elapsed_seconds=10.0))

    report = observer.report()
    assert "run_001" in report
    assert "DetectionStage" in report
    assert "MidlineStage" in report
    assert "TrackingStage" in report
    assert "TOTAL" in report
    assert "10.00" in report
    # Check percentage: DetectionStage is 2/10 = 20%
    assert "20.0%" in report


def test_timing_observer_writes_file(tmp_path: object) -> None:
    """When output_path is set, report is written to file on PipelineComplete."""
    from pathlib import Path

    output_file = Path(str(tmp_path)) / "timing_report.txt"
    observer = TimingObserver(output_path=output_file)

    observer.on_event(PipelineStart(run_id="file_test"))
    observer.on_event(
        StageComplete(stage_name="Stage1", stage_index=0, elapsed_seconds=1.0)
    )
    observer.on_event(PipelineComplete(run_id="file_test", elapsed_seconds=1.0))

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "file_test" in content
    assert "Stage1" in content


def test_timing_observer_handles_failure() -> None:
    """PipelineFailed sets total_time and report contains failure indication."""
    observer = TimingObserver()
    observer.on_event(PipelineStart(run_id="fail_run"))
    observer.on_event(
        StageComplete(stage_name="DetectionStage", stage_index=0, elapsed_seconds=1.0)
    )
    observer.on_event(
        PipelineFailed(run_id="fail_run", error="boom", elapsed_seconds=1.5)
    )

    assert observer.total_time == 1.5
    report = observer.report()
    assert "FAILED" in report
    assert "DetectionStage" in report
