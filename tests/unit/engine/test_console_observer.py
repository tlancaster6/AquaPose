"""Unit tests for ConsoleObserver."""

from __future__ import annotations

import io
from unittest.mock import patch

from aquapose.engine.console_observer import ConsoleObserver
from aquapose.engine.events import (
    FrameProcessed,
    PipelineComplete,
    PipelineFailed,
    PipelineStart,
    StageComplete,
)


class TestConsoleObserver:
    """Tests for ConsoleObserver event handling and output format."""

    def test_stage_complete_prints_progress_line(self) -> None:
        obs = ConsoleObserver(verbose=False, total_stages=5)
        event = StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=12.3,
            summary={},
        )
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            obs.on_event(event)
        output = buf.getvalue()
        assert "[1/5] DetectionStage... done (12.3s)" in output

    def test_stage_complete_second_stage(self) -> None:
        obs = ConsoleObserver(verbose=False, total_stages=5)
        event = StageComplete(
            stage_name="MidlineStage",
            stage_index=1,
            elapsed_seconds=5.7,
            summary={},
        )
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            obs.on_event(event)
        output = buf.getvalue()
        assert "[2/5] MidlineStage... done (5.7s)" in output

    def test_pipeline_complete_prints_summary(self) -> None:
        obs = ConsoleObserver(verbose=False, total_stages=5)
        # Simulate PipelineStart to set output_dir
        start_config = type("Config", (), {"output_dir": "/tmp/test_run"})()
        obs.on_event(PipelineStart(run_id="test", config=start_config))

        event = PipelineComplete(
            run_id="test",
            elapsed_seconds=47.2,
        )
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            obs.on_event(event)
        output = buf.getvalue()
        assert "Run complete: /tmp/test_run (47.2s)" in output

    def test_pipeline_failed_prints_error(self) -> None:
        obs = ConsoleObserver(verbose=False, total_stages=5)
        event = PipelineFailed(
            run_id="test",
            error="Something went wrong",
            elapsed_seconds=10.5,
        )
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            obs.on_event(event)
        output = buf.getvalue()
        assert "Run FAILED after 10.5s: Something went wrong" in output

    def test_verbose_frame_processed_prints_detail(self) -> None:
        obs = ConsoleObserver(verbose=True, total_stages=5)
        event = FrameProcessed(
            stage_name="DetectionStage",
            frame_index=4,
            frame_count=30,
        )
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            obs.on_event(event)
        output = buf.getvalue()
        assert "frame 5/30" in output
        assert "DetectionStage" in output

    def test_non_verbose_frame_processed_silent(self) -> None:
        obs = ConsoleObserver(verbose=False, total_stages=5)
        event = FrameProcessed(
            stage_name="DetectionStage",
            frame_index=4,
            frame_count=30,
        )
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            obs.on_event(event)
        output = buf.getvalue()
        assert output == ""

    def test_pipeline_start_captures_output_dir(self) -> None:
        obs = ConsoleObserver(verbose=False, total_stages=5)
        config = type("Config", (), {"output_dir": "/my/output/dir"})()
        obs.on_event(PipelineStart(run_id="test", config=config))
        assert obs._output_dir == "/my/output/dir"
