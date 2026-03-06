"""Timing observer for per-stage and total pipeline wall-clock profiling."""

from __future__ import annotations

import logging
from pathlib import Path

from aquapose.engine.events import (
    Event,
    PipelineComplete,
    PipelineFailed,
    PipelineStart,
    StageComplete,
)

logger = logging.getLogger(__name__)


class TimingObserver:
    """Records per-stage and total pipeline wall-clock time from lifecycle events.

    Subscribes to StageComplete, PipelineStart, PipelineComplete, and
    PipelineFailed events to build a formatted timing report with per-stage
    breakdown and percentages.

    Args:
        output_path: If set, the timing report is written to this file path
            on pipeline completion.

    Example::

        observer = TimingObserver(output_path="/tmp/timing.txt")
        pipeline = PosePipeline(stages=stages, config=config, observers=[observer])
        context = pipeline.run()
        print(observer.report())
    """

    def __init__(self, output_path: str | Path | None = None) -> None:
        self._output_path = Path(output_path) if output_path is not None else None
        self.stage_times: dict[str, float] = {}
        self.total_time: float | None = None
        self.run_id: str = ""
        self._failed: bool = False

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event and record timing data.

        Args:
            event: The event instance from the pipeline event bus.
        """
        if isinstance(event, PipelineStart):
            self.run_id = event.run_id
        elif isinstance(event, StageComplete):
            self.stage_times[event.stage_name] = event.elapsed_seconds
        elif isinstance(event, PipelineComplete):
            self.total_time = event.elapsed_seconds
            self._failed = False
            self._finalize()
        elif isinstance(event, PipelineFailed):
            self.total_time = event.elapsed_seconds
            self._failed = True
            self._finalize()

    def report(self) -> str:
        """Return a formatted multi-line timing report.

        The report includes a header with the run ID, per-stage rows with
        elapsed time and percentage of total, and a total time row. If the
        pipeline failed, a failure note is appended.

        Returns:
            Formatted timing report string.
        """
        lines: list[str] = []
        lines.append(f"Timing Report — run: {self.run_id}")
        lines.append("=" * 50)

        total = self.total_time if self.total_time and self.total_time > 0 else None

        for stage_name, elapsed in self.stage_times.items():
            pct = f" ({elapsed / total * 100:5.1f}%)" if total else ""
            lines.append(f"  {stage_name:<30s} {elapsed:8.2f}s{pct}")

        lines.append("-" * 50)

        if total is not None:
            lines.append(f"  {'TOTAL':<30s} {total:8.2f}s")
        else:
            lines.append(f"  {'TOTAL':<30s}      N/A")

        if self._failed:
            lines.append("")
            lines.append("  ** Pipeline FAILED — partial timing report **")

        return "\n".join(lines)

    def _finalize(self) -> None:
        """Log the report and optionally write to file."""
        report_text = self.report()
        logger.info("\n%s", report_text)

        if self._output_path is not None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_path.write_text(report_text, encoding="utf-8")
