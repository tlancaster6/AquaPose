"""ConsoleObserver — prints stage-level progress to stderr."""

from __future__ import annotations

import sys

from aquapose.engine.events import (
    Event,
    FrameProcessed,
    PipelineComplete,
    PipelineFailed,
    PipelineStart,
    StageComplete,
)


class ConsoleObserver:
    """Observer that prints human-readable stage progress to stderr.

    Output goes to stderr to keep stdout clean for piping. Stage progress
    lines use the format ``[1/5] DetectionStage... done (12.3s)``.

    Args:
        verbose: When True, emit per-frame detail lines on FrameProcessed events.
        total_stages: Total number of stages in the pipeline (for progress display).
    """

    def __init__(self, verbose: bool = False, total_stages: int = 5) -> None:
        self._verbose = verbose
        self._total_stages = total_stages
        self._output_dir: str = ""

    def on_event(self, event: Event) -> None:
        """Handle a pipeline event by printing progress to stderr.

        Args:
            event: The pipeline event to handle.
        """
        if isinstance(event, PipelineStart):
            config = event.config
            if config is not None and hasattr(config, "output_dir"):
                self._output_dir = config.output_dir  # type: ignore[union-attr]

        elif isinstance(event, StageComplete):
            line = (
                f"[{event.stage_index + 1}/{self._total_stages}] "
                f"{event.stage_name}... done ({event.elapsed_seconds:.1f}s)\n"
            )
            sys.stderr.write(line)
            sys.stderr.flush()

        elif isinstance(event, PipelineComplete):
            sys.stderr.write("\n")
            sys.stderr.write(
                f"Run complete: {self._output_dir} ({event.elapsed_seconds:.1f}s)\n"
            )
            sys.stderr.flush()

        elif isinstance(event, PipelineFailed):
            sys.stderr.write(
                f"Run FAILED after {event.elapsed_seconds:.1f}s: {event.error}\n"
            )
            sys.stderr.flush()

        elif isinstance(event, FrameProcessed) and self._verbose:
            sys.stderr.write(
                f"  frame {event.frame_index + 1}/{event.frame_count} "
                f"({event.stage_name})\n"
            )
            sys.stderr.flush()
