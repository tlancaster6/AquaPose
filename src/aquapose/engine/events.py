"""Typed event dataclasses for the AquaPose pipeline event system.

Events use a 3-tier taxonomy:
- Pipeline lifecycle: PipelineStart, PipelineComplete, PipelineFailed
- Stage lifecycle: StageStart, StageComplete
- Frame-level: FrameProcessed

All events are frozen dataclasses with an auto-populated timestamp field.
Events are the sole communication channel between the pipeline and observers —
observers react to events without mutating pipeline state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    """Base class for all pipeline events.

    All concrete event types inherit from this class so that EventBus can
    match subscriptions by type hierarchy (subscribing to ``Event`` receives
    every event).

    Attributes:
        timestamp: Unix timestamp (seconds) at event construction time.
            Populated automatically via ``field(default_factory=time.time)``.
    """

    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Pipeline lifecycle events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStart(Event):
    """Emitted when the pipeline begins execution.

    Attributes:
        run_id: Unique identifier for this pipeline run (e.g. "run_20260225_143022").
        config: The pipeline configuration object. Typed as ``object`` to avoid
            circular imports between engine and config modules; callers pass
            the actual ``PipelineConfig`` instance.
        timestamp: Unix timestamp at event construction.
    """

    run_id: str = ""
    config: object = field(default=None, compare=False)


@dataclass(frozen=True)
class PipelineComplete(Event):
    """Emitted after all stages complete successfully.

    Attributes:
        run_id: Unique identifier for this pipeline run.
        elapsed_seconds: Wall-clock time for the entire pipeline run.
        timestamp: Unix timestamp at event construction.
    """

    run_id: str = ""
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class PipelineFailed(Event):
    """Emitted when the pipeline terminates due to an unhandled exception.

    Attributes:
        run_id: Unique identifier for this pipeline run.
        error: String representation of the exception that caused the failure.
        elapsed_seconds: Wall-clock time elapsed before failure.
        timestamp: Unix timestamp at event construction.
    """

    run_id: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Stage lifecycle events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageStart(Event):
    """Emitted immediately before a stage begins execution.

    Attributes:
        stage_name: Human-readable name of the stage.
        stage_index: Zero-based position of the stage in the pipeline sequence.
        timestamp: Unix timestamp at event construction.
    """

    stage_name: str = ""
    stage_index: int = 0


@dataclass(frozen=True)
class StageComplete(Event):
    """Emitted after a stage finishes execution.

    Attributes:
        stage_name: Human-readable name of the stage.
        stage_index: Zero-based position of the stage in the pipeline sequence.
        elapsed_seconds: Wall-clock time for this stage.
        summary: Stage-specific metrics dict (e.g. ``{"detection_count": 42}``).
            Values are arbitrary objects; keys are short metric names.
        timestamp: Unix timestamp at event construction.
    """

    stage_name: str = ""
    stage_index: int = 0
    elapsed_seconds: float = 0.0
    summary: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Frame-level events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameProcessed(Event):
    """Emitted by frame-based stages after each frame is processed.

    Allows observers (e.g. progress bars, live visualizers) to track
    per-frame progress without the stage needing to know who is listening.

    Attributes:
        stage_name: Name of the stage emitting this event.
        frame_index: Zero-based index of the frame just processed.
        frame_count: Total number of frames in the sequence.
        timestamp: Unix timestamp at event construction.
    """

    stage_name: str = ""
    frame_index: int = 0
    frame_count: int = 0


__all__ = [
    "Event",
    "FrameProcessed",
    "PipelineComplete",
    "PipelineFailed",
    "PipelineStart",
    "StageComplete",
    "StageStart",
]
