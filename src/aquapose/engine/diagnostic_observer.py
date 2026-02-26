"""Diagnostic observer for capturing intermediate stage outputs in memory."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aquapose.engine.events import Event, StageComplete

logger = logging.getLogger(__name__)

# PipelineContext field names that hold per-frame data (list-typed).
_PER_FRAME_FIELDS = (
    "detections",
    "annotated_detections",
    "associated_bundles",
    "tracks",
    "midlines_3d",
)

# PipelineContext field names that hold scalar data.
_SCALAR_FIELDS = (
    "frame_count",
    "camera_ids",
)


@dataclass
class StageSnapshot:
    """Immutable snapshot of PipelineContext state after a stage completes.

    Stores *references* (not deep copies) to PipelineContext fields, relying
    on the freeze-on-populate invariant for correctness.

    Subscript access (``snapshot[frame_idx]``) returns a dict of all non-None
    per-frame fields at that index, enabling convenient exploration in Jupyter
    notebooks.

    Attributes:
        stage_name: Name of the stage that produced this snapshot.
        stage_index: Zero-based position in the pipeline sequence.
        elapsed_seconds: Wall-clock time for this stage.
        frame_count: Number of frames (from PipelineContext), or None.
        camera_ids: Active camera IDs (from PipelineContext), or None.
        detections: Reference to PipelineContext.detections, or None.
        annotated_detections: Reference to PipelineContext.annotated_detections, or None.
        associated_bundles: Reference to PipelineContext.associated_bundles, or None.
        tracks: Reference to PipelineContext.tracks, or None.
        midlines_3d: Reference to PipelineContext.midlines_3d, or None.
    """

    stage_name: str = ""
    stage_index: int = 0
    elapsed_seconds: float = 0.0
    frame_count: int | None = None
    camera_ids: list[str] | None = None
    detections: list | None = None
    annotated_detections: list | None = None
    associated_bundles: list | None = None
    tracks: list | None = None
    midlines_3d: list | None = None

    # Keep a frozen set of per-frame field names for __getitem__.
    _per_frame_fields: tuple[str, ...] = field(
        default=_PER_FRAME_FIELDS, init=False, repr=False, compare=False
    )

    def __getitem__(self, frame_idx: int) -> dict[str, object]:
        """Return a dict of all non-None per-frame fields at *frame_idx*.

        Args:
            frame_idx: Zero-based frame index.

        Returns:
            Dict mapping field name to the value at ``frame_idx``.

        Raises:
            IndexError: If *frame_idx* is out of range for any field.
        """
        result: dict[str, object] = {}
        for name in self._per_frame_fields:
            value = getattr(self, name, None)
            if value is not None and isinstance(value, list):
                result[name] = value[frame_idx]
        return result


class DiagnosticObserver:
    """Captures intermediate stage outputs in memory for post-hoc analysis.

    After each stage completes, the observer takes a snapshot of the
    PipelineContext (by reference, not deep copy) and stores it keyed by
    stage name. All 5 stages are captured without selective filtering.

    Designed for interactive exploration in Jupyter notebooks::

        observer = DiagnosticObserver()
        pipeline = PosePipeline(stages=stages, config=config, observers=[observer])
        context = pipeline.run()

        # Explore detection stage results
        snapshot = observer.stages["DetectionStage"]
        frame_0 = snapshot[0]  # dict of per-frame fields
        print(frame_0["detections"])
    """

    def __init__(self) -> None:
        self.stages: dict[str, StageSnapshot] = {}

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event and capture stage snapshots.

        Args:
            event: The event instance from the pipeline event bus.
        """
        if not isinstance(event, StageComplete):
            return

        context = event.context
        if context is None:
            return

        snapshot = StageSnapshot(
            stage_name=event.stage_name,
            stage_index=event.stage_index,
            elapsed_seconds=event.elapsed_seconds,
            frame_count=getattr(context, "frame_count", None),
            camera_ids=getattr(context, "camera_ids", None),
            detections=getattr(context, "detections", None),
            annotated_detections=getattr(context, "annotated_detections", None),
            associated_bundles=getattr(context, "associated_bundles", None),
            tracks=getattr(context, "tracks", None),
            midlines_3d=getattr(context, "midlines_3d", None),
        )

        self.stages[event.stage_name] = snapshot
