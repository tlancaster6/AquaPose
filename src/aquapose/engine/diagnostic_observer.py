"""Diagnostic observer for capturing intermediate stage outputs in memory."""

from __future__ import annotations

import datetime
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from aquapose.core.context import context_fingerprint
from aquapose.engine.events import Event, PipelineComplete, PipelineStart, StageComplete

logger = logging.getLogger(__name__)

# PipelineContext field names that hold per-frame data (list-typed, indexed by frame).
# detections, annotated_detections, and midlines_3d are all list[...] indexed by frame.
_PER_FRAME_FIELDS = (
    "detections",
    "annotated_detections",
    "midlines_3d",
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
        tracks_2d: Reference to PipelineContext.tracks_2d (dict, not per-frame), or None.
        tracklet_groups: Reference to PipelineContext.tracklet_groups (flat list), or None.
        midlines_3d: Reference to PipelineContext.midlines_3d, or None.
    """

    stage_name: str = ""
    stage_index: int = 0
    elapsed_seconds: float = 0.0
    frame_count: int | None = None
    camera_ids: list[str] | None = None
    detections: list | None = None
    annotated_detections: list | None = None
    tracks_2d: dict | None = None
    tracklet_groups: list | None = None
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

    def __init__(
        self,
        output_dir: str | Path | None = None,
    ) -> None:
        self.stages: dict[str, StageSnapshot] = {}
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._run_id: str = ""

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event and capture stage snapshots.

        Args:
            event: The event instance from the pipeline event bus.
        """
        if isinstance(event, PipelineStart):
            self._run_id = event.run_id
            return

        if isinstance(event, PipelineComplete):
            self._on_pipeline_complete()
            return

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
            tracks_2d=getattr(context, "tracks_2d", None),
            tracklet_groups=getattr(context, "tracklet_groups", None),
            midlines_3d=getattr(context, "midlines_3d", None),
        )

        self.stages[event.stage_name] = snapshot

        if self._output_dir is not None:
            self._write_stage_cache(event, context)

    def _write_stage_cache(self, event: StageComplete, context: object) -> None:
        """Write a pickle cache file for the completed stage.

        The cache is stored as a metadata envelope dict containing run_id,
        timestamp, stage_name, version_fingerprint, and the full PipelineContext
        snapshot at that stage.

        Args:
            event: The StageComplete event carrying stage name and index.
            context: The PipelineContext after this stage has run.
        """
        diagnostics_dir = self._output_dir / "diagnostics"  # type: ignore[operator]
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        # Normalize stage name for filename: "DetectionStage" -> "detection"
        stage_key = event.stage_name.removesuffix("Stage").lower()
        cache_path = diagnostics_dir / f"{stage_key}_cache.pkl"

        envelope = {
            "run_id": self._run_id,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "stage_name": event.stage_name,
            "version_fingerprint": context_fingerprint(context),  # type: ignore[arg-type]
            "context": context,
        }

        cache_path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
        logger.info("Stage cache written: %s", cache_path)

    def _on_pipeline_complete(self) -> None:
        """No-op hook called when the pipeline completes. Reserved for future extensibility."""
        pass
