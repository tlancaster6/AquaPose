"""Diagnostic observer for capturing intermediate stage outputs in memory."""

from __future__ import annotations

import datetime
import json
import logging
import os
import pickle
import tempfile
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

    On PipelineComplete, writes a single ``cache.pkl`` under
    ``diagnostics/chunk_{chunk_idx:03d}/`` containing the full PipelineContext
    from the last stage. Also writes/appends ``diagnostics/manifest.json``
    with a summary entry for this chunk.

    The in-memory ``stages`` dict is retained for interactive Jupyter exploration.

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
        chunk_idx: int = 0,
        chunk_start: int = 0,
    ) -> None:
        """Initialize the DiagnosticObserver.

        Args:
            output_dir: Directory where diagnostics are written. When None,
                no files are written (in-memory-only mode).
            chunk_idx: Zero-based chunk index used to name the cache directory
                (``diagnostics/chunk_NNN/``). Default 0.
            chunk_start: Global frame index of this chunk's first frame. Written
                to ``manifest.json`` as ``start_frame`` for each chunk entry so
                callers can map chunk data back to the full video timeline.
                Default 0.
        """
        self.stages: dict[str, StageSnapshot] = {}
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._chunk_idx = chunk_idx
        self._chunk_start = chunk_start
        self._run_id: str = ""
        self._last_context: object = None

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event and capture stage snapshots.

        Args:
            event: The event instance from the pipeline event bus.
        """
        if isinstance(event, PipelineStart):
            self._run_id = event.run_id
            return

        if isinstance(event, PipelineComplete):
            self._on_pipeline_complete(event)
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
        # Keep a reference to the most recent context for the chunk cache
        self._last_context = context

    def _write_chunk_cache(self, context: object) -> None:
        """Write a single cache.pkl for this chunk using atomic write.

        Creates ``diagnostics/chunk_{chunk_idx:03d}/cache.pkl`` containing an
        envelope dict with run_id, timestamp, version_fingerprint, and the full
        PipelineContext.

        Args:
            context: The final PipelineContext after all stages have run.
        """
        if self._output_dir is None:
            return

        chunk_dir = self._output_dir / "diagnostics" / f"chunk_{self._chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        cache_path = chunk_dir / "cache.pkl"

        envelope = {
            "run_id": self._run_id,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "version_fingerprint": context_fingerprint(context),  # type: ignore[arg-type]
            "context": context,
        }

        with tempfile.NamedTemporaryFile(
            dir=chunk_dir, delete=False, suffix=".tmp"
        ) as tmp:
            tmp_path = tmp.name
            pickle.dump(envelope, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        logger.info("Chunk cache written: %s", cache_path)

    def _write_manifest(self, context: object) -> None:
        """Write or append this chunk's entry to diagnostics/manifest.json.

        Reads the existing manifest (if any), adds this chunk's entry, and
        writes back atomically.

        Args:
            context: The final PipelineContext for this chunk.
        """
        if self._output_dir is None:
            return

        diagnostics_dir = self._output_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = diagnostics_dir / "manifest.json"

        # Load existing manifest or create fresh
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
        else:
            existing = {}

        frame_count = getattr(context, "frame_count", None)
        chunk_size_val = getattr(context, "chunk_size", None)

        chunk_entry = {
            "index": self._chunk_idx,
            "start_frame": self._chunk_start,
            "end_frame": frame_count,
            "stages_cached": list(self.stages.keys()),
        }

        # Build updated manifest
        chunks: list[dict] = existing.get("chunks", [])
        # Replace entry if same index already present (re-run scenario)
        chunks = [c for c in chunks if c.get("index") != self._chunk_idx]
        chunks.append(chunk_entry)
        # Sort by chunk index for readability
        chunks.sort(key=lambda c: c.get("index", 0))

        manifest: dict = {
            "run_id": self._run_id or existing.get("run_id", ""),
            "total_frames": frame_count,
            "chunk_size": chunk_size_val,
            "version_fingerprint": context_fingerprint(context),  # type: ignore[arg-type]
            "chunks": chunks,
        }

        # Atomic write
        with tempfile.NamedTemporaryFile(
            dir=diagnostics_dir,
            delete=False,
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
        ) as tmp:
            tmp_path = tmp.name
            json.dump(manifest, tmp, indent=2)
        os.replace(tmp_path, manifest_path)
        logger.info("Manifest written: %s", manifest_path)

    def _on_pipeline_complete(self, event: PipelineComplete) -> None:
        """Write chunk cache and update manifest when the pipeline completes.

        Args:
            event: The PipelineComplete event (may carry context).
        """
        if self._output_dir is None:
            return

        # Use event context if available; otherwise fall back to last StageComplete context
        context = getattr(event, "context", None) or self._last_context
        if context is None:
            return

        self._write_chunk_cache(context)
        self._write_manifest(context)
