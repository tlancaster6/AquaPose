"""ChunkOrchestrator and chunk-mode infrastructure for AquaPose.

ChunkHandoff is defined in aquapose.core.context and re-exported here for
backward compatibility and public API surface (aquapose.engine.ChunkHandoff).
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from aquapose.core.context import ChunkHandoff

if TYPE_CHECKING:
    from aquapose.engine.config import PipelineConfig

logger = logging.getLogger(__name__)


def _stitch_identities(
    tracklet_groups: list,
    prev_handoff: ChunkHandoff | None,
    next_global_id: int,
) -> tuple[dict[int, int], int]:
    """Map chunk-local fish IDs to globally consistent IDs.

    Uses track ID continuity: checks each tracklet's (camera_id, track_id)
    against prev_handoff.track_id_to_global. Majority vote resolves conflicts.

    Args:
        tracklet_groups: List of TrackletGroup from the completed chunk.
        prev_handoff: Handoff from the previous chunk. None for chunk 0.
        next_global_id: Next fresh global ID to assign to unmatched groups.

    Returns:
        (identity_map, updated_next_global_id) where identity_map maps
        local fish_id -> global fish_id for every group in tracklet_groups.
    """
    from collections import Counter

    track_to_global: dict[tuple[str, int], int] = {}
    if prev_handoff is not None:
        track_to_global = dict(prev_handoff.track_id_to_global)

    identity_map: dict[int, int] = {}
    for group in tracklet_groups:
        local_id = group.fish_id
        candidate_global_ids: list[int] = []
        for tracklet in group.tracklets:
            key = (tracklet.camera_id, tracklet.track_id)
            if key in track_to_global:
                candidate_global_ids.append(track_to_global[key])

        if not candidate_global_ids:
            identity_map[local_id] = next_global_id
            next_global_id += 1
        else:
            counts = Counter(candidate_global_ids)
            winner_global_id, winner_count = counts.most_common(1)[0]
            if len(counts) > 1:
                logger.warning(
                    "Identity conflict for local fish %d: matched global IDs %s — "
                    "using majority winner %d (%d/%d tracklets)",
                    local_id,
                    dict(counts),
                    winner_global_id,
                    winner_count,
                    len(candidate_global_ids),
                )
            identity_map[local_id] = winner_global_id

    prev_next = prev_handoff.next_global_id if prev_handoff is not None else 0
    n_new = next_global_id - prev_next
    n_continued = len(identity_map) - n_new
    logger.info(
        "Identity stitching: %d continued, %d new",
        max(0, n_continued),
        n_new,
    )
    return identity_map, next_global_id


def _build_stages_for_chunk(config: object, chunk_source: object) -> list:
    """Build pipeline stages with an injected chunk frame source.

    Calls build_stages() with frame_source=chunk_source, injecting the
    ChunkFrameSource view instead of constructing a new VideoFrameSource.

    Args:
        config: PipelineConfig instance.
        chunk_source: ChunkFrameSource to inject into DetectionStage and MidlineStage.

    Returns:
        List of stage instances in pipeline order.
    """
    from aquapose.engine.pipeline import build_stages

    return build_stages(config, frame_source=chunk_source)  # type: ignore[arg-type]


def _format_eta(chunk_times: list[float], chunk_idx: int, n_chunks: int) -> str:
    """Format an ETA string based on average chunk duration.

    Args:
        chunk_times: Elapsed seconds for completed chunks.
        chunk_idx: Zero-based index of the just-completed chunk.
        n_chunks: Total number of chunks.

    Returns:
        A human-readable ETA string (e.g. ``"~45s left"`` or ``"~3m left"``).
    """
    if not chunk_times:
        return "..."
    avg = sum(chunk_times) / len(chunk_times)
    remaining = (n_chunks - chunk_idx - 1) * avg
    if remaining < 60:
        return f"~{remaining:.0f}s left"
    return f"~{remaining / 60:.0f}m left"


def _write_skipped_metadata(
    hdf5_path: Path, skipped_ranges: list[tuple[int, int]]
) -> None:
    """Write skipped chunk frame ranges to HDF5 root attrs.

    Args:
        hdf5_path: Path to the existing HDF5 output file.
        skipped_ranges: List of (start_frame, end_frame) tuples for failed chunks.
    """
    import h5py
    import numpy as np

    try:
        with h5py.File(hdf5_path, "a") as f:
            arr = np.array(skipped_ranges, dtype=np.int64)
            f.attrs["skipped_chunks"] = arr
    except Exception:
        logger.warning("Failed to write skipped_chunks metadata to HDF5", exc_info=True)


class ChunkOrchestrator:
    """Processes a full video in fixed-size temporal chunks.

    Each chunk runs the full 5-stage PosePipeline independently via a
    ChunkFrameSource view. State is carried across chunk boundaries via
    ChunkHandoff (tracker state + identity map). Per-chunk 3D midlines are
    flushed to HDF5 with the correct global frame offset.

    When chunk_size is None or 0, the entire video is processed as a single
    chunk — a degenerate case that matches non-chunked PosePipeline.run() behavior.

    Args:
        config: Frozen pipeline config. chunk_size controls chunk loop behavior.
        verbose: If True, attach ConsoleObserver for per-stage output within each
            chunk. Default False (quiet chunk mode for long runs).
        max_chunks: If not None, process at most this many chunks then stop.
            Useful for quick testing (e.g. max_chunks=1 for single-chunk dry run).
        stop_after: Stage name at which to stop the pipeline (e.g. "detection").
            Injected into config via CLI overrides before construction.
        extra_observers: Additional observer names from ``--add-observer`` flags,
            forwarded to build_observers() for each chunk.
        frame_source: Optional pre-built FrameSource. When provided the caller
            owns the lifecycle (no open/close). When None, ChunkOrchestrator
            constructs and manages a VideoFrameSource internally.
    """

    def __init__(
        self,
        config: PipelineConfig,
        verbose: bool = False,
        max_chunks: int | None = None,
        stop_after: str | None = None,
        extra_observers: tuple[str, ...] = (),
        frame_source: object = None,
    ) -> None:
        self._config = config
        self._verbose = verbose
        self._max_chunks = max_chunks
        self._stop_after = stop_after
        self._extra_observers = extra_observers
        self._frame_source = frame_source

    def run(self) -> None:
        """Execute the full video in chunks and write HDF5 output."""
        import contextlib
        import time

        from aquapose.core.context import PipelineContext
        from aquapose.core.types.frame_source import ChunkFrameSource, VideoFrameSource
        from aquapose.engine.pipeline import PosePipeline
        from aquapose.io.midline_writer import Midline3DWriter

        config = self._config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from aquapose.logging import setup_file_logging

        setup_file_logging(output_dir, "run")

        chunk_size = config.chunk_size or None

        hdf5_path = output_dir / "midlines.h5"
        next_global_id = 0
        prev_handoff: ChunkHandoff | None = None
        skipped_ranges: list[tuple[int, int]] = []

        # Build ExitStack: always wrap Midline3DWriter; only wrap VideoFrameSource
        # when we own the lifecycle (i.e. caller did not pass frame_source).
        with contextlib.ExitStack() as stack:
            if self._frame_source is not None:
                video_source = self._frame_source
            else:
                video_source = stack.enter_context(
                    VideoFrameSource(
                        video_dir=config.video_dir,
                        calibration_path=config.calibration_path,
                    )
                )
            writer = stack.enter_context(
                Midline3DWriter(
                    output_path=hdf5_path,
                    max_fish=config.n_animals,
                    n_sample_points=config.n_sample_points,
                )
            )

            total_frames = len(video_source)

            if chunk_size is None or chunk_size <= 0:
                boundaries = [(0, total_frames)]
            else:
                boundaries = [
                    (s, min(s + chunk_size, total_frames))
                    for s in range(0, total_frames, chunk_size)
                ]

            # Limit to max_chunks if specified
            if self._max_chunks is not None:
                boundaries = boundaries[: self._max_chunks]

            n_chunks = len(boundaries)
            chunk_times: list[float] = []

            for chunk_idx, (chunk_start, chunk_end) in enumerate(boundaries):
                t0 = time.monotonic()
                chunk_source = ChunkFrameSource(
                    video_source, start_frame=chunk_start, end_frame=chunk_end
                )
                stages = _build_stages_for_chunk(config, chunk_source)

                # PipelineContext is a mutable dataclass — direct field assignment is safe
                # Pass the full ChunkHandoff directly so TrackingStage can preserve
                # identity fields (identity_map, track_id_to_global, next_global_id).
                initial_context = PipelineContext()
                if prev_handoff is not None:
                    initial_context.carry_forward = prev_handoff

                # Build observers for this chunk; suppress ConsoleObserver unless verbose.
                try:
                    from aquapose.engine.observer_factory import build_observers

                    observers = build_observers(
                        config=config,
                        mode=config.mode,
                        verbose=self._verbose,
                        total_stages=len(stages),
                        extra_observers=self._extra_observers,
                        chunk_idx=chunk_idx,
                        chunk_start=chunk_start,
                    )
                except Exception:
                    observers = []

                # Remove ConsoleObserver unless verbose
                if not self._verbose:
                    try:
                        from aquapose.engine.console_observer import ConsoleObserver

                        observers = [
                            o for o in observers if not isinstance(o, ConsoleObserver)
                        ]
                    except ImportError:
                        pass

                try:
                    pipeline = PosePipeline(
                        stages=stages,
                        config=config,
                        observers=observers,
                    )
                    context = pipeline.run(initial_context=initial_context)
                except Exception as exc:
                    logger.error(
                        "Chunk %d/%d (%d-%d) FAILED: %s — skipping",
                        chunk_idx + 1,
                        n_chunks,
                        chunk_start,
                        chunk_end - 1,
                        exc,
                        exc_info=True,
                    )
                    skipped_ranges.append((chunk_start, chunk_end))
                    prev_handoff = None
                    next_global_id += 1
                    elapsed = time.monotonic() - t0
                    chunk_times.append(elapsed)
                    continue

                tracklet_groups = getattr(context, "tracklet_groups", None) or []
                identity_map, next_global_id = _stitch_identities(
                    tracklet_groups=tracklet_groups,
                    prev_handoff=prev_handoff,
                    next_global_id=next_global_id,
                )

                n_continued = sum(
                    1
                    for gid in identity_map.values()
                    if prev_handoff and gid in prev_handoff.identity_map.values()
                )
                n_new = len(identity_map) - n_continued

                midlines_3d = getattr(context, "midlines_3d", None) or []
                for local_idx, frame_midlines in enumerate(midlines_3d):
                    global_frame_idx = chunk_start + local_idx
                    remapped: dict[int, object] = {}
                    for lid, ml in frame_midlines.items():  # type: ignore[union-attr]
                        global_id = identity_map.get(int(lid), int(lid))
                        remapped[global_id] = ml
                    writer.write_frame(global_frame_idx, remapped)  # type: ignore[arg-type]

                carry_out = getattr(context, "carry_forward", None)
                tracks_2d_state = carry_out.tracks_2d_state if carry_out else {}

                new_track_id_to_global: dict[tuple[str, int], int] = {}
                for group in tracklet_groups:
                    gid: int = identity_map.get(group.fish_id, group.fish_id)  # type: ignore[assignment]
                    for tracklet in group.tracklets:
                        new_track_id_to_global[
                            (tracklet.camera_id, tracklet.track_id)
                        ] = gid

                prev_handoff = ChunkHandoff(
                    tracks_2d_state=tracks_2d_state,
                    identity_map=identity_map,
                    track_id_to_global=new_track_id_to_global,
                    next_global_id=next_global_id,
                )
                write_handoff(output_dir / "handoff.pkl", prev_handoff)

                elapsed = time.monotonic() - t0
                chunk_times.append(elapsed)
                eta_str = _format_eta(chunk_times, chunk_idx, n_chunks)
                print(
                    f"Chunk {chunk_idx + 1}/{n_chunks} ({chunk_start}-{chunk_end - 1}) — "
                    f"{len(identity_map)} fish ({n_new} new), {elapsed:.0f}s, {eta_str}"
                )

        if skipped_ranges:
            _write_skipped_metadata(hdf5_path, skipped_ranges)

        # Update manifest.json with the authoritative total_frames after all chunks complete
        if config.mode == "diagnostic":
            _update_manifest_total_frames(output_dir, total_frames, chunk_size)


def _update_manifest_total_frames(
    output_dir: Path, total_frames: int, chunk_size: int | None
) -> None:
    """Update manifest.json with the authoritative total_frames after all chunks complete.

    Reads the existing manifest (written per-chunk by DiagnosticObserver), updates
    total_frames and chunk_size with the values known only after the full run, and
    writes back atomically.

    Args:
        output_dir: Run output directory (parent of diagnostics/).
        total_frames: Authoritative total frame count from the video source.
        chunk_size: Chunk size used for this run (None for single-chunk).
    """
    import json
    import tempfile

    manifest_path = output_dir / "diagnostics" / "manifest.json"
    if not manifest_path.exists():
        return

    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read manifest.json for total_frames update")
        return

    existing["total_frames"] = total_frames
    existing["chunk_size"] = chunk_size

    try:
        with tempfile.NamedTemporaryFile(
            dir=manifest_path.parent,
            delete=False,
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
        ) as tmp:
            tmp_path = tmp.name
            json.dump(existing, tmp, indent=2)
        os.replace(tmp_path, manifest_path)
        logger.info("Manifest updated with total_frames=%d", total_frames)
    except OSError:
        logger.warning("Failed to update manifest.json total_frames", exc_info=True)


def write_handoff(path: Path | str, handoff: ChunkHandoff) -> None:
    """Write a ChunkHandoff to disk atomically using temp-file + rename.

    Args:
        path: Destination path for the handoff pickle file.
        handoff: ChunkHandoff instance to serialize.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dest.parent, delete=False, suffix=".tmp"
    ) as tmp:
        tmp_path = tmp.name
        pickle.dump(handoff, tmp)
    os.replace(tmp_path, dest)
