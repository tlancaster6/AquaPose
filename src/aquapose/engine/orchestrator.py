"""ChunkOrchestrator, ChunkHandoff, and chunk-mode infrastructure for AquaPose."""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkHandoff:
    """Cross-chunk state carried between chunk invocations.

    Frozen so it is replaced wholesale each chunk, never mutated.

    Attributes:
        tracks_2d_state: Per-camera opaque OC-SORT tracker state blobs.
            Keys are camera IDs; values are dicts from OcSortTracker.get_state().
            Used to restore tracker continuity at the start of the next chunk.
        identity_map: Maps chunk-local fish IDs to globally consistent fish IDs.
            Keys are chunk-local fish IDs (from TrackletGroup.fish_id in the
            just-completed chunk); values are global fish IDs.
            Built by the identity stitcher after each chunk.
        track_id_to_global: Maps (camera_id, track_id) tuples to global fish IDs.
            Used for track-continuity-based identity stitching across chunk boundaries.
        next_global_id: Next globally unique fish ID to assign to an unmatched
            fish. Monotonically increasing across chunks to prevent ID reuse.
    """

    tracks_2d_state: dict  # camera_id -> OcSortTracker.get_state() blob
    identity_map: dict  # local_fish_id -> global_fish_id
    track_id_to_global: dict  # (camera_id, track_id) -> global_fish_id
    next_global_id: int


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
