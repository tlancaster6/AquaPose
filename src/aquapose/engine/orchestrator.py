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
        next_global_id: Next globally unique fish ID to assign to an unmatched
            fish. Monotonically increasing across chunks to prevent ID reuse.
    """

    tracks_2d_state: dict  # camera_id -> OcSortTracker.get_state() blob
    identity_map: dict  # local_fish_id -> global_fish_id
    next_global_id: int


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
