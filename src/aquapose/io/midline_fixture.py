"""MidlineFixture dataclass, NPZ key convention, and loader for midline capture.

Defines the data contract for serialising per-frame MidlineSet data captured
during a pipeline run to a compressed NPZ file.  ``load_midline_fixture``
deserialises a file produced by ``DiagnosticObserver.export_midline_fixtures``.

NPZ key convention
------------------
All keys use forward-slash separators and fall into two groups:

Meta keys (scalars / 1-D arrays):
  ``meta/version``        — str "1.0"
  ``meta/camera_ids``     — 1-D object array of camera-ID strings
  ``meta/frame_indices``  — 1-D int64 array of captured frame indices
  ``meta/frame_count``    — scalar int64 (total frames in original run)
  ``meta/timestamp``      — str ISO-8601 timestamp (UTC)

Midline data keys (per fish x camera x frame):
  ``midline/{frame_idx}/{fish_id}/{camera_id}/points``          — (N, 2) float32
  ``midline/{frame_idx}/{fish_id}/{camera_id}/half_widths``     — (N,)   float32
  ``midline/{frame_idx}/{fish_id}/{camera_id}/point_confidence``— (N,)   float32
      (uniform 1.0 when the Midline2D field is None)
  ``midline/{frame_idx}/{fish_id}/{camera_id}/is_head_to_tail`` — scalar bool

Note: forward slashes in NPZ keys are stored literally; ``numpy.load`` returns
them as-is and they must be split on ``"/"`` by the loader.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aquapose.core.types.midline import Midline2D
from aquapose.core.types.reconstruction import MidlineSet

__all__ = ["NPZ_VERSION", "MidlineFixture", "load_midline_fixture"]

NPZ_VERSION = "1.0"


@dataclass(frozen=True)
class MidlineFixture:
    """Immutable container for per-frame MidlineSet data from a pipeline run.

    Represents the data captured by DiagnosticObserver and serialised to an
    NPZ file.  ``load_midline_fixture`` deserialises an NPZ into this form.

    Attributes:
        frames: Sequence of per-frame MidlineSets.  Each element is a
            ``MidlineSet`` (``dict[int, dict[str, Midline2D]]``) mapping
            ``fish_id -> camera_id -> Midline2D`` for one frame.  Only frames
            that contain at least one midline are included.
        frame_indices: Original frame indices (same length as ``frames``).
        camera_ids: All camera IDs that appear in any frame's midlines.
        metadata: Arbitrary metadata dict; must contain at least
            ``"version"``, ``"timestamp"``, and ``"frame_count"`` keys.
    """

    frames: tuple[MidlineSet, ...]
    frame_indices: tuple[int, ...]
    camera_ids: tuple[str, ...]
    metadata: dict[str, object]


def load_midline_fixture(path: Path | str) -> MidlineFixture:
    """Deserialise a midline_fixtures.npz file into a MidlineFixture.

    Reads an NPZ file produced by
    ``DiagnosticObserver.export_midline_fixtures`` and reconstructs the
    structured ``MidlineFixture`` with per-frame ``MidlineSet`` data.

    Args:
        path: Path to the ``midline_fixtures.npz`` file.

    Returns:
        A ``MidlineFixture`` whose ``frames`` tuple contains one
        ``MidlineSet`` per captured frame, ordered by ascending frame index.

    Raises:
        ValueError: If ``meta/version`` is absent, if the version does not
            match ``NPZ_VERSION``, or if midline array keys are malformed.
    """
    path = Path(path)
    data = np.load(str(path), allow_pickle=True)

    # ------------------------------------------------------------------
    # Validate metadata
    # ------------------------------------------------------------------
    if "meta/version" not in data:
        raise ValueError(f"Missing meta/version in fixture file {path}")

    version = str(data["meta/version"])
    if version != NPZ_VERSION:
        raise ValueError(
            f"Unsupported fixture version '{version}' (expected '{NPZ_VERSION}') in {path}"
        )

    camera_ids_arr: np.ndarray = data["meta/camera_ids"]

    # ------------------------------------------------------------------
    # Parse midline keys into per-frame MidlineSet structures
    # ------------------------------------------------------------------
    # Group arrays by (frame_idx_int, fish_id_int, camera_id_str)
    # Each group accumulates a dict of field_name -> array
    groups: dict[tuple[int, int, str], dict[str, Any]] = defaultdict(dict)

    for key in data.files:
        if not key.startswith("midline/"):
            continue
        parts = key.split("/")
        if len(parts) != 5:
            raise ValueError(f"Malformed midline key '{key}' in {path}")
        _, frame_idx_str, fish_id_str, camera_id, field_name = parts
        group_key: tuple[int, int, str] = (
            int(frame_idx_str),
            int(fish_id_str),
            camera_id,
        )
        groups[group_key][field_name] = data[key]

    # ------------------------------------------------------------------
    # Assemble per-frame MidlineSet dicts
    # ------------------------------------------------------------------
    # frame_idx -> fish_id -> camera_id -> Midline2D
    frames_dict: dict[int, dict[int, dict[str, Midline2D]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for (frame_idx, fish_id, camera_id), fields in groups.items():
        is_head_to_tail = bool(fields["is_head_to_tail"])
        point_confidence: np.ndarray | None = fields.get("point_confidence")

        midline = Midline2D(
            points=fields["points"],
            half_widths=fields["half_widths"],
            fish_id=fish_id,
            camera_id=camera_id,
            frame_index=frame_idx,
            is_head_to_tail=is_head_to_tail,
            point_confidence=point_confidence,
        )
        frames_dict[frame_idx][fish_id][camera_id] = midline

    # ------------------------------------------------------------------
    # Build ordered frames tuple aligned with sorted frame indices
    # ------------------------------------------------------------------
    sorted_frame_indices = sorted(frames_dict.keys())
    ordered_frames: list[MidlineSet] = [
        dict(frames_dict[fi]) for fi in sorted_frame_indices
    ]

    # ------------------------------------------------------------------
    # Build metadata dict from meta keys
    # ------------------------------------------------------------------
    metadata: dict[str, object] = {
        "version": version,
    }
    if "meta/timestamp" in data:
        metadata["timestamp"] = str(data["meta/timestamp"])
    if "meta/frame_count" in data:
        metadata["frame_count"] = int(data["meta/frame_count"])

    return MidlineFixture(
        frames=tuple(ordered_frames),
        frame_indices=tuple(sorted_frame_indices),
        camera_ids=tuple(str(c) for c in camera_ids_arr),
        metadata=metadata,
    )
