"""MidlineFixture dataclass and NPZ key convention for midline capture.

Defines the data contract for serialising per-frame MidlineSet data captured
during a pipeline run to a compressed NPZ file.  The loader (Plan 02) reads
files produced by this contract.

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

from dataclasses import dataclass

from aquapose.core.types.reconstruction import MidlineSet

__all__ = ["NPZ_VERSION", "MidlineFixture"]

NPZ_VERSION = "1.0"


@dataclass(frozen=True)
class MidlineFixture:
    """Immutable container for per-frame MidlineSet data from a pipeline run.

    Represents the data captured by DiagnosticObserver and serialised to an
    NPZ file.  The loader (Plan 02) deserialises an NPZ into this form.

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
