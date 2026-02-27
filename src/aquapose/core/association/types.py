"""Domain types for the Association stage (Stage 3) and downstream consumers.

Defines TrackletGroup — the cross-camera identity cluster produced by Stage 3 — and
retains AssociationBundle for reconstruction compatibility until Phase 26 replaces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AssociationBundle", "HandoffState", "TrackletGroup"]


# ---------------------------------------------------------------------------
# v2.1 domain type: TrackletGroup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackletGroup:
    """Cross-camera identity cluster from Stage 3 (Association).

    Groups Tracklet2D objects from multiple cameras into a single fish identity.
    Each TrackletGroup represents one physical fish whose per-camera tracklets
    have been matched by the Association stage (Stage 3).

    The ``tracklets`` field uses a generic ``tuple`` (not ``tuple[Tracklet2D, ...]``)
    to preserve the core/ import boundary — association types must not import
    tracking types at runtime. The actual element type is ``Tracklet2D``; see
    aquapose.core.tracking.types.

    Attributes:
        fish_id: Global fish identity assigned by the Association stage.
            Unique within the batch; persistent IDs are assigned in Phase 24+.
        tracklets: Tuple of Tracklet2D objects, one per contributing camera.
            Type: ``tuple[Tracklet2D, ...]`` (generic tuple at runtime to avoid
            cross-package import within core/).
        confidence: Association confidence score. ``None`` until Phase 25/26
            populates it with the Leiden clustering confidence.
        per_frame_confidence: Per-frame confidence values aligned with the
            union of all constituent tracklets' frame ranges. ``None`` until
            refinement runs. Each element is a float in [0, 1].
    """

    fish_id: int
    tracklets: tuple
    confidence: float | None = None
    per_frame_confidence: tuple | None = None


# ---------------------------------------------------------------------------
# v2.1 domain type: HandoffState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandoffState:
    """Chunk handoff state emitted by the Association stage.

    Typed stub for future chunk-boundary orchestration. Fields are defined
    but populated with None/empty values in v2.1 (single-chunk mode).

    Attributes:
        active_fish_ids: Global fish IDs active at chunk end. None in v2.1.
        per_id_3d_state: Per-fish 3D position/velocity at chunk end. None in v2.1.
        per_id_2d_state: Per-fish per-camera 2D state at chunk end. None in v2.1.
        per_id_confidence: Per-fish association confidence at boundary. None in v2.1.
    """

    active_fish_ids: tuple | None = None
    per_id_3d_state: dict | None = None
    per_id_2d_state: dict | None = None
    per_id_confidence: dict | None = None


# ---------------------------------------------------------------------------
# Legacy compatibility type: AssociationBundle
#
# Retained for reconstruction compatibility until Phase 26 replaces it with
# TrackletGroup-based reconstruction. Do not use in new code.
# ---------------------------------------------------------------------------


@dataclass
class AssociationBundle:
    """Cross-camera detection grouping for a single physical fish in one frame.

    Legacy type from v1.0. In v2.1, Stage 3 produces TrackletGroup objects.
    AssociationBundle is retained here so reconstruction code that still
    references it continues to import without modification until Phase 26.

    Unlike the v1.0 ``AssociationResult``, ``fish_idx`` is a 0-indexed
    per-frame position (not a persistent track ID). Persistent fish IDs are
    assigned by the Tracking stage (Stage 4) which consumes these bundles.

    Attributes:
        fish_idx: 0-indexed fish position within this frame's bundle list.
            Not a persistent ID — use only for intra-frame ordering.
        centroid_3d: Estimated 3D centroid in world frame, shape (3,).
        camera_detections: Mapping from camera_id to detection index in the
            per-camera detection list for that camera.
        n_cameras: Number of cameras contributing to this bundle.
        reprojection_residual: Mean pixel distance from projected 3D centroid
            to assigned detection centroids. 0.0 for single-view fallback
            entries.
        confidence: Association confidence. 1.0 for high-confidence multi-view
            bundles; lower for single-view fallback entries.
    """

    fish_idx: int
    centroid_3d: np.ndarray  # shape (3,), world coordinates
    camera_detections: dict  # camera_id -> detection_index
    n_cameras: int
    reprojection_residual: float
    confidence: float
