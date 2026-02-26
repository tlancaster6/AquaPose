"""Stage-specific types for the Association stage (Stage 3).

Defines AssociationBundle — a cross-camera detection grouping representing
one physical fish in a single frame, produced by the RANSAC centroid
clustering backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AssociationBundle"]


@dataclass
class AssociationBundle:
    """Cross-camera detection grouping for a single physical fish in one frame.

    Produced by the Association stage (Stage 3). Each bundle represents one
    physical fish identified by triangulating detection centroids across cameras.

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
    camera_detections: dict[str, int]  # camera_id -> detection_index
    n_cameras: int
    reprojection_residual: float
    confidence: float
