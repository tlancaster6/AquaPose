"""Domain types for the 2D Tracking stage (Stage 2).

Defines Tracklet2D — the per-camera temporal tracklet produced by Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Tracklet2D"]

# ---------------------------------------------------------------------------
# v2.1 domain type: Tracklet2D
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tracklet2D:
    """Per-camera temporal tracklet from Stage 2 (2D Tracking).

    Represents a single fish's track within one camera over a batch of frames.
    Tracklets are the outputs of the per-camera 2D tracker and
    the inputs to Stage 3 (Association), which groups them across cameras into
    TrackletGroup objects.

    All sequence fields use ``tuple`` (not ``list``) for immutability — the
    dataclass is frozen and these fields must not be mutated after creation.

    Attributes:
        camera_id: Which camera produced this tracklet.
        track_id: Unique tracklet identifier within this camera (local, not global).
            Track IDs are NOT comparable across cameras.
        frames: Ordered frame indices where the tracklet is active.
            Type: ``tuple[int, ...]``
        centroids: Per-frame (u, v) pixel centroids, one per entry in ``frames``.
            Deprecated: prefer ``keypoints[:, centroid_idx, :]`` when
            ``keypoints`` is not None.
            Type: ``tuple[tuple[float, float], ...]``
        bboxes: Per-frame bounding boxes as (x, y, w, h), one per entry in ``frames``.
            Type: ``tuple[tuple[float, float, float, float], ...]``
        frame_status: Per-frame detection status, one per entry in ``frames``.
            Each value is ``"detected"`` (directly observed) or ``"coasted"``
            (position interpolated during a missed detection).
            Type: ``tuple[str, ...]``
        keypoints: Per-frame keypoint positions, shape ``(T, K, 2)``, float32.
            ``None`` when keypoint data is unavailable (e.g. tracklets from
            fragment merging).
        keypoint_conf: Per-frame keypoint confidences, shape ``(T, K)``, float32.
            Values are 0.0 for coasted/interpolated frames and retain raw
            detector confidence for detected frames. ``None`` when unavailable.
    """

    camera_id: str
    track_id: int
    frames: tuple
    centroids: tuple
    bboxes: tuple
    frame_status: tuple
    keypoints: np.ndarray | None = None
    keypoint_conf: np.ndarray | None = None
