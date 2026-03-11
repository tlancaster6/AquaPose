"""Shared type: detection result for a single object in a single camera frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single fish detection.

    Attributes:
        bbox: Bounding box as (x, y, w, h) in pixel coordinates.
        mask: Full-frame binary mask (uint8, 0/255) for this component only.
            ``None`` when the mask is equivalent to the bbox (e.g. YOLO
            detections without instance segmentation).
        area: Pixel area of the detected component.
        confidence: Detection confidence in [0, 1].
        angle: OBB rotation angle in radians, standard math convention,
            range ``[-pi, pi]``. ``None`` for non-OBB detectors (YOLO
            axis-aligned). Set by YOLO-OBB backend when available.
        obb_points: Oriented bounding box corner points, shape ``(4, 2)``,
            clockwise from top-left of the oriented box. ``None`` for
            non-OBB detectors. Set by YOLO-OBB backend when available.
        keypoints: Raw anatomical keypoint positions in full-frame pixel
            coordinates, shape ``(K, 2)`` float32. Keypoint indices:
            0=nose, 1=head, 2=spine1, 3=spine2, 4=spine3, 5=tail.
            ``None`` until PoseStage has run.
        keypoint_conf: Per-keypoint confidence scores in ``[0, 1]``,
            shape ``(K,)`` float32. ``None`` until PoseStage has run.
    """

    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None
    area: int
    confidence: float
    angle: float | None = None
    obb_points: np.ndarray | None = None
    keypoints: np.ndarray | None = None
    keypoint_conf: np.ndarray | None = None
