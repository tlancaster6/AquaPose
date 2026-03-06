"""Shared type: 2D midline for a single fish in a single camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Midline2D:
    """Ordered 2D midline for a single fish in a single camera.

    Attributes:
        points: Full-frame pixel coordinates, shape (N, 2), float32.
            Column order is (x, y). Ordered from head to tail when
            ``is_head_to_tail`` is True.
        half_widths: Half-width of the fish at each midline point,
            shape (N,), float32, in full-frame pixels.
        fish_id: Globally unique fish identifier.
        camera_id: Camera identifier string.
        frame_index: Frame index within the video.
        is_head_to_tail: True when point[0] is the head end. False when
            orientation has not yet been established (first few frames).
        point_confidence: Per-point confidence scores, shape (N,), float32,
            values in ``[0, 1]``. ``None`` when confidence is not available
            (e.g. segment-then-extract uses uniform 1.0s; keypoint backends
            provide per-point model confidence). Used for confidence-weighted
            triangulation in Stage 5.
    """

    points: np.ndarray
    half_widths: np.ndarray
    fish_id: int
    camera_id: str
    frame_index: int
    is_head_to_tail: bool = False
    point_confidence: np.ndarray | None = None
