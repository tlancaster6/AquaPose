"""Shared types: crop region and affine crop containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CropRegion:
    """A padded crop region within a full-frame image.

    Attributes:
        x1: Left edge of the crop in the full frame (clipped to 0).
        y1: Top edge of the crop in the full frame (clipped to 0).
        x2: Right edge of the crop in the full frame (clipped to frame width).
        y2: Bottom edge of the crop in the full frame (clipped to frame height).
        frame_h: Height of the full frame.
        frame_w: Width of the full frame.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    frame_h: int
    frame_w: int

    @property
    def width(self) -> int:
        """Width of the crop region in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the crop region in pixels."""
        return self.y2 - self.y1


@dataclass
class AffineCrop:
    """A rotation-aligned crop extracted via an affine warp.

    The transform matrix ``M`` maps frame coordinates to crop coordinates.
    Use :func:`invert_affine_point` or :func:`invert_affine_points` to
    back-project crop-space predictions to frame space.

    Attributes:
        image: Cropped image array of shape ``(crop_h, crop_w, C)`` or
            ``(crop_h, crop_w)``. Pixels outside the source frame are
            zero-filled.
        M: Affine transform matrix of shape ``(2, 3)``, float64, mapping
            frame coordinates to crop coordinates.
        crop_size: Output canvas size as ``(width, height)`` in pixels.
        frame_shape: Source frame dimensions as ``(height, width)``.
    """

    image: np.ndarray
    M: np.ndarray
    crop_size: tuple[int, int]
    frame_shape: tuple[int, int]
