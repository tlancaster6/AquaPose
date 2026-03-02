"""Crop-segment-paste utilities for ROI-based mask prediction.

Provides shared machinery to crop an image region around a bounding box,
run a segmentation model on the crop, and paste the resulting mask back
into the full frame. Used by midline extraction backends and crop preprocessing.

Also provides affine crop utilities for OBB-aligned crops:
:func:`extract_affine_crop`, :func:`invert_affine_point`, and
:func:`invert_affine_points`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
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


def compute_crop_region(
    bbox: tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
    padding: float = 0.25,
) -> CropRegion:
    """Compute a padded crop region around a bounding box.

    Adds fractional padding relative to bbox dimensions, then clips
    to frame boundaries.

    Args:
        bbox: Bounding box as (x, y, w, h) in pixel coordinates.
        frame_h: Full frame height.
        frame_w: Full frame width.
        padding: Fractional padding relative to bbox size. 0.25 means
            25% of the bbox dimension is added on each side.

    Returns:
        CropRegion with clipped coordinates.
    """
    x, y, w, h = bbox
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_w, x + w + pad_x)
    y2 = min(frame_h, y + h + pad_y)

    return CropRegion(x1=x1, y1=y1, x2=x2, y2=y2, frame_h=frame_h, frame_w=frame_w)


def extract_crop(image: np.ndarray, region: CropRegion) -> np.ndarray:
    """Extract a crop from a full-frame image.

    Args:
        image: Full-frame image array of shape (H, W) or (H, W, C).
        region: CropRegion specifying the area to extract.

    Returns:
        Cropped image array.
    """
    return image[region.y1 : region.y2, region.x1 : region.x2].copy()


def paste_mask(
    crop_mask: np.ndarray,
    region: CropRegion,
) -> np.ndarray:
    """Paste a crop-sized mask back into a full-frame binary mask.

    Args:
        crop_mask: Binary mask of the crop region, shape (crop_h, crop_w).
            Values should be 0 or 255 (uint8).
        region: CropRegion that defines where to paste.

    Returns:
        Full-frame binary mask (uint8, 0/255) of shape (frame_h, frame_w).
    """
    full_mask = np.zeros((region.frame_h, region.frame_w), dtype=np.uint8)

    # Resize crop_mask if it doesn't match the region dimensions
    if crop_mask.shape[:2] != (region.height, region.width):
        crop_mask = cv2.resize(
            crop_mask,
            (region.width, region.height),
            interpolation=cv2.INTER_NEAREST,
        )

    full_mask[region.y1 : region.y2, region.x1 : region.x2] = crop_mask
    return full_mask


# ---------------------------------------------------------------------------
# Affine crop utilities (OBB-aligned, invertible)
# ---------------------------------------------------------------------------


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


def extract_affine_crop(
    frame: np.ndarray,
    center_xy: tuple[float, float],
    angle_math_rad: float,
    obb_w: float,
    obb_h: float,
    crop_size: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> AffineCrop:
    """Extract a rotation-aligned crop centred on an OBB detection.

    Builds an affine warp that rotates the frame so the OBB's long axis is
    horizontal, then centres the OBB centre in the crop canvas at native
    pixel scale.  Pixels outside the source frame are zero-filled.

    Args:
        frame: Source image of shape ``(H, W)`` or ``(H, W, C)``.
        center_xy: OBB centre in frame pixel coordinates ``(cx, cy)``.
        angle_math_rad: OBB orientation in standard math radians (CCW,
            range ``[-pi, pi]``).
        obb_w: OBB width in pixels (along the fish's long axis).
        obb_h: OBB height in pixels (across the fish body).
        crop_size: Output canvas size ``(width, height)`` in pixels.
        interpolation: OpenCV interpolation flag for ``cv2.warpAffine``.
            Defaults to ``cv2.INTER_LINEAR``.

    Returns:
        :class:`AffineCrop` containing the warped image, the transform
        matrix, and metadata.
    """
    cx, cy = center_xy
    crop_w, crop_h = crop_size

    # Convert standard math angle (CCW, radians) -> cv2 angle (CW, degrees)
    angle_cv2_deg = math.degrees(-angle_math_rad)

    # Build a rotation matrix about the OBB centre (native scale)
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_cv2_deg, scale=1.0)

    # Translate so that the OBB centre lands at the centre of the crop canvas
    M_rot[0, 2] += crop_w / 2 - cx
    M_rot[1, 2] += crop_h / 2 - cy

    crop_image = cv2.warpAffine(
        frame,
        M_rot,
        (crop_w, crop_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return AffineCrop(
        image=crop_image,
        M=M_rot,
        crop_size=crop_size,
        frame_shape=frame.shape[:2],
    )


def invert_affine_point(
    crop_xy: tuple[float, float],
    M: np.ndarray,
) -> tuple[float, float]:
    """Back-project a single crop-space point to frame coordinates.

    Uses the inverse of the affine transform ``M`` produced by
    :func:`extract_affine_crop`.  For a pure rotation + translation,
    :func:`cv2.invertAffineTransform` is numerically exact.

    Args:
        crop_xy: Point in crop pixel coordinates ``(x, y)``.
        M: Affine transform matrix of shape ``(2, 3)``, float64, as
            returned by :func:`extract_affine_crop`.

    Returns:
        Corresponding point in frame pixel coordinates ``(x, y)``.
    """
    M_inv = cv2.invertAffineTransform(M)
    pt = np.array([[[crop_xy[0], crop_xy[1]]]], dtype=np.float64)
    result = cv2.transform(pt, M_inv)
    return (float(result[0, 0, 0]), float(result[0, 0, 1]))


def invert_affine_points(
    crop_points: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Back-project multiple crop-space points to frame coordinates (batch).

    Equivalent to calling :func:`invert_affine_point` for each point, but
    more efficient for large arrays.

    Args:
        crop_points: Points in crop pixel coordinates, shape ``(N, 2)``.
        M: Affine transform matrix of shape ``(2, 3)``, float64, as
            returned by :func:`extract_affine_crop`.

    Returns:
        Corresponding points in frame pixel coordinates, shape ``(N, 2)``.
    """
    M_inv = cv2.invertAffineTransform(M)
    pts = crop_points.reshape(1, -1, 2).astype(np.float64)
    result = cv2.transform(pts, M_inv)
    return result.reshape(-1, 2)
