"""Affine crop extraction and inverse projection utilities."""

from __future__ import annotations

import math

import cv2
import numpy as np

from aquapose.core.types.crop import AffineCrop


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
