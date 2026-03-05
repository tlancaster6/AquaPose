"""IRLS-weighted SVD plane fitting and projection for z-denoising.

Fits a best-fit plane to triangulated 3D body points using camera-count
weights, then projects the points onto that plane to remove z-axis noise
before spline fitting.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fit_plane_weighted", "project_onto_plane"]

# Singular value ratio threshold below which the point distribution is
# considered degenerate (collinear or coincident).  When s[1]/s[0] falls
# below this, the plane normal is unreliable so we default to [0, 0, 1].
_DEGENERACY_THRESHOLD: float = 0.01


def fit_plane_weighted(
    pts_3d: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fit a weighted best-fit plane to 3D points via SVD.

    Uses camera-count weights so that body points observed by more cameras
    contribute more to the plane orientation.

    Args:
        pts_3d: 3D points, shape ``(M, 3)``, float64.
        weights: Per-point weights (e.g. camera counts as floats),
            shape ``(M,)``, float64.  Must be non-negative.

    Returns:
        Tuple of ``(normal, centroid, is_degenerate)`` where:

        - ``normal``: Unit normal of the best-fit plane, shape ``(3,)``.
        - ``centroid``: Weighted centroid, shape ``(3,)``.
        - ``is_degenerate``: True if the point distribution is too
          degenerate (collinear or coincident) for a reliable plane fit.
          When True, ``normal`` defaults to ``[0, 0, 1]``.
    """
    pts_3d = np.asarray(pts_3d, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Weighted centroid
    w_sum = weights.sum()
    if w_sum == 0:
        centroid = pts_3d.mean(axis=0)
    else:
        centroid = (weights[:, None] * pts_3d).sum(axis=0) / w_sum

    # Centre and weight rows
    centred = pts_3d - centroid
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))
    weighted_centred = centred * sqrt_w[:, None]

    # SVD
    _U, S, Vt = np.linalg.svd(weighted_centred, full_matrices=False)

    # Degeneracy check: if second singular value is too small relative to
    # the first, the points are essentially collinear.
    if S[0] == 0 or S[1] / S[0] < _DEGENERACY_THRESHOLD:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), centroid, True

    normal = Vt[2]  # row corresponding to smallest singular value

    # Ensure consistent orientation: normal points "up" (positive z component)
    # This is arbitrary but prevents sign flips between frames.
    if normal[2] < 0:
        normal = -normal

    return normal, centroid, False


def project_onto_plane(
    pts_3d: np.ndarray,
    normal: np.ndarray,
    centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points onto a plane defined by normal and centroid.

    Args:
        pts_3d: 3D points to project, shape ``(M, 3)``.
        normal: Unit normal of the target plane, shape ``(3,)``.
        centroid: A point on the plane (e.g. weighted centroid),
            shape ``(3,)``.

    Returns:
        Tuple of ``(pts_projected, signed_residuals)`` where:

        - ``pts_projected``: Projected points, shape ``(M, 3)``.
        - ``signed_residuals``: Signed off-plane distances, shape ``(M,)``.
          Positive means the point was on the same side as the normal.
    """
    pts_3d = np.asarray(pts_3d, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64)

    # Signed distances from points to the plane
    d = (pts_3d - centroid) @ normal  # (M,)

    # Project: subtract the normal component
    pts_projected = pts_3d - d[:, None] * normal[None, :]

    return pts_projected, d
