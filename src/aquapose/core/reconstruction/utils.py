"""Shared helper functions for reconstruction backends."""

from __future__ import annotations

import logging

import numpy as np
import scipy.interpolate
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (shared by all reconstruction backends)
# ---------------------------------------------------------------------------

SPLINE_K: int = 3
SPLINE_N_CTRL: int = 7
MIN_BODY_POINTS: int = 9  # SPLINE_N_CTRL + 2
SPLINE_KNOTS: np.ndarray = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
)

__all__ = [
    "MIN_BODY_POINTS",
    "SPLINE_K",
    "SPLINE_KNOTS",
    "SPLINE_N_CTRL",
    "build_spline_knots",
    "fit_spline",
    "pixel_half_width_to_metres",
    "weighted_triangulate_rays",
]


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def build_spline_knots(n_control_points: int) -> np.ndarray:
    """Build a clamped cubic B-spline knot vector for the given control point count.

    Args:
        n_control_points: Number of B-spline control points.

    Returns:
        Knot vector of length ``n_control_points + SPLINE_K + 1``.
    """
    n_interior = n_control_points - SPLINE_K - 1
    if n_interior < 0:
        n_interior = 0
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    knots = np.concatenate(
        [
            np.zeros(SPLINE_K + 1),
            interior,
            np.ones(SPLINE_K + 1),
        ]
    )
    return knots.astype(np.float64)


def weighted_triangulate_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted DLT triangulation using normal equations with per-camera weights.

    Each camera's contribution to the normal equations (A, b) is scaled by
    ``weights[i]``. When all weights are 1.0, produces identical output to
    ``triangulate_rays()``.

    Implementation mirrors ``triangulate_rays`` in ``calibration/projection.py``
    (normal equations: A = sum_i w_i * (I - d_i d_i^T), b = A_i @ o_i) but
    applies a scalar weight to each camera's A and b contribution.

    Args:
        origins: Ray origin points, shape (N, 3).
        directions: Unit ray direction vectors, shape (N, 3). Must be unit vectors.
        weights: Per-camera scalar weights, shape (N,). Typically sqrt(confidence).

    Returns:
        Triangulated 3D point, shape (3,).
    """
    device = origins.device
    dtype = origins.dtype

    A = torch.zeros(3, 3, device=device, dtype=dtype)
    b = torch.zeros(3, device=device, dtype=dtype)

    eye3 = torch.eye(3, device=device, dtype=dtype)
    for i in range(origins.shape[0]):
        d = directions[i]  # (3,)
        o = origins[i]  # (3,)
        w = weights[i]  # scalar
        M = eye3 - d.unsqueeze(1) @ d.unsqueeze(0)  # (3, 3)
        A = A + w * M
        b = b + w * (M @ o)

    result = torch.linalg.lstsq(A, b.unsqueeze(1))
    return result.solution.squeeze(1)


def fit_spline(
    u_param: np.ndarray,
    pts_3d: np.ndarray,
    knots: np.ndarray | None = None,
    min_body_points: int | None = None,
) -> tuple[np.ndarray, float] | None:
    """Fit a cubic B-spline to 3D body positions.

    Uses scipy.interpolate.make_lsq_spline with the provided knot vector.

    Args:
        u_param: Arc-length parameter values in [0, 1], shape (M,), float64.
            Must be strictly increasing.
        pts_3d: 3D point positions, shape (M, 3), float64.
        knots: B-spline knot vector. Defaults to ``build_spline_knots(7)``.
        min_body_points: Minimum observations required. Defaults to
            module-level MIN_BODY_POINTS.

    Returns:
        Tuple of (control_points, arc_length) where control_points has shape
        (n_ctrl, 3), float32 and arc_length is the numerical integral of the
        spline curve length. Returns None if too few points or fitting fails.
    """
    if knots is None:
        knots = build_spline_knots(7)
    if min_body_points is None:
        min_body_points = MIN_BODY_POINTS

    if len(u_param) < min_body_points:
        return None

    try:
        spl = scipy.interpolate.make_lsq_spline(u_param, pts_3d, knots, k=SPLINE_K)
    except (ValueError, np.linalg.LinAlgError) as exc:
        # Schoenberg-Whitney condition violation or singular matrix
        logger.debug("Spline fitting failed: %s", exc)
        return None

    control_points = spl.c.astype(np.float32)  # shape (7, 3)

    # Compute arc length via 1000-point numerical integration
    u_fine = np.linspace(0.0, 1.0, 1000)
    curve_pts = spl(u_fine)  # shape (1000, 3)
    diffs = np.diff(curve_pts, axis=0)  # shape (999, 3)
    seg_lengths = np.linalg.norm(diffs, axis=1)  # shape (999,)
    arc_length = float(np.sum(seg_lengths))

    return control_points, arc_length


def pixel_half_width_to_metres(
    hw_px: float,
    depth_m: float,
    focal_px: float,
) -> float:
    """Convert pixel half-width to world metres using pinhole approximation.

    Uses the formula: hw_m = hw_px * depth_m / focal_px

    This is an approximation valid near the optical axis. Sufficient for
    width profile estimation (not used in triangulation geometry).

    Args:
        hw_px: Half-width in pixels.
        depth_m: Depth of the body point below the water surface in metres.
        focal_px: Camera focal length in pixels (mean of fx and fy).

    Returns:
        Half-width in world metres.
    """
    return hw_px * depth_m / focal_px
