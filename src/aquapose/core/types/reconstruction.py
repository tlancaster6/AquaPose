"""Shared types: 3D midline and midline set for multi-view reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aquapose.core.types.midline import Midline2D

# fish_id -> camera_id -> Midline2D
MidlineSet = dict[int, dict[str, Midline2D]]


@dataclass
class Midline3D:
    """Continuous 3D midline for a single fish in a single frame.

    Attributes:
        fish_id: Globally unique fish identifier.
        frame_index: Frame index within the video.
        control_points: B-spline control points, shape (7, 3), float32.
        knots: B-spline knot vector, shape (11,), float32.
        degree: B-spline degree (always 3).
        arc_length: Total arc length of the spline in world metres.
        half_widths: Half-width of the fish at each sample body position
            in world metres, shape (n_sample_points,), float32.
        n_cameras: Minimum number of camera observations across body points.
        mean_residual: Mean spline reprojection residual in pixels.  Computed
            by evaluating the fitted spline at n_sample_points positions,
            reprojecting into every observing camera, and averaging the pixel
            distance to the corresponding observed 2D midline point.
        max_residual: Maximum single-point spline reprojection residual in
            pixels across all (camera, body_point) pairs.
        is_low_confidence: True when more than 20% of body points were
            triangulated from fewer than 3 cameras.
        per_camera_residuals: Mean spline reprojection residual per camera.
            Maps camera_id to mean pixel error across all body points for
            that camera.  Useful for diagnosing cross-view identity errors.
        plane_normal: Unit normal of the best-fit bending plane, shape (3,),
            float32.  None when plane projection is disabled.
        plane_centroid: Weighted centroid of the best-fit plane, shape (3,),
            float32.  None when plane projection is disabled.
        off_plane_residuals: Signed off-plane distances for each body sample
            point, shape (n_sample_points,), float32.  NaN for body points
            that were not triangulated.  None when plane projection is
            disabled.
        is_degenerate_plane: True when the point distribution was too
            degenerate (collinear) for a reliable plane fit.  The plane
            normal defaults to [0, 0, 1] in this case.
    """

    fish_id: int
    frame_index: int
    control_points: np.ndarray  # shape (7, 3), float32
    knots: np.ndarray  # shape (11,), float32
    degree: int
    arc_length: float
    half_widths: np.ndarray  # shape (N,), float32 in world metres
    n_cameras: int
    mean_residual: float
    max_residual: float
    is_low_confidence: bool = False
    per_camera_residuals: dict[str, float] | None = None
    plane_normal: np.ndarray | None = None
    plane_centroid: np.ndarray | None = None
    off_plane_residuals: np.ndarray | None = None
    is_degenerate_plane: bool = False
