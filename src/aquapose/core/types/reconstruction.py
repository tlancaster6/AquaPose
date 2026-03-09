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
        centroid_z: Z-coordinate of the body centroid before flattening.
            Used by temporal smoothing to reduce frame-to-frame z-jitter.
            None when z-flattening is disabled.
        z_offsets: Per-body-point z-offset from centroid before flattening,
            shape (n_sample_points,), float32.  Preserves the raw z-structure
            for potential future use.  NaN for body points that were not
            triangulated.  None when z-flattening is disabled.
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
    centroid_z: float | None = None
    z_offsets: np.ndarray | None = None
    triangulated_points: np.ndarray | None = None
    """Raw triangulated 3D body points before spline fitting.

    Shape (n_body_points, 3), float32. NaN for body points that failed
    triangulation. None when not populated (backward compatibility).
    """
    per_point_inlier_cameras: list[list[str]] | None = None
    """Per-body-point inlier camera IDs after outlier rejection.

    Length n_body_points. Empty list for body points that failed
    triangulation. None when not populated (backward compatibility).
    """
