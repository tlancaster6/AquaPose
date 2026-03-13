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

    Supports two representations depending on whether spline fitting was
    performed:

    **Raw-keypoint mode** (``spline_enabled=False``): ``points`` holds the
    primary reconstruction output — raw triangulated 3D keypoints, shape
    ``(N, 3)``.  ``control_points``, ``knots``, ``degree``, and
    ``arc_length`` are all ``None``.

    **Spline-fitted mode** (``spline_enabled=True``): ``control_points``
    holds B-spline control points, shape ``(7, 3)``.  ``points`` may also
    be populated for consistency (same data as ``triangulated_points``).

    At least one of ``points`` or ``control_points`` must be non-None.

    Attributes:
        fish_id: Globally unique fish identifier.
        frame_index: Frame index within the video.
        half_widths: Half-width of the fish at each body point in world metres,
            shape (N,), float32.  N matches the point count in ``points`` (variable;
            typically 6 for raw-keypoint mode, higher for interpolated midlines).
        n_cameras: Minimum number of camera observations across body points.
        mean_residual: Mean reprojection residual in pixels.  In spline mode,
            computed by evaluating the fitted spline at n_sample_points
            positions, reprojecting into every observing camera, and averaging
            the pixel distance to the corresponding observed 2D midline point.
            In raw mode, the mean of per-point triangulation residuals.
        max_residual: Maximum single-point residual in pixels.
        points: Raw triangulated 3D keypoints, shape (N, 3), float32.  N is
            variable: typically 6 for raw-keypoint mode (one per anatomical
            keypoint), or higher when keypoints were interpolated before
            triangulation.  This is the primary output in raw-keypoint mode
            (spline disabled).  NaN for body points that failed triangulation.
            In spline mode, populated with the same data as
            ``triangulated_points`` for consistency.  None when not populated.
        control_points: B-spline control points, shape (7, 3), float32.
            Populated only when spline fitting was performed.  None otherwise.
        knots: B-spline knot vector, shape (11,), float32.
            None when spline fitting was not performed.
        degree: B-spline degree (always 3 when set).
            None when spline fitting was not performed.
        arc_length: Total arc length of the spline in world metres.
            None when spline fitting was not performed.
        is_low_confidence: True when more than 20% of body points were
            triangulated from fewer than 3 cameras.
        per_camera_residuals: Mean reprojection residual per camera.
            Maps camera_id to mean pixel error across all body points for
            that camera.  Useful for diagnosing cross-view identity errors.
        centroid_z: Z-coordinate of the body centroid before flattening.
            Used by temporal smoothing to reduce frame-to-frame z-jitter.
            None when z-flattening is disabled.
        z_offsets: Per-body-point z-offset from centroid before flattening,
            shape (N,), float32, matching the point count in ``points``.
            Preserves the raw z-structure for potential future use.  NaN for
            body points that were not triangulated.  None when z-flattening
            is disabled.
        triangulated_points: Raw triangulated 3D body points before spline
            fitting.  Shape (n_body_points, 3), float32.  NaN for body points
            that failed triangulation.  None when not populated (backward
            compatibility).  In spline mode ``points`` mirrors this field.
        per_point_inlier_cameras: Per-body-point inlier camera IDs after
            outlier rejection.  Length n_body_points.  Empty list for body
            points that failed triangulation.  None when not populated
            (backward compatibility).
    """

    # Required fields
    fish_id: int
    frame_index: int
    half_widths: np.ndarray  # shape (N,), float32 in world metres
    n_cameras: int
    mean_residual: float
    max_residual: float
    # Optional fields (with defaults)
    points: np.ndarray | None = None
    control_points: np.ndarray | None = None
    knots: np.ndarray | None = None
    degree: int | None = None
    arc_length: float | None = None
    is_low_confidence: bool = False
    per_camera_residuals: dict[str, float] | None = None
    centroid_z: float | None = None
    z_offsets: np.ndarray | None = None
    triangulated_points: np.ndarray | None = None
    per_point_inlier_cameras: list[list[str]] | None = None
