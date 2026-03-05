"""Core pseudo-label generation: spline reprojection, confidence scoring, per-fish label generation."""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.core.types.reconstruction import Midline3D
from aquapose.training.geometry import (
    extrapolate_edge_keypoints,
    format_obb_annotation,
    format_pose_annotation,
    pca_obb,
)


def reproject_spline_keypoints(
    midline: Midline3D,
    keypoint_t_values: list[float],
    projection_model: RefractiveProjectionModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a Midline3D B-spline at given arc fractions and project to 2D.

    Args:
        midline: 3D midline with control_points, knots, and degree.
        keypoint_t_values: Arc-length fractions in [0, 1] at which to evaluate
            the spline. These are mapped to the knot domain.
        projection_model: Refractive camera projection model.

    Returns:
        Tuple of:
        - pixels: float64 array of shape ``(N, 2)`` with (u, v) pixel coords.
        - valid: bool array of shape ``(N,)``, True if projection succeeded.
    """
    # Map t-values from [0, 1] to the spline's knot domain
    knots = midline.knots
    t_min = knots[midline.degree]
    t_max = knots[-(midline.degree + 1)]

    t_eval = np.array(
        [t_min + t * (t_max - t_min) for t in keypoint_t_values], dtype=np.float64
    )

    # Evaluate B-spline at each t-value to get 3D points
    # BSpline expects control_points as (n_control, ndim) and knots as full knot vector
    spl_x = BSpline(knots, midline.control_points[:, 0], midline.degree)
    spl_y = BSpline(knots, midline.control_points[:, 1], midline.degree)
    spl_z = BSpline(knots, midline.control_points[:, 2], midline.degree)

    pts_3d = np.column_stack([spl_x(t_eval), spl_y(t_eval), spl_z(t_eval)])

    # Project via refractive model
    pts_tensor = torch.from_numpy(pts_3d).float()
    pixels_t, valid_t = projection_model.project(pts_tensor)

    # CUDA safety: always .cpu().numpy()
    pixels_np = pixels_t.cpu().detach().numpy().astype(np.float64)
    valid_np = valid_t.cpu().detach().numpy().astype(bool)

    return pixels_np, valid_np


def compute_confidence_score(
    mean_residual: float,
    n_cameras: int,
    per_camera_residuals: dict[str, float] | None,
) -> tuple[float, dict]:
    """Compute a composite 0-1 confidence score for a 3D reconstruction.

    Higher n_cameras and lower residuals yield higher scores.

    Args:
        mean_residual: Mean spline reprojection residual in pixels.
        n_cameras: Number of cameras that contributed observations.
        per_camera_residuals: Per-camera mean residuals, or None.

    Returns:
        Tuple of:
        - score: Composite confidence in [0, 1].
        - raw_metrics: Dict with component scores and input values.
    """
    # Residual component: 10 px = zero confidence
    residual_score = max(0.0, 1.0 - mean_residual / 10.0)

    # Camera count component: 2 cameras = 0, 8+ cameras = 1.0
    camera_score = min(1.0, max(0.0, (n_cameras - 2) / 6.0))

    # Per-camera residual variance component
    if per_camera_residuals and len(per_camera_residuals) > 1:
        values = list(per_camera_residuals.values())
        variance = float(np.var(values))
    else:
        variance = 0.0

    # 25 px^2 variance = zero confidence
    variance_score = max(0.0, 1.0 - variance / 25.0)

    # Weighted composite
    score = 0.5 * residual_score + 0.3 * camera_score + 0.2 * variance_score

    raw_metrics = {
        "mean_residual": mean_residual,
        "n_cameras": n_cameras,
        "per_camera_variance": variance,
        "residual_score": residual_score,
        "camera_score": camera_score,
        "variance_score": variance_score,
    }

    return score, raw_metrics


def generate_fish_labels(
    midline: Midline3D,
    projection_model: RefractiveProjectionModel,
    img_w: int,
    img_h: int,
    keypoint_t_values: list[float],
    lateral_pad: float,
    max_camera_residual_px: float,
    camera_id: str,
) -> dict | None:
    """Generate OBB and pose labels for one fish in one camera view.

    Returns None if the camera did not contribute to the reconstruction,
    or if its per-camera residual exceeds the threshold, or if fewer than
    2 keypoints are visible after projection.

    Args:
        midline: 3D midline reconstruction for the fish.
        projection_model: Refractive projection model for the target camera.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        keypoint_t_values: Arc-length fractions for keypoint evaluation.
        lateral_pad: OBB lateral padding in pixels.
        max_camera_residual_px: Per-camera residual threshold in pixels.
        camera_id: Camera identifier.

    Returns:
        Dict with keys ``obb_line``, ``pose_line``, ``confidence``,
        ``raw_metrics``, ``keypoints_2d``, ``visibility``; or None if
        the label should be skipped.
    """
    # Check camera contributed to reconstruction
    if (
        midline.per_camera_residuals is None
        or camera_id not in midline.per_camera_residuals
    ):
        return None

    # Check per-camera residual threshold
    if midline.per_camera_residuals[camera_id] > max_camera_residual_px:
        return None

    # Reproject spline keypoints
    keypoints_2d, visibility = reproject_spline_keypoints(
        midline, keypoint_t_values, projection_model
    )

    # Need at least 2 visible keypoints
    if visibility.sum() < 2:
        return None

    # Edge extrapolation and OBB computation
    coords_ext, vis_ext = extrapolate_edge_keypoints(
        keypoints_2d, visibility, img_w, img_h, lateral_pad
    )
    obb_corners = pca_obb(coords_ext, vis_ext, lateral_pad)

    # Format OBB annotation line
    obb_row = format_obb_annotation(obb_corners, img_w, img_h)
    obb_line = " ".join(str(v) for v in obb_row)

    # Compute AABB from OBB corners for pose bbox
    x_min = float(obb_corners[:, 0].min())
    x_max = float(obb_corners[:, 0].max())
    y_min = float(obb_corners[:, 1].min())
    y_max = float(obb_corners[:, 1].max())

    cx = ((x_min + x_max) / 2.0) / img_w
    cy = ((y_min + y_max) / 2.0) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    # Clamp bbox to [0, 1]
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))

    # Format pose annotation line
    pose_row = format_pose_annotation(
        cx, cy, w, h, keypoints_2d, visibility, img_w, img_h
    )
    pose_line = " ".join(str(v) for v in pose_row)

    # Compute confidence score
    confidence, raw_metrics = compute_confidence_score(
        midline.mean_residual, midline.n_cameras, midline.per_camera_residuals
    )

    return {
        "obb_line": obb_line,
        "pose_line": pose_line,
        "confidence": confidence,
        "raw_metrics": raw_metrics,
        "keypoints_2d": keypoints_2d,
        "visibility": visibility,
    }
