"""Core pseudo-label generation: spline reprojection, confidence scoring, per-fish label generation."""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline

from aquapose.calibration.luts import InverseLUT, ghost_point_lookup
from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.core.types.reconstruction import Midline3D
from aquapose.training.geometry import (
    compute_arc_length,
    extrapolate_edge_keypoints,
    format_obb_annotation,
    format_pose_annotation,
    pca_obb,
)


def compute_curvature(control_points: np.ndarray) -> float:
    """Compute mean absolute curvature from control points via finite differences.

    Uses finite differences of tangent vectors:
    ``T[i] = cp[i+1] - cp[i]``, then
    ``k[i] = |T[i+1] - T[i]| / (0.5 * (|T[i]| + |T[i+1]|))``.

    Args:
        control_points: Shape ``(N, D)`` array of control points (2D or 3D).

    Returns:
        Mean absolute curvature (scalar). Near zero for straight lines.
    """
    tangents = np.diff(control_points, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1)

    dt = np.diff(tangents, axis=0)
    dt_norms = np.linalg.norm(dt, axis=1)

    # Average adjacent tangent magnitudes for normalization
    avg_norms = 0.5 * (tangent_norms[:-1] + tangent_norms[1:])

    # Avoid division by zero for degenerate cases
    safe_norms = np.where(avg_norms > 1e-12, avg_norms, 1.0)
    curvatures = dt_norms / safe_norms

    return float(np.mean(curvatures))


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
    lateral_ratio: float,
    max_camera_residual_px: float,
    camera_id: str,
    min_visible_keypoints: int = 2,
    edge_factor: float = 2.0,
) -> dict | None:
    """Generate OBB and pose labels for one fish in one camera view.

    Returns None if the camera did not contribute to the reconstruction,
    or if its per-camera residual exceeds the threshold, or if fewer than
    2 keypoints are visible after projection. The ``pose_line`` key is
    only included when at least *min_visible_keypoints* are visible.

    Args:
        midline: 3D midline reconstruction for the fish.
        projection_model: Refractive projection model for the target camera.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        keypoint_t_values: Arc-length fractions for keypoint evaluation.
        lateral_ratio: Fraction of arc length for lateral OBB padding.
        max_camera_residual_px: Per-camera residual threshold in pixels.
        camera_id: Camera identifier.
        min_visible_keypoints: Minimum visible keypoints to include pose label.
        edge_factor: Multiplier on lateral_pad for edge extrapolation threshold.

    Returns:
        Dict with keys ``obb_line``, ``confidence``, ``raw_metrics``,
        ``keypoints_2d``, ``visibility``, ``lateral_pad``, and optionally
        ``pose_line`` (only when ``n_visible >= min_visible_keypoints``);
        or None if the label should be skipped entirely.
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

    # Mark out-of-bounds keypoints as invisible
    oob = (
        (keypoints_2d[:, 0] < 0)
        | (keypoints_2d[:, 0] >= img_w)
        | (keypoints_2d[:, 1] < 0)
        | (keypoints_2d[:, 1] >= img_h)
    )
    visibility = visibility & ~oob

    # OBB needs at least 2 visible keypoints for PCA orientation
    n_visible = int(visibility.sum())
    if n_visible < 2:
        return None

    # Compute data-driven lateral pad from arc length (matches manual pipeline)
    arc_length = compute_arc_length(keypoints_2d, visibility)
    lateral_pad = max(arc_length * lateral_ratio, 5.0)  # floor at 5px

    # Edge extrapolation (matches manual pipeline)
    kp_ext, vis_ext = extrapolate_edge_keypoints(
        keypoints_2d, visibility, img_w, img_h, lateral_pad, edge_factor
    )

    # OBB computation from edge-extrapolated keypoints
    obb_corners = pca_obb(kp_ext, vis_ext, lateral_pad)

    # Format OBB annotation line
    obb_row = format_obb_annotation(obb_corners, img_w, img_h)
    obb_line = " ".join(str(v) for v in obb_row)

    # Compute confidence score
    confidence, raw_metrics = compute_confidence_score(
        midline.mean_residual, midline.n_cameras, midline.per_camera_residuals
    )

    result: dict = {
        "obb_line": obb_line,
        "confidence": confidence,
        "raw_metrics": raw_metrics,
        "keypoints_2d": keypoints_2d,
        "visibility": visibility,
        "lateral_pad": lateral_pad,
    }

    # Pose label requires meeting the higher min_visible_keypoints threshold
    if n_visible >= min_visible_keypoints:
        x_min = float(obb_corners[:, 0].min())
        x_max = float(obb_corners[:, 0].max())
        y_min = float(obb_corners[:, 1].min())
        y_max = float(obb_corners[:, 1].max())

        cx = float(np.clip(((x_min + x_max) / 2.0) / img_w, 0.0, 1.0))
        cy = float(np.clip(((y_min + y_max) / 2.0) / img_h, 0.0, 1.0))
        w = float(np.clip((x_max - x_min) / img_w, 0.0, 1.0))
        h = float(np.clip((y_max - y_min) / img_h, 0.0, 1.0))

        pose_row = format_pose_annotation(
            cx, cy, w, h, keypoints_2d, visibility, img_w, img_h
        )
        result["pose_line"] = " ".join(str(v) for v in pose_row)

    return result


def detect_gaps(
    midline: Midline3D,
    inverse_lut: InverseLUT,
    proj_models: dict[str, RefractiveProjectionModel],
    frame_detections: dict[str, list],
    frame_tracks: dict[str, list],
    min_cameras: int = 3,
) -> list[tuple[str, str]]:
    """Identify gap cameras and classify each by failure reason.

    A gap camera is one that the InverseLUT indicates should see the fish
    (based on its 3D centroid) but that did not contribute to the
    reconstruction (not present in ``per_camera_residuals``).

    Args:
        midline: 3D midline reconstruction for the fish.
        inverse_lut: Inverse lookup table for visibility queries.
        proj_models: Per-camera refractive projection models.
        frame_detections: Camera-keyed detection lists for this frame.
        frame_tracks: Camera-keyed tracklet lists for this frame.
        min_cameras: Minimum contributing cameras to activate gap detection.

    Returns:
        List of ``(camera_id, reason)`` where reason is one of
        ``'no-detection'``, ``'no-tracklet'``, ``'failed-midline'``.
        Empty list if ``per_camera_residuals`` is None or fewer than
        ``min_cameras`` cameras contributed.
    """
    if midline.per_camera_residuals is None:
        return []

    contributing = set(midline.per_camera_residuals.keys())
    if len(contributing) < min_cameras:
        return []

    # Compute centroid (mean of control points)
    centroid_3d = midline.control_points.mean(axis=0)  # shape (3,)
    centroid_tensor = torch.from_numpy(centroid_3d[None].astype(np.float32))

    # Query InverseLUT for visibility
    visible_list = ghost_point_lookup(inverse_lut, centroid_tensor)
    visible_cam_ids = {cam_id for cam_id, _, _ in visible_list[0]}

    # Gap cameras = visible - contributing
    gap_cam_ids = visible_cam_ids - contributing

    # Classify each gap
    gaps: list[tuple[str, str]] = []
    for cam_id in gap_cam_ids:
        reason = _classify_gap(
            cam_id, midline, proj_models[cam_id], frame_detections, frame_tracks
        )
        gaps.append((cam_id, reason))
    return gaps


def _classify_gap(
    cam_id: str,
    midline: Midline3D,
    proj_model: RefractiveProjectionModel,
    frame_detections: dict[str, list],
    frame_tracks: dict[str, list],
) -> str:
    """Classify a gap camera by checking pipeline stages in reverse order.

    Uses the projected 3D centroid (via RefractiveProjectionModel) for the
    bbox overlap check rather than LUT pixel coordinates.

    Args:
        cam_id: Camera identifier for the gap camera.
        midline: 3D midline reconstruction for the fish.
        proj_model: Refractive projection model for this camera.
        frame_detections: Camera-keyed detection lists for this frame.
        frame_tracks: Camera-keyed tracklet lists for this frame.

    Returns:
        One of ``'no-detection'``, ``'no-tracklet'``, ``'failed-midline'``.
    """
    # Project 3D centroid into this camera
    centroid_3d = midline.control_points.mean(axis=0)
    px, valid = proj_model.project(
        torch.from_numpy(centroid_3d[None].astype(np.float32))
    )

    if not valid[0]:
        return "no-detection"

    # CUDA safety: .cpu().detach().numpy()
    cx = float(px[0, 0].cpu().detach())
    cy = float(px[0, 1].cpu().detach())

    # Check detections: does any bbox contain the projected centroid?
    cam_dets = frame_detections.get(cam_id, [])
    matching_det = None
    for det in cam_dets:
        x, y, w, h = det.bbox
        if x <= cx <= x + w and y <= cy <= y + h:
            matching_det = det
            break

    if matching_det is None:
        return "no-detection"

    # Check tracklets: does any tracklet cover this frame with "detected" status?
    cam_tracks = frame_tracks.get(cam_id, [])
    has_tracklet = False
    for tracklet in cam_tracks:
        try:
            idx = tracklet.frames.index(midline.frame_index)
        except ValueError:
            continue
        if tracklet.frame_status[idx] == "detected":
            has_tracklet = True
            break

    if not has_tracklet:
        return "no-tracklet"

    return "failed-midline"


def _passes_bounds_check(
    keypoints_2d: np.ndarray,
    visibility: np.ndarray,
    img_w: int,
    img_h: int,
) -> bool:
    """Check that most visible keypoints fall within image bounds.

    Args:
        keypoints_2d: Keypoint pixel coordinates, shape ``(N, 2)``.
        visibility: Boolean visibility mask, shape ``(N,)``.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        True if at least 50% of visible keypoints are within bounds.
    """
    vis_pts = keypoints_2d[visibility]
    if len(vis_pts) < 2:
        return False
    in_bounds = (
        (vis_pts[:, 0] >= 0)
        & (vis_pts[:, 0] < img_w)
        & (vis_pts[:, 1] >= 0)
        & (vis_pts[:, 1] < img_h)
    )
    return bool(in_bounds.sum() >= len(vis_pts) * 0.5)


def generate_gap_fish_labels(
    midline: Midline3D,
    projection_model: RefractiveProjectionModel,
    img_w: int,
    img_h: int,
    keypoint_t_values: list[float],
    lateral_ratio: float,
    min_visible_keypoints: int = 2,
    edge_factor: float = 2.0,
) -> dict | None:
    """Generate OBB and pose labels for a gap camera.

    Same logic as :func:`generate_fish_labels` but without the
    ``per_camera_residuals`` membership check or per-camera residual
    threshold check. Uses the fish-level confidence score.

    Args:
        midline: 3D midline reconstruction for the fish.
        projection_model: Refractive projection model for the target camera.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        keypoint_t_values: Arc-length fractions for keypoint evaluation.
        lateral_ratio: Fraction of arc length for lateral OBB padding.
        min_visible_keypoints: Minimum visible keypoints to include pose label.
        edge_factor: Multiplier on lateral_pad for edge extrapolation threshold.

    Returns:
        Dict with keys ``obb_line``, ``confidence``, ``raw_metrics``,
        ``keypoints_2d``, ``visibility``, ``lateral_pad``, and optionally
        ``pose_line`` (only when ``n_visible >= min_visible_keypoints``);
        or None if fewer than 2 keypoints are visible or bounds check fails.
    """
    # Reproject spline keypoints
    keypoints_2d, visibility = reproject_spline_keypoints(
        midline, keypoint_t_values, projection_model
    )

    # Mark out-of-bounds keypoints as invisible
    oob = (
        (keypoints_2d[:, 0] < 0)
        | (keypoints_2d[:, 0] >= img_w)
        | (keypoints_2d[:, 1] < 0)
        | (keypoints_2d[:, 1] >= img_h)
    )
    visibility = visibility & ~oob

    # OBB needs at least 2 visible keypoints for PCA orientation
    n_visible = int(visibility.sum())
    if n_visible < 2:
        return None

    # Bounds check
    if not _passes_bounds_check(keypoints_2d, visibility, img_w, img_h):
        return None

    # Compute data-driven lateral pad from arc length (matches manual pipeline)
    arc_length = compute_arc_length(keypoints_2d, visibility)
    lateral_pad = max(arc_length * lateral_ratio, 5.0)  # floor at 5px

    # Edge extrapolation (matches manual pipeline)
    kp_ext, vis_ext = extrapolate_edge_keypoints(
        keypoints_2d, visibility, img_w, img_h, lateral_pad, edge_factor
    )

    # OBB computation from edge-extrapolated keypoints
    obb_corners = pca_obb(kp_ext, vis_ext, lateral_pad)

    # Format OBB annotation line
    obb_row = format_obb_annotation(obb_corners, img_w, img_h)
    obb_line = " ".join(str(v) for v in obb_row)

    # Compute confidence score (fish-level, no discount for gaps)
    confidence, raw_metrics = compute_confidence_score(
        midline.mean_residual, midline.n_cameras, midline.per_camera_residuals
    )

    result: dict = {
        "obb_line": obb_line,
        "confidence": confidence,
        "raw_metrics": raw_metrics,
        "keypoints_2d": keypoints_2d,
        "visibility": visibility,
        "lateral_pad": lateral_pad,
    }

    # Pose label requires meeting the higher min_visible_keypoints threshold
    if n_visible >= min_visible_keypoints:
        x_min = float(obb_corners[:, 0].min())
        x_max = float(obb_corners[:, 0].max())
        y_min = float(obb_corners[:, 1].min())
        y_max = float(obb_corners[:, 1].max())

        cx = float(np.clip(((x_min + x_max) / 2.0) / img_w, 0.0, 1.0))
        cy = float(np.clip(((y_min + y_max) / 2.0) / img_h, 0.0, 1.0))
        w = float(np.clip((x_max - x_min) / img_w, 0.0, 1.0))
        h = float(np.clip((y_max - y_min) / img_h, 0.0, 1.0))

        pose_row = format_pose_annotation(
            cx, cy, w, h, keypoints_2d, visibility, img_w, img_h
        )
        result["pose_line"] = " ".join(str(v) for v in pose_row)

    return result
