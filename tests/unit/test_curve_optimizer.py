"""Unit tests for the curve-based 3D midline optimizer module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.reconstruction.curve_optimizer import (
    CurveOptimizer,
    CurveOptimizerConfig,
    _build_basis_matrix,
    _chamfer_distance_2d,
    _cold_start,
    _curvature_penalty,
    _length_penalty,
    _smoothness_penalty,
    get_basis,
)
from aquapose.reconstruction.midline import Midline2D
from aquapose.reconstruction.triangulation import (
    N_SAMPLE_POINTS,
    SPLINE_K,
    SPLINE_KNOTS,
)

# ---------------------------------------------------------------------------
# Synthetic camera helpers (mirrors test_triangulation.py pattern)
# ---------------------------------------------------------------------------


def _make_camera(
    cam_x: float,
    cam_y: float,
    water_z: float = 0.75,
    fx: float = 1400.0,
    cx: float = 800.0,
    cy: float = 600.0,
) -> RefractiveProjectionModel:
    """Create a synthetic camera looking down at water from above.

    Camera placed at (cam_x, cam_y, 0) with identity rotation.
    Water surface at world Z = water_z.

    Args:
        cam_x: Camera X position in world frame.
        cam_y: Camera Y position in world frame.
        water_z: Z coordinate of water surface.
        fx: Focal length in pixels.
        cx: Principal point x.
        cy: Principal point y.

    Returns:
        Configured RefractiveProjectionModel.
    """
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.tensor([-cam_x, -cam_y, 0.0], dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return RefractiveProjectionModel(K, R, t, water_z, normal, 1.0, 1.333)


def _build_synthetic_rig(
    n_cameras: int = 4,
    water_z: float = 0.75,
    radius: float = 0.5,
) -> dict[str, RefractiveProjectionModel]:
    """Build N synthetic cameras arranged in a circle above the water.

    Args:
        n_cameras: Number of cameras.
        water_z: Z coordinate of water surface.
        radius: Camera arrangement radius.

    Returns:
        Mapping from camera_id strings to RefractiveProjectionModel.
    """
    models: dict[str, RefractiveProjectionModel] = {}
    for i in range(n_cameras):
        angle = 2.0 * np.pi * i / n_cameras
        cam_x = radius * np.cos(float(angle))
        cam_y = radius * np.sin(float(angle))
        cam_id = f"cam_{i:02d}"
        models[cam_id] = _make_camera(cam_x, cam_y, water_z=water_z)
    return models


def _make_gt_spline_pts(
    n_pts: int = N_SAMPLE_POINTS, water_z: float = 0.75
) -> np.ndarray:
    """Create ground-truth 3D body points along a simple arc.

    Fish is positioned below the water surface at ~0.5m depth, oriented
    along the X axis with a slight Y curve.

    Args:
        n_pts: Number of sample points.
        water_z: Z coordinate of water surface.

    Returns:
        3D points array, shape (n_pts, 3), float32.
    """
    t = np.linspace(0.0, 1.0, n_pts)
    length = 0.085  # 85mm nominal
    x = np.linspace(-length / 2, length / 2, n_pts)
    y = 0.005 * np.sin(np.pi * t)  # slight arc
    z = np.full(n_pts, water_z + 0.5)  # 0.5m below water
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _project_points(
    pts_3d: np.ndarray,
    model: RefractiveProjectionModel,
) -> np.ndarray:
    """Project N 3D points through a camera model.

    Args:
        pts_3d: 3D points, shape (N, 3), float32.
        model: Projection model.

    Returns:
        2D pixel coordinates, shape (N, 2), float32.
    """
    pts_torch = torch.from_numpy(pts_3d).float()
    proj_px, valid = model.project(pts_torch)
    pts_2d = proj_px.numpy()
    pts_2d[~valid.numpy()] = np.nan
    return pts_2d


# ---------------------------------------------------------------------------
# B-spline basis matrix tests
# ---------------------------------------------------------------------------


def test_basis_matrix_shape() -> None:
    """Verify _build_basis_matrix returns the expected shape."""
    B = _build_basis_matrix(20, 7)
    assert B.shape == (20, 7), f"Expected (20, 7), got {B.shape}"


def test_basis_matrix_small_shape() -> None:
    """Verify basis matrix shape for coarse control point count."""
    B = _build_basis_matrix(15, 4)
    assert B.shape == (15, 4), f"Expected (15, 4), got {B.shape}"


def test_basis_matrix_partition_of_unity() -> None:
    """Each row of the basis matrix must sum to approximately 1.0.

    This is the B-spline partition-of-unity property.
    """
    B = _build_basis_matrix(20, 7)
    row_sums = B.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(20), atol=1e-5), (
        f"Row sums not 1.0: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
    )


def test_basis_matrix_endpoints() -> None:
    """First row must activate only first CP; last row only last CP.

    This is the clamped knot (endpoint interpolation) property.
    """
    B = _build_basis_matrix(20, 7)

    # First row: weight on first control point should be 1.0
    assert abs(B[0, 0].item() - 1.0) < 1e-5, f"First row first weight: {B[0, 0].item()}"
    assert B[0, 1:].abs().max().item() < 1e-5, (
        f"First row non-zero off-diagonal: {B[0, 1:]}"
    )

    # Last row: weight on last control point should be 1.0
    assert abs(B[-1, -1].item() - 1.0) < 1e-5, (
        f"Last row last weight: {B[-1, -1].item()}"
    )
    assert B[-1, :-1].abs().max().item() < 1e-5, (
        f"Last row non-zero off-diagonal: {B[-1, :-1]}"
    )


def test_basis_cache() -> None:
    """get_basis should return identical object on cache hit."""
    B1 = get_basis(20, 7, "cpu")
    B2 = get_basis(20, 7, "cpu")
    # Same tensor object in cache (data_ptr equality)
    assert B1.data_ptr() == B2.data_ptr(), (
        "Cache miss: different tensor objects returned"
    )


# ---------------------------------------------------------------------------
# Chamfer distance tests
# ---------------------------------------------------------------------------


def test_chamfer_identical_points() -> None:
    """Chamfer distance between identical point sets must be 0."""
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    dist = _chamfer_distance_2d(pts, pts)
    assert dist.item() < 1e-6, f"Expected 0, got {dist.item()}"


def test_chamfer_known_distance() -> None:
    """Chamfer distance between two known point sets matches expected value.

    Set A = [(0,0)], Set B = [(3,4)]. Only one point each.
    Distance = 5.0. Mean both directions = 5.0.
    """
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[3.0, 4.0]])
    dist = _chamfer_distance_2d(a, b)
    assert abs(dist.item() - 5.0) < 1e-4, f"Expected 5.0, got {dist.item()}"


def test_chamfer_empty_input() -> None:
    """Chamfer distance with empty tensors should return 0 without error."""
    pts = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    empty = torch.zeros(0, 2)
    dist1 = _chamfer_distance_2d(empty, pts)
    dist2 = _chamfer_distance_2d(pts, empty)
    assert dist1.item() == 0.0, f"Expected 0 for empty proj, got {dist1.item()}"
    assert dist2.item() == 0.0, f"Expected 0 for empty obs, got {dist2.item()}"


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


def test_length_penalty_nominal() -> None:
    """Length penalty must be 0 when arc length equals nominal."""
    config = CurveOptimizerConfig(nominal_length_m=0.085, length_tolerance=0.30)
    # Create a straight spline of exactly nominal length
    K = 7
    n_eval = 20
    length = 0.085
    ctrl = torch.zeros(1, K, 3)
    for k in range(K):
        ctrl[0, k, 0] = -length / 2 + k * length / (K - 1)

    B = get_basis(n_eval, K, "cpu")
    penalty = _length_penalty(ctrl, B, config)
    assert penalty.item() < 1e-6, f"Expected 0 length penalty, got {penalty.item()}"


def test_length_penalty_outside_tolerance() -> None:
    """Length penalty must be > 0 when arc length is outside ±30% of nominal."""
    config = CurveOptimizerConfig(nominal_length_m=0.085, length_tolerance=0.30)
    K = 7
    n_eval = 20
    # Spline with length 3x nominal (well outside tolerance)
    length = 0.085 * 3.0
    ctrl = torch.zeros(1, K, 3)
    for k in range(K):
        ctrl[0, k, 0] = -length / 2 + k * length / (K - 1)

    B = get_basis(n_eval, K, "cpu")
    penalty = _length_penalty(ctrl, B, config)
    assert penalty.item() > 0.0, (
        f"Expected positive length penalty, got {penalty.item()}"
    )


def test_curvature_penalty_straight() -> None:
    """Curvature penalty must be 0 for collinear control points."""
    config = CurveOptimizerConfig(max_bend_angle_deg=30.0)
    # Straight line: all points on X axis
    K = 7
    ctrl = torch.zeros(1, K, 3)
    for k in range(K):
        ctrl[0, k, 0] = float(k)

    penalty = _curvature_penalty(ctrl, config)
    assert penalty.item() < 1e-6, f"Expected 0 curvature penalty, got {penalty.item()}"


def test_curvature_penalty_sharp_bend() -> None:
    """Curvature penalty must be > 0 for a very sharp bend (near 90 degrees)."""
    config = CurveOptimizerConfig(max_bend_angle_deg=30.0)
    # Very sharp 90-degree bend at middle point (3 control points)
    ctrl = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])

    penalty = _curvature_penalty(ctrl, config)
    assert penalty.item() > 0.0, (
        f"Expected positive curvature penalty for sharp bend, got {penalty.item()}"
    )


def test_smoothness_penalty_straight() -> None:
    """Smoothness penalty must be 0 for evenly-spaced collinear points."""
    # Evenly spaced along X axis: second differences are exactly 0
    K = 7
    ctrl = torch.zeros(1, K, 3)
    for k in range(K):
        ctrl[0, k, 0] = float(k)

    penalty = _smoothness_penalty(ctrl)
    assert penalty.item() < 1e-6, f"Expected 0 smoothness penalty, got {penalty.item()}"


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


def test_cold_start_shape() -> None:
    """cold_start must return control points of shape (K, 3)."""
    K = 7
    centroid = torch.tensor([0.1, 0.2, 1.25])
    ctrl = _cold_start(centroid, None, K, 0.085, "cpu")
    assert ctrl.shape == (K, 3), f"Expected shape ({K}, 3), got {ctrl.shape}"


def test_cold_start_centered() -> None:
    """Mean of cold-start control points must be near the centroid."""
    K = 7
    centroid = torch.tensor([0.1, 0.2, 1.25])
    ctrl = _cold_start(centroid, None, K, 0.085, "cpu")
    mean_cp = ctrl.mean(dim=0)
    assert torch.allclose(mean_cp, centroid, atol=1e-5), (
        f"Mean {mean_cp} != centroid {centroid}"
    )


def test_cold_start_length() -> None:
    """Span of cold-start control points must match nominal_length along orientation."""
    K = 7
    centroid = torch.tensor([0.0, 0.0, 1.25])
    orientation = torch.tensor([1.0, 0.0, 0.0])
    nominal = 0.085
    ctrl = _cold_start(centroid, orientation, K, nominal, "cpu")

    # Span along X axis
    x_span = ctrl[-1, 0].item() - ctrl[0, 0].item()
    assert abs(x_span - nominal) < 1e-5, f"Expected span {nominal}, got {x_span}"


def test_cold_start_with_orientation() -> None:
    """Cold start with custom orientation produces points along that direction."""
    K = 5
    centroid = torch.tensor([0.0, 0.0, 1.25])
    orientation = torch.tensor([0.0, 1.0, 0.0])  # along Y
    nominal = 0.085
    ctrl = _cold_start(centroid, orientation, K, nominal, "cpu")

    # Points should vary along Y, not X
    x_range = (ctrl[:, 0].max() - ctrl[:, 0].min()).item()
    y_range = (ctrl[:, 1].max() - ctrl[:, 1].min()).item()
    assert y_range > x_range, (
        f"Expected variation along Y, got x_range={x_range}, y_range={y_range}"
    )


# ---------------------------------------------------------------------------
# Integration test: synthetic optimization
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_optimize_synthetic_fish() -> None:
    """End-to-end test: optimizer converges on synthetic 3-camera data.

    Creates a known 3D ground-truth spline (simple arc), projects into 3
    synthetic cameras, then runs CurveOptimizer and verifies:
    - Output dict is non-empty
    - Arc length is within 30% of known ground truth
    - mean_residual < 20px
    """
    water_z = 0.75
    models = _build_synthetic_rig(n_cameras=3, water_z=water_z, radius=0.5)

    # Ground-truth 3D body points
    gt_pts_3d = _make_gt_spline_pts(n_pts=N_SAMPLE_POINTS, water_z=water_z)

    # Compute ground-truth arc length
    diffs_gt = np.diff(gt_pts_3d, axis=0)
    gt_arc_length = float(np.sum(np.linalg.norm(diffs_gt, axis=1)))

    # Build MidlineSet by projecting GT points into each camera
    fish_id = 42
    cam_midlines: dict[str, Midline2D] = {}
    for cam_id, model in models.items():
        pts_2d = _project_points(gt_pts_3d, model)
        # Skip cameras where any projection is NaN
        if np.any(np.isnan(pts_2d)):
            continue
        cam_midlines[cam_id] = Midline2D(
            points=pts_2d,
            half_widths=np.full(N_SAMPLE_POINTS, 5.0, dtype=np.float32),
            fish_id=fish_id,
            camera_id=cam_id,
            frame_index=0,
            is_head_to_tail=True,
        )

    assert len(cam_midlines) >= 2, "Need at least 2 cameras for test"

    midline_set = {fish_id: cam_midlines}

    # Use fewer iterations for test speed
    config = CurveOptimizerConfig(
        nominal_length_m=0.085,
        length_tolerance=0.30,
        n_eval_points=15,
        lbfgs_max_iter_coarse=20,
        lbfgs_max_iter_fine=30,
        lambda_length=10.0,
        lambda_curvature=2.0,
        lambda_smoothness=0.5,
    )

    # Use a known centroid (GT fish center)
    gt_centroid = torch.from_numpy(gt_pts_3d.mean(axis=0)).float()
    fish_centroids = {fish_id: gt_centroid}

    optimizer = CurveOptimizer(config=config)
    results = optimizer.optimize_midlines(
        midline_set, models, frame_index=0, fish_centroids=fish_centroids
    )

    # Verify output
    assert fish_id in results, f"Fish {fish_id} not in results"
    midline_3d = results[fish_id]

    # Arc length within 30% of ground truth
    tol = 0.30
    assert midline_3d.arc_length > gt_arc_length * (1 - tol), (
        f"Arc length {midline_3d.arc_length:.4f} too small vs GT {gt_arc_length:.4f}"
    )
    assert midline_3d.arc_length < gt_arc_length * (1 + tol), (
        f"Arc length {midline_3d.arc_length:.4f} too large vs GT {gt_arc_length:.4f}"
    )

    # Mean residual below 20px
    assert midline_3d.mean_residual < 20.0, (
        f"mean_residual {midline_3d.mean_residual:.2f}px >= 20px threshold"
    )

    # Control points shape matches Midline3D contract
    assert midline_3d.control_points.shape == (7, 3), (
        f"Expected (7, 3) control points, got {midline_3d.control_points.shape}"
    )

    # Knots match SPLINE_KNOTS
    assert np.allclose(midline_3d.knots, SPLINE_KNOTS.astype(np.float32)), (
        "Knots do not match SPLINE_KNOTS"
    )

    # Degree matches
    assert midline_3d.degree == SPLINE_K

    # n_cameras reported correctly
    assert midline_3d.n_cameras == len(cam_midlines)


def test_optimizer_warm_start_stored() -> None:
    """Verify that after optimize_midlines, warm-start is stored for next frame."""
    water_z = 0.75
    models = _build_synthetic_rig(n_cameras=3, water_z=water_z, radius=0.5)
    gt_pts = _make_gt_spline_pts(water_z=water_z)

    fish_id = 1
    cam_midlines: dict[str, Midline2D] = {}
    for cam_id, model in models.items():
        pts_2d = _project_points(gt_pts, model)
        if np.any(np.isnan(pts_2d)):
            continue
        cam_midlines[cam_id] = Midline2D(
            points=pts_2d,
            half_widths=np.full(N_SAMPLE_POINTS, 5.0, dtype=np.float32),
            fish_id=fish_id,
            camera_id=cam_id,
            frame_index=0,
        )

    midline_set = {fish_id: cam_midlines}
    config = CurveOptimizerConfig(lbfgs_max_iter_coarse=5, lbfgs_max_iter_fine=5)

    optimizer = CurveOptimizer(config=config)
    optimizer.optimize_midlines(midline_set, models, frame_index=0)

    # Warm-start should now contain this fish's solution
    assert fish_id in optimizer._warm_starts, "Warm-start not stored after frame 0"
    ws = optimizer._warm_starts[fish_id]
    assert ws.shape == (config.n_fine_ctrl, 3), f"Warm-start shape mismatch: {ws.shape}"


def test_empty_midline_set_returns_empty() -> None:
    """optimize_midlines with empty input must return empty dict."""
    models = _build_synthetic_rig(n_cameras=3)
    optimizer = CurveOptimizer()
    results = optimizer.optimize_midlines({}, models, frame_index=0)
    assert results == {}, f"Expected empty dict, got {results}"


def test_config_defaults() -> None:
    """CurveOptimizerConfig must expose all regularization weights with sensible defaults."""
    cfg = CurveOptimizerConfig()
    assert cfg.nominal_length_m == 0.085
    assert cfg.length_tolerance == 0.30
    assert cfg.n_coarse_ctrl == 4
    assert cfg.n_fine_ctrl == 7
    assert cfg.max_bend_angle_deg == 30.0
    assert cfg.lambda_length > 0
    assert cfg.lambda_curvature > 0
    assert cfg.lambda_smoothness > 0
    assert cfg.warm_start_loss_ratio == 2.0
