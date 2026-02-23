"""Unit tests for the synthetic data generation module."""

from __future__ import annotations

import math

import numpy as np
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.reconstruction.triangulation import (
    N_SAMPLE_POINTS,
    SPLINE_K,
    SPLINE_KNOTS,
    SPLINE_N_CTRL,
    Midline3D,
)
from aquapose.synthetic import (
    FishConfig,
    build_fabricated_rig,
    generate_fish_3d,
    generate_fish_half_widths,
    generate_synthetic_midline_sets,
    make_ground_truth_midline3d,
    project_fish_to_midline2d,
)

# ---------------------------------------------------------------------------
# Rig tests
# ---------------------------------------------------------------------------


def test_build_fabricated_rig_default() -> None:
    """Default 3x3 grid produces 9 cameras with correct IDs and types."""
    rig = build_fabricated_rig()
    assert len(rig) == 9
    for cam_id, model in rig.items():
        assert cam_id.startswith("syn_"), f"ID {cam_id!r} should start with 'syn_'"
        assert isinstance(model, RefractiveProjectionModel)


def test_build_fabricated_rig_custom() -> None:
    """Custom 2x4 grid returns 8 cameras."""
    rig = build_fabricated_rig(
        n_cameras_x=2, n_cameras_y=4, spacing_x=0.3, spacing_y=0.3
    )
    assert len(rig) == 8


# ---------------------------------------------------------------------------
# Fish 3D shape tests
# ---------------------------------------------------------------------------


def test_generate_fish_3d_straight() -> None:
    """Straight fish (curvature=0) has correct point count and arc length."""
    cfg = FishConfig(
        position=(0.0, 0.0, 1.25), heading_rad=0.0, curvature=0.0, scale=0.085
    )
    pts = generate_fish_3d(cfg)

    assert pts.shape == (N_SAMPLE_POINTS, 3)
    assert pts.dtype == np.float32

    # Total arc length should be ~= scale (0.085m)
    diffs = np.diff(pts, axis=0)
    arc_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    assert abs(arc_len - 0.085) < 1e-4, f"Arc length {arc_len:.6f} != 0.085"


def test_generate_fish_3d_arc() -> None:
    """Curved fish (curvature=10) has correct arc length and non-trivial curve."""
    cfg = FishConfig(
        position=(0.0, 0.0, 1.25), heading_rad=0.0, curvature=10.0, scale=0.085
    )
    pts = generate_fish_3d(cfg)

    assert pts.shape == (N_SAMPLE_POINTS, 3)

    # Arc length should be ~= scale (0.085m)
    diffs = np.diff(pts, axis=0)
    arc_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    assert abs(arc_len - 0.085) < 1e-4, f"Arc length {arc_len:.6f} != 0.085"

    # Points should not all be collinear (non-trivial curvature)
    x_range = float(pts[:, 0].max() - pts[:, 0].min())
    y_range = float(pts[:, 1].max() - pts[:, 1].min())
    # Both X and Y should have non-trivial extent for a curved fish
    assert x_range > 1e-4 or y_range > 1e-4


def test_generate_fish_3d_heading() -> None:
    """Heading=pi/2 rotates fish to spread along Y instead of X."""
    cfg_x = FishConfig(heading_rad=0.0, curvature=0.0, scale=0.085)
    cfg_y = FishConfig(heading_rad=math.pi / 2, curvature=0.0, scale=0.085)

    pts_x = generate_fish_3d(cfg_x)
    pts_y = generate_fish_3d(cfg_y)

    x_range_x = float(pts_x[:, 0].max() - pts_x[:, 0].min())
    y_range_x = float(pts_x[:, 1].max() - pts_x[:, 1].min())
    x_range_y = float(pts_y[:, 0].max() - pts_y[:, 0].min())
    y_range_y = float(pts_y[:, 1].max() - pts_y[:, 1].min())

    # heading=0: spread along X
    assert x_range_x > y_range_x * 10, "heading=0 should spread along X"
    # heading=pi/2: spread along Y
    assert y_range_y > x_range_y * 10, "heading=pi/2 should spread along Y"


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------


def _make_test_camera(
    cam_x: float = 0.0,
    cam_y: float = 0.0,
    water_z: float = 0.75,
) -> RefractiveProjectionModel:
    """Create a single synthetic camera for testing."""
    K = torch.tensor(
        [[1400.0, 0.0, 800.0], [0.0, 1400.0, 600.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.tensor([-cam_x, -cam_y, 0.0], dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return RefractiveProjectionModel(K, R, t, water_z, normal, 1.0, 1.333)


def test_project_fish_returns_midline2d() -> None:
    """Project straight fish through one camera; verify Midline2D structure."""
    cfg = FishConfig(
        position=(0.0, 0.0, 1.25), heading_rad=0.0, curvature=0.0, scale=0.085
    )
    pts_3d = generate_fish_3d(cfg)
    half_widths = generate_fish_half_widths(n_points=cfg.n_points, scale=cfg.scale)
    model = _make_test_camera()

    result = project_fish_to_midline2d(
        pts_3d=pts_3d,
        half_widths_3d=half_widths,
        model=model,
        fish_id=0,
        camera_id="test_cam",
        frame_index=0,
    )

    assert result is not None
    assert result.points.shape == (N_SAMPLE_POINTS, 2)
    assert result.half_widths.shape == (N_SAMPLE_POINTS,)
    assert result.fish_id == 0
    assert result.camera_id == "test_cam"
    assert result.frame_index == 0
    assert result.is_head_to_tail is True


def test_project_fish_returns_none_when_not_visible() -> None:
    """Camera placed far away with fish behind it returns None."""
    # Fish at Z=1.25 (underwater), camera looking down but fish at water_z=5.0
    # so the fish is actually above the water from this model's perspective
    model_far = _make_test_camera(cam_x=0.0, cam_y=0.0, water_z=5.0)

    cfg = FishConfig(position=(0.0, 0.0, 1.25), curvature=0.0, scale=0.085)
    pts_3d = generate_fish_3d(cfg)
    half_widths = generate_fish_half_widths()

    result = project_fish_to_midline2d(
        pts_3d=pts_3d,
        half_widths_3d=half_widths,
        model=model_far,
        fish_id=0,
        camera_id="far_cam",
        frame_index=0,
    )

    # Fish is above water_z=5.0 (fish_z=1.25 < 5.0), so no valid projections
    assert result is None


# ---------------------------------------------------------------------------
# MidlineSet generation tests
# ---------------------------------------------------------------------------


def test_generate_synthetic_midline_sets_structure() -> None:
    """Default call returns correct structure with 1 frame and 1 fish."""
    rig = build_fabricated_rig()
    midline_sets, ground_truths = generate_synthetic_midline_sets(rig)

    assert len(midline_sets) == 1
    assert len(ground_truths) == 1

    # Fish 0 should be in the midline set with multiple camera entries
    assert 0 in midline_sets[0], "Fish 0 should be present in midline_set"
    cam_midlines = midline_sets[0][0]
    assert len(cam_midlines) >= 2, "Fish should be visible in at least 2 cameras"

    # All entries should be Midline2D-compatible (have points, half_widths, fish_id)
    for _cam_id, ml in cam_midlines.items():
        assert ml.fish_id == 0
        assert ml.points.shape[1] == 2


def test_generate_synthetic_midline_sets_multi_fish() -> None:
    """3 fish configs result in all 3 fish_ids in the MidlineSet."""
    rig = build_fabricated_rig()
    configs = [
        FishConfig(position=(-0.1, 0.0, 1.25), curvature=0.0),
        FishConfig(position=(0.0, 0.0, 1.25), curvature=0.0),
        FishConfig(position=(0.1, 0.0, 1.25), curvature=15.0),
    ]
    midline_sets, ground_truths = generate_synthetic_midline_sets(
        rig, fish_configs=configs
    )

    assert len(midline_sets) == 1
    ms = midline_sets[0]
    gt = ground_truths[0]

    assert 0 in ms, "Fish 0 should be present in midline_set"
    assert 1 in ms, "Fish 1 should be present in midline_set"
    assert 2 in ms, "Fish 2 should be present in midline_set"
    assert 0 in gt, "Fish 0 GT should be present"
    assert 1 in gt, "Fish 1 GT should be present"
    assert 2 in gt, "Fish 2 GT should be present"


# ---------------------------------------------------------------------------
# Ground truth Midline3D tests
# ---------------------------------------------------------------------------


def test_ground_truth_midline3d_valid() -> None:
    """Ground truth Midline3D has correct shapes and valid values."""
    cfg = FishConfig(position=(0.0, 0.0, 1.25), curvature=0.0, scale=0.085)
    pts_3d = generate_fish_3d(cfg)
    half_widths = generate_fish_half_widths(n_points=cfg.n_points, scale=cfg.scale)

    gt = make_ground_truth_midline3d(
        fish_id=0,
        frame_index=5,
        pts_3d=pts_3d,
        half_widths=half_widths,
    )

    assert isinstance(gt, Midline3D)
    assert gt.control_points.shape == (SPLINE_N_CTRL, 3)
    assert gt.knots.shape == (len(SPLINE_KNOTS),)
    assert gt.degree == SPLINE_K
    assert gt.arc_length > 0.0
    assert gt.fish_id == 0
    assert gt.frame_index == 5
    assert gt.n_cameras == 99
    assert gt.mean_residual == 0.0
    assert gt.max_residual == 0.0


# ---------------------------------------------------------------------------
# Round-trip accuracy test
# ---------------------------------------------------------------------------


def test_round_trip_accuracy() -> None:
    """Projected 2D points are within 1px of direct model.project() call."""
    cfg = FishConfig(position=(0.0, 0.0, 1.25), curvature=0.0, scale=0.085)
    pts_3d = generate_fish_3d(cfg)
    half_widths = generate_fish_half_widths(n_points=cfg.n_points, scale=cfg.scale)

    model = _make_test_camera()
    ml = project_fish_to_midline2d(
        pts_3d=pts_3d,
        half_widths_3d=half_widths,
        model=model,
        fish_id=0,
        camera_id="cam",
        frame_index=0,
    )
    assert ml is not None

    # Direct projection for comparison
    pts_torch = torch.from_numpy(pts_3d)
    direct_px, valid = model.project(pts_torch)
    direct_np = direct_px.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()

    # For every valid point, the projected pixel should match directly
    for i in range(N_SAMPLE_POINTS):
        if valid_np[i] and not np.any(np.isnan(ml.points[i])):
            err = float(np.linalg.norm(ml.points[i] - direct_np[i]))
            assert err < 1.0, f"Point {i}: round-trip error {err:.3f}px > 1px"
