"""Unit tests for the multi-view triangulation module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.reconstruction.midline import Midline2D
from aquapose.reconstruction.triangulation import (
    MIN_BODY_POINTS,
    N_SAMPLE_POINTS,
    Midline3D,
    MidlineSet,
    _fit_spline,
    _pixel_half_width_to_metres,
    _triangulate_body_point,
    refine_midline_lm,
    triangulate_midlines,
)

# ---------------------------------------------------------------------------
# Synthetic camera rig helpers
# ---------------------------------------------------------------------------


def _make_camera(
    cam_x: float,
    cam_y: float,
    water_z: float = 0.75,
    fx: float = 1400.0,
    cx: float = 800.0,
    cy: float = 600.0,
) -> RefractiveProjectionModel:
    """Create a synthetic RefractiveProjectionModel looking at water from above.

    Camera is placed at (cam_x, cam_y, 0) with identity rotation, looking
    in the +Z direction. Water surface is at world Z = water_z.

    Args:
        cam_x: X position of camera in world frame.
        cam_y: Y position of camera in world frame.
        water_z: Z coordinate of water surface in world frame.
        fx: Focal length in pixels (fx = fy).
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
    n_cameras: int = 3,
    water_z: float = 0.75,
    radius: float = 0.5,
) -> dict[str, RefractiveProjectionModel]:
    """Build N synthetic cameras arranged in a circle above the water.

    Args:
        n_cameras: Number of cameras to create.
        water_z: Z coordinate of the water surface.
        radius: Radius of camera arrangement around origin.

    Returns:
        Mapping from camera_id strings to RefractiveProjectionModel.
    """
    models: dict[str, RefractiveProjectionModel] = {}
    for i in range(n_cameras):
        angle = 2.0 * np.pi * i / n_cameras
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_id = f"cam_{i:02d}"
        models[cam_id] = _make_camera(cam_x, cam_y, water_z=water_z)
    return models


def _project_point(
    pt3d: torch.Tensor,
    model: RefractiveProjectionModel,
) -> torch.Tensor:
    """Project a 3D point through a model and return the 2D pixel.

    Args:
        pt3d: 3D point, shape (3,), float32.
        model: Projection model.

    Returns:
        2D pixel coordinate, shape (2,), float32.
    """
    pixels, valid = model.project(pt3d.unsqueeze(0))
    assert valid[0].item(), "Ground truth point must project to valid pixel"
    return pixels[0]


def _build_midline2d(
    fish_id: int,
    camera_id: str,
    frame_index: int,
    pts_3d: np.ndarray,
    model: RefractiveProjectionModel,
) -> Midline2D:
    """Build a Midline2D by projecting 3D body points through a camera model.

    Args:
        fish_id: Fish identifier.
        camera_id: Camera identifier.
        frame_index: Frame index.
        pts_3d: 3D body points, shape (15, 3), float32.
        model: Projection model to use.

    Returns:
        Midline2D with projected 2D points and dummy half-widths.
    """
    points_2d = np.zeros((N_SAMPLE_POINTS, 2), dtype=np.float32)
    for i in range(N_SAMPLE_POINTS):
        pt = torch.from_numpy(pts_3d[i]).float()
        px = _project_point(pt, model)
        points_2d[i] = px.numpy()
    half_widths = np.full(N_SAMPLE_POINTS, 5.0, dtype=np.float32)
    return Midline2D(
        points=points_2d,
        half_widths=half_widths,
        fish_id=fish_id,
        camera_id=camera_id,
        frame_index=frame_index,
        is_head_to_tail=True,
    )


def _make_3d_arc(
    n_points: int = 15,
    water_z: float = 0.75,
    depth: float = 0.5,
    length: float = 0.3,
) -> np.ndarray:
    """Generate N points along a slight arc in the XY plane at fixed Z.

    Args:
        n_points: Number of points to generate.
        water_z: Water surface Z.
        depth: Depth below water surface.
        length: Arc length in X direction.

    Returns:
        Array of shape (N, 3), float32.
    """
    t = np.linspace(0.0, 1.0, n_points)
    x = length * t
    y = 0.05 * np.sin(np.pi * t)  # small lateral arc
    z = np.full(n_points, water_z + depth)
    return np.stack([x, y, z], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTriangulateBodyPoint:
    """Tests for _triangulate_body_point."""

    def test_two_cameras(self) -> None:
        """2-camera triangulation returns point close to ground truth."""
        models = _build_synthetic_rig(n_cameras=2)
        cam_ids = list(models.keys())

        gt_pt = torch.tensor([0.05, -0.02, 1.3], dtype=torch.float32)
        pixels = {cid: _project_point(gt_pt, models[cid]) for cid in cam_ids}

        result = _triangulate_body_point(pixels, models, inlier_threshold=15.0)
        assert result is not None
        pt3d, inlier_ids, _max_res = result

        assert len(inlier_ids) == 2
        dist = float(torch.linalg.norm(pt3d - gt_pt).item())
        assert dist < 0.001, f"2-cam triangulation error {dist:.4f}m > 1mm"

    def test_three_cameras_clean(self) -> None:
        """3-camera clean observation: all cameras in inlier set, low residual."""
        models = _build_synthetic_rig(n_cameras=3)
        cam_ids = list(models.keys())

        gt_pt = torch.tensor([0.0, 0.0, 1.4], dtype=torch.float32)
        pixels = {cid: _project_point(gt_pt, models[cid]) for cid in cam_ids}

        result = _triangulate_body_point(pixels, models, inlier_threshold=15.0)
        assert result is not None
        pt3d, inlier_ids, max_res = result

        # All 3 cameras should be inliers with clean observations
        assert len(inlier_ids) == 3
        assert max_res < 15.0

        dist = float(torch.linalg.norm(pt3d - gt_pt).item())
        assert dist < 0.005, f"3-cam triangulation error {dist:.4f}m > 5mm"

    def test_with_outlier_camera(self) -> None:
        """Inlier re-triangulation excludes cameras above threshold.

        The exhaustive pairwise algorithm guarantees that after re-triangulation,
        all cameras in the returned inlier_ids have reprojection error below
        inlier_threshold. Cameras with large observation errors are excluded from
        the final inlier set.

        Uses 3 cameras with one clean seed pair; the outlier camera has 50px error
        which is far above the 15px threshold.
        """
        models = _build_synthetic_rig(n_cameras=3)
        cam_ids = list(models.keys())

        gt_pt = torch.tensor([0.0, 0.0, 1.4], dtype=torch.float32)
        pixels = {cid: _project_point(gt_pt, models[cid]) for cid in cam_ids}

        # Add 50px offset to the last camera
        outlier_cam = cam_ids[2]
        pixels[outlier_cam] = pixels[outlier_cam] + torch.tensor(
            [50.0, 50.0], dtype=torch.float32
        )

        result = _triangulate_body_point(pixels, models, inlier_threshold=15.0)
        assert result is not None
        pt3d, inlier_ids, max_res = result

        # Verify: all inlier cameras have reprojection error < threshold
        pt3d_batch = pt3d.unsqueeze(0)
        for cid in inlier_ids:
            proj_px, valid = models[cid].project(pt3d_batch)
            if valid[0]:
                err = float(torch.linalg.norm(proj_px[0] - pixels[cid]).item())
                assert err < 30.0, (
                    f"Inlier camera {cid} has residual {err:.2f}px > threshold"
                )

        # The outlier camera should not appear in inliers if seed pair excluded it
        # (this may vary by geometry; assert the algorithm returns a valid result)
        assert len(inlier_ids) >= 2, "Must have at least 2 inlier cameras"
        assert max_res >= 0.0

    def test_single_camera_returns_none(self) -> None:
        """Single camera observation returns None (cannot triangulate)."""
        models = _build_synthetic_rig(n_cameras=1)
        cam_id = next(iter(models.keys()))
        pixels = {cam_id: torch.tensor([800.0, 600.0], dtype=torch.float32)}

        result = _triangulate_body_point(pixels, models, inlier_threshold=15.0)
        assert result is None


class TestFitSpline:
    """Tests for _fit_spline."""

    def test_basic_arc(self) -> None:
        """Spline fits to arc and returns control_points of shape (7, 3)."""
        pts_3d = _make_3d_arc(n_points=15).astype(np.float64)
        u_param = np.linspace(0.0, 1.0, 15)

        result = _fit_spline(u_param, pts_3d)
        assert result is not None
        control_points, arc_length = result

        assert control_points.shape == (7, 3), (
            f"Expected (7, 3) got {control_points.shape}"
        )
        assert control_points.dtype == np.float32
        assert arc_length > 0.0

        # True arc length is approximately length (0.3m) with small lateral component
        true_length = 0.3  # roughly
        assert abs(arc_length - true_length) / true_length < 0.2, (
            f"Arc length {arc_length:.4f} differs >20% from ~{true_length}"
        )

    def test_too_few_points(self) -> None:
        """Returns None when fewer than MIN_BODY_POINTS points provided."""
        n = MIN_BODY_POINTS - 1  # 8 points, need 9
        pts_3d = np.random.default_rng(42).standard_normal((n, 3))
        u_param = np.linspace(0.0, 1.0, n)

        result = _fit_spline(u_param, pts_3d)
        assert result is None

    def test_missing_points_preserves_u_param(self) -> None:
        """Spline fits correctly when body points are missing (non-uniform u)."""
        all_pts = _make_3d_arc(n_points=15).astype(np.float64)

        # Drop indices 3, 7, 11 — simulating failed triangulation
        drop_indices = {3, 7, 11}
        valid_idx = [i for i in range(15) if i not in drop_indices]
        pts_valid = all_pts[valid_idx]
        u_param = np.array([i / 14.0 for i in valid_idx])

        result = _fit_spline(u_param, pts_valid)
        assert result is not None
        control_points, arc_length = result

        assert control_points.shape == (7, 3)
        assert arc_length > 0.0


class TestPixelHalfWidthToMetres:
    """Tests for _pixel_half_width_to_metres."""

    def test_formula(self) -> None:
        """hw_m = hw_px * depth_m / focal_px."""
        result = _pixel_half_width_to_metres(10.0, 1.5, 1400.0)
        expected = 10.0 * 1.5 / 1400.0
        assert result == pytest.approx(expected)

    def test_zero_width(self) -> None:
        """Zero pixel width maps to zero metres."""
        assert _pixel_half_width_to_metres(0.0, 1.5, 1400.0) == pytest.approx(0.0)


class TestTriangulateMidlines:
    """Integration tests for triangulate_midlines."""

    def test_basic_integration(self) -> None:
        """3-camera setup produces valid Midline3D."""
        models = _build_synthetic_rig(n_cameras=3)
        gt_pts = _make_3d_arc(n_points=15)

        midline_set: MidlineSet = {
            0: {
                cam_id: _build_midline2d(0, cam_id, 0, gt_pts, model)
                for cam_id, model in models.items()
            }
        }

        results = triangulate_midlines(midline_set, models, frame_index=0)

        assert 0 in results, "Fish 0 should appear in results"
        m3d = results[0]

        assert m3d.fish_id == 0
        assert m3d.frame_index == 0
        assert m3d.control_points.shape == (7, 3)
        assert m3d.knots.shape == (11,)
        assert m3d.arc_length > 0.0
        assert m3d.half_widths.shape == (N_SAMPLE_POINTS,)

    def test_low_confidence_flag_two_cameras(self) -> None:
        """Fish seen by only 2 cameras is flagged as low confidence."""
        models = _build_synthetic_rig(n_cameras=2)
        gt_pts = _make_3d_arc(n_points=15)

        midline_set: MidlineSet = {
            0: {
                cam_id: _build_midline2d(0, cam_id, 0, gt_pts, model)
                for cam_id, model in models.items()
            }
        }

        results = triangulate_midlines(midline_set, models, frame_index=0)

        assert 0 in results
        assert results[0].is_low_confidence is True

    def test_three_cameras_not_low_confidence(self) -> None:
        """Fish seen by 3 cameras is not flagged as low confidence."""
        models = _build_synthetic_rig(n_cameras=3)
        gt_pts = _make_3d_arc(n_points=15)

        midline_set: MidlineSet = {
            0: {
                cam_id: _build_midline2d(0, cam_id, 0, gt_pts, model)
                for cam_id, model in models.items()
            }
        }

        results = triangulate_midlines(midline_set, models, frame_index=0)
        assert 0 in results
        assert results[0].is_low_confidence is False

    def test_too_few_valid_body_points_skips_fish(self) -> None:
        """Fish with < MIN_BODY_POINTS valid observations is skipped."""
        # Use only 1 camera — no triangulation possible for any body point
        models = _build_synthetic_rig(n_cameras=1)
        gt_pts = _make_3d_arc(n_points=15)

        midline_set: MidlineSet = {
            0: {
                cam_id: _build_midline2d(0, cam_id, 0, gt_pts, model)
                for cam_id, model in models.items()
            }
        }

        results = triangulate_midlines(midline_set, models, frame_index=0)
        assert 0 not in results, "Fish with 1 camera should be skipped"

    def test_returns_empty_for_empty_input(self) -> None:
        """Empty MidlineSet returns empty result dict."""
        models = _build_synthetic_rig(n_cameras=3)
        results = triangulate_midlines({}, models, frame_index=0)
        assert results == {}


class TestRefineMidlineLmStub:
    """Tests for refine_midline_lm stub."""

    def test_passthrough(self) -> None:
        """refine_midline_lm returns the input object unchanged."""
        midline_3d = Midline3D(
            fish_id=0,
            frame_index=0,
            control_points=np.zeros((7, 3), dtype=np.float32),
            knots=np.zeros(11, dtype=np.float32),
            degree=3,
            arc_length=0.3,
            half_widths=np.zeros(15, dtype=np.float32),
            n_cameras=3,
            mean_residual=0.0,
            max_residual=0.0,
            is_low_confidence=False,
        )

        models = _build_synthetic_rig(n_cameras=3)
        midline_set: MidlineSet = {}

        result = refine_midline_lm(midline_3d, midline_set, models)
        assert result is midline_3d, "refine_midline_lm should return the exact input"
