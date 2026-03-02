"""Unit tests for confidence-weighted reconstruction in triangulation and curve optimizer.

Covers:
- _weighted_triangulate_rays: uniform weights == triangulate_rays output
- _weighted_triangulate_rays: high-weight cameras bias the result
- triangulate_midlines: backward compatibility when point_confidence=None
- triangulate_midlines: confidence weights change result vs uniform
- NaN points excluded from DLT (not passed as zero-weight rows)
- _weighted_chamfer_distance_2d: uniform weights == _chamfer_distance_2d output
- _weighted_chamfer_distance_2d: low-weight outlier reduces influence
- _data_loss: backward compat when confidence_per_fish=None
- _data_loss: non-uniform confidence changes loss value
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aquapose.calibration.projection import RefractiveProjectionModel, triangulate_rays
from aquapose.core.reconstruction.curve_optimizer import (
    CurveOptimizerConfig,
    _chamfer_distance_2d,
    _data_loss,
    _weighted_chamfer_distance_2d,
    get_basis,
)
from aquapose.core.reconstruction.triangulation import (
    _weighted_triangulate_rays,
    triangulate_midlines,
)
from aquapose.core.types.midline import Midline2D
from aquapose.core.types.reconstruction import MidlineSet

# ---------------------------------------------------------------------------
# Shared camera rig helpers (mirrors tests/unit/test_triangulation.py)
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

    Args:
        cam_x: X position of camera in world frame.
        cam_y: Y position of camera in world frame.
        water_z: Z coordinate of water surface in world frame.
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
        angle = 2.0 * math.pi * i / n_cameras
        cam_x = radius * math.cos(angle)
        cam_y = radius * math.sin(angle)
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
    point_confidence: np.ndarray | None = None,
) -> Midline2D:
    """Build a Midline2D by projecting 3D body points through a camera model.

    Args:
        fish_id: Fish identifier.
        camera_id: Camera identifier.
        frame_index: Frame index.
        pts_3d: 3D body positions, shape (N, 3), float32.
        model: Camera projection model.
        point_confidence: Optional per-point confidence, shape (N,), float32.

    Returns:
        Midline2D with projected 2D points and unit half-widths.
    """
    pts_2d = []
    for pt in pts_3d:
        px = _project_point(torch.from_numpy(pt).float(), model)
        pts_2d.append(px.numpy())
    points = np.stack(pts_2d, axis=0).astype(np.float32)
    half_widths = np.full(len(pts_3d), 2.0, dtype=np.float32)
    return Midline2D(
        points=points,
        half_widths=half_widths,
        fish_id=fish_id,
        camera_id=camera_id,
        frame_index=frame_index,
        is_head_to_tail=True,
        point_confidence=point_confidence,
    )


def _make_synthetic_fish_pts(
    n_pts: int = 15,
    water_z: float = 0.75,
    depth: float = 0.3,
) -> np.ndarray:
    """Generate a straight-line set of 3D body points underwater.

    Args:
        n_pts: Number of body points.
        water_z: Z coordinate of water surface.
        depth: Depth below water surface.

    Returns:
        3D body positions, shape (n_pts, 3), float32.
    """
    xs = np.linspace(-0.1, 0.1, n_pts, dtype=np.float32)
    ys = np.zeros(n_pts, dtype=np.float32)
    zs = np.full(n_pts, water_z + depth, dtype=np.float32)
    return np.stack([xs, ys, zs], axis=1)


# ---------------------------------------------------------------------------
# Tests: _weighted_triangulate_rays
# ---------------------------------------------------------------------------


class TestWeightedTriangulateRays:
    """Tests for the _weighted_triangulate_rays helper."""

    def test_uniform_weights_match_unweighted(self):
        """Uniform weights should produce identical output to triangulate_rays."""
        # Build two non-parallel rays
        origins = torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=torch.float32)
        # Rays pointing roughly toward (0, 0, 1)
        d0 = torch.tensor([-0.3, 0.0, 1.0], dtype=torch.float32)
        d1 = torch.tensor([0.3, 0.0, 1.0], dtype=torch.float32)
        d0 = d0 / d0.norm()
        d1 = d1 / d1.norm()
        directions = torch.stack([d0, d1])

        weights = torch.ones(2, dtype=torch.float32)
        pt_weighted = _weighted_triangulate_rays(origins, directions, weights)
        pt_unweighted = triangulate_rays(origins, directions)

        assert torch.allclose(pt_weighted, pt_unweighted, atol=1e-5), (
            f"Uniform-weighted result {pt_weighted} differs from "
            f"unweighted result {pt_unweighted}"
        )

    def test_high_weight_biases_result(self):
        """High weight on one camera should produce a different result than low weight.

        We set up three cameras: two close together (almost parallel, conflicting)
        and one orthogonal (good geometry). We compare:
        - w=[1,1,1]: uniform result is pulled toward the conflicting pair
        - w=[0.01,0.01,1]: result is dominated by the orthogonal camera's ray

        The two results must differ, confirming weight scaling is effective.
        """
        # Three camera origins
        o0 = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32)
        o1 = torch.tensor([0.05, 0.5, 0.0], dtype=torch.float32)  # near-duplicate of o0
        o2 = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)  # orthogonal

        # True point at (0.1, 0.1, 0.8)
        true_pt = torch.tensor([0.1, 0.1, 0.8], dtype=torch.float32)

        # Cameras 0 and 1 point toward a different (wrong) target (0.3, 0.3, 0.8)
        wrong_pt = torch.tensor([0.3, 0.3, 0.8], dtype=torch.float32)

        d0 = wrong_pt - o0
        d0 = d0 / d0.norm()
        d1 = wrong_pt - o1
        d1 = d1 / d1.norm()

        # Camera 2 points toward true_pt
        d2 = true_pt - o2
        d2 = d2 / d2.norm()

        origins = torch.stack([o0, o1, o2])
        directions = torch.stack([d0, d1, d2])

        # Uniform weights — conflicting pair has more votes
        w_uniform = torch.ones(3, dtype=torch.float32)
        pt_uniform = _weighted_triangulate_rays(origins, directions, w_uniform)

        # Near-zero weight on conflicting cameras — orthogonal camera dominates
        w_biased = torch.tensor([0.01, 0.01, 1.0], dtype=torch.float32)
        pt_biased = _weighted_triangulate_rays(origins, directions, w_biased)

        # Results must differ
        diff = float((pt_uniform - pt_biased).norm().item())
        assert diff > 0.01, (
            f"Biased and uniform results should differ (diff={diff:.4f})"
        )

        # Biased result should be closer to true_pt (camera 2 observes it)
        dist_biased = float((pt_biased - true_pt).norm().item())
        dist_uniform = float((pt_uniform - true_pt).norm().item())
        assert dist_biased < dist_uniform, (
            f"High-weight camera 2 observes true_pt; biased result (dist={dist_biased:.4f}) "
            f"should be closer to true_pt than uniform (dist={dist_uniform:.4f})"
        )

    def test_three_cameras_uniform_weights(self):
        """Three cameras with uniform weights should match unweighted."""
        origins = torch.tensor(
            [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=torch.float32,
        )
        true_pt = torch.tensor([0.0, 0.0, 0.8], dtype=torch.float32)
        directions = []
        for i in range(3):
            d = true_pt - origins[i]
            directions.append(d / d.norm())
        dirs = torch.stack(directions)

        w = torch.ones(3, dtype=torch.float32)
        pt_weighted = _weighted_triangulate_rays(origins, dirs, w)
        pt_unweighted = triangulate_rays(origins, dirs)

        assert torch.allclose(pt_weighted, pt_unweighted, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: triangulate_midlines with confidence weighting
# ---------------------------------------------------------------------------


class TestTriangulateMidlinesConfidence:
    """Tests for confidence weighting in triangulate_midlines."""

    @pytest.fixture
    def rig(self) -> dict[str, RefractiveProjectionModel]:
        """Three-camera synthetic rig."""
        return _build_synthetic_rig(n_cameras=3, water_z=0.75, radius=0.5)

    @pytest.fixture
    def fish_pts(self) -> np.ndarray:
        """Synthetic straight fish body, 15 points."""
        return _make_synthetic_fish_pts(n_pts=15, water_z=0.75, depth=0.3)

    def test_none_confidence_backward_compat(
        self, rig: dict[str, RefractiveProjectionModel], fish_pts: np.ndarray
    ):
        """point_confidence=None should produce a valid result (backward compat).

        We do not compare to an exact reference — the key requirement is that
        the function runs without error and produces a Midline3D for the fish.
        """
        midline_set: MidlineSet = {
            0: {
                cam_id: _build_midline2d(0, cam_id, 0, fish_pts, model)
                for cam_id, model in rig.items()
            }
        }
        result = triangulate_midlines(midline_set, rig)
        assert 0 in result, "Fish 0 should be reconstructed from 3 cameras"
        assert isinstance(result[0].control_points, np.ndarray)

    def test_with_confidence_produces_result(
        self, rig: dict[str, RefractiveProjectionModel], fish_pts: np.ndarray
    ):
        """High-confidence observations should produce a valid Midline3D."""
        confs = np.ones(len(fish_pts), dtype=np.float32)
        midline_set: MidlineSet = {
            0: {
                cam_id: _build_midline2d(0, cam_id, 0, fish_pts, model, confs)
                for cam_id, model in rig.items()
            }
        }
        result = triangulate_midlines(midline_set, rig)
        assert 0 in result

    def test_confidence_vs_no_confidence_differ_when_noisy(
        self, rig: dict[str, RefractiveProjectionModel], fish_pts: np.ndarray
    ):
        """Low-confidence noisy camera should bias result differently than uniform weights.

        We add significant noise to one camera and assign it low confidence.
        The weighted result's control points should differ from the unweighted result.
        """
        rng = np.random.RandomState(42)
        cam_ids = sorted(rig.keys())

        # Build midlines: cam 0 gets its points displaced, cam 1&2 are clean
        noisy_cam = cam_ids[0]

        midline_set_unweighted: MidlineSet = {0: {}}
        midline_set_weighted: MidlineSet = {0: {}}

        for cam_id, model in rig.items():
            ml_clean = _build_midline2d(0, cam_id, 0, fish_pts, model)
            if cam_id == noisy_cam:
                # Add significant noise to this camera's 2D observations
                noisy_pts = ml_clean.points.copy()
                noisy_pts += rng.normal(0, 15.0, noisy_pts.shape).astype(np.float32)
                ml_noisy = Midline2D(
                    points=noisy_pts,
                    half_widths=ml_clean.half_widths.copy(),
                    fish_id=0,
                    camera_id=cam_id,
                    frame_index=0,
                )
                ml_noisy_low_conf = Midline2D(
                    points=noisy_pts,
                    half_widths=ml_clean.half_widths.copy(),
                    fish_id=0,
                    camera_id=cam_id,
                    frame_index=0,
                    point_confidence=np.full(len(fish_pts), 0.05, dtype=np.float32),
                )
                midline_set_unweighted[0][cam_id] = ml_noisy
                midline_set_weighted[0][cam_id] = ml_noisy_low_conf
            else:
                midline_set_unweighted[0][cam_id] = ml_clean
                midline_set_weighted[0][cam_id] = Midline2D(
                    points=ml_clean.points.copy(),
                    half_widths=ml_clean.half_widths.copy(),
                    fish_id=0,
                    camera_id=cam_id,
                    frame_index=0,
                    point_confidence=np.ones(len(fish_pts), dtype=np.float32),
                )

        result_unweighted = triangulate_midlines(midline_set_unweighted, rig)
        result_weighted = triangulate_midlines(midline_set_weighted, rig)

        assert 0 in result_unweighted, "Unweighted result should contain fish 0"
        assert 0 in result_weighted, "Weighted result should contain fish 0"

        # Control points should differ because noisy camera has low weight
        ctrl_unw = result_unweighted[0].control_points
        ctrl_w = result_weighted[0].control_points
        diff = float(np.linalg.norm(ctrl_unw - ctrl_w))
        assert diff > 1e-3, (
            f"Weighted and unweighted control points should differ (diff={diff:.6f})"
        )

    def test_nan_points_excluded_from_dlt(
        self, rig: dict[str, RefractiveProjectionModel], fish_pts: np.ndarray
    ):
        """A Midline2D with NaN at body point i should not contribute that point to triangulation.

        We set point index 5 to NaN in one camera. The function must still succeed
        (excluding the NaN from that camera's contribution for body point 5) rather
        than crashing or passing zeros.
        """
        cam_ids = sorted(rig.keys())
        midline_set: MidlineSet = {0: {}}
        for cam_id, model in rig.items():
            ml = _build_midline2d(0, cam_id, 0, fish_pts, model)
            if cam_id == cam_ids[0]:
                # Set point 5 to NaN — should be excluded, not passed as zero
                pts = ml.points.copy()
                pts[5] = np.nan
                ml = Midline2D(
                    points=pts,
                    half_widths=ml.half_widths,
                    fish_id=ml.fish_id,
                    camera_id=ml.camera_id,
                    frame_index=ml.frame_index,
                )
            midline_set[0][cam_id] = ml

        # Should not crash; body point 5 is triangulated from remaining cameras
        result = triangulate_midlines(midline_set, rig)
        assert 0 in result, "Fish should still reconstruct even with one NaN point"


# ---------------------------------------------------------------------------
# Tests: _weighted_chamfer_distance_2d
# ---------------------------------------------------------------------------


class TestWeightedChamferDistance2D:
    """Tests for the _weighted_chamfer_distance_2d helper."""

    def test_uniform_weights_match_unweighted(self):
        """Uniform weights should produce the same value as _chamfer_distance_2d."""
        torch.manual_seed(0)
        proj = torch.rand(10, 2)
        obs = torch.rand(8, 2)
        weights = torch.ones(8, dtype=torch.float32)

        chamfer_unweighted = _chamfer_distance_2d(proj, obs)
        chamfer_weighted = _weighted_chamfer_distance_2d(proj, obs, weights)

        assert torch.isclose(chamfer_unweighted, chamfer_weighted, atol=1e-5), (
            f"Uniform-weight chamfer {chamfer_weighted.item():.6f} should match "
            f"unweighted {chamfer_unweighted.item():.6f}"
        )

    def test_low_weight_outlier_reduces_influence(self):
        """An outlier obs point with low weight should reduce its chamfer contribution.

        We have close inlier obs points plus one far outlier. With uniform weights
        the outlier increases chamfer; with near-zero weight on the outlier it
        should be lower.
        """
        # proj: a cluster near (5, 5)
        proj = torch.tensor(
            [[4.9, 5.0], [5.0, 5.0], [5.1, 5.0], [5.0, 4.9], [5.0, 5.1]],
            dtype=torch.float32,
        )
        # obs: 4 inliers close to proj cluster + 1 outlier far away
        obs = torch.tensor(
            [[5.0, 5.0], [5.1, 5.1], [4.9, 4.9], [5.0, 4.8], [100.0, 100.0]],
            dtype=torch.float32,
        )

        # Uniform weights
        w_uniform = torch.ones(5, dtype=torch.float32)
        ch_uniform = _weighted_chamfer_distance_2d(proj, obs, w_uniform)

        # Zero (near-zero) weight on outlier
        w_downweighted = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.001], dtype=torch.float32)
        ch_downweighted = _weighted_chamfer_distance_2d(proj, obs, w_downweighted)

        assert ch_downweighted.item() < ch_uniform.item(), (
            f"Downweighted outlier chamfer ({ch_downweighted.item():.4f}) should be "
            f"less than uniform ({ch_uniform.item():.4f})"
        )

    def test_empty_proj_returns_zero(self):
        """Empty proj tensor should return zero scalar."""
        proj = torch.zeros(0, 2)
        obs = torch.rand(5, 2)
        w = torch.ones(5)
        result = _weighted_chamfer_distance_2d(proj, obs, w)
        assert result.item() == 0.0

    def test_empty_obs_returns_zero(self):
        """Empty obs tensor should return zero scalar."""
        proj = torch.rand(5, 2)
        obs = torch.zeros(0, 2)
        w = torch.zeros(0)
        result = _weighted_chamfer_distance_2d(proj, obs, w)
        assert result.item() == 0.0


# ---------------------------------------------------------------------------
# Tests: _data_loss with confidence_per_fish
# ---------------------------------------------------------------------------


class TestDataLossConfidence:
    """Tests for confidence weighting in _data_loss."""

    @pytest.fixture
    def simple_config(self) -> CurveOptimizerConfig:
        """Minimal CurveOptimizerConfig for data-loss tests."""
        return CurveOptimizerConfig()

    @pytest.fixture
    def simple_model(self) -> RefractiveProjectionModel:
        """Single camera at x=0.5, water_z=0.75."""
        return _make_camera(0.5, 0.0, water_z=0.75)

    def _make_ctrl_below_water(
        self,
        n_fish: int,
        n_ctrl: int,
        water_z: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Create control points below the water surface.

        Args:
            n_fish: Number of fish.
            n_ctrl: Number of control points per fish.
            water_z: Water surface Z coordinate.
            device: Target device.

        Returns:
            Control point tensor of shape (n_fish, n_ctrl, 3).
        """
        ctrl = torch.zeros(n_fish, n_ctrl, 3, device=device)
        # Spread points along x, keep underwater (z > water_z)
        for k in range(n_ctrl):
            ctrl[:, k, 0] = (k - n_ctrl / 2) * 0.02
            ctrl[:, k, 2] = water_z + 0.3
        return ctrl

    def test_none_confidence_backward_compat(
        self,
        simple_config: CurveOptimizerConfig,
        simple_model: RefractiveProjectionModel,
    ):
        """_data_loss with confidence_per_fish=None should equal call without kwarg."""
        device = torch.device("cpu")
        n_ctrl = 4
        basis = get_basis(15, n_ctrl, device)

        ctrl = self._make_ctrl_below_water(1, n_ctrl, simple_model.water_z, device)

        # Build a small obs set
        obs = torch.tensor([[800.0, 600.0], [810.0, 605.0]], dtype=torch.float32)
        midlines_per_fish = [{"cam0": obs}]
        models = {"cam0": simple_model}

        loss_no_kwarg = _data_loss(
            ctrl, basis, midlines_per_fish, models, simple_config
        )
        loss_none = _data_loss(
            ctrl,
            basis,
            midlines_per_fish,
            models,
            simple_config,
            confidence_per_fish=None,
        )

        assert torch.isclose(loss_no_kwarg, loss_none, atol=1e-6), (
            f"None confidence should produce identical loss: "
            f"{loss_no_kwarg.item():.6f} vs {loss_none.item():.6f}"
        )

    def test_with_confidence_changes_loss(
        self,
        simple_config: CurveOptimizerConfig,
        simple_model: RefractiveProjectionModel,
    ):
        """Non-uniform confidence should produce a different loss than no confidence.

        We create an outlier obs point far from the spline projection. With
        uniform confidence it contributes fully to obs->proj chamfer.
        With near-zero weight it contributes minimally, changing the loss.
        """
        device = torch.device("cpu")
        n_ctrl = 4
        basis = get_basis(15, n_ctrl, device)

        ctrl = self._make_ctrl_below_water(1, n_ctrl, simple_model.water_z, device)

        # Observations: 2 inliers near proj + 1 outlier far away
        obs = torch.tensor(
            [[800.0, 600.0], [810.0, 605.0], [2000.0, 2000.0]], dtype=torch.float32
        )
        midlines_per_fish = [{"cam0": obs}]
        models = {"cam0": simple_model}

        # No confidence — uniform
        loss_none = _data_loss(
            ctrl,
            basis,
            midlines_per_fish,
            models,
            simple_config,
            confidence_per_fish=None,
        )

        # Near-zero weight on outlier
        w = torch.tensor([1.0, 1.0, 0.001], dtype=torch.float32)
        confidence_per_fish = [{"cam0": w}]
        loss_conf = _data_loss(
            ctrl,
            basis,
            midlines_per_fish,
            models,
            simple_config,
            confidence_per_fish=confidence_per_fish,
        )

        assert not torch.isclose(loss_none, loss_conf, atol=1e-3), (
            f"Non-uniform confidence should change loss: "
            f"none={loss_none.item():.4f}, conf={loss_conf.item():.4f}"
        )

    def test_all_none_weights_uses_unweighted_chamfer(
        self,
        simple_config: CurveOptimizerConfig,
        simple_model: RefractiveProjectionModel,
    ):
        """confidence_per_fish with all None values should use unweighted chamfer.

        This simulates the segmentation backend path where point_confidence is None
        on all cameras — should produce identical output to confidence_per_fish=None.
        """
        device = torch.device("cpu")
        n_ctrl = 4
        basis = get_basis(15, n_ctrl, device)

        ctrl = self._make_ctrl_below_water(1, n_ctrl, simple_model.water_z, device)

        obs = torch.tensor([[800.0, 600.0], [810.0, 605.0]], dtype=torch.float32)
        midlines_per_fish = [{"cam0": obs}]
        models = {"cam0": simple_model}

        loss_none = _data_loss(
            ctrl,
            basis,
            midlines_per_fish,
            models,
            simple_config,
            confidence_per_fish=None,
        )
        # All-None confidence_per_fish — should behave identically
        loss_all_none = _data_loss(
            ctrl,
            basis,
            midlines_per_fish,
            models,
            simple_config,
            confidence_per_fish=[{"cam0": None}],
        )

        assert torch.isclose(loss_none, loss_all_none, atol=1e-6), (
            f"All-None confidence_per_fish should equal no-confidence: "
            f"{loss_none.item():.6f} vs {loss_all_none.item():.6f}"
        )
