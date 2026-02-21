"""Tests for soft_iou_loss, compute_angular_diversity_weights, multi_objective_loss."""

from __future__ import annotations

import math

import pytest
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.mesh.state import FishState
from aquapose.optimization.loss import (
    compute_angular_diversity_weights,
    multi_objective_loss,
    soft_iou_loss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(
    R: torch.Tensor,
    t: torch.Tensor,
    water_z: float = 1.0,
) -> RefractiveProjectionModel:
    """Create a minimal RefractiveProjectionModel with given R and t."""
    K = torch.tensor(
        [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return RefractiveProjectionModel(K, R, t, water_z, normal, 1.0, 1.333)


def _simple_state(
    requires_grad: bool = False,
    kappa: float = 0.0,
    theta: float = 0.0,
    s: float = 0.15,
) -> FishState:
    """Create a simple horizontal FishState."""
    return FishState(
        p=torch.tensor(
            [0.0, 0.0, 1.5], dtype=torch.float32, requires_grad=requires_grad
        ),
        psi=torch.tensor(0.0, dtype=torch.float32, requires_grad=requires_grad),
        theta=torch.tensor(theta, dtype=torch.float32, requires_grad=requires_grad),
        kappa=torch.tensor(kappa, dtype=torch.float32, requires_grad=requires_grad),
        s=torch.tensor(s, dtype=torch.float32, requires_grad=requires_grad),
    )


# ---------------------------------------------------------------------------
# soft_iou_loss tests
# ---------------------------------------------------------------------------


class TestSoftIoULoss:
    """Tests for soft_iou_loss."""

    def test_perfect_overlap(self):
        """When pred == target, loss should be near zero."""
        mask = torch.zeros(10, 10)
        mask[3:7, 3:7] = 1.0
        loss = soft_iou_loss(mask.clone(), mask.clone())
        assert float(loss) < 1e-4, f"Expected ~0, got {float(loss)}"

    def test_no_overlap(self):
        """When pred and target are disjoint, loss should be near 1."""
        pred = torch.zeros(10, 10)
        pred[:5, :] = 1.0
        target = torch.zeros(10, 10)
        target[5:, :] = 1.0
        loss = soft_iou_loss(pred, target)
        assert float(loss) > 0.9, f"Expected ~1, got {float(loss)}"

    def test_partial_overlap_value(self):
        """Partial overlap: verify loss matches expected 1 - IoU."""
        pred = torch.zeros(4, 4)
        pred[:, :2] = 1.0  # left half
        target = torch.zeros(4, 4)
        target[:, 1:3] = 1.0  # middle two columns

        # Intersection: column 1 (4 pixels)
        # Union: columns 0, 1, 2 (12 pixels), but soft: same math applies
        # Exact soft IoU: intersection=4, union=12, iou=1/3, loss=2/3
        loss = soft_iou_loss(pred, target)
        expected = 2.0 / 3.0
        assert abs(float(loss) - expected) < 1e-4, (
            f"Expected {expected:.4f}, got {float(loss):.4f}"
        )

    def test_crop_region_applied(self):
        """With crop_region, only the cropped area contributes."""
        pred = torch.zeros(10, 10)
        pred[2:5, 2:5] = 1.0  # fish in top-left quadrant

        target = torch.zeros(10, 10)
        target[2:5, 2:5] = 1.0  # perfect overlap in top-left

        # Crop to exactly the fish region
        loss_cropped = soft_iou_loss(pred, target, crop_region=(2, 2, 5, 5))
        assert float(loss_cropped) < 1e-4, "Cropped perfect overlap should have loss ~0"

        # Without crop: still perfect overlap globally
        loss_full = soft_iou_loss(pred, target)
        assert float(loss_full) < 1e-4

    def test_crop_region_excludes_outside(self):
        """Predictions outside crop region should not affect cropped loss."""
        pred = torch.zeros(10, 10)
        pred[7:10, 7:10] = 1.0  # prediction far from crop

        target = torch.zeros(10, 10)
        target[2:5, 2:5] = 1.0  # target in crop

        # Crop to [2:5, 2:5] — pred has nothing there, target has fish
        loss_cropped = soft_iou_loss(pred, target, crop_region=(2, 2, 5, 5))
        assert float(loss_cropped) > 0.9, "Crop excludes pred, should be high loss"

    def test_gradient_flows_through_pred(self):
        """Backward through pred_alpha completes without error."""
        pred = torch.rand(10, 10, requires_grad=True)
        target = (torch.rand(10, 10) > 0.5).float()
        loss = soft_iou_loss(pred, target)
        loss.backward()
        assert pred.grad is not None, "No gradient to pred_alpha"
        assert torch.isfinite(pred.grad).all(), "NaN gradient to pred_alpha"

    def test_output_is_scalar(self):
        """Return value is a scalar tensor."""
        pred = torch.rand(8, 8)
        target = (torch.rand(8, 8) > 0.5).float()
        loss = soft_iou_loss(pred, target)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_loss_in_range(self):
        """Loss is always in [0, 1]."""
        for _ in range(5):
            pred = torch.rand(10, 10)
            target = (torch.rand(10, 10) > 0.5).float()
            loss = soft_iou_loss(pred, target)
            assert 0.0 <= float(loss) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# compute_angular_diversity_weights tests
# ---------------------------------------------------------------------------


class TestAngularDiversityWeights:
    """Tests for compute_angular_diversity_weights.

    Camera view directions are constructed by using R to define how the camera
    is oriented. R maps world to camera. The camera looks along its +Z axis in
    camera frame, which corresponds to R.T @ [0, 0, 1] in world frame.

    To create a camera looking in direction d_world, set R = I and override
    the view direction via a rotation that maps world +Z to d_world. The easiest
    approach: set R.T @ [0, 0, 1] = d_world, so R maps d_world -> [0, 0, 1]
    in camera frame.

    For simplicity, we construct cameras with different t vectors and identity R
    to give identical view directions — OR we rotate the camera so its -Z axis
    in camera frame (the look direction) points in different world-space directions.

    Actually: the look direction in world frame for a camera with rotation R is
    R.T @ [0, 0, -1] (camera looks along -Z in camera space). To make it point
    along a given world direction, we need an appropriate R.
    """

    def _camera_looking_toward(self, azimuth_deg: float) -> RefractiveProjectionModel:
        """Create a camera whose world-space view direction is at azimuth_deg in XZ plane.

        At azimuth=0: looks toward +X in world.
        At azimuth=90: looks toward +Z (down into water).
        """
        az = math.radians(azimuth_deg)
        # World view direction: (sin(az), 0, cos(az))
        # Camera -Z_cam = view_dir in world -> camera +Z in camera = -view_dir
        # R needs to map world_view_dir to [0, 0, -1] in camera.
        # The easiest construction: set camera +Z (in world) = -view_dir
        # and camera +X = perpendicular in XZ plane = (cos(az), 0, -sin(az))
        # and camera +Y = world Y up
        vx = math.cos(az)
        vz = -math.sin(az)
        R_world_to_cam = torch.tensor(
            [[vx, 0.0, vz], [0.0, 1.0, 0.0], [math.sin(az), 0.0, math.cos(az)]],
            dtype=torch.float32,
        )
        # Verify: R.T @ [0,0,-1] = [sin(az), 0, -cos(az)] -- that's the look dir in world
        # Actually verify in the test
        t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        return _make_camera(R_world_to_cam, t)

    def test_clustered_cameras_lower_weight(self):
        """Two cameras with nearly identical view direction get lower weight than isolated one."""
        # Camera A and B: 2 degrees apart
        cam_a = self._camera_looking_toward(0.0)
        cam_b = self._camera_looking_toward(2.0)
        # Camera C: 90 degrees from A and B
        cam_c = self._camera_looking_toward(90.0)

        models = [cam_a, cam_b, cam_c]
        ids = ["a", "b", "c"]
        weights = compute_angular_diversity_weights(models, ids)

        # C should have higher weight than A or B (C is isolated)
        assert weights["c"] > weights["a"], (
            f"Isolated camera (c={weights['c']:.3f}) should outweigh clustered (a={weights['a']:.3f})"
        )
        assert weights["c"] > weights["b"], (
            f"Isolated camera (c={weights['c']:.3f}) should outweigh clustered (b={weights['b']:.3f})"
        )

    def test_diverse_cameras_higher_weight(self):
        """Cameras uniformly spread at 0, 90, 180, 270 degrees get equal weights."""
        models = [self._camera_looking_toward(a) for a in [0, 90, 180, 270]]
        ids = ["a", "b", "c", "d"]
        weights = compute_angular_diversity_weights(models, ids)

        values = list(weights.values())
        assert max(values) - min(values) < 1e-4, (
            f"Symmetric cameras should get equal weights, got {weights}"
        )

    def test_weights_in_range(self):
        """All weights are in (0, 1]."""
        models = [self._camera_looking_toward(a) for a in [0, 45, 90, 135]]
        ids = ["a", "b", "c", "d"]
        weights = compute_angular_diversity_weights(models, ids)
        for cam_id, w in weights.items():
            assert 0.0 < w <= 1.0 + 1e-8, f"Weight {cam_id}={w} out of (0, 1]"

    def test_max_weight_is_one(self):
        """The maximum weight across cameras is 1.0."""
        models = [self._camera_looking_toward(a) for a in [0, 60, 120, 180]]
        ids = ["a", "b", "c", "d"]
        weights = compute_angular_diversity_weights(models, ids)
        assert max(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_returns_all_camera_ids(self):
        """Result dict contains exactly the input camera IDs."""
        models = [self._camera_looking_toward(0.0), self._camera_looking_toward(90.0)]
        ids = ["front", "down"]
        weights = compute_angular_diversity_weights(models, ids)
        assert set(weights.keys()) == set(ids)

    def test_mismatched_lengths_raises(self):
        """Raises ValueError when models and camera_ids have different lengths."""
        models = [self._camera_looking_toward(0.0)]
        with pytest.raises(ValueError, match="must have the same length"):
            compute_angular_diversity_weights(models, ["a", "b"])

    def test_temperature_affects_spread(self):
        """Higher temperature produces larger spread in weights (more differentiation).

        Formula: weight = (min_angle / max_min_angle) ** temperature.
        Clustered cameras have small min_angle, so their weight base is < 1.
        Raising this to a higher power amplifies the difference: high temperature
        makes clustered cameras' weights drop faster toward 0 relative to isolated ones.
        """
        # 3 cameras: A and B are close (5 deg), C is isolated (90 deg from A/B).
        models = [
            self._camera_looking_toward(0.0),
            self._camera_looking_toward(5.0),
            self._camera_looking_toward(90.0),
        ]
        ids = ["a", "b", "c"]
        weights_low_temp = compute_angular_diversity_weights(
            models, ids, temperature=0.1
        )
        weights_high_temp = compute_angular_diversity_weights(
            models, ids, temperature=2.0
        )

        spread_low = max(weights_low_temp.values()) - min(weights_low_temp.values())
        spread_high = max(weights_high_temp.values()) - min(weights_high_temp.values())
        # High temperature exaggerates differences: clustered cameras get lower relative weights.
        assert spread_high >= spread_low, (
            f"Higher temperature should give wider spread: high={spread_high:.3f}, low={spread_low:.3f}"
        )


# ---------------------------------------------------------------------------
# multi_objective_loss tests
# ---------------------------------------------------------------------------


class TestMultiObjectiveLoss:
    """Tests for multi_objective_loss."""

    def _dummy_pred_target(
        self, overlap: float = 0.5, size: int = 10
    ) -> tuple[dict, dict, dict]:
        """Create simple pred/target/crop dicts for one camera."""
        pred = torch.zeros(size, size)
        pred[: int(size * overlap), :] = 0.8
        target = torch.zeros(size, size)
        target[int(size * (1 - overlap)) :, :] = 1.0
        return (
            {"cam0": pred},
            {"cam0": target},
            {"cam0": None},
        )

    def test_returns_all_keys(self):
        """Result dict contains exactly the 5 expected keys."""
        state = _simple_state()
        pred_a, target_a, crops_a = self._dummy_pred_target()
        losses = multi_objective_loss(
            state,
            pred_a,
            target_a,
            crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights={"iou": 1.0, "gravity": 0.05, "morph": 0.2},
        )
        assert set(losses.keys()) == {"total", "iou", "gravity", "morph", "temporal"}

    def test_all_values_are_scalar_tensors(self):
        """All returned loss values are scalar tensors."""
        state = _simple_state()
        pred_a, target_a, crops_a = self._dummy_pred_target()
        losses = multi_objective_loss(
            state,
            pred_a,
            target_a,
            crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights={"iou": 1.0, "gravity": 0.05, "morph": 0.2},
        )
        for key, val in losses.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a tensor"
            assert val.shape == (), f"{key} is not scalar, got shape {val.shape}"

    def test_gravity_increases_with_theta(self):
        """Gravity loss increases as theta (pitch) increases."""
        state_flat = _simple_state(theta=0.0)
        state_tilted = _simple_state(theta=0.5)

        pred_a, target_a, crops_a = self._dummy_pred_target()

        loss_flat = multi_objective_loss(
            state_flat,
            pred_a,
            target_a,
            crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights={"iou": 1.0, "gravity": 1.0, "morph": 0.0},
        )
        loss_tilted = multi_objective_loss(
            state_tilted,
            pred_a,
            target_a,
            crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights={"iou": 1.0, "gravity": 1.0, "morph": 0.0},
        )

        assert float(loss_tilted["gravity"]) > float(loss_flat["gravity"]), (
            "Higher theta should give higher gravity loss"
        )

    def test_morph_zero_within_bounds(self):
        """Morphological loss is zero when kappa and s are within bounds."""
        state = _simple_state(kappa=5.0, s=0.15)  # kappa_max=10, s in [0.05, 0.30]
        pred_a, target_a, crops_a = self._dummy_pred_target()
        losses = multi_objective_loss(
            state,
            pred_a,
            target_a,
            crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights={"iou": 1.0, "gravity": 0.05, "morph": 0.2},
            kappa_max=10.0,
            s_min=0.05,
            s_max=0.30,
        )
        assert float(losses["morph"]) == pytest.approx(0.0, abs=1e-6)

    def test_morph_nonzero_kappa_out_of_bounds(self):
        """Morphological loss is positive when |kappa| > kappa_max."""
        state = _simple_state(kappa=15.0)  # exceeds kappa_max=10
        pred_a, target_a, crops_a = self._dummy_pred_target()
        losses = multi_objective_loss(
            state,
            pred_a,
            target_a,
            crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights={"iou": 1.0, "gravity": 0.05, "morph": 0.2},
            kappa_max=10.0,
            s_min=0.05,
            s_max=0.30,
        )
        assert float(losses["morph"]) > 0.0, (
            "kappa out of bounds should give positive morph loss"
        )

    def test_morph_nonzero_scale_out_of_bounds(self):
        """Morphological loss is positive when s < s_min or s > s_max."""
        state_small = _simple_state(s=0.01)  # below s_min=0.05
        state_large = _simple_state(s=0.50)  # above s_max=0.30

        pred_a, target_a, crops_a = self._dummy_pred_target()
        base_kwargs = dict(
            pred_alphas=pred_a,
            target_masks=target_a,
            crop_regions=crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights={"iou": 1.0, "gravity": 0.05, "morph": 0.2},
            kappa_max=10.0,
            s_min=0.05,
            s_max=0.30,
        )

        loss_small = multi_objective_loss(state_small, **base_kwargs)
        loss_large = multi_objective_loss(state_large, **base_kwargs)

        assert float(loss_small["morph"]) > 0.0, (
            "s too small should give positive morph"
        )
        assert float(loss_large["morph"]) > 0.0, (
            "s too large should give positive morph"
        )

    def test_temporal_inactive_when_none(self):
        """Temporal loss is exactly 0 when temporal_state is None."""
        state = _simple_state()
        pred_a, target_a, crops_a = self._dummy_pred_target()
        losses = multi_objective_loss(
            state,
            pred_a,
            target_a,
            crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights={"iou": 1.0, "gravity": 0.05, "morph": 0.2},
            temporal_state=None,
        )
        assert float(losses["temporal"]) == 0.0, "temporal should be 0 with no tracking"

    def test_temporal_active_when_state_provided(self):
        """Temporal loss is positive when current and previous states differ."""
        state_now = _simple_state()
        state_prev = FishState(
            p=torch.tensor([1.0, 0.0, 1.5], dtype=torch.float32),  # offset in X
            psi=torch.tensor(0.5, dtype=torch.float32),
            theta=torch.tensor(0.0, dtype=torch.float32),
            kappa=torch.tensor(0.0, dtype=torch.float32),
            s=torch.tensor(0.15, dtype=torch.float32),
        )
        pred_a, target_a, crops_a = self._dummy_pred_target()
        losses = multi_objective_loss(
            state_now,
            pred_a,
            target_a,
            crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights={"iou": 1.0, "gravity": 0.05, "morph": 0.2},
            temporal_state=state_prev,
            temporal_weight=1.0,
        )
        assert float(losses["temporal"]) > 0.0, (
            "Temporal should be positive for different states"
        )

    def test_total_is_weighted_sum(self):
        """Total loss equals weighted sum of individual terms."""
        state = _simple_state(theta=0.2, kappa=5.0, s=0.15)
        pred_a, target_a, crops_a = self._dummy_pred_target()
        loss_weights = {"iou": 1.0, "gravity": 0.05, "morph": 0.2}
        losses = multi_objective_loss(
            state,
            pred_a,
            target_a,
            crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights=loss_weights,
        )
        expected_total = (
            1.0 * float(losses["iou"])
            + 0.05 * float(losses["gravity"])
            + 0.2 * float(losses["morph"])
            + 0.1 * float(losses["temporal"])  # default temporal_weight
        )
        assert float(losses["total"]) == pytest.approx(expected_total, rel=1e-5)

    def test_gradient_flows_through_state_params(self):
        """Backward through total loss reaches all FishState parameters."""
        state = _simple_state(requires_grad=True, theta=0.1, kappa=1.0, s=0.15)
        pred_a, target_a, crops_a = self._dummy_pred_target()
        losses = multi_objective_loss(
            state,
            pred_a,
            target_a,
            crops_a,
            camera_weights={"cam0": 1.0},
            loss_weights={"iou": 1.0, "gravity": 0.05, "morph": 0.2},
        )
        losses["total"].backward()

        # Position gradient: comes only from temporal (inactive here) or IoU
        # theta/kappa/s: from gravity and morph terms directly
        assert state.theta.grad is not None, "No gradient to theta"
        assert state.kappa.grad is not None, "No gradient to kappa"
        assert state.s.grad is not None, "No gradient to s"
        assert torch.isfinite(state.theta.grad), "NaN gradient to theta"
        assert torch.isfinite(state.kappa.grad), "NaN gradient to kappa"
        assert torch.isfinite(state.s.grad), "NaN gradient to s"

    def test_multiple_cameras_combined(self):
        """Loss runs correctly with multiple cameras."""
        state = _simple_state()
        pred = {"cam0": torch.rand(8, 8), "cam1": torch.rand(8, 8)}
        target = {
            "cam0": (torch.rand(8, 8) > 0.5).float(),
            "cam1": (torch.rand(8, 8) > 0.5).float(),
        }
        crops = {"cam0": None, "cam1": None}
        losses = multi_objective_loss(
            state,
            pred,
            target,
            crops,
            camera_weights={"cam0": 1.0, "cam1": 0.8},
            loss_weights={"iou": 1.0, "gravity": 0.05, "morph": 0.2},
        )
        assert torch.isfinite(losses["total"]), "Total loss should be finite"
