"""Unit tests for multi-camera refractive triangulation and FishState initialization."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.initialization.triangulator import (
    init_fish_state,
    init_fish_states_from_masks,
    triangulate_keypoint,
)
from aquapose.mesh.state import FishState  # used in isinstance check

# ---------------------------------------------------------------------------
# Minimal synthetic rig helper (4 cameras, no aquacal dependency)
# ---------------------------------------------------------------------------


def _make_overhead_camera(
    cam_x: float,
    cam_y: float,
    water_z: float,
    fx: float = 1400.0,
    cx: float = 800.0,
    cy: float = 600.0,
) -> RefractiveProjectionModel:
    """Build a downward-looking camera at world position (cam_x, cam_y, 0).

    In AquaPose world frame, cameras are at Z=0 with Z increasing downward.
    water_z > 0 means water surface is water_z meters below the cameras.

    With R=I (world frame = camera frame), camera center C = -R^T @ t = -t.
    So t = (-cam_x, -cam_y, 0) places camera at (cam_x, cam_y, 0) in world.
    """
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.tensor([-cam_x, -cam_y, 0.0], dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return RefractiveProjectionModel(
        K=K,
        R=R,
        t=t,
        water_z=water_z,
        normal=normal,
        n_air=1.0,
        n_water=1.333,
    )


def _make_4_camera_rig() -> list[RefractiveProjectionModel]:
    """Build 4 overhead cameras arranged in a square, looking straight down.

    Cameras at Z=0 in world, water surface at Z=1.0, fish at Z~1.5.
    """
    water_z = 1.0  # water surface at Z=1.0 in world frame
    positions = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)]
    return [_make_overhead_camera(x, y, water_z) for x, y in positions]


# ---------------------------------------------------------------------------
# Tests for triangulate_keypoint
# ---------------------------------------------------------------------------


class TestTriangulateRoundTrip:
    """Test that projection + triangulation round-trips within 1mm."""

    def test_triangulate_round_trip(self):
        """Project a known 3D point through 4 cameras, triangulate back, verify <1mm."""
        models = _make_4_camera_rig()

        # Known 3D point below water surface
        true_point = torch.tensor([0.1, -0.05, 1.5], dtype=torch.float32)

        # Project to each camera
        pixel_coords = []
        valid_models = []
        for model in models:
            pixels, valid = model.project(true_point.unsqueeze(0))
            if valid[0]:
                uv = pixels[0]
                pixel_coords.append((float(uv[0]), float(uv[1])))
                valid_models.append(model)

        assert len(valid_models) >= 3, "Need at least 3 valid projections for test"

        # Triangulate back
        point_3d = triangulate_keypoint(pixel_coords, valid_models)

        # Error should be within 1mm (0.001m)
        error = torch.linalg.norm(point_3d - true_point).item()
        assert error < 0.001, f"Round-trip error = {error * 1000:.3f}mm (limit: 1mm)"

    def test_triangulate_output_shape(self):
        """triangulate_keypoint returns shape (3,)."""
        models = _make_4_camera_rig()
        true_point = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float32)

        pixel_coords = []
        valid_models = []
        for model in models:
            pixels, valid = model.project(true_point.unsqueeze(0))
            if valid[0]:
                pixel_coords.append((float(pixels[0, 0]), float(pixels[0, 1])))
                valid_models.append(model)

        result = triangulate_keypoint(pixel_coords[:3], valid_models[:3])
        assert result.shape == (3,)
        assert result.dtype == torch.float32


class TestTriangulateRequires3Cameras:
    """Test that fewer than 3 cameras raises ValueError."""

    def test_two_cameras_raises(self):
        models = _make_4_camera_rig()
        pixel_coords = [(100.0, 200.0), (300.0, 400.0)]
        with pytest.raises(ValueError, match=r"[Aa]t least 3|3 camera"):
            triangulate_keypoint(pixel_coords, models[:2])

    def test_one_camera_raises(self):
        models = _make_4_camera_rig()
        pixel_coords = [(100.0, 200.0)]
        with pytest.raises(ValueError, match=r"[Aa]t least 3|3 camera"):
            triangulate_keypoint(pixel_coords, models[:1])

    def test_three_cameras_accepted(self):
        """Exactly 3 cameras should be accepted (no error)."""
        models = _make_4_camera_rig()
        true_point = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float32)
        pixel_coords = []
        valid_models = []
        for model in models:
            pixels, valid = model.project(true_point.unsqueeze(0))
            if valid[0]:
                pixel_coords.append((float(pixels[0, 0]), float(pixels[0, 1])))
                valid_models.append(model)

        # Should not raise
        result = triangulate_keypoint(pixel_coords[:3], valid_models[:3])
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# Tests for init_fish_state
# ---------------------------------------------------------------------------


class TestInitFishStateHorizontal:
    """Test FishState initialization from horizontal fish endpoints."""

    def test_horizontal_fish_psi_zero(self):
        """Fish pointing along +X: psi should be 0."""
        center = torch.tensor([0.0, 0.0, 1.5])
        ep_a = torch.tensor([0.075, 0.0, 1.5])  # head side (+X)
        ep_b = torch.tensor([-0.075, 0.0, 1.5])  # tail side (-X)

        state = init_fish_state(center, ep_a, ep_b)
        assert isinstance(state, FishState)
        assert abs(state.psi.item()) < 0.01, (
            f"psi = {state.psi.item():.4f} (expected ~0)"
        )

    def test_horizontal_fish_theta_zero(self):
        """Horizontal fish (no pitch): theta should be 0."""
        center = torch.tensor([0.0, 0.0, 1.5])
        ep_a = torch.tensor([0.075, 0.0, 1.5])
        ep_b = torch.tensor([-0.075, 0.0, 1.5])

        state = init_fish_state(center, ep_a, ep_b)
        assert abs(state.theta.item()) < 0.01, (
            f"theta = {state.theta.item():.4f} (expected ~0)"
        )

    def test_horizontal_fish_scale(self):
        """Scale s should equal distance between endpoints (0.15m)."""
        center = torch.tensor([0.0, 0.0, 1.5])
        ep_a = torch.tensor([0.075, 0.0, 1.5])
        ep_b = torch.tensor([-0.075, 0.0, 1.5])

        state = init_fish_state(center, ep_a, ep_b)
        expected_s = 0.15
        assert abs(state.s.item() - expected_s) < 0.001, (
            f"s = {state.s.item():.4f} (expected {expected_s})"
        )


class TestInitFishStateAngled:
    """Test FishState initialization from angled fish endpoints."""

    def test_45_degree_fish_psi(self):
        """Fish pointing at 45 degrees in XY: psi should be pi/4."""
        center = torch.tensor([0.0, 0.0, 1.5])
        half_len = 0.15 / 2 / math.sqrt(2)
        ep_a = torch.tensor([half_len, half_len, 1.5])
        ep_b = torch.tensor([-half_len, -half_len, 1.5])

        state = init_fish_state(center, ep_a, ep_b)
        expected_psi = math.pi / 4
        assert abs(state.psi.item() - expected_psi) < 0.01, (
            f"psi = {state.psi.item():.4f} (expected {expected_psi:.4f})"
        )

    def test_vertical_fish_theta(self):
        """Fish pointing straight down: theta should be pi/2."""
        center = torch.tensor([0.0, 0.0, 1.5])
        ep_a = torch.tensor([0.0, 0.0, 1.65])  # deeper end
        ep_b = torch.tensor([0.0, 0.0, 1.35])  # shallower end

        state = init_fish_state(center, ep_a, ep_b)
        expected_theta = math.pi / 2
        assert abs(state.theta.item() - expected_theta) < 0.01, (
            f"theta = {state.theta.item():.4f} (expected {expected_theta:.4f})"
        )

    def test_position_equals_center(self):
        """FishState.p should equal the provided center_3d."""
        center = torch.tensor([0.3, -0.1, 1.8])
        ep_a = torch.tensor([0.375, -0.1, 1.8])
        ep_b = torch.tensor([0.225, -0.1, 1.8])

        state = init_fish_state(center, ep_a, ep_b)
        assert torch.allclose(state.p, center.float(), atol=1e-5)


class TestInitFishStateKappaZero:
    """Test that kappa is always 0 at initialization."""

    def test_kappa_is_zero(self):
        center = torch.tensor([0.0, 0.0, 1.5])
        ep_a = torch.tensor([0.075, 0.0, 1.5])
        ep_b = torch.tensor([-0.075, 0.0, 1.5])

        state = init_fish_state(center, ep_a, ep_b)
        assert state.kappa.item() == 0.0

    def test_kappa_is_zero_for_angled(self):
        center = torch.tensor([0.1, 0.2, 1.5])
        ep_a = torch.tensor([0.175, 0.275, 1.5])
        ep_b = torch.tensor([0.025, 0.125, 1.5])

        state = init_fish_state(center, ep_a, ep_b)
        assert state.kappa.item() == 0.0


class TestInitFishStateScaleIsEndpointDistance:
    """Test that scale s equals the distance between endpoints."""

    def test_scale_matches_distance(self):
        center = torch.tensor([0.0, 0.0, 1.5])
        ep_a = torch.tensor([0.1, 0.05, 1.6])
        ep_b = torch.tensor([-0.1, -0.05, 1.4])

        state = init_fish_state(center, ep_a, ep_b)
        expected_s = torch.linalg.norm(ep_a - ep_b).item()
        assert abs(state.s.item() - expected_s) < 1e-4, (
            f"s = {state.s.item():.5f}, expected {expected_s:.5f}"
        )


# ---------------------------------------------------------------------------
# Full pipeline test: masks → FishState
# ---------------------------------------------------------------------------


def _rasterize_line_mask(
    p1_uv: tuple[float, float],
    p2_uv: tuple[float, float],
    H: int,
    W: int,
    thickness: int = 5,
) -> np.ndarray:
    """Create a binary mask with a thick line between two pixel coordinates."""
    mask = np.zeros((H, W), dtype=np.uint8)
    u1, v1 = p1_uv
    u2, v2 = p2_uv

    # Rasterize using Bresenham-like approach with thickness
    n_steps = max(int(abs(u2 - u1) + abs(v2 - v1)) * 2, 1)
    for i in range(n_steps + 1):
        t = i / n_steps
        u = u1 + t * (u2 - u1)
        v = v1 + t * (v2 - v1)
        u_int, v_int = round(u), round(v)
        # Draw thick point
        for dv in range(-thickness, thickness + 1):
            for du in range(-thickness, thickness + 1):
                pu, pv = u_int + du, v_int + dv
                if 0 <= pu < W and 0 <= pv < H:
                    mask[pv, pu] = 255

    return mask


class TestFullPipelineSynthetic:
    """Test the full pipeline from masks to FishState."""

    def test_full_pipeline_synthetic(self):
        """Generate synthetic masks for a known fish, run init_fish_states_from_masks.

        Verifies:
        - Position error < 5mm
        - Heading error < 10 degrees
        """
        models = _make_4_camera_rig()

        # Known fish axis: center at (0.0, 0.0, 1.5), pointing along X
        true_center = torch.tensor([0.05, 0.02, 1.5], dtype=torch.float32)
        true_ep_a = torch.tensor([0.125, 0.02, 1.5], dtype=torch.float32)
        true_ep_b = torch.tensor([-0.025, 0.02, 1.5], dtype=torch.float32)

        H, W = 1200, 1600

        # Project keypoints into each camera and create synthetic masks
        masks_per_camera: list[list[np.ndarray | None]] = []
        for model in models:
            pts = torch.stack([true_center, true_ep_a, true_ep_b], dim=0)  # (3, 3)
            pixels, valid = model.project(pts)

            if valid.all():
                # Create mask as a thick line between ep_a and ep_b projections
                ep_a_uv = (float(pixels[1, 0]), float(pixels[1, 1]))
                ep_b_uv = (float(pixels[2, 0]), float(pixels[2, 1]))
                mask = _rasterize_line_mask(ep_a_uv, ep_b_uv, H, W, thickness=8)
                masks_per_camera.append([mask])
            else:
                masks_per_camera.append([None])

        # Count valid cameras
        n_valid = sum(1 for cam_masks in masks_per_camera if cam_masks[0] is not None)
        assert n_valid >= 3, f"Need >=3 valid cameras, got {n_valid}"

        # Run full pipeline
        states = init_fish_states_from_masks(masks_per_camera, models)
        assert len(states) == 1
        state = states[0]

        # Position error < 5mm
        pos_error = torch.linalg.norm(state.p - true_center).item()
        assert pos_error < 0.005, (
            f"Position error = {pos_error * 1000:.2f}mm (limit: 5mm)"
        )

        # Heading error < 10 degrees
        true_axis = true_ep_a - true_ep_b
        true_heading = true_axis / torch.linalg.norm(true_axis)
        true_psi = math.atan2(float(true_heading[1]), float(true_heading[0]))
        psi_diff = abs(state.psi.item() - true_psi)
        # Handle wraparound
        psi_diff = min(psi_diff, 2 * math.pi - psi_diff)
        psi_deg = math.degrees(psi_diff)
        assert psi_deg < 10.0, f"Heading error = {psi_deg:.2f} deg (limit: 10 deg)"

    def test_full_pipeline_too_few_cameras_raises(self):
        """Full pipeline should raise when fewer than 3 cameras have valid masks."""
        models = _make_4_camera_rig()
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[40:60, 30:70] = 255

        # Only 2 cameras have valid masks
        masks_per_camera: list[list[np.ndarray | None]] = [
            [mask],
            [mask],
            [None],
            [None],
        ]

        with pytest.raises(ValueError, match=r"[Aa]t least 3|3 camera|need.*3"):
            init_fish_states_from_masks(masks_per_camera, models)
