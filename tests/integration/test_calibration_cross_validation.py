"""Cross-validation tests comparing AquaPose RefractiveProjectionModel against AquaCal NumPy reference."""

import math

import numpy as np
import pytest
import torch
from aquacal.config.schema import CameraExtrinsics, CameraIntrinsics
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project, trace_ray_air_to_water

from aquapose.calibration import RefractiveProjectionModel


@pytest.fixture
def reference_camera_params() -> dict:
    """Realistic camera parameters from the reference rig geometry.

    Camera on the ring at radius 0.635m, positioned at angle 0 (positive X).
    Focal length ~1400 pixels, image 1600x1200, water_z=0.978m.

    Returns:
        Dictionary of camera parameter arrays.
    """
    return {
        "fx": 1400.0,
        "fy": 1400.0,
        "cx": 800.0,
        "cy": 600.0,
        "image_size": (1600, 1200),
        "R": np.eye(3, dtype=np.float64),
        "t": np.array([-0.635, 0.0, 0.0], dtype=np.float64),
        "water_z": 0.978,
        "normal": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "n_air": 1.0,
        "n_water": 1.333,
        "dist_coeffs": np.zeros(5, dtype=np.float64),
    }


@pytest.fixture
def rotated_camera_params() -> dict:
    """Camera rotated 30 degrees around Z (simulating off-axis ring camera).

    Returns:
        Dictionary of camera parameter arrays.
    """
    angle = math.radians(30.0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    R = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    C_world = np.array([0.635 * cos_a, 0.635 * sin_a, 0.0], dtype=np.float64)
    t = -R @ C_world

    return {
        "fx": 1400.0,
        "fy": 1400.0,
        "cx": 800.0,
        "cy": 600.0,
        "image_size": (1600, 1200),
        "R": R,
        "t": t,
        "water_z": 0.978,
        "normal": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "n_air": 1.0,
        "n_water": 1.333,
        "dist_coeffs": np.zeros(5, dtype=np.float64),
    }


def _build_aquacal(params: dict) -> tuple[Camera, Interface]:
    """Construct AquaCal Camera and Interface from parameter dict.

    Args:
        params: Camera parameter dictionary.

    Returns:
        Tuple of (Camera, Interface) AquaCal objects.
    """
    p = params
    K = np.array(
        [[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]], dtype=np.float64
    )
    intrinsics = CameraIntrinsics(
        K=K, dist_coeffs=p["dist_coeffs"], image_size=p["image_size"]
    )
    extrinsics = CameraExtrinsics(R=p["R"], t=p["t"])
    camera = Camera("test_cam", intrinsics, extrinsics)

    interface = Interface(
        normal=p["normal"],
        camera_distances={"test_cam": p["water_z"]},
        n_air=p["n_air"],
        n_water=p["n_water"],
    )
    return camera, interface


def _build_aquapose(params: dict) -> RefractiveProjectionModel:
    """Construct AquaPose RefractiveProjectionModel from parameter dict.

    Args:
        params: Camera parameter dictionary.

    Returns:
        Configured RefractiveProjectionModel.
    """
    p = params
    K = torch.tensor(
        [[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]],
        dtype=torch.float32,
    )
    R = torch.tensor(p["R"], dtype=torch.float32)
    t = torch.tensor(p["t"], dtype=torch.float32)
    normal = torch.tensor(p["normal"], dtype=torch.float32)
    return RefractiveProjectionModel(
        K, R, t, p["water_z"], normal, p["n_air"], p["n_water"]
    )


@pytest.mark.slow
class TestCastRayCrossValidation:
    """Cross-validation tests for cast_ray against AquaCal trace_ray_air_to_water."""

    def test_cast_ray_grid(self, reference_camera_params: dict):
        """Test cast_ray on a 10x10 pixel grid across the image."""
        camera, interface = _build_aquacal(reference_camera_params)
        model = _build_aquapose(reference_camera_params)

        u_grid = np.linspace(100, 1500, 10)
        v_grid = np.linspace(100, 1100, 10)
        pixels_list = [[u, v] for u in u_grid for v in v_grid]
        pixels_np = np.array(pixels_list, dtype=np.float64)

        origins_aquacal = []
        directions_aquacal = []
        valid_indices = []

        for i, pixel_np in enumerate(pixels_np):
            result = trace_ray_air_to_water(camera, interface, pixel_np)
            if result[0] is not None:
                origins_aquacal.append(result[0])
                directions_aquacal.append(result[1])
                valid_indices.append(i)

        if len(valid_indices) == 0:
            pytest.skip("No valid pixels in grid")

        pixels_pt = torch.tensor(pixels_np, dtype=torch.float32)
        origins_pt, directions_pt = model.cast_ray(pixels_pt)

        for list_pos, idx in enumerate(valid_indices):
            origin_ref = torch.tensor(origins_aquacal[list_pos], dtype=torch.float32)
            dir_ref = torch.tensor(directions_aquacal[list_pos], dtype=torch.float32)

            assert torch.allclose(origins_pt[idx], origin_ref, atol=1e-5), (
                f"Origin mismatch at pixel {pixels_np[idx]}: "
                f"AquaPose={origins_pt[idx].numpy()}, AquaCal={origin_ref.numpy()}"
            )
            assert torch.allclose(directions_pt[idx], dir_ref, atol=1e-5), (
                f"Direction mismatch at pixel {pixels_np[idx]}: "
                f"AquaPose={directions_pt[idx].numpy()}, AquaCal={dir_ref.numpy()}"
            )

    def test_cast_ray_principal_point(self, reference_camera_params: dict):
        """Test cast_ray at the principal point (zero refraction case)."""
        camera, interface = _build_aquacal(reference_camera_params)
        model = _build_aquapose(reference_camera_params)

        pixel_np = np.array([800.0, 600.0], dtype=np.float64)
        result_np = trace_ray_air_to_water(camera, interface, pixel_np)

        if result_np[0] is None:
            pytest.skip("AquaCal returned None for principal point")

        pixel_pt = torch.tensor([[800.0, 600.0]], dtype=torch.float32)
        origins_pt, directions_pt = model.cast_ray(pixel_pt)

        origin_ref = torch.tensor(result_np[0], dtype=torch.float32)
        dir_ref = torch.tensor(result_np[1], dtype=torch.float32)

        assert torch.allclose(origins_pt[0], origin_ref, atol=1e-5)
        assert torch.allclose(directions_pt[0], dir_ref, atol=1e-5)

    def test_cast_ray_image_corners(self, reference_camera_params: dict):
        """Test cast_ray at image corners (highest incidence angles)."""
        camera, interface = _build_aquacal(reference_camera_params)
        model = _build_aquapose(reference_camera_params)

        corners = np.array(
            [[0.0, 0.0], [1600.0, 0.0], [0.0, 1200.0], [1600.0, 1200.0]],
            dtype=np.float64,
        )

        for corner in corners:
            result_np = trace_ray_air_to_water(camera, interface, corner)
            if result_np[0] is None:
                continue

            pixel_pt = torch.tensor(corner, dtype=torch.float32).unsqueeze(0)
            origins_pt, directions_pt = model.cast_ray(pixel_pt)

            origin_ref = torch.tensor(result_np[0], dtype=torch.float32)
            dir_ref = torch.tensor(result_np[1], dtype=torch.float32)

            assert torch.allclose(origins_pt[0], origin_ref, atol=1e-5), (
                f"Origin mismatch at corner {corner}"
            )
            assert torch.allclose(directions_pt[0], dir_ref, atol=1e-5), (
                f"Direction mismatch at corner {corner}"
            )

    def test_cast_ray_rotated_camera_grid(self, rotated_camera_params: dict):
        """Test cast_ray on an 8x8 pixel grid with rotated camera."""
        camera, interface = _build_aquacal(rotated_camera_params)
        model = _build_aquapose(rotated_camera_params)

        u_grid = np.linspace(200, 1400, 8)
        v_grid = np.linspace(200, 1000, 8)
        pixels_list = [[u, v] for u in u_grid for v in v_grid]
        pixels_np = np.array(pixels_list, dtype=np.float64)

        origins_aquacal = []
        directions_aquacal = []
        valid_indices = []

        for i, pixel_np in enumerate(pixels_np):
            result = trace_ray_air_to_water(camera, interface, pixel_np)
            if result[0] is not None:
                origins_aquacal.append(result[0])
                directions_aquacal.append(result[1])
                valid_indices.append(i)

        if len(valid_indices) == 0:
            pytest.skip("No valid pixels in rotated camera grid")

        pixels_pt = torch.tensor(pixels_np, dtype=torch.float32)
        origins_pt, directions_pt = model.cast_ray(pixels_pt)

        for list_pos, idx in enumerate(valid_indices):
            origin_ref = torch.tensor(origins_aquacal[list_pos], dtype=torch.float32)
            dir_ref = torch.tensor(directions_aquacal[list_pos], dtype=torch.float32)

            assert torch.allclose(origins_pt[idx], origin_ref, atol=1e-5)
            assert torch.allclose(directions_pt[idx], dir_ref, atol=1e-5)


@pytest.mark.slow
class TestProjectCrossValidation:
    """Cross-validation tests for project against AquaCal refractive_project."""

    def test_project_grid(self, reference_camera_params: dict):
        """Test project on a 3D point grid at multiple depths and XY positions."""
        camera, interface = _build_aquacal(reference_camera_params)
        model = _build_aquapose(reference_camera_params)
        water_z = reference_camera_params["water_z"]

        x_grid = np.linspace(0.135, 1.135, 5)
        y_grid = np.linspace(-0.5, 0.5, 5)
        z_grid = np.linspace(water_z + 0.2, water_z + 0.7, 4)

        points_list = [[x, y, z] for x in x_grid for y in y_grid for z in z_grid]
        points_np = np.array(points_list, dtype=np.float64)

        pixels_aquacal = []
        valid_indices = []

        for i, point_np in enumerate(points_np):
            pixel_np = refractive_project(camera, interface, point_np)
            if pixel_np is not None:
                pixels_aquacal.append(pixel_np)
                valid_indices.append(i)

        if len(valid_indices) == 0:
            pytest.skip("No valid points in grid")

        points_pt = torch.tensor(points_np, dtype=torch.float32)
        pixels_pt, valid_pt = model.project(points_pt)

        for list_pos, idx in enumerate(valid_indices):
            pixel_ref = torch.tensor(pixels_aquacal[list_pos], dtype=torch.float32)

            assert valid_pt[idx].item(), (
                f"AquaPose marked point {points_np[idx]} invalid, AquaCal succeeded"
            )
            assert torch.allclose(pixels_pt[idx], pixel_ref, atol=1e-5), (
                f"Pixel mismatch for point {points_np[idx]}: "
                f"AquaPose={pixels_pt[idx].numpy()}, AquaCal={pixel_ref.numpy()}"
            )

    def test_project_nadir_point(self, reference_camera_params: dict):
        """Test projection of point directly below the camera center (nadir)."""
        camera, interface = _build_aquacal(reference_camera_params)
        model = _build_aquapose(reference_camera_params)
        water_z = reference_camera_params["water_z"]

        point_np = np.array([0.635, 0.0, water_z + 0.5], dtype=np.float64)
        pixel_np = refractive_project(camera, interface, point_np)

        if pixel_np is None:
            pytest.skip("AquaCal returned None for nadir point")

        point_pt = torch.tensor(point_np, dtype=torch.float32).unsqueeze(0)
        pixels_pt, valid_pt = model.project(point_pt)

        assert valid_pt[0].item()
        pixel_ref = torch.tensor(pixel_np, dtype=torch.float32)
        assert torch.allclose(pixels_pt[0], pixel_ref, atol=1e-5)

    def test_project_shallow_depth(self, reference_camera_params: dict):
        """Test project at shallow depth (water_z + 0.1)."""
        camera, interface = _build_aquacal(reference_camera_params)
        model = _build_aquapose(reference_camera_params)
        water_z = reference_camera_params["water_z"]

        point_np = np.array([0.635, 0.0, water_z + 0.1], dtype=np.float64)
        pixel_np = refractive_project(camera, interface, point_np)

        if pixel_np is None:
            pytest.skip("AquaCal returned None for shallow point")

        point_pt = torch.tensor(point_np, dtype=torch.float32).unsqueeze(0)
        pixels_pt, valid_pt = model.project(point_pt)

        assert valid_pt[0].item()
        pixel_ref = torch.tensor(pixel_np, dtype=torch.float32)
        assert torch.allclose(pixels_pt[0], pixel_ref, atol=1e-5)

    def test_project_mid_depth(self, reference_camera_params: dict):
        """Test project at mid depth (water_z + 0.3)."""
        camera, interface = _build_aquacal(reference_camera_params)
        model = _build_aquapose(reference_camera_params)
        water_z = reference_camera_params["water_z"]

        point_np = np.array([0.635, 0.0, water_z + 0.3], dtype=np.float64)
        pixel_np = refractive_project(camera, interface, point_np)

        if pixel_np is None:
            pytest.skip("AquaCal returned None for mid depth point")

        point_pt = torch.tensor(point_np, dtype=torch.float32).unsqueeze(0)
        pixels_pt, valid_pt = model.project(point_pt)

        assert valid_pt[0].item()
        pixel_ref = torch.tensor(pixel_np, dtype=torch.float32)
        assert torch.allclose(pixels_pt[0], pixel_ref, atol=1e-5)

    def test_project_deep_depth(self, reference_camera_params: dict):
        """Test project at deep depth (water_z + 0.5)."""
        camera, interface = _build_aquacal(reference_camera_params)
        model = _build_aquapose(reference_camera_params)
        water_z = reference_camera_params["water_z"]

        point_np = np.array([0.635, 0.0, water_z + 0.5], dtype=np.float64)
        pixel_np = refractive_project(camera, interface, point_np)

        if pixel_np is None:
            pytest.skip("AquaCal returned None for deep point")

        point_pt = torch.tensor(point_np, dtype=torch.float32).unsqueeze(0)
        pixels_pt, valid_pt = model.project(point_pt)

        assert valid_pt[0].item()
        pixel_ref = torch.tensor(pixel_np, dtype=torch.float32)
        assert torch.allclose(pixels_pt[0], pixel_ref, atol=1e-5)

    def test_project_off_center_positions(self, reference_camera_params: dict):
        """Test project at multiple off-center XY positions."""
        camera, interface = _build_aquacal(reference_camera_params)
        model = _build_aquapose(reference_camera_params)
        water_z = reference_camera_params["water_z"]

        xy_positions = [
            (0.635, 0.0),
            (0.835, 0.0),
            (0.635, 0.2),
            (0.435, -0.2),
            (0.835, 0.2),
        ]
        depth = water_z + 0.3

        for x, y in xy_positions:
            point_np = np.array([x, y, depth], dtype=np.float64)
            pixel_np = refractive_project(camera, interface, point_np)

            if pixel_np is None:
                continue

            point_pt = torch.tensor(point_np, dtype=torch.float32).unsqueeze(0)
            pixels_pt, valid_pt = model.project(point_pt)

            assert valid_pt[0].item(), f"AquaPose invalid for point ({x}, {y}, {depth})"
            pixel_ref = torch.tensor(pixel_np, dtype=torch.float32)
            assert torch.allclose(pixels_pt[0], pixel_ref, atol=1e-5), (
                f"Mismatch at ({x}, {y}): AquaPose={pixels_pt[0].numpy()}, AquaCal={pixel_ref.numpy()}"
            )

    def test_project_varying_depths(self, reference_camera_params: dict):
        """Test project at same XY but varying depths."""
        camera, interface = _build_aquacal(reference_camera_params)
        model = _build_aquapose(reference_camera_params)
        water_z = reference_camera_params["water_z"]

        x, y = 0.835, 0.1
        depths = np.linspace(0.2, 0.7, 10)
        points_np = np.array([[x, y, water_z + d] for d in depths], dtype=np.float64)

        pixels_aquacal = []
        valid_indices = []
        for i, point_np in enumerate(points_np):
            pixel_np = refractive_project(camera, interface, point_np)
            if pixel_np is not None:
                pixels_aquacal.append(pixel_np)
                valid_indices.append(i)

        if len(valid_indices) == 0:
            pytest.skip("No valid points at varying depths")

        points_pt = torch.tensor(points_np, dtype=torch.float32)
        pixels_pt, valid_pt = model.project(points_pt)

        for list_pos, idx in enumerate(valid_indices):
            pixel_ref = torch.tensor(pixels_aquacal[list_pos], dtype=torch.float32)
            assert valid_pt[idx].item()
            assert torch.allclose(pixels_pt[idx], pixel_ref, atol=1e-5)

    def test_project_rotated_camera(self, rotated_camera_params: dict):
        """Test project on a 3D point grid with rotated camera."""
        camera, interface = _build_aquacal(rotated_camera_params)
        model = _build_aquapose(rotated_camera_params)
        water_z = rotated_camera_params["water_z"]

        R_inv = rotated_camera_params["R"].T
        C = -R_inv @ rotated_camera_params["t"]

        x_grid = np.linspace(C[0] - 0.3, C[0] + 0.3, 4)
        y_grid = np.linspace(C[1] - 0.3, C[1] + 0.3, 4)
        z_grid = np.linspace(water_z + 0.3, water_z + 0.6, 3)

        points_list = [[x, y, z] for x in x_grid for y in y_grid for z in z_grid]
        points_np = np.array(points_list, dtype=np.float64)

        pixels_aquacal = []
        valid_indices = []
        for i, point_np in enumerate(points_np):
            pixel_np = refractive_project(camera, interface, point_np)
            if pixel_np is not None:
                pixels_aquacal.append(pixel_np)
                valid_indices.append(i)

        if len(valid_indices) == 0:
            pytest.skip("No valid points in rotated camera grid")

        points_pt = torch.tensor(points_np, dtype=torch.float32)
        pixels_pt, valid_pt = model.project(points_pt)

        for list_pos, idx in enumerate(valid_indices):
            pixel_ref = torch.tensor(pixels_aquacal[list_pos], dtype=torch.float32)
            assert valid_pt[idx].item()
            assert torch.allclose(pixels_pt[idx], pixel_ref, atol=1e-5)
