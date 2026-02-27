"""Unit tests for ForwardLUT and InverseLUT generation, accuracy, and serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from aquapose.calibration.luts import (
    camera_overlap_graph,
    compute_lut_hash,
    generate_forward_lut,
    generate_inverse_lut,
    ghost_point_lookup,
    load_forward_lut,
    load_inverse_lut,
    save_forward_lut,
    save_inverse_lut,
    validate_forward_lut,
    validate_inverse_lut,
)
from aquapose.calibration.projection import RefractiveProjectionModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_model() -> RefractiveProjectionModel:
    """Build a RefractiveProjectionModel with simple, known parameters.

    Camera configuration:
    - 640x480 image, focal length ~500, principal point at (320, 240)
    - Identity rotation (camera frame aligned with world frame)
    - Camera center 0.5 m above the water surface (at z=0)
    - Standard air-water refractive indices

    Returns:
        RefractiveProjectionModel configured for CPU.
    """
    K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    # t = -R @ C → camera at (0, 0, -0.5) in world → t = [0, 0, 0.5]
    # But we want camera center C = (0, 0, -0.5): C = -R^T @ t → t = -R @ C = [0, 0, 0.5]
    t = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32)
    water_z = 0.0
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    n_air = 1.0
    n_water = 1.333

    return RefractiveProjectionModel(
        K=K, R=R, t=t, water_z=water_z, normal=normal, n_air=n_air, n_water=n_water
    )


class _SimpleLutConfig:
    """Minimal LutConfigLike implementation for testing."""

    def __init__(
        self,
        tank_diameter: float = 2.0,
        tank_height: float = 1.0,
        voxel_resolution_m: float = 0.02,
        margin_fraction: float = 0.1,
        forward_grid_step: int = 1,
    ) -> None:
        self.tank_diameter = tank_diameter
        self.tank_height = tank_height
        self.voxel_resolution_m = voxel_resolution_m
        self.margin_fraction = margin_fraction
        self.forward_grid_step = forward_grid_step


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_forward_lut_grid_shape() -> None:
    """Grid arrays must have the expected shape for a given grid_step."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = make_test_model()
    image_size = (640, 480)
    grid_step = 10

    lut = generate_forward_lut("cam_test", model, image_size, grid_step=grid_step)

    expected_grid_w = len(range(0, 640, 10))  # 64
    expected_grid_h = len(range(0, 480, 10))  # 48

    assert lut.grid_origins.shape == (
        expected_grid_h,
        expected_grid_w,
        3,
    ), f"grid_origins shape mismatch: {lut.grid_origins.shape}"
    assert lut.grid_directions.shape == (
        expected_grid_h,
        expected_grid_w,
        3,
    ), f"grid_directions shape mismatch: {lut.grid_directions.shape}"

    # All direction vectors must be unit vectors
    norms = np.linalg.norm(lut.grid_directions.reshape(-1, 3), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"Directions not unit-length: {norms.min():.6f}"
    )

    assert lut.camera_id == "cam_test"
    assert lut.grid_step == grid_step
    assert lut.image_size == image_size


def test_forward_lut_cast_ray_matches_model() -> None:
    """With grid_step=1, LUT.cast_ray() must closely match model.cast_ray()."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = make_test_model()
    image_size = (640, 480)

    lut = generate_forward_lut("cam_test", model, image_size, grid_step=1)

    # Sample 50 pixel coordinates (not on edges to avoid boundary effects)
    rng = np.random.default_rng(42)
    us = rng.uniform(10, 630, 50).astype(np.float32)
    vs = rng.uniform(10, 470, 50).astype(np.float32)
    samples = torch.from_numpy(np.stack([us, vs], axis=-1))

    with torch.no_grad():
        lut_origins, lut_dirs = lut.cast_ray(samples)
        model_origins, model_dirs = model.cast_ray(samples)

    lut_origins = lut_origins.cpu()
    lut_dirs = lut_dirs.cpu()
    model_origins = model_origins.cpu()
    model_dirs = model_dirs.cpu()

    origin_dists = torch.linalg.norm(lut_origins - model_origins, dim=-1)
    assert float(origin_dists.max()) < 1e-4, (
        f"Max origin distance {float(origin_dists.max()):.2e} m exceeds 1e-4 m threshold"
    )

    dot = (lut_dirs * model_dirs).sum(dim=-1).clamp(-1.0, 1.0)
    angular_errors = torch.acos(dot).abs() * (180.0 / torch.pi)
    assert float(angular_errors.max()) < 0.01, (
        f"Max angular error {float(angular_errors.max()):.4f}° exceeds 0.01° threshold"
    )


def test_forward_lut_interpolation_accuracy() -> None:
    """With grid_step=4, bilinear interpolation must stay within 0.1° angular error."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = make_test_model()
    image_size = (640, 480)

    lut = generate_forward_lut("cam_test", model, image_size, grid_step=4)

    # Sample 100 random sub-pixel coordinates (not on grid points)
    rng = np.random.default_rng(42)
    us = rng.uniform(5.0, 630.0, 100).astype(np.float32)
    vs = rng.uniform(5.0, 470.0, 100).astype(np.float32)
    samples = torch.from_numpy(np.stack([us, vs], axis=-1))

    with torch.no_grad():
        lut_origins, lut_dirs = lut.cast_ray(samples)
        model_origins, model_dirs = model.cast_ray(samples)

    lut_origins = lut_origins.cpu()
    lut_dirs = lut_dirs.cpu()
    model_origins = model_origins.cpu()
    model_dirs = model_dirs.cpu()

    origin_dists = torch.linalg.norm(lut_origins - model_origins, dim=-1)
    assert float(origin_dists.max()) < 1e-3, (
        f"Max origin distance {float(origin_dists.max()):.2e} m exceeds 1e-3 m threshold"
    )

    dot = (lut_dirs * model_dirs).sum(dim=-1).clamp(-1.0, 1.0)
    angular_errors = torch.acos(dot).abs() * (180.0 / torch.pi)
    assert float(angular_errors.max()) < 0.1, (
        f"Max angular error {float(angular_errors.max()):.4f}° exceeds 0.1° threshold"
    )


def test_forward_lut_serialization_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    """Save and reload a ForwardLUT — all arrays and metadata must match exactly."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = make_test_model()
    image_size = (640, 480)
    lut = generate_forward_lut("cam_roundtrip", model, image_size, grid_step=8)

    test_hash = "test-hash-for-roundtrip"
    out_path = tmp_path / "test_lut.npz"
    save_forward_lut(lut, out_path, config_hash=test_hash)

    loaded_lut, loaded_hash = load_forward_lut(out_path)

    assert loaded_lut.camera_id == lut.camera_id
    assert loaded_lut.grid_step == lut.grid_step
    assert loaded_lut.image_size == lut.image_size
    assert loaded_hash == test_hash
    assert np.allclose(loaded_lut.grid_origins, lut.grid_origins), (
        "grid_origins mismatch"
    )
    assert np.allclose(loaded_lut.grid_directions, lut.grid_directions), (
        "grid_directions mismatch"
    )


def test_compute_lut_hash_changes_with_config(tmp_path: pytest.TempPathFactory) -> None:
    """compute_lut_hash must produce different hashes for different configs."""
    # Write a minimal dummy calibration file
    dummy_cal = tmp_path / "calibration.json"
    dummy_cal.write_text(json.dumps({"version": "test"}), encoding="utf-8")

    cfg_a = _SimpleLutConfig(voxel_resolution_m=0.02)
    cfg_b = _SimpleLutConfig(voxel_resolution_m=0.05)
    cfg_c = _SimpleLutConfig(voxel_resolution_m=0.02)  # same as cfg_a

    hash_a = compute_lut_hash(dummy_cal, cfg_a)
    hash_b = compute_lut_hash(dummy_cal, cfg_b)
    hash_c = compute_lut_hash(dummy_cal, cfg_c)

    assert hash_a != hash_b, "Hashes must differ for different voxel_resolution_m"
    assert hash_a == hash_c, "Hashes must be equal for identical configs"

    # Verify hash length (first 16 chars of sha256 hex)
    assert len(hash_a) == 16


def test_validate_forward_lut_passes() -> None:
    """validate_forward_lut must pass for a LUT generated with grid_step=1."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = make_test_model()
    image_size = (640, 480)

    lut = generate_forward_lut("cam_validate", model, image_size, grid_step=1)
    result = validate_forward_lut(lut, model, n_samples=100, seed=42)

    # validate_forward_lut itself enforces < 0.1°; here we verify it's well below that.
    # With grid_step=1, sub-pixel bilinear interpolation yields errors < 0.05°.
    assert result["max_angular_error_deg"] < 0.05, (
        f"Max angular error {result['max_angular_error_deg']:.4f}° too high"
    )
    assert "mean_angular_error_deg" in result
    assert "max_origin_error_m" in result
    assert "mean_origin_error_m" in result


def test_forward_lut_edge_pixels() -> None:
    """LUT.cast_ray() must return valid (non-NaN) results for image corner pixels."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = make_test_model()
    image_size = (640, 480)
    width, height = image_size

    lut = generate_forward_lut("cam_edge", model, image_size, grid_step=1)

    corners = torch.tensor(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [0.0, float(height - 1)],
            [float(width - 1), float(height - 1)],
        ],
        dtype=torch.float32,
    )

    origins, directions = lut.cast_ray(corners)

    assert not torch.any(torch.isnan(origins)), "NaN origins at edge pixels"
    assert not torch.any(torch.isnan(directions)), "NaN directions at edge pixels"

    # Directions must be unit vectors
    norms = torch.linalg.norm(directions, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"Edge pixel directions not unit-length: {norms}"
    )


# ---------------------------------------------------------------------------
# Inverse LUT test helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockCameraData:
    """Minimal CameraData substitute for InverseLUT tests."""

    name: str
    K: torch.Tensor
    dist_coeffs: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    image_size: tuple[int, int]
    is_fisheye: bool
    is_auxiliary: bool


@dataclass
class _MockCalibrationData:
    """Minimal CalibrationData substitute for InverseLUT tests."""

    cameras: dict[str, _MockCameraData]
    water_z: float
    interface_normal: torch.Tensor
    n_air: float
    n_water: float

    @property
    def ring_cameras(self) -> list[str]:
        """Return sorted non-auxiliary camera names."""
        return sorted(
            [name for name, cam in self.cameras.items() if not cam.is_auxiliary]
        )

    def camera_positions(self) -> dict[str, torch.Tensor]:
        """Compute world-frame camera centres as C = -R^T @ t."""
        return {name: -cam.R.T @ cam.t for name, cam in self.cameras.items()}


def _look_at_rotation(cam_pos: np.ndarray, target: np.ndarray) -> torch.Tensor:
    """Build a rotation matrix R (world-to-camera) for a camera looking at target.

    Args:
        cam_pos: Camera centre in world frame, shape (3,).
        target: Look-at target in world frame, shape (3,).

    Returns:
        Rotation matrix (world to camera), shape (3, 3), float32.
    """
    # Forward: direction from camera toward target (in camera +Z direction)
    fwd = target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)

    # World up is y-axis; use x if forward is near y
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(fwd, world_up)) > 0.9:
        world_up = np.array([1.0, 0.0, 0.0])

    right = np.cross(fwd, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, fwd)
    up = up / np.linalg.norm(up)

    # Rows of R (world-to-camera): right, -up (camera y points down), fwd
    R = np.stack([right, -up, fwd], axis=0).astype(np.float32)
    return torch.from_numpy(R)


def make_test_rig(
    tank_diameter: float = 0.5,
    tank_height: float = 0.3,
    water_z: float = 0.0,
) -> _MockCalibrationData:
    """Build a synthetic 3-camera CalibrationData for InverseLUT tests.

    Places three cameras in a ring above the water surface, all looking
    toward the tank centre. Uses a small tank for fast test execution.

    Args:
        tank_diameter: Tank diameter in metres.
        tank_height: Water column height in metres.
        water_z: Z-coordinate of the water surface.

    Returns:
        A CalibrationData-like object with 3 ring cameras.
    """
    # Tank centre target: midpoint of the water column
    tank_center = np.array([0.0, 0.0, water_z + tank_height * 0.5], dtype=np.float32)

    # Camera positions: arranged in a ring, above (negative z = above water surface)
    cam_positions_world = {
        "cam_a": np.array([0.3, 0.0, water_z - 0.3], dtype=np.float32),
        "cam_b": np.array([-0.3, 0.0, water_z - 0.3], dtype=np.float32),
        "cam_c": np.array([0.0, 0.3, water_z - 0.3], dtype=np.float32),
    }

    # Simple intrinsics: 640x480, focal length 400
    K = torch.tensor(
        [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    cameras: dict[str, _MockCameraData] = {}
    for cam_id, c_world in cam_positions_world.items():
        R = _look_at_rotation(c_world, tank_center)
        # t = -R @ C (world-to-camera translation)
        t = -(R @ torch.from_numpy(c_world))
        cameras[cam_id] = _MockCameraData(
            name=cam_id,
            K=K.clone(),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=R,
            t=t,
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

    return _MockCalibrationData(
        cameras=cameras,
        water_z=water_z,
        interface_normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
        n_air=1.0,
        n_water=1.333,
    )


class _CoarseLutConfig:
    """LutConfigLike with coarse voxel resolution for fast test execution."""

    def __init__(
        self,
        tank_diameter: float = 0.5,
        tank_height: float = 0.3,
        voxel_resolution_m: float = 0.05,
        margin_fraction: float = 0.1,
        forward_grid_step: int = 1,
    ) -> None:
        self.tank_diameter = tank_diameter
        self.tank_height = tank_height
        self.voxel_resolution_m = voxel_resolution_m
        self.margin_fraction = margin_fraction
        self.forward_grid_step = forward_grid_step


# ---------------------------------------------------------------------------
# Inverse LUT tests
# ---------------------------------------------------------------------------


def test_cylindrical_voxel_grid_shape() -> None:
    """All voxels must lie within the expanded cylindrical volume."""
    from aquapose.calibration.luts import _build_cylindrical_voxel_grid

    diameter = 0.5
    height = 0.3
    water_z = 0.0
    resolution = 0.05
    margin = 0.1
    cx, cy = 0.0, 0.0

    voxels, _ = _build_cylindrical_voxel_grid(
        tank_center_xy=(cx, cy),
        tank_diameter=diameter,
        tank_height=height,
        water_z=water_z,
        voxel_resolution=resolution,
        margin_fraction=margin,
    )

    assert voxels.ndim == 2
    assert voxels.shape[1] == 3
    assert len(voxels) > 0, "Should have at least some voxels"

    radius = (diameter / 2.0) * (1.0 + margin)
    z_max = water_z + height * (1.0 + margin)

    dist_xy = np.sqrt((voxels[:, 0] - cx) ** 2 + (voxels[:, 1] - cy) ** 2)
    assert np.all(dist_xy <= radius + 1e-5), "Voxel outside cylinder radius"
    assert np.all(voxels[:, 2] >= water_z - 1e-5), "Voxel below water surface"
    assert np.all(voxels[:, 2] <= z_max + 1e-5), "Voxel above expanded tank top"


def test_inverse_lut_visibility_mask_shape() -> None:
    """InverseLUT visibility_mask and projected_pixels must have expected shapes."""
    torch.manual_seed(42)
    rig = make_test_rig()
    cfg = _CoarseLutConfig()

    lut = generate_inverse_lut(rig, cfg)  # type: ignore[arg-type]

    n_voxels = lut.voxel_centers.shape[0]
    n_cams = 3

    assert lut.visibility_mask.shape == (n_voxels, n_cams), (
        f"visibility_mask shape: {lut.visibility_mask.shape}"
    )
    assert lut.projected_pixels.shape == (n_voxels, n_cams, 2), (
        f"projected_pixels shape: {lut.projected_pixels.shape}"
    )

    all_visible = lut.visibility_mask.all(axis=1)
    assert all_visible.any(), "Expected some voxels visible to all 3 cameras"

    for cam_idx in range(n_cams):
        not_vis = ~lut.visibility_mask[:, cam_idx]
        if not_vis.any():
            assert np.all(np.isnan(lut.projected_pixels[not_vis, cam_idx, 0])), (
                "Expected NaN pixels for non-visible voxels"
            )


def test_inverse_lut_projected_pixels_match_model() -> None:
    """Stored projected pixels must match RefractiveProjectionModel.project() within 0.01 px."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    rig = make_test_rig()
    cfg = _CoarseLutConfig()
    lut = generate_inverse_lut(rig, cfg)  # type: ignore[arg-type]

    cam_idx = 0
    cam_id = lut.camera_ids[cam_idx]
    cam = rig.cameras[cam_id]

    model = RefractiveProjectionModel(
        K=cam.K,
        R=cam.R,
        t=cam.t,
        water_z=rig.water_z,
        normal=rig.interface_normal,
        n_air=rig.n_air,
        n_water=rig.n_water,
    )

    visible = np.where(lut.visibility_mask[:, cam_idx])[0]
    assert len(visible) > 0, "No visible voxels for camera 0"

    sample = rng.choice(visible, size=min(20, len(visible)), replace=False)
    centers = torch.from_numpy(lut.voxel_centers[sample])

    with torch.no_grad():
        model_pix, valid = model.project(centers)

    model_pix_np = model_pix.cpu().numpy()
    valid_np = valid.cpu().numpy()
    stored = lut.projected_pixels[sample, cam_idx, :]

    valid_both = valid_np & ~np.isnan(stored[:, 0])
    assert valid_both.any(), "No valid voxels to compare"

    diffs = np.linalg.norm(stored[valid_both] - model_pix_np[valid_both], axis=-1)
    assert float(diffs.max()) < 0.01, (
        f"Max pixel error {diffs.max():.4f} px exceeds 0.01 px"
    )


def test_camera_overlap_graph() -> None:
    """All 3 camera pairs must appear in the overlap graph with positive shared voxels."""
    torch.manual_seed(42)
    rig = make_test_rig()
    cfg = _CoarseLutConfig()
    lut = generate_inverse_lut(rig, cfg)  # type: ignore[arg-type]

    graph = camera_overlap_graph(lut, min_shared_voxels=1)

    cam_ids = sorted(lut.camera_ids)
    expected_pairs = {
        (min(cam_ids[i], cam_ids[j]), max(cam_ids[i], cam_ids[j]))
        for i in range(3)
        for j in range(i + 1, 3)
    }

    for pair in expected_pairs:
        assert pair in graph, f"Expected pair {pair} in overlap graph"
        assert graph[pair] > 0, f"Expected positive shared voxels for {pair}"


def test_ghost_point_lookup_returns_visible_cameras() -> None:
    """Ghost-point lookup returns correct cameras for in-tank points; empty for outside."""
    torch.manual_seed(42)
    rig = make_test_rig()
    cfg = _CoarseLutConfig()
    lut = generate_inverse_lut(rig, cfg)  # type: ignore[arg-type]

    all_visible = np.where(lut.visibility_mask.all(axis=1))[0]
    assert len(all_visible) > 0, "Expected some voxels visible to all cameras"

    voxel_center = lut.voxel_centers[int(all_visible[0])]
    query_pt = torch.from_numpy(voxel_center).unsqueeze(0)
    results = ghost_point_lookup(lut, query_pt)

    assert len(results) == 1
    cam_results = results[0]
    assert len(cam_results) == 3, (
        f"Expected 3 cameras, got {len(cam_results)}: {[r[0] for r in cam_results]}"
    )

    returned_cam_ids = {r[0] for r in cam_results}
    assert returned_cam_ids == set(lut.camera_ids)

    for _, u, v in cam_results:
        assert np.isfinite(u), f"Non-finite u coordinate: {u}"
        assert np.isfinite(v), f"Non-finite v coordinate: {v}"

    far_point = torch.tensor([[100.0, 100.0, 100.0]])
    out_results = ghost_point_lookup(lut, far_point)
    assert len(out_results) == 1
    assert out_results[0] == [], "Expected empty result for point outside tank"


def test_inverse_lut_serialization_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    """Save and reload InverseLUT — all arrays and metadata must match exactly."""
    torch.manual_seed(42)
    rig = make_test_rig()
    cfg = _CoarseLutConfig()
    lut = generate_inverse_lut(rig, cfg)  # type: ignore[arg-type]

    out_path = tmp_path / "inverse.npz"
    test_hash = "inverse-roundtrip-test"
    save_inverse_lut(lut, out_path, config_hash=test_hash)

    loaded_lut, loaded_hash = load_inverse_lut(out_path)

    assert loaded_hash == test_hash
    assert loaded_lut.camera_ids == lut.camera_ids
    assert loaded_lut.voxel_resolution == lut.voxel_resolution
    assert loaded_lut.grid_bounds == lut.grid_bounds

    assert np.array_equal(loaded_lut.voxel_centers, lut.voxel_centers)
    assert np.array_equal(loaded_lut.visibility_mask, lut.visibility_mask)

    stored = lut.projected_pixels
    reloaded = loaded_lut.projected_pixels
    nan_mask = np.isnan(stored)
    assert np.array_equal(np.isnan(reloaded), nan_mask), "NaN pattern mismatch"
    assert np.allclose(stored[~nan_mask], reloaded[~nan_mask]), (
        "projected_pixels value mismatch"
    )


def test_validate_inverse_lut_passes() -> None:
    """validate_inverse_lut must not raise and report sub-pixel error."""
    torch.manual_seed(42)
    rig = make_test_rig()
    cfg = _CoarseLutConfig()
    lut = generate_inverse_lut(rig, cfg)  # type: ignore[arg-type]

    result = validate_inverse_lut(lut, rig, n_samples=20, seed=42)  # type: ignore[arg-type]

    assert result["max_pixel_error"] < 0.01, (
        f"Max pixel error {result['max_pixel_error']:.4f} px too high"
    )
    assert "mean_pixel_error" in result


def test_coverage_histogram_output(capsys: pytest.CaptureFixture[str]) -> None:
    """generate_inverse_lut must print a coverage histogram to stdout."""
    torch.manual_seed(42)
    rig = make_test_rig()
    cfg = _CoarseLutConfig()

    generate_inverse_lut(rig, cfg)  # type: ignore[arg-type]

    captured = capsys.readouterr()
    assert "Camera coverage histogram" in captured.out
    assert "1+ cameras:" in captured.out
    assert "%" in captured.out
