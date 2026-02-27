"""Unit tests for ForwardLUT generation, interpolation accuracy, and serialization."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aquapose.calibration.luts import (
    compute_lut_hash,
    generate_forward_lut,
    load_forward_lut,
    save_forward_lut,
    validate_forward_lut,
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
