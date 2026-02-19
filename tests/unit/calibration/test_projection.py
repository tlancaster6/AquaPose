"""Tests for RefractiveProjectionModel."""

import math

import pytest
import torch

from aquapose.calibration import RefractiveProjectionModel


@pytest.fixture
def simple_model() -> RefractiveProjectionModel:
    """Synthetic reference geometry from plan spec.

    Camera at (0.635, 0, 0), R=eye(3), t=(-0.635, 0, 0),
    K with fx=fy=1400, cx=800, cy=600, water_z=0.978,
    normal=(0,0,-1), n_air=1.0, n_water=1.333, no distortion.
    """
    K = torch.tensor(
        [[1400.0, 0.0, 800.0], [0.0, 1400.0, 600.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.tensor([-0.635, 0.0, 0.0], dtype=torch.float32)
    water_z = 0.978
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    n_air = 1.0
    n_water = 1.333
    return RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)


@pytest.fixture
def origin_model() -> RefractiveProjectionModel:
    """Simple camera at world origin looking straight down."""
    K = torch.tensor(
        [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.zeros(3, dtype=torch.float32)
    water_z = 1.0
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return RefractiveProjectionModel(K, R, t, water_z, normal, 1.0, 1.333)


class TestConstructor:
    """Tests for RefractiveProjectionModel initialization."""

    def test_stores_all_parameters(self, simple_model: RefractiveProjectionModel):
        """Test that constructor stores all parameters correctly."""
        assert simple_model.water_z == 0.978
        assert simple_model.n_air == 1.0
        assert simple_model.n_water == 1.333
        assert torch.allclose(
            simple_model.normal, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        )

    def test_precomputes_K_inv(self, simple_model: RefractiveProjectionModel):
        """Test that K_inv is the true inverse of K.

        float32 inversion of large focal-length matrices (fx=1400) has
        limited precision; atol=1e-4 is appropriate for float32 here.
        """
        identity = simple_model.K @ simple_model.K_inv
        expected = torch.eye(3, dtype=torch.float32)
        assert torch.allclose(identity, expected, atol=1e-4)

    def test_precomputes_camera_center(self):
        """Test camera center C = -R^T @ t is precomputed correctly."""
        R = torch.eye(3, dtype=torch.float32)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        K = torch.eye(3, dtype=torch.float32)
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        model = RefractiveProjectionModel(K, R, t, 1.0, normal, 1.0, 1.333)

        expected_C = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
        assert torch.allclose(model.C, expected_C)

    def test_precomputes_n_ratio(self, simple_model: RefractiveProjectionModel):
        """Test that n_ratio = n_air / n_water is precomputed."""
        expected = 1.0 / 1.333
        assert abs(simple_model.n_ratio - expected) < 1e-6

    def test_to_method_returns_self(self, simple_model: RefractiveProjectionModel):
        """Test that to() returns self for method chaining."""
        result = simple_model.to("cpu")
        assert result is simple_model

    def test_to_method_moves_tensors(self, simple_model: RefractiveProjectionModel):
        """Test that to() moves all internal tensors to the target device."""
        model = simple_model.to("cpu")
        assert model.K.device.type == "cpu"
        assert model.K_inv.device.type == "cpu"
        assert model.R.device.type == "cpu"
        assert model.t.device.type == "cpu"
        assert model.C.device.type == "cpu"
        assert model.normal.device.type == "cpu"


class TestProject:
    """Tests for RefractiveProjectionModel.project method."""

    def test_valid_for_underwater_points(self, origin_model: RefractiveProjectionModel):
        """Test project() returns valid pixels for underwater points (Z > water_z)."""
        points = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32)
        pixels, valid = origin_model.project(points)

        assert valid[0].item() is True
        assert torch.all(torch.isfinite(pixels[0])).item()

    def test_invalid_for_points_above_water(
        self, origin_model: RefractiveProjectionModel
    ):
        """Test project() returns invalid for points above water (Z < water_z)."""
        points = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32)
        pixels, valid = origin_model.project(points)

        assert valid[0].item() is False
        assert torch.isnan(pixels[0, 0]).item()
        assert torch.isnan(pixels[0, 1]).item()

    def test_nadir_point_projects_to_principal_point(
        self, origin_model: RefractiveProjectionModel
    ):
        """Test point directly below camera projects to principal point."""
        points = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32)
        pixels, valid = origin_model.project(points)

        assert valid[0].item() is True
        expected = torch.tensor([[500.0, 500.0]], dtype=torch.float32)
        assert torch.allclose(pixels, expected, atol=1e-3)

    def test_output_shapes(self, origin_model: RefractiveProjectionModel):
        """Test that project() output shapes are correct."""
        points = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32)
        pixels, valid = origin_model.project(points)

        assert pixels.shape == (1, 2)
        assert valid.shape == (1,)
        assert pixels.dtype == torch.float32
        assert valid.dtype == torch.bool

    def test_batch_processing(self, simple_model: RefractiveProjectionModel):
        """Test project() handles batch of N points simultaneously."""
        torch.manual_seed(42)
        N = 50
        points = torch.rand(N, 3, dtype=torch.float32)
        points[:, 0] += 0.3
        points[:, 1] -= 0.2
        points[:, 2] = points[:, 2] * 0.5 + 1.1  # Z in [1.1, 1.6]

        pixels, valid = simple_model.project(points)

        assert pixels.shape == (N, 2)
        assert valid.shape == (N,)
        assert pixels.dtype == torch.float32

    def test_mixed_valid_invalid(self, origin_model: RefractiveProjectionModel):
        """Test handling of mixed valid and invalid points in one batch."""
        points = torch.tensor(
            [
                [0.0, 0.0, 1.5],  # valid: below water
                [0.0, 0.0, 0.5],  # invalid: above water
                [0.2, 0.1, 1.3],  # valid: below water
                [0.0, 0.0, 0.9],  # invalid: above water
            ],
            dtype=torch.float32,
        )

        pixels, valid = origin_model.project(points)

        assert valid[0].item() is True
        assert valid[1].item() is False
        assert valid[2].item() is True
        assert valid[3].item() is False

        assert torch.all(torch.isfinite(pixels[0])).item()
        assert torch.all(torch.isnan(pixels[1])).item()
        assert torch.all(torch.isfinite(pixels[2])).item()
        assert torch.all(torch.isnan(pixels[3])).item()

    def test_autograd_backward_completes(self, simple_model: RefractiveProjectionModel):
        """Test that autograd backward pass completes through project()."""
        points = torch.tensor(
            [[0.635, 0.0, 1.5], [0.835, 0.1, 1.3]],
            dtype=torch.float32,
            requires_grad=True,
        )

        pixels, valid = simple_model.project(points)
        loss = pixels[valid].sum()
        loss.backward()

        assert points.grad is not None
        assert points.grad.shape == (2, 3)
        assert torch.all(torch.isfinite(points.grad)).item()

    def test_gradcheck_float64(self, simple_model: RefractiveProjectionModel):
        """Test torch.autograd.gradcheck passes with float64 inputs."""
        K64 = simple_model.K.double()
        R64 = simple_model.R.double()
        t64 = simple_model.t.double()
        normal64 = simple_model.normal.double()

        model64 = RefractiveProjectionModel(
            K64,
            R64,
            t64,
            simple_model.water_z,
            normal64,
            simple_model.n_air,
            simple_model.n_water,
        )

        # Single point directly in front of camera, well below water
        points = torch.tensor(
            [[0.635, 0.0, 1.5]], dtype=torch.float64, requires_grad=True
        )

        def project_fn(pts: torch.Tensor) -> torch.Tensor:
            px, v = model64.project(pts)
            return px[v]

        assert torch.autograd.gradcheck(project_fn, (points,), atol=1e-4, rtol=1e-3)


class TestCastRay:
    """Tests for RefractiveProjectionModel.cast_ray method."""

    def test_origins_on_water_surface(self, origin_model: RefractiveProjectionModel):
        """Test that all ray origins lie on the water surface (Z = water_z)."""
        torch.manual_seed(42)
        pixels = torch.rand(100, 2, dtype=torch.float32) * 1000

        origins, _ = origin_model.cast_ray(pixels)

        expected_z = torch.full((100,), origin_model.water_z, dtype=torch.float32)
        assert torch.allclose(origins[:, 2], expected_z, atol=1e-5)

    def test_directions_are_unit_vectors(self, origin_model: RefractiveProjectionModel):
        """Test that all ray directions are unit vectors."""
        torch.manual_seed(42)
        pixels = torch.rand(100, 2, dtype=torch.float32) * 1000

        _, directions = origin_model.cast_ray(pixels)

        norms = torch.linalg.norm(directions, dim=-1)
        expected = torch.ones(100, dtype=torch.float32)
        assert torch.allclose(norms, expected, atol=1e-5)

    def test_nadir_ray_no_refraction(self, origin_model: RefractiveProjectionModel):
        """Test ray at principal point has no refraction (straight down)."""
        pixels = torch.tensor([[500.0, 500.0]], dtype=torch.float32)
        origins, directions = origin_model.cast_ray(pixels)

        expected_origin = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        expected_dir = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        assert torch.allclose(origins, expected_origin, atol=1e-5)
        assert torch.allclose(directions, expected_dir, atol=1e-5)

    def test_snells_law_at_known_angle(self, origin_model: RefractiveProjectionModel):
        """Test that cast_ray directions satisfy Snell's law for a known incidence angle."""
        theta_air_deg = 10.0
        dx = 1000.0 * math.tan(math.radians(theta_air_deg))
        pixels = torch.tensor([[500.0 + dx, 500.0]], dtype=torch.float32)

        _, directions = origin_model.cast_ray(pixels)

        theta_air = math.radians(theta_air_deg)
        sin_theta_water = (1.0 / 1.333) * math.sin(theta_air)
        theta_water = math.asin(sin_theta_water)
        expected_tan = math.tan(theta_water)
        actual_tan = directions[0, 0].item() / directions[0, 2].item()

        assert abs(actual_tan - expected_tan) < 1e-4

    def test_refraction_bends_toward_normal(
        self, origin_model: RefractiveProjectionModel
    ):
        """Test that off-axis rays refract toward the normal (air to water)."""
        pixels = torch.tensor([[600.0, 500.0]], dtype=torch.float32)
        _, directions = origin_model.cast_ray(pixels)

        incident_tan = 100.0 / 1000.0
        refracted_tan = directions[0, 0].item() / directions[0, 2].item()

        assert directions[0, 0] > 0
        assert directions[0, 2] > 0
        assert refracted_tan < incident_tan

    def test_batch_processing(self, simple_model: RefractiveProjectionModel):
        """Test cast_ray handles batch of N pixels simultaneously."""
        torch.manual_seed(42)
        N = 50
        pixels = torch.rand(N, 2, dtype=torch.float32)
        pixels[:, 0] = pixels[:, 0] * 1200 + 200
        pixels[:, 1] = pixels[:, 1] * 800 + 100

        origins, directions = simple_model.cast_ray(pixels)

        assert origins.shape == (N, 3)
        assert directions.shape == (N, 3)
        assert origins.dtype == torch.float32
        assert directions.dtype == torch.float32

    def test_autograd_backward_completes(self, simple_model: RefractiveProjectionModel):
        """Test that autograd backward pass completes through cast_ray()."""
        pixels = torch.tensor(
            [[800.0, 600.0], [900.0, 650.0]],
            dtype=torch.float32,
            requires_grad=True,
        )

        origins, directions = simple_model.cast_ray(pixels)
        loss = origins.sum() + directions.sum()
        loss.backward()

        assert pixels.grad is not None
        assert pixels.grad.shape == (2, 2)


class TestRoundtrip:
    """Tests for project/cast_ray roundtrip consistency."""

    def test_project_then_cast_ray(self, simple_model: RefractiveProjectionModel):
        """Test 3D point -> project -> cast_ray -> reconstruct 3D point."""
        original = torch.tensor([[0.835, 0.1, 1.5]], dtype=torch.float32)

        pixels, valid = simple_model.project(original)
        assert valid[0].item() is True

        origins, directions = simple_model.cast_ray(pixels)
        depth = (original[0, 2] - origins[0, 2]) / directions[0, 2]
        reconstructed = origins + depth * directions

        assert torch.allclose(reconstructed, original, atol=1e-4)

    def test_cast_ray_then_project(self, simple_model: RefractiveProjectionModel):
        """Test pixel -> cast_ray -> 3D point -> project -> pixel roundtrip."""
        original_pixels = torch.tensor(
            [[800.0, 600.0], [950.0, 650.0], [700.0, 580.0]],
            dtype=torch.float32,
        )

        origins, directions = simple_model.cast_ray(original_pixels)
        depth = 0.5
        points_3d = origins + depth * directions

        recovered_pixels, valid = simple_model.project(points_3d)

        assert torch.all(valid).item()
        assert torch.allclose(recovered_pixels, original_pixels, atol=1e-4)

    def test_multiple_depths(self, simple_model: RefractiveProjectionModel):
        """Test roundtrip consistency at shallow, mid, and deep underwater depths."""
        water_z = simple_model.water_z
        depths = [water_z + 0.1, water_z + 0.3, water_z + 0.5]

        for depth in depths:
            original = torch.tensor([[0.635, 0.0, depth]], dtype=torch.float32)
            pixels, valid = simple_model.project(original)

            if not valid[0].item():
                continue

            origins, directions = simple_model.cast_ray(pixels)
            d = (original[0, 2] - origins[0, 2]) / directions[0, 2]
            reconstructed = origins + d * directions

            assert torch.allclose(reconstructed, original, atol=1e-4), (
                f"Roundtrip failed at depth={depth}"
            )

    def test_multiple_xy_positions(self, simple_model: RefractiveProjectionModel):
        """Test roundtrip at multiple XY positions including off-center points."""
        xy_positions = [
            (0.635, 0.0),
            (0.835, 0.1),
            (0.435, -0.2),
            (0.635, 0.3),
        ]
        depth = simple_model.water_z + 0.3

        for x, y in xy_positions:
            original = torch.tensor([[x, y, depth]], dtype=torch.float32)
            pixels, valid = simple_model.project(original)

            if not valid[0].item():
                continue

            origins, directions = simple_model.cast_ray(pixels)
            d = (original[0, 2] - origins[0, 2]) / directions[0, 2]
            reconstructed = origins + d * directions

            assert torch.allclose(reconstructed, original, atol=1e-4), (
                f"Roundtrip failed at position=({x}, {y})"
            )
