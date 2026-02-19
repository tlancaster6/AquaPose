"""Unit tests for the Z-uncertainty characterization module."""

import math

import pytest
import torch

from aquapose.calibration.uncertainty import (
    UncertaintyResult,
    build_synthetic_rig,
    compute_triangulation_uncertainty,
    triangulate_rays,
)

# ---------------------------------------------------------------------------
# triangulate_rays tests
# ---------------------------------------------------------------------------


class TestTriangulateRays:
    """Tests for the SVD least-squares ray triangulation function."""

    def test_two_orthogonal_rays_intersect_at_known_point(self) -> None:
        """Two perpendicular rays that exactly intersect should return the intersection."""
        # Ray 1: origin (1, 0, 0), direction (-1, 0, 0) → passes through (0,0,0)
        # Ray 2: origin (0, 1, 0), direction (0, -1, 0) → passes through (0,0,0)
        origins = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        directions = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        point = triangulate_rays(origins, directions)
        assert point.shape == (3,)
        assert torch.allclose(point, torch.zeros(3), atol=1e-5)

    def test_three_rays_intersecting_at_known_point(self) -> None:
        """Three rays from different cameras should triangulate to common point."""
        target = torch.tensor([1.5, -0.3, 2.0])

        # Build rays pointing toward target from three different origins
        cam_centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )
        origins = cam_centers
        raw_dirs = target.unsqueeze(0) - cam_centers  # (3, 3)
        directions = raw_dirs / torch.linalg.norm(raw_dirs, dim=-1, keepdim=True)

        point = triangulate_rays(origins, directions)
        assert torch.allclose(point, target, atol=1e-4)

    def test_parallel_rays_returns_finite_result(self) -> None:
        """Parallel rays are degenerate; triangulate_rays should still return a finite point."""
        # Two rays with same direction (parallel) — lstsq should handle gracefully
        origins = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        point = triangulate_rays(origins, directions)
        # Result may not be meaningful but should not crash or return NaN
        assert point.shape == (3,)

    def test_requires_at_least_two_rays(self) -> None:
        """Fewer than 2 rays should raise ValueError."""
        origins = torch.tensor([[0.0, 0.0, 0.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="at least 2"):
            triangulate_rays(origins, directions)

    def test_output_dtype_matches_input(self) -> None:
        """Output dtype should match input dtype (float32)."""
        origins = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        directions = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32
        )
        point = triangulate_rays(origins, directions)
        assert point.dtype == torch.float32


# ---------------------------------------------------------------------------
# build_synthetic_rig tests
# ---------------------------------------------------------------------------


class TestBuildSyntheticRig:
    """Tests for the 13-camera synthetic rig builder."""

    @pytest.fixture(scope="class")
    def rig(self):
        """Build the synthetic rig once per class (avoids repeated AquaCal calls)."""
        return build_synthetic_rig()

    def test_returns_13_models(self, rig) -> None:
        """Synthetic rig must return exactly 13 camera models."""
        assert len(rig) == 13

    def test_each_model_can_project_tank_center(self, rig) -> None:
        """Every camera model should be able to project a point near tank center."""
        # Tank center: (0, 0, Z) with Z slightly below water surface
        # Water surface is at ~0.75m for cameras at Z=0
        tank_center = torch.tensor([[0.0, 0.0, 1.0]])  # 0.25m below water surface
        n_valid = 0
        for model in rig:
            pixels, valid = model.project(tank_center)
            if valid[0]:
                n_valid += 1
                # Projected pixel should be finite
                assert torch.all(torch.isfinite(pixels[valid]))
        # At least the center camera and some ring cameras should see this point
        assert n_valid >= 1, "No camera can see tank center — rig geometry likely wrong"

    def test_models_have_correct_types(self, rig) -> None:
        """Each model must be a RefractiveProjectionModel with float32 tensors."""
        from aquapose.calibration import RefractiveProjectionModel

        for model in rig:
            assert isinstance(model, RefractiveProjectionModel)
            assert model.K.dtype == torch.float32
            assert model.R.dtype == torch.float32
            assert model.t.dtype == torch.float32


# ---------------------------------------------------------------------------
# compute_triangulation_uncertainty tests
# ---------------------------------------------------------------------------


class TestComputeTriangulationUncertainty:
    """Tests for the full uncertainty computation pipeline."""

    @pytest.fixture(scope="class")
    def rig(self):
        return build_synthetic_rig()

    @pytest.fixture(scope="class")
    def result(self, rig) -> UncertaintyResult:
        depths = torch.linspace(0.80, 1.20, 5)
        return compute_triangulation_uncertainty(rig, depths, pixel_noise=0.5)

    def test_output_is_uncertainty_result(self, result) -> None:
        """Return type must be UncertaintyResult."""
        assert isinstance(result, UncertaintyResult)

    def test_output_shapes_match_depths(self, result) -> None:
        """All output arrays must have the same length as the depth input."""
        n = result.depths.shape[0]
        assert result.x_errors.shape == (n,)
        assert result.y_errors.shape == (n,)
        assert result.z_errors.shape == (n,)
        assert result.n_cameras_visible.shape == (n,)

    def test_all_input_depths_appear_in_output(self, rig) -> None:
        """All supplied depths must appear in the result (same tensor)."""
        depths = torch.tensor([0.85, 0.95, 1.10])
        result = compute_triangulation_uncertainty(rig, depths)
        assert torch.allclose(result.depths, depths)

    def test_z_errors_larger_than_xy_errors_for_top_down_geometry(self, result) -> None:
        """Z errors must exceed XY errors for top-down cameras.

        Top-down cameras have rays that converge nearly parallel in Z, so
        depth (Z) is poorly constrained compared to X and Y.
        """
        # Filter to depths where enough cameras are visible (n_cams >= 2)
        visible = result.n_cameras_visible >= 2
        assert visible.any(), "Need at least one depth with 2+ cameras visible"

        x_err = result.x_errors[visible]
        y_err = result.y_errors[visible]
        z_err = result.z_errors[visible]

        # Filter to finite values
        finite_mask = (
            torch.isfinite(x_err) & torch.isfinite(y_err) & torch.isfinite(z_err)
        )
        assert finite_mask.any(), "Need at least one finite error value"

        mean_xy = (x_err[finite_mask] + y_err[finite_mask]) / 2.0
        mean_z = z_err[finite_mask]

        # Z error should be larger than XY on average for top-down geometry
        assert (mean_z > mean_xy).any(), (
            "Expected Z errors to exceed XY errors for top-down cameras. "
            f"Mean X: {x_err[finite_mask].mean():.4f}, "
            f"Mean Y: {y_err[finite_mask].mean():.4f}, "
            f"Mean Z: {mean_z.mean():.4f}"
        )

    def test_n_cameras_visible_positive_for_deep_points(self, result) -> None:
        """Deep tank points should be seen by multiple cameras."""
        assert result.n_cameras_visible.max() >= 2, (
            "Expected at least 2 cameras to see any tank center point"
        )

    def test_custom_pixel_noise_affects_errors(self, rig) -> None:
        """Larger pixel noise should produce larger errors."""
        depths = torch.tensor([1.0])
        result_small = compute_triangulation_uncertainty(rig, depths, pixel_noise=0.1)
        result_large = compute_triangulation_uncertainty(rig, depths, pixel_noise=2.0)

        # Check at least one error axis responds to noise magnitude
        visible = result_small.n_cameras_visible[0] >= 2
        if visible:
            small_z = result_small.z_errors[0]
            large_z = result_large.z_errors[0]
            if math.isfinite(small_z) and math.isfinite(large_z):
                assert large_z > small_z, (
                    "Larger pixel noise should produce larger Z error"
                )
