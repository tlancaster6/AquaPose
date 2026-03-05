"""Unit tests for IRLS-weighted SVD plane fit and projection."""

from __future__ import annotations

import numpy as np

from aquapose.core.reconstruction.plane_fit import (
    fit_plane_weighted,
    project_onto_plane,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_coplanar_points(
    normal: np.ndarray,
    centroid: np.ndarray,
    n_points: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Generate random points lying on a plane defined by normal and centroid.

    Args:
        normal: Unit normal of the plane, shape (3,).
        centroid: A point on the plane, shape (3,).
        n_points: Number of points to generate.
        seed: Random seed.

    Returns:
        Points on the plane, shape (n_points, 3).
    """
    rng = np.random.default_rng(seed)
    normal = normal / np.linalg.norm(normal)

    # Build two orthogonal in-plane vectors
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Random in-plane coordinates
    coords = rng.standard_normal((n_points, 2)) * 0.05
    pts = centroid + coords[:, 0:1] * u + coords[:, 1:2] * v
    return pts


# ---------------------------------------------------------------------------
# Tests: fit_plane_weighted
# ---------------------------------------------------------------------------


class TestFitPlaneWeighted:
    """Tests for fit_plane_weighted."""

    def test_coplanar_uniform_weights(self) -> None:
        """Uniform weights on coplanar points return correct normal and zero residuals."""
        true_normal = np.array([0.0, 0.0, 1.0])
        centroid = np.array([0.1, 0.2, 0.3])
        pts = _make_coplanar_points(true_normal, centroid, n_points=10)
        weights = np.ones(10)

        normal, cent, is_deg = fit_plane_weighted(pts, weights)

        assert not is_deg
        # Normal should be parallel to [0,0,1] (up to sign)
        assert abs(abs(np.dot(normal, true_normal)) - 1.0) < 1e-6
        # Residuals should be zero
        _, residuals = project_onto_plane(pts, normal, cent)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_camera_count_weights_downweight_endpoints(self) -> None:
        """Camera-count weights reduce influence of low-camera endpoints."""
        true_normal = np.array([0.0, 0.0, 1.0])
        centroid = np.array([0.0, 0.0, 0.5])
        pts = _make_coplanar_points(true_normal, centroid, n_points=8, seed=10)

        # Add one outlier point that is off-plane
        outlier = centroid + np.array([0.03, 0.0, 0.05])
        pts = np.vstack([pts, outlier])

        # Uniform weights: outlier has full influence
        uniform_w = np.ones(9)
        normal_uniform, _, _ = fit_plane_weighted(pts, uniform_w)

        # Down-weight the outlier (last point)
        weighted_w = np.ones(9)
        weighted_w[-1] = 0.01
        normal_weighted, _, _ = fit_plane_weighted(pts, weighted_w)

        # Weighted normal should be closer to true normal
        err_uniform = abs(abs(np.dot(normal_uniform, true_normal)) - 1.0)
        err_weighted = abs(abs(np.dot(normal_weighted, true_normal)) - 1.0)
        assert err_weighted < err_uniform

    def test_collinear_points_degenerate(self) -> None:
        """Collinear points are detected as degenerate with default normal."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        weights = np.ones(3)

        normal, _cent, is_deg = fit_plane_weighted(pts, weights)

        assert is_deg
        np.testing.assert_array_equal(normal, [0.0, 0.0, 1.0])

    def test_coincident_points_degenerate(self) -> None:
        """Coincident (identical) points are detected as degenerate."""
        pts = np.tile([1.0, 2.0, 3.0], (5, 1))
        weights = np.ones(5)

        normal, _cent, is_deg = fit_plane_weighted(pts, weights)

        assert is_deg
        np.testing.assert_array_equal(normal, [0.0, 0.0, 1.0])

    def test_tilted_plane(self) -> None:
        """Tilted plane (non-axis-aligned) is correctly recovered."""
        true_normal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        centroid = np.array([0.0, 0.0, 0.5])
        pts = _make_coplanar_points(true_normal, centroid, n_points=15, seed=99)
        weights = np.ones(15)

        normal, _cent, is_deg = fit_plane_weighted(pts, weights)

        assert not is_deg
        # Normal should be parallel to true_normal (up to sign)
        assert abs(abs(np.dot(normal, true_normal)) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests: project_onto_plane
# ---------------------------------------------------------------------------


class TestProjectOntoPlane:
    """Tests for project_onto_plane."""

    def test_returns_signed_residuals(self) -> None:
        """Projection returns correct signed off-plane residuals."""
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.array([0.0, 0.0, 0.5])
        pts = np.array(
            [
                [1.0, 0.0, 0.5],  # on plane -> residual 0
                [1.0, 0.0, 0.6],  # above -> residual +0.1
                [1.0, 0.0, 0.4],  # below -> residual -0.1
            ]
        )

        projected, residuals = project_onto_plane(pts, normal, centroid)

        np.testing.assert_allclose(residuals, [0.0, 0.1, -0.1], atol=1e-10)
        # All projected points should have z = 0.5 (the plane)
        np.testing.assert_allclose(projected[:, 2], 0.5, atol=1e-10)
        # x, y should be unchanged
        np.testing.assert_allclose(projected[:, 0], pts[:, 0], atol=1e-10)
        np.testing.assert_allclose(projected[:, 1], pts[:, 1], atol=1e-10)

    def test_preserves_centroid(self) -> None:
        """Mean of projected points matches centroid projected onto plane."""
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.array([0.1, 0.2, 0.3])
        rng = np.random.default_rng(42)
        pts = centroid + rng.standard_normal((20, 3)) * 0.05

        projected, _ = project_onto_plane(pts, normal, centroid)

        # The centroid of the projected points should have z = centroid's z
        # (since all projected z values are centroid[2])
        np.testing.assert_allclose(projected[:, 2].mean(), centroid[2], atol=1e-10)

    def test_already_on_plane(self) -> None:
        """Points already on the plane are unchanged."""
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.array([0.0, 0.0, 0.5])
        pts = np.array(
            [
                [0.1, 0.2, 0.5],
                [0.3, 0.4, 0.5],
            ]
        )

        projected, residuals = project_onto_plane(pts, normal, centroid)

        np.testing.assert_allclose(projected, pts, atol=1e-10)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Midline3D new fields
# ---------------------------------------------------------------------------


class TestMidline3DPlaneFields:
    """Verify Midline3D plane metadata fields have correct defaults."""

    def test_defaults(self) -> None:
        """New fields default to None/False."""
        from aquapose.core.types.reconstruction import Midline3D

        m = Midline3D(
            fish_id=0,
            frame_index=0,
            control_points=np.zeros((7, 3), dtype=np.float32),
            knots=np.zeros(11, dtype=np.float32),
            degree=3,
            arc_length=0.3,
            half_widths=np.zeros(15, dtype=np.float32),
            n_cameras=3,
            mean_residual=1.0,
            max_residual=2.0,
        )

        assert m.plane_normal is None
        assert m.plane_centroid is None
        assert m.off_plane_residuals is None
        assert m.is_degenerate_plane is False


# ---------------------------------------------------------------------------
# Tests: PlaneProjectionConfig
# ---------------------------------------------------------------------------


class TestPlaneProjectionConfig:
    """Verify PlaneProjectionConfig defaults and presence in ReconstructionConfig."""

    def test_default_enabled(self) -> None:
        """PlaneProjectionConfig.enabled defaults to True."""
        from aquapose.engine.config import PlaneProjectionConfig

        cfg = PlaneProjectionConfig()
        assert cfg.enabled is True

    def test_in_reconstruction_config(self) -> None:
        """ReconstructionConfig has plane_projection field."""
        from aquapose.engine.config import ReconstructionConfig

        cfg = ReconstructionConfig()
        assert cfg.plane_projection.enabled is True

    def test_disabled(self) -> None:
        """PlaneProjectionConfig can be constructed with enabled=False."""
        from aquapose.engine.config import PlaneProjectionConfig, ReconstructionConfig

        pp = PlaneProjectionConfig(enabled=False)
        assert pp.enabled is False
        rc = ReconstructionConfig(plane_projection=pp)
        assert rc.plane_projection.enabled is False
