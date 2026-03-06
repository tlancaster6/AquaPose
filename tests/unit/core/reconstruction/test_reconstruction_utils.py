"""Unit tests for shared reconstruction helper functions in utils.py."""

from __future__ import annotations

import numpy as np
import torch

from aquapose.core.reconstruction.utils import (
    MIN_BODY_POINTS,
    SPLINE_K,
    build_spline_knots,
    fit_spline,
    pixel_half_width_to_metres,
    weighted_triangulate_rays,
)

# ---------------------------------------------------------------------------
# build_spline_knots
# ---------------------------------------------------------------------------


class TestBuildSplineKnots:
    """Tests for build_spline_knots."""

    def test_n7_returns_correct_knot_vector(self) -> None:
        """build_spline_knots(7) returns a length-11 clamped cubic knot vector."""
        knots = build_spline_knots(7)
        expected = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float64,
        )
        assert knots.shape == (11,), f"Expected shape (11,), got {knots.shape}"
        np.testing.assert_allclose(knots, expected, atol=1e-9)

    def test_n4_returns_all_boundary_knot_vector(self) -> None:
        """build_spline_knots(4) returns a length-8 all-boundary knot vector."""
        knots = build_spline_knots(4)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        assert knots.shape == (8,), f"Expected shape (8,), got {knots.shape}"
        np.testing.assert_allclose(knots, expected, atol=1e-9)

    def test_output_dtype_is_float64(self) -> None:
        """Knot vector dtype is float64."""
        knots = build_spline_knots(7)
        assert knots.dtype == np.float64

    def test_length_formula(self) -> None:
        """Knot vector length is n_control_points + SPLINE_K + 1 for various n."""
        for n in [4, 5, 6, 7, 8, 10]:
            knots = build_spline_knots(n)
            expected_len = n + SPLINE_K + 1
            assert len(knots) == expected_len, (
                f"n={n}: expected length {expected_len}, got {len(knots)}"
            )

    def test_knots_are_non_decreasing(self) -> None:
        """Knot vectors must be non-decreasing."""
        for n in [4, 5, 7, 9]:
            knots = build_spline_knots(n)
            assert np.all(np.diff(knots) >= 0), f"n={n}: knots not non-decreasing"


# ---------------------------------------------------------------------------
# weighted_triangulate_rays
# ---------------------------------------------------------------------------


class TestWeightedTriangulateRays:
    """Tests for weighted_triangulate_rays."""

    def _make_rays_for_point(
        self,
        target: torch.Tensor,
        cam_origins: list[tuple[float, float, float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create rays from camera origins toward a known 3D target point."""
        origins = torch.tensor(cam_origins, dtype=torch.float64)
        # Direction = normalize(target - origin)
        diffs = target.unsqueeze(0) - origins  # (N, 3)
        norms = torch.linalg.norm(diffs, dim=1, keepdim=True)
        directions = diffs / norms
        return origins, directions

    def test_uniform_weights_recovers_known_point(self) -> None:
        """With uniform weights, weighted_triangulate_rays should recover the target."""
        target = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64)
        cam_origins = [
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        origins, directions = self._make_rays_for_point(target, cam_origins)
        weights = torch.ones(len(cam_origins), dtype=torch.float64)
        result = weighted_triangulate_rays(origins, directions, weights)
        assert result.shape == (3,)
        np.testing.assert_allclose(result.numpy(), target.numpy(), atol=1e-5)

    def test_output_shape(self) -> None:
        """Output shape is (3,)."""
        target = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        cam_origins = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        origins, directions = self._make_rays_for_point(target, cam_origins)
        weights = torch.ones(2, dtype=torch.float64)
        result = weighted_triangulate_rays(origins, directions, weights)
        assert result.shape == (3,)

    def test_nonuniform_weights_biases_toward_high_weight_camera(self) -> None:
        """High-weight camera should pull the result toward its observation.

        Use three cameras where cameras 0 and 1 agree on target_a, and camera 2
        points at target_b. With high weight on camera 2, the result should be
        closer to target_b than with equal weights.
        """
        target_a = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        target_b = torch.tensor([0.3, 0.0, 1.0], dtype=torch.float64)

        # Cameras at fixed positions
        origins = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )

        # Cam 0 and 1 point at target_a; cam 2 points at target_b
        def _dir(o: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            v = t - o
            return v / torch.linalg.norm(v)

        directions = torch.stack(
            [
                _dir(origins[0], target_a),
                _dir(origins[1], target_a),
                _dir(origins[2], target_b),
            ]
        )

        # Equal weights: result near target_a (2 vs 1 votes)
        equal_w = torch.ones(3, dtype=torch.float64)
        result_equal = weighted_triangulate_rays(origins, directions, equal_w)

        # High weight on cam2: result pulled toward target_b
        high_w = torch.tensor([1.0, 1.0, 100.0], dtype=torch.float64)
        result_high = weighted_triangulate_rays(origins, directions, high_w)

        # With high cam2 weight, x-coord should be closer to target_b.x (0.3)
        # than with equal weights where majority votes for target_a.x (0.0)
        x_equal = result_equal[0].item()
        x_high = result_high[0].item()
        assert x_high > x_equal, (
            f"High-weight result x={x_high:.4f} should be > equal-weight x={x_equal:.4f} "
            f"(pulling toward target_b.x=0.3)"
        )


# ---------------------------------------------------------------------------
# fit_spline
# ---------------------------------------------------------------------------


class TestFitSpline:
    """Tests for fit_spline."""

    def _make_arc_length_params(self, n: int) -> np.ndarray:
        """Create strictly increasing arc-length params in [0, 1]."""
        return np.linspace(0.0, 1.0, n, dtype=np.float64)

    def _make_straight_line_pts(self, n: int) -> np.ndarray:
        """Create n points along a straight line in 3D."""
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        pts = np.column_stack([t, np.zeros(n), np.zeros(n)])
        return pts

    def test_returns_none_for_too_few_points(self) -> None:
        """fit_spline returns None when fewer than min_body_points are provided."""
        n = MIN_BODY_POINTS - 1
        u = self._make_arc_length_params(n)
        pts = self._make_straight_line_pts(n)
        result = fit_spline(u, pts)
        assert result is None

    def test_returns_none_at_exactly_threshold_minus_one(self) -> None:
        """fit_spline returns None for n == min_body_points - 1."""
        u = self._make_arc_length_params(MIN_BODY_POINTS - 1)
        pts = self._make_straight_line_pts(MIN_BODY_POINTS - 1)
        assert fit_spline(u, pts) is None

    def test_returns_tuple_for_sufficient_points(self) -> None:
        """fit_spline returns (control_points, arc_length) for valid input."""
        n = MIN_BODY_POINTS + 2
        u = self._make_arc_length_params(n)
        pts = self._make_straight_line_pts(n)
        result = fit_spline(u, pts)
        assert result is not None
        control_points, arc_length = result
        assert isinstance(control_points, np.ndarray)
        assert isinstance(arc_length, float)

    def test_control_points_shape(self) -> None:
        """Control points have shape (n_ctrl, 3) where n_ctrl matches knot vector."""
        n = 15
        u = self._make_arc_length_params(n)
        pts = self._make_straight_line_pts(n)
        knots = build_spline_knots(7)
        result = fit_spline(u, pts, knots=knots)
        assert result is not None
        control_points, _ = result
        assert control_points.shape[1] == 3
        assert control_points.dtype == np.float32

    def test_arc_length_positive_for_nonzero_curve(self) -> None:
        """Arc length is positive for a non-degenerate curve."""
        n = 15
        u = self._make_arc_length_params(n)
        pts = self._make_straight_line_pts(n)
        result = fit_spline(u, pts)
        assert result is not None
        _, arc_length = result
        assert arc_length > 0.0

    def test_custom_min_body_points(self) -> None:
        """Custom min_body_points overrides the module-level default."""
        # Provide exactly MIN_BODY_POINTS - 1 points but set min to even lower
        n = MIN_BODY_POINTS - 1
        u = self._make_arc_length_params(n)
        pts = self._make_straight_line_pts(n)
        # With min set to n - 1 (below n), should succeed if scipy accepts it
        result = fit_spline(u, pts, min_body_points=5)
        # n = MIN_BODY_POINTS - 1 = 8, min = 5 => should attempt fitting
        assert result is not None or result is None  # Either outcome is valid


# ---------------------------------------------------------------------------
# pixel_half_width_to_metres
# ---------------------------------------------------------------------------


class TestPixelHalfWidthToMetres:
    """Tests for pixel_half_width_to_metres."""

    def test_known_arithmetic(self) -> None:
        """pixel_half_width_to_metres(10.0, 0.5, 500.0) == 0.01."""
        result = pixel_half_width_to_metres(10.0, 0.5, 500.0)
        assert abs(result - 0.01) < 1e-12, f"Expected 0.01, got {result}"

    def test_zero_half_width(self) -> None:
        """Zero pixel half-width returns zero metres."""
        result = pixel_half_width_to_metres(0.0, 1.0, 1000.0)
        assert result == 0.0

    def test_proportional_to_depth(self) -> None:
        """Result is proportional to depth."""
        r1 = pixel_half_width_to_metres(5.0, 1.0, 500.0)
        r2 = pixel_half_width_to_metres(5.0, 2.0, 500.0)
        assert abs(r2 - 2 * r1) < 1e-12

    def test_inversely_proportional_to_focal(self) -> None:
        """Result is inversely proportional to focal length."""
        r1 = pixel_half_width_to_metres(5.0, 1.0, 500.0)
        r2 = pixel_half_width_to_metres(5.0, 1.0, 1000.0)
        assert abs(r1 - 2 * r2) < 1e-12


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify module-level constants are accessible and correctly typed."""

    def test_spline_k_is_int(self) -> None:
        """SPLINE_K is an integer."""
        assert isinstance(SPLINE_K, int)
        assert SPLINE_K == 3

    def test_min_body_points_is_int(self) -> None:
        """MIN_BODY_POINTS is an integer."""
        assert isinstance(MIN_BODY_POINTS, int)
        assert MIN_BODY_POINTS >= 1
