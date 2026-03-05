"""Unit tests for temporal smoothing of plane normals and control point rotation."""

from __future__ import annotations

import numpy as np

from aquapose.core.reconstruction.temporal_smoothing import (
    rotate_control_points_to_plane,
    smooth_plane_normals,
)

# ---------------------------------------------------------------------------
# Tests: smooth_plane_normals
# ---------------------------------------------------------------------------


class TestSmoothPlaneNormals:
    """Tests for smooth_plane_normals."""

    def test_constant_normals_unchanged(self) -> None:
        """Constant normals are returned unchanged."""
        T = 20
        normals = np.tile([0.0, 0.0, 1.0], (T, 1))
        is_degen = np.zeros(T, dtype=bool)
        fish_ids = np.zeros(T, dtype=int)
        frames = np.arange(T)

        result = smooth_plane_normals(
            normals, is_degen, fish_ids, frames, sigma_frames=3
        )

        np.testing.assert_allclose(result, normals, atol=1e-10)

    def test_noisy_normals_reduced_variation(self) -> None:
        """Smoothing reduces frame-to-frame angular variation."""
        T = 50
        rng = np.random.default_rng(42)
        base_normal = np.array([0.0, 0.0, 1.0])
        # Small random perturbations
        noise = rng.standard_normal((T, 3)) * 0.05
        normals = base_normal + noise
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        is_degen = np.zeros(T, dtype=bool)
        fish_ids = np.zeros(T, dtype=int)
        frames = np.arange(T)

        smoothed = smooth_plane_normals(
            normals, is_degen, fish_ids, frames, sigma_frames=3
        )

        # Compute angular differences between consecutive frames
        def angular_diffs(n: np.ndarray) -> np.ndarray:
            dots = np.sum(n[:-1] * n[1:], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            return np.arccos(dots)

        raw_diffs = angular_diffs(normals)
        smooth_diffs = angular_diffs(smoothed)

        assert smooth_diffs.mean() < raw_diffs.mean()

    def test_sign_flips_resolved(self) -> None:
        """Sign-flipped normals produce consistent output."""
        T = 10
        normals = np.tile([0.0, 0.0, 1.0], (T, 1))
        # Flip some signs
        normals[3] = [0.0, 0.0, -1.0]
        normals[7] = [0.0, 0.0, -1.0]

        is_degen = np.zeros(T, dtype=bool)
        fish_ids = np.zeros(T, dtype=int)
        frames = np.arange(T)

        result = smooth_plane_normals(
            normals, is_degen, fish_ids, frames, sigma_frames=1
        )

        # All normals should point in the same direction after smoothing
        dots = np.sum(result[:-1] * result[1:], axis=1)
        assert np.all(dots > 0.9)

    def test_gap_creates_independent_segments(self) -> None:
        """Frame gap > 1 creates independent segments."""
        # Segment 1: frames 0-4, normal [0,0,1]
        # Segment 2: frames 10-14, normal [0,1,0] (different direction)
        T = 10
        normals = np.zeros((T, 3))
        normals[:5] = [0.0, 0.0, 1.0]
        normals[5:] = [0.0, 1.0, 0.0]

        is_degen = np.zeros(T, dtype=bool)
        fish_ids = np.zeros(T, dtype=int)
        frames = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])

        result = smooth_plane_normals(
            normals, is_degen, fish_ids, frames, sigma_frames=3
        )

        # Segment 1 should still point mostly in z direction
        assert np.allclose(result[2], [0, 0, 1], atol=0.01)
        # Segment 2 should still point mostly in y direction
        assert np.allclose(result[7], [0, 1, 0], atol=0.01)

    def test_degenerate_frames_interpolated(self) -> None:
        """Degenerate frames are interpolated through."""
        T = 10
        normals = np.tile([0.0, 0.0, 1.0], (T, 1))
        is_degen = np.zeros(T, dtype=bool)
        is_degen[4] = True
        is_degen[5] = True
        # Set degenerate normals to junk (should be overwritten)
        normals[4] = [1.0, 0.0, 0.0]
        normals[5] = [1.0, 0.0, 0.0]

        fish_ids = np.zeros(T, dtype=int)
        frames = np.arange(T)

        result = smooth_plane_normals(
            normals, is_degen, fish_ids, frames, sigma_frames=1
        )

        # Degenerate frames should have been interpolated and smoothed
        # to approximately [0, 0, 1]
        assert abs(result[4, 2]) > 0.9
        assert abs(result[5, 2]) > 0.9

    def test_empty_input(self) -> None:
        """Empty input returns empty array."""
        normals = np.zeros((0, 3))
        is_degen = np.zeros(0, dtype=bool)
        fish_ids = np.zeros(0, dtype=int)
        frames = np.zeros(0, dtype=int)

        result = smooth_plane_normals(normals, is_degen, fish_ids, frames)
        assert result.shape == (0, 3)

    def test_single_frame(self) -> None:
        """Single frame returns the same normal."""
        normals = np.array([[0.0, 0.0, 1.0]])
        is_degen = np.array([False])
        fish_ids = np.array([0])
        frames = np.array([0])

        result = smooth_plane_normals(normals, is_degen, fish_ids, frames)
        np.testing.assert_allclose(result, normals, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: rotate_control_points_to_plane
# ---------------------------------------------------------------------------


class TestRotateControlPoints:
    """Tests for rotate_control_points_to_plane."""

    def test_identity_rotation(self) -> None:
        """Same raw and smoothed normal returns original control points."""
        rng = np.random.default_rng(42)
        cp = rng.random((7, 3))
        centroid = np.array([0.5, 0.5, 0.5])
        normal = np.array([0.0, 0.0, 1.0])

        result = rotate_control_points_to_plane(cp, centroid, normal, normal)

        np.testing.assert_allclose(result, cp, atol=1e-10)

    def test_90_degree_rotation(self) -> None:
        """90-degree rotation produces geometrically correct result."""
        # Single point offset from centroid in z direction
        cp = np.array([[0.0, 0.0, 1.0]] * 7)
        centroid = np.array([0.0, 0.0, 0.0])
        raw_normal = np.array([0.0, 0.0, 1.0])
        smoothed_normal = np.array([1.0, 0.0, 0.0])

        result = rotate_control_points_to_plane(
            cp, centroid, raw_normal, smoothed_normal
        )

        # After rotating z-axis to x-axis, point at (0,0,1) relative to
        # centroid should move. The rotation axis is cross(z, x) = -y,
        # so the rotation is 90 degrees around -y axis.
        # (0,0,1) rotated 90 degrees around -y gives (1,0,0)
        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0], atol=1e-10)

    def test_opposite_normals(self) -> None:
        """180-degree rotation (opposite normals) handled correctly."""
        cp = np.array([[0.0, 0.0, 1.0]] * 7)
        centroid = np.array([0.0, 0.0, 0.0])
        raw_normal = np.array([0.0, 0.0, 1.0])
        smoothed_normal = np.array([0.0, 0.0, -1.0])

        result = rotate_control_points_to_plane(
            cp, centroid, raw_normal, smoothed_normal
        )

        # 180-degree rotation should flip z
        np.testing.assert_allclose(result[0, 2], -1.0, atol=1e-10)

    def test_centroid_invariant(self) -> None:
        """Rotation is relative to centroid (centroid stays fixed)."""
        rng = np.random.default_rng(99)
        cp = rng.random((7, 3))
        centroid = np.array([1.0, 2.0, 3.0])
        raw_normal = np.array([0.0, 0.0, 1.0])
        smoothed_normal = np.array([0.0, 1.0, 0.0])

        result = rotate_control_points_to_plane(
            cp, centroid, raw_normal, smoothed_normal
        )

        # Distances from centroid should be preserved (rotation is isometric)
        raw_dists = np.linalg.norm(cp - centroid, axis=1)
        rot_dists = np.linalg.norm(result - centroid, axis=1)
        np.testing.assert_allclose(rot_dists, raw_dists, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: PlaneSmoothingConfig
# ---------------------------------------------------------------------------


class TestPlaneSmoothingConfig:
    """Verify PlaneSmoothingConfig defaults."""

    def test_defaults(self) -> None:
        """PlaneSmoothingConfig has expected defaults."""
        from aquapose.engine.config import PlaneSmoothingConfig

        cfg = PlaneSmoothingConfig()
        assert cfg.enabled is True
        assert cfg.sigma_frames == 3

    def test_custom_sigma(self) -> None:
        """PlaneSmoothingConfig accepts custom sigma_frames."""
        from aquapose.engine.config import PlaneSmoothingConfig

        cfg = PlaneSmoothingConfig(sigma_frames=5)
        assert cfg.sigma_frames == 5
