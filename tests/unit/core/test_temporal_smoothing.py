"""Unit tests for temporal smoothing of centroid z values."""

from __future__ import annotations

import numpy as np

from aquapose.core.reconstruction.temporal_smoothing import smooth_centroid_z


class TestSmoothCentroidZ:
    """Tests for smooth_centroid_z function."""

    def test_6_point_contiguous_array_shape(self) -> None:
        """smooth_centroid_z on a 6-element contiguous array returns same shape (T,)."""
        centroid_z = np.array([0.1, 0.15, 0.12, 0.18, 0.09, 0.14], dtype=np.float64)
        frame_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

        result = smooth_centroid_z(centroid_z, frame_indices, sigma_frames=2)

        assert result.shape == (6,), f"Expected shape (6,), got {result.shape}"
        assert result.dtype == np.float64

    def test_6_point_contiguous_array_values_smoothed(self) -> None:
        """smooth_centroid_z on 6-element array returns smoothed values."""
        # Use a step function — Gaussian filter should smooth the transition
        centroid_z = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        frame_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

        result = smooth_centroid_z(centroid_z, frame_indices, sigma_frames=1)

        # After smoothing, the step should be spread out — middle values should differ
        assert not np.allclose(result, centroid_z), (
            "Expected smoothing to change values for a step function"
        )
        # Output should be bounded between 0 and 1 for a step
        assert np.all(result >= 0.0 - 1e-6)
        assert np.all(result <= 1.0 + 1e-6)

    def test_6_point_nan_values_interpolated_and_restored(self) -> None:
        """smooth_centroid_z with NaN in 6-point input interpolates through NaN, restores NaN at original positions."""
        centroid_z = np.array([0.1, np.nan, 0.3, 0.4, np.nan, 0.6], dtype=np.float64)
        frame_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

        result = smooth_centroid_z(centroid_z, frame_indices, sigma_frames=1)

        assert result.shape == (6,)
        # NaN positions should remain NaN
        assert np.isnan(result[1]), "Position 1 (originally NaN) should remain NaN"
        assert np.isnan(result[4]), "Position 4 (originally NaN) should remain NaN"
        # Non-NaN positions should have finite values
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])
        assert np.isfinite(result[3])
        assert np.isfinite(result[5])

    def test_6_point_all_nan_returns_all_nan(self) -> None:
        """smooth_centroid_z with all-NaN 6-point input returns all-NaN array."""
        centroid_z = np.full(6, np.nan, dtype=np.float64)
        frame_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

        result = smooth_centroid_z(centroid_z, frame_indices, sigma_frames=2)

        assert result.shape == (6,)
        assert np.all(np.isnan(result))

    def test_6_point_with_gap_creates_two_segments(self) -> None:
        """smooth_centroid_z with a gap in frame indices treats them as separate segments."""
        # Gap between frame 2 and frame 10 — two segments
        centroid_z = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0], dtype=np.float64)
        frame_indices = np.array([0, 1, 2, 10, 11, 12], dtype=np.int64)

        result = smooth_centroid_z(centroid_z, frame_indices, sigma_frames=1)

        assert result.shape == (6,)
        # Both segments should be independently smoothed
        assert np.all(np.isfinite(result))
