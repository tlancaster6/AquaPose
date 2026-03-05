"""Unit tests for frame selection functions."""

from __future__ import annotations

import numpy as np
import pytest
from aquapose.training.frame_selection import (
    compute_curvature,
    diversity_sample,
    filter_empty_frames,
    temporal_subsample,
)

from aquapose.core.types.reconstruction import Midline3D


def _make_midline3d(
    fish_id: int,
    frame_index: int,
    control_points: np.ndarray | None = None,
) -> Midline3D:
    """Create a minimal Midline3D for testing."""
    if control_points is None:
        # Straight line along x-axis
        control_points = np.column_stack(
            [np.linspace(0, 1, 7), np.zeros(7), np.zeros(7)]
        ).astype(np.float32)
    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=control_points,
        knots=np.linspace(0, 1, 11).astype(np.float32),
        degree=3,
        arc_length=1.0,
        half_widths=np.ones(13, dtype=np.float32) * 0.01,
        n_cameras=5,
        mean_residual=2.0,
        max_residual=5.0,
    )


def _curved_control_points(curvature_scale: float = 1.0) -> np.ndarray:
    """Control points forming an arc in the XY plane."""
    t = np.linspace(0, np.pi * curvature_scale, 7)
    return np.column_stack([np.cos(t), np.sin(t), np.zeros(7)]).astype(np.float32)


# --- temporal_subsample ---


class TestTemporalSubsample:
    """Tests for temporal_subsample."""

    def test_every_kth_frame(self) -> None:
        result = temporal_subsample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], step=3)
        assert result == [0, 3, 6, 9]

    def test_step_one_returns_all(self) -> None:
        frames = [5, 10, 15, 20]
        result = temporal_subsample(frames, step=1)
        assert result == [5, 10, 15, 20]

    def test_sorts_input(self) -> None:
        result = temporal_subsample([9, 3, 0, 6], step=3)
        assert result == [0, 3, 6, 9]

    def test_empty_input(self) -> None:
        assert temporal_subsample([], step=5) == []


# --- filter_empty_frames ---


class TestFilterEmptyFrames:
    """Tests for filter_empty_frames."""

    def test_removes_empty_frames(self) -> None:
        midlines_3d: list[dict[int, Midline3D]] = [
            {0: _make_midline3d(0, 0)},  # frame 0: has fish
            {},  # frame 1: empty
            {0: _make_midline3d(0, 2)},  # frame 2: has fish
        ]
        result = filter_empty_frames([0, 1, 2], midlines_3d)
        assert result == [0, 2]

    def test_all_empty_returns_empty(self) -> None:
        midlines_3d: list[dict[int, Midline3D]] = [{}, {}, {}]
        result = filter_empty_frames([0, 1, 2], midlines_3d)
        assert result == []

    def test_out_of_range_index_removed(self) -> None:
        midlines_3d: list[dict[int, Midline3D]] = [
            {0: _make_midline3d(0, 0)},
        ]
        result = filter_empty_frames([0, 5], midlines_3d)
        assert result == [0]


# --- compute_curvature ---


class TestComputeCurvature:
    """Tests for compute_curvature."""

    def test_straight_line_near_zero(self) -> None:
        # Straight line along x-axis
        cp = np.column_stack([np.linspace(0, 1, 7), np.zeros(7), np.zeros(7)]).astype(
            np.float32
        )
        curv = compute_curvature(cp)
        assert curv == pytest.approx(0.0, abs=1e-6)

    def test_curved_line_positive(self) -> None:
        cp = _curved_control_points(curvature_scale=0.5)
        curv = compute_curvature(cp)
        assert curv > 0.01

    def test_more_curved_is_larger(self) -> None:
        cp_mild = _curved_control_points(curvature_scale=0.3)
        cp_strong = _curved_control_points(curvature_scale=1.0)
        assert compute_curvature(cp_strong) > compute_curvature(cp_mild)


# --- diversity_sample ---


class TestDiversitySample:
    """Tests for diversity_sample."""

    def test_max_per_bin_limits_output(self) -> None:
        # Build midlines with varying curvatures across frames
        midlines_3d: list[dict[int, Midline3D]] = []
        for i in range(20):
            scale = 0.1 + (i / 20) * 0.9  # varying curvature
            cp = _curved_control_points(curvature_scale=scale)
            midlines_3d.append({0: _make_midline3d(0, i, control_points=cp)})

        frame_indices = list(range(20))
        result = diversity_sample(
            midlines_3d, frame_indices, n_bins=5, max_per_bin=1, seed=42
        )
        # At most 5 frames (1 per bin, 1 fish)
        assert len(result) <= 5
        # All returned indices are valid
        assert all(idx in frame_indices for idx in result)

    def test_max_per_bin_none_preserves_all(self) -> None:
        midlines_3d: list[dict[int, Midline3D]] = []
        for i in range(10):
            cp = _curved_control_points(curvature_scale=0.1 + i * 0.1)
            midlines_3d.append({0: _make_midline3d(0, i, control_points=cp)})

        frame_indices = list(range(10))
        result = diversity_sample(
            midlines_3d, frame_indices, n_bins=3, max_per_bin=None, seed=42
        )
        assert sorted(result) == frame_indices

    def test_returns_sorted_indices(self) -> None:
        midlines_3d: list[dict[int, Midline3D]] = []
        for i in range(10):
            cp = _curved_control_points(curvature_scale=0.1 + i * 0.1)
            midlines_3d.append({0: _make_midline3d(0, i, control_points=cp)})

        result = diversity_sample(
            midlines_3d, list(range(10)), n_bins=3, max_per_bin=2, seed=42
        )
        assert result == sorted(result)

    def test_deterministic_with_seed(self) -> None:
        midlines_3d: list[dict[int, Midline3D]] = []
        for i in range(15):
            cp = _curved_control_points(curvature_scale=0.1 + i * 0.05)
            midlines_3d.append({0: _make_midline3d(0, i, control_points=cp)})

        frames = list(range(15))
        r1 = diversity_sample(midlines_3d, frames, n_bins=3, max_per_bin=2, seed=42)
        r2 = diversity_sample(midlines_3d, frames, n_bins=3, max_per_bin=2, seed=42)
        assert r1 == r2
