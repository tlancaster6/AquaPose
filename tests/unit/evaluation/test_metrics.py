"""Unit tests for select_frames, compute_tier1, and compute_tier2 in metrics.py."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aquapose.core.types.reconstruction import Midline3D
from aquapose.evaluation.metrics import (
    Tier1Result,
    Tier2Result,
    compute_tier1,
    compute_tier2,
    select_frames,
)

# ---------------------------------------------------------------------------
# select_frames tests
# ---------------------------------------------------------------------------


def test_select_frames_returns_15_from_30() -> None:
    """select_frames with 30 available and n_frames=15 returns exactly 15."""
    frame_indices = tuple(range(10, 310, 10))  # 30 frames: 10, 20, ..., 300
    result = select_frames(frame_indices, n_frames=15)
    assert len(result) == 15


def test_select_frames_first_and_last_included() -> None:
    """First and last frame indices are included in the selection."""
    frame_indices = tuple(range(10, 310, 10))  # 30 frames
    result = select_frames(frame_indices, n_frames=15)
    assert result[0] == 10
    assert result[-1] == 300


def test_select_frames_fewer_than_requested_returns_all_with_warning() -> None:
    """When fixture has fewer than requested frames, return all with a warning."""
    frame_indices = (0, 10, 20, 30, 40)  # 5 frames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = select_frames(frame_indices, n_frames=15)
    assert len(result) == 5
    assert any("fewer" in str(warning.message).lower() for warning in w)


def test_select_frames_empty_tuple_returns_empty() -> None:
    """select_frames with empty tuple returns empty list."""
    result = select_frames((), n_frames=15)
    assert result == []


def test_select_frames_deterministic() -> None:
    """Same inputs always produce the same outputs."""
    frame_indices = tuple(range(0, 300, 1))  # 300 frames
    result1 = select_frames(frame_indices, n_frames=15)
    result2 = select_frames(frame_indices, n_frames=15)
    assert result1 == result2


def test_select_frames_exact_count_no_warning() -> None:
    """When count equals n_frames, return all without warning."""
    frame_indices = tuple(range(15))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = select_frames(frame_indices, n_frames=15)
    assert len(result) == 15
    assert not any("fewer" in str(warning.message).lower() for warning in w)


# ---------------------------------------------------------------------------
# Helper: build a synthetic Midline3D
# ---------------------------------------------------------------------------


def _make_midline3d(
    fish_id: int = 0,
    frame_index: int = 0,
    mean_residual: float = 1.0,
    max_residual: float = 2.0,
    per_camera_residuals: dict[str, float] | None = None,
    control_points: np.ndarray | None = None,
) -> Midline3D:
    if control_points is None:
        control_points = np.zeros((7, 3), dtype=np.float32)
    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=control_points,
        knots=np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], dtype=np.float32),
        degree=3,
        arc_length=0.2,
        half_widths=np.zeros(15, dtype=np.float32),
        n_cameras=3,
        mean_residual=mean_residual,
        max_residual=max_residual,
        per_camera_residuals=per_camera_residuals,
    )


# ---------------------------------------------------------------------------
# compute_tier1 tests
# ---------------------------------------------------------------------------


def test_compute_tier1_aggregates_per_camera() -> None:
    """compute_tier1 aggregates per-camera residuals correctly."""
    midline = _make_midline3d(
        fish_id=0,
        frame_index=0,
        mean_residual=2.0,
        max_residual=4.0,
        per_camera_residuals={"cam0": 1.0, "cam1": 3.0},
    )
    frame_results = [(0, {0: midline})]
    result = compute_tier1(frame_results)
    assert isinstance(result, Tier1Result)
    assert "cam0" in result.per_camera
    assert "cam1" in result.per_camera
    assert result.per_camera["cam0"]["mean_px"] == pytest.approx(1.0)
    assert result.per_camera["cam1"]["mean_px"] == pytest.approx(3.0)


def test_compute_tier1_per_fish_aggregates() -> None:
    """compute_tier1 aggregates per-fish mean/max residuals across frames."""
    m0 = _make_midline3d(fish_id=0, frame_index=0, mean_residual=2.0, max_residual=5.0)
    m1 = _make_midline3d(fish_id=0, frame_index=10, mean_residual=4.0, max_residual=8.0)
    frame_results = [(0, {0: m0}), (10, {0: m1})]
    result = compute_tier1(frame_results)
    assert 0 in result.per_fish
    assert result.per_fish[0]["mean_px"] == pytest.approx(3.0)
    assert result.per_fish[0]["max_px"] == pytest.approx(8.0)


def test_compute_tier1_handles_none_per_camera_residuals() -> None:
    """compute_tier1 gracefully skips Midline3D with per_camera_residuals=None."""
    midline = _make_midline3d(
        fish_id=0,
        frame_index=0,
        mean_residual=2.0,
        max_residual=4.0,
        per_camera_residuals=None,
    )
    frame_results = [(0, {0: midline})]
    result = compute_tier1(frame_results)
    # per_camera should be empty since no per_camera_residuals
    assert result.per_camera == {}
    # per_fish should still be populated
    assert 0 in result.per_fish


def test_compute_tier1_overall_aggregates() -> None:
    """compute_tier1 computes correct overall mean and max."""
    m0 = _make_midline3d(
        fish_id=0,
        frame_index=0,
        mean_residual=2.0,
        max_residual=3.0,
        per_camera_residuals={"cam0": 2.0},
    )
    m1 = _make_midline3d(
        fish_id=1,
        frame_index=0,
        mean_residual=4.0,
        max_residual=6.0,
        per_camera_residuals={"cam0": 4.0},
    )
    frame_results = [(0, {0: m0, 1: m1})]
    result = compute_tier1(frame_results)
    # Overall max should be 6.0 (max of max_residuals)
    assert result.overall_max_px == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# compute_tier2 tests
# ---------------------------------------------------------------------------


def test_compute_tier2_with_known_displacements() -> None:
    """compute_tier2 returns max displacement for each (fish, dropout_cam)."""
    tier2_data = {
        0: {
            "cam0": [1.0, 2.0, 3.0],  # max = 3.0
            "cam1": [0.5, 0.5],  # max = 0.5
        }
    }
    result = compute_tier2(tier2_data)
    assert isinstance(result, Tier2Result)
    assert result.per_fish_dropout[0]["cam0"] == pytest.approx(3.0)
    assert result.per_fish_dropout[0]["cam1"] == pytest.approx(0.5)


def test_compute_tier2_none_entries() -> None:
    """compute_tier2 returns None when all displacement values are None."""
    tier2_data = {
        0: {
            "cam0": [None, None],  # all None -> result is None
            "cam1": [None, 2.0],  # mixed -> max of non-None = 2.0
        }
    }
    result = compute_tier2(tier2_data)
    assert result.per_fish_dropout[0]["cam0"] is None
    assert result.per_fish_dropout[0]["cam1"] == pytest.approx(2.0)


def test_compute_tier2_empty_input() -> None:
    """compute_tier2 handles empty tier2_data."""
    result = compute_tier2({})
    assert result.per_fish_dropout == {}
