"""Unit tests for TrackingMetrics and evaluate_tracking in stages/tracking.py."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aquapose.core.tracking.types import Tracklet2D
from aquapose.evaluation.stages.tracking import TrackingMetrics, evaluate_tracking

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracklet(
    camera_id: str = "cam0",
    track_id: int = 0,
    n_frames: int = 5,
    n_coasted: int = 0,
    start_frame: int = 0,
) -> Tracklet2D:
    """Create a synthetic Tracklet2D.

    Args:
        camera_id: Camera identifier.
        track_id: Track identifier.
        n_frames: Total number of frames in the tracklet.
        n_coasted: Number of frames with "coasted" status (first n_coasted frames).
        start_frame: Starting frame index.

    Returns:
        A synthetic Tracklet2D with the specified parameters.
    """
    frames = tuple(range(start_frame, start_frame + n_frames))
    centroids = tuple((float(i), float(i)) for i in range(n_frames))
    bboxes = tuple((0.0, 0.0, 10.0, 10.0) for _ in range(n_frames))
    statuses = tuple(
        "coasted" if i < n_coasted else "detected" for i in range(n_frames)
    )
    return Tracklet2D(
        camera_id=camera_id,
        track_id=track_id,
        frames=frames,
        centroids=centroids,
        bboxes=bboxes,
        frame_status=statuses,
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_evaluate_tracking_empty_returns_zeroes() -> None:
    """evaluate_tracking([]) returns TrackingMetrics with all zeroes."""
    result = evaluate_tracking([])
    assert isinstance(result, TrackingMetrics)
    assert result.track_count == 0
    assert result.length_median == 0.0
    assert result.length_mean == 0.0
    assert result.length_min == 0
    assert result.length_max == 0
    assert result.coast_frequency == 0.0
    assert result.detection_coverage == 0.0


# ---------------------------------------------------------------------------
# Track count and length stats
# ---------------------------------------------------------------------------


def test_evaluate_tracking_track_count() -> None:
    """evaluate_tracking returns correct track_count."""
    tracklets = [
        _make_tracklet(track_id=0, n_frames=10),
        _make_tracklet(track_id=1, n_frames=5),
        _make_tracklet(track_id=2, n_frames=15),
    ]
    result = evaluate_tracking(tracklets)
    assert result.track_count == 3


def test_evaluate_tracking_length_stats() -> None:
    """evaluate_tracking returns correct length statistics for known tracklets."""
    # Lengths: 10, 5, 15 → median=10.0, mean=10.0, min=5, max=15
    tracklets = [
        _make_tracklet(track_id=0, n_frames=10),
        _make_tracklet(track_id=1, n_frames=5),
        _make_tracklet(track_id=2, n_frames=15),
    ]
    result = evaluate_tracking(tracklets)
    assert result.length_median == pytest.approx(10.0)
    assert result.length_mean == pytest.approx(10.0)
    assert result.length_min == 5
    assert result.length_max == 15


# ---------------------------------------------------------------------------
# Coast frequency
# ---------------------------------------------------------------------------


def test_evaluate_tracking_coast_frequency() -> None:
    """Tracklet with 2/4 coasted frames contributes 0.5 coast_frequency."""
    # 1 tracklet, 4 frames, 2 coasted → 2/4 = 0.5
    tracklets = [_make_tracklet(n_frames=4, n_coasted=2)]
    result = evaluate_tracking(tracklets)
    assert result.coast_frequency == pytest.approx(0.5)


def test_evaluate_tracking_coast_frequency_none_coasted() -> None:
    """No coasted frames → coast_frequency == 0.0."""
    tracklets = [_make_tracklet(n_frames=5, n_coasted=0)]
    result = evaluate_tracking(tracklets)
    assert result.coast_frequency == pytest.approx(0.0)


def test_evaluate_tracking_coast_frequency_all_coasted() -> None:
    """All coasted frames → coast_frequency == 1.0."""
    tracklets = [_make_tracklet(n_frames=5, n_coasted=5)]
    result = evaluate_tracking(tracklets)
    assert result.coast_frequency == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Detection coverage
# ---------------------------------------------------------------------------


def test_evaluate_tracking_detection_coverage() -> None:
    """detection_coverage is 1.0 - coast_frequency."""
    # 10 detected, 5 coasted → total 15 frames → detection_coverage = 10/15
    t1 = _make_tracklet(n_frames=10, n_coasted=0)
    t2 = _make_tracklet(track_id=1, n_frames=5, n_coasted=5)
    result = evaluate_tracking([t1, t2])
    assert result.detection_coverage == pytest.approx(10.0 / 15.0)


# ---------------------------------------------------------------------------
# Edge case: zero-length tracklet
# ---------------------------------------------------------------------------


def test_evaluate_tracking_zero_length_tracklet_no_divzero() -> None:
    """A zero-length tracklet does not cause division by zero."""
    zero_tracklet = Tracklet2D(
        camera_id="cam0",
        track_id=99,
        frames=(),
        centroids=(),
        bboxes=(),
        frame_status=(),
    )
    normal_tracklet = _make_tracklet(n_frames=5)
    result = evaluate_tracking([zero_tracklet, normal_tracklet])
    # Should not raise; zero tracklet contributes 0 frames
    assert result.track_count == 2
    assert result.length_min == 0


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


def test_evaluate_tracking_to_dict_python_native_types() -> None:
    """to_dict() returns only Python-native types."""
    tracklets = [
        _make_tracklet(track_id=0, n_frames=10, n_coasted=2),
        _make_tracklet(track_id=1, n_frames=5),
    ]
    result = evaluate_tracking(tracklets)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert isinstance(d["track_count"], int)
    assert isinstance(d["length_median"], float)
    assert isinstance(d["length_mean"], float)
    assert isinstance(d["length_min"], int)
    assert isinstance(d["length_max"], int)
    assert isinstance(d["coast_frequency"], float)
    assert isinstance(d["detection_coverage"], float)


# ---------------------------------------------------------------------------
# No engine imports
# ---------------------------------------------------------------------------


def test_tracking_py_has_no_engine_imports() -> None:
    """tracking.py must have zero imports from aquapose.engine."""
    tracking_py = (
        Path(__file__).parent.parent.parent.parent
        / "src"
        / "aquapose"
        / "evaluation"
        / "stages"
        / "tracking.py"
    )
    tree = ast.parse(tracking_py.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "aquapose.engine" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "aquapose.engine" not in module, (
                f"Forbidden import from {module!r} in tracking.py"
            )
