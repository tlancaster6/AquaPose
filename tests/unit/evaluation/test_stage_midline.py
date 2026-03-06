"""Unit tests for MidlineMetrics and evaluate_midline()."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from aquapose.core.types.midline import Midline2D
from aquapose.evaluation.stages.midline import MidlineMetrics, evaluate_midline

# ---------------------------------------------------------------------------
# Helper: build synthetic Midline2D
# ---------------------------------------------------------------------------


def _make_midline(
    fish_id: int = 0,
    frame_index: int = 0,
    points: np.ndarray | None = None,
    point_confidence: np.ndarray | None = None,
) -> Midline2D:
    """Build a minimal Midline2D for testing.

    Args:
        fish_id: Fish identifier.
        frame_index: Frame index.
        points: Shape (N, 2) float32 array of pixel coordinates.
            Defaults to 5 points at the origin.
        point_confidence: Shape (N,) float32 confidence array or None.

    Returns:
        A Midline2D stub with the specified data.
    """
    if points is None:
        points = np.zeros((5, 2), dtype=np.float32)
    n = points.shape[0]
    return Midline2D(
        points=points,
        half_widths=np.zeros(n, dtype=np.float32),
        fish_id=fish_id,
        camera_id="cam0",
        frame_index=frame_index,
        is_head_to_tail=False,
        point_confidence=point_confidence,
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_evaluate_midline_empty_returns_zeroed_metrics() -> None:
    """evaluate_midline([]) returns MidlineMetrics with zeroed values."""
    result = evaluate_midline([])
    assert isinstance(result, MidlineMetrics)
    assert result.mean_confidence == pytest.approx(0.0)
    assert result.std_confidence == pytest.approx(0.0)
    assert result.completeness == pytest.approx(0.0)
    assert result.temporal_smoothness == pytest.approx(0.0)
    assert result.total_midlines == 0


# ---------------------------------------------------------------------------
# Confidence stats
# ---------------------------------------------------------------------------


def test_evaluate_midline_known_confidence_mean_std() -> None:
    """evaluate_midline with known confidences returns correct mean/std."""
    # All 4 points have known confidence values
    conf = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    midline = _make_midline(
        fish_id=0,
        frame_index=0,
        points=np.zeros((4, 2), dtype=np.float32),
        point_confidence=conf,
    )
    result = evaluate_midline([{0: midline}])
    assert result.mean_confidence == pytest.approx(float(np.mean(conf)), abs=1e-5)
    assert result.std_confidence == pytest.approx(float(np.std(conf)), abs=1e-5)
    assert result.total_midlines == 1


def test_evaluate_midline_none_confidence_treated_as_1() -> None:
    """When point_confidence=None, all points are treated as confidence 1.0."""
    midline = _make_midline(
        fish_id=0,
        frame_index=0,
        points=np.zeros((5, 2), dtype=np.float32),
        point_confidence=None,
    )
    result = evaluate_midline([{0: midline}])
    assert result.mean_confidence == pytest.approx(1.0)
    assert result.std_confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def test_evaluate_midline_completeness_all_ones() -> None:
    """Midline with all-1.0 confidence has completeness=1.0."""
    conf = np.ones(6, dtype=np.float32)
    midline = _make_midline(
        points=np.zeros((6, 2), dtype=np.float32), point_confidence=conf
    )
    result = evaluate_midline([{0: midline}])
    assert result.completeness == pytest.approx(1.0)


def test_evaluate_midline_completeness_half_zero() -> None:
    """Midline with half-zero confidence has completeness=0.5."""
    conf = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    midline = _make_midline(
        points=np.zeros((4, 2), dtype=np.float32), point_confidence=conf
    )
    result = evaluate_midline([{0: midline}])
    assert result.completeness == pytest.approx(0.5)


def test_evaluate_midline_completeness_none_confidence_is_1() -> None:
    """point_confidence=None counts as all points complete (completeness=1.0)."""
    midline = _make_midline(
        points=np.zeros((4, 2), dtype=np.float32), point_confidence=None
    )
    result = evaluate_midline([{0: midline}])
    assert result.completeness == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Temporal smoothness
# ---------------------------------------------------------------------------


def test_evaluate_midline_smoothness_stationary_fish() -> None:
    """Fish with identical centroids across frames has temporal_smoothness=0.0."""
    pts = np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]], dtype=np.float32)
    frame0 = {0: _make_midline(fish_id=0, frame_index=0, points=pts)}
    frame1 = {0: _make_midline(fish_id=0, frame_index=1, points=pts)}
    frame2 = {0: _make_midline(fish_id=0, frame_index=2, points=pts)}
    result = evaluate_midline([frame0, frame1, frame2])
    assert result.temporal_smoothness == pytest.approx(0.0, abs=1e-5)


def test_evaluate_midline_smoothness_moving_fish() -> None:
    """Fish centroid moving 3 units per frame yields smoothness=3.0."""
    # Centroid moves from (0,0) to (3,0) to (6,0): L2 distance = 3 per step
    pts0 = np.array([[0.0, 0.0]], dtype=np.float32)
    pts1 = np.array([[3.0, 0.0]], dtype=np.float32)
    pts2 = np.array([[6.0, 0.0]], dtype=np.float32)
    frame0 = {0: _make_midline(fish_id=0, frame_index=0, points=pts0)}
    frame1 = {0: _make_midline(fish_id=0, frame_index=1, points=pts1)}
    frame2 = {0: _make_midline(fish_id=0, frame_index=2, points=pts2)}
    result = evaluate_midline([frame0, frame1, frame2])
    assert result.temporal_smoothness == pytest.approx(3.0, abs=1e-5)


def test_evaluate_midline_smoothness_single_frame_is_zero() -> None:
    """Single frame yields temporal_smoothness=0.0 (no consecutive pairs)."""
    frame = {0: _make_midline(fish_id=0, frame_index=0)}
    result = evaluate_midline([frame])
    assert result.temporal_smoothness == pytest.approx(0.0)


def test_evaluate_midline_smoothness_multiple_fish() -> None:
    """Smoothness averaged across fish: one stationary, one moving 4 units."""
    pts_still = np.array([[5.0, 5.0]], dtype=np.float32)
    pts_move0 = np.array([[0.0, 0.0]], dtype=np.float32)
    pts_move1 = np.array([[4.0, 0.0]], dtype=np.float32)
    frame0 = {
        0: _make_midline(fish_id=0, frame_index=0, points=pts_still),
        1: _make_midline(fish_id=1, frame_index=0, points=pts_move0),
    }
    frame1 = {
        0: _make_midline(fish_id=0, frame_index=1, points=pts_still),
        1: _make_midline(fish_id=1, frame_index=1, points=pts_move1),
    }
    result = evaluate_midline([frame0, frame1])
    # Fish 0 smoothness = 0.0, fish 1 smoothness = 4.0; average = 2.0
    assert result.temporal_smoothness == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


def test_to_dict_returns_json_serializable() -> None:
    """MidlineMetrics.to_dict() returns a JSON-serializable dict."""
    midline = _make_midline(
        points=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        point_confidence=np.array([0.9, 0.7], dtype=np.float32),
    )
    result = evaluate_midline([{0: midline}])
    d = result.to_dict()
    json.dumps(d)  # must not raise
    assert isinstance(d, dict)
    assert "mean_confidence" in d
    assert "std_confidence" in d
    assert "completeness" in d
    assert "temporal_smoothness" in d
    assert "total_midlines" in d


def test_to_dict_types_are_python_scalars() -> None:
    """to_dict() fields are plain Python float/int, not numpy types."""
    midline = _make_midline(
        point_confidence=np.array([0.5, 0.5], dtype=np.float32),
        points=np.zeros((2, 2), dtype=np.float32),
    )
    result = evaluate_midline([{0: midline}])
    d = result.to_dict()
    assert isinstance(d["mean_confidence"], float)
    assert isinstance(d["total_midlines"], int)


# ---------------------------------------------------------------------------
# No engine imports
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Confidence percentiles (EVAL-02)
# ---------------------------------------------------------------------------


def test_evaluate_midline_confidence_percentiles_known_data() -> None:
    """evaluate_midline with known confidences returns correct p10/p50/p90."""
    # 10 points with confidences 0.1, 0.2, ..., 1.0
    conf = np.array([0.1 * (i + 1) for i in range(10)], dtype=np.float32)
    midline = _make_midline(
        fish_id=0,
        frame_index=0,
        points=np.zeros((10, 2), dtype=np.float32),
        point_confidence=conf,
    )
    result = evaluate_midline([{0: midline}])
    assert result.p10_confidence == pytest.approx(
        float(np.percentile(conf, 10)), abs=1e-4
    )
    assert result.p50_confidence == pytest.approx(
        float(np.percentile(conf, 50)), abs=1e-4
    )
    assert result.p90_confidence == pytest.approx(
        float(np.percentile(conf, 90)), abs=1e-4
    )


def test_evaluate_midline_confidence_percentiles_empty_are_none() -> None:
    """evaluate_midline([]) returns None for all confidence percentile fields."""
    result = evaluate_midline([])
    assert result.p10_confidence is None
    assert result.p50_confidence is None
    assert result.p90_confidence is None


def test_midline_to_dict_includes_confidence_percentiles() -> None:
    """to_dict includes confidence percentile fields."""
    conf = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    midline = _make_midline(
        points=np.zeros((3, 2), dtype=np.float32),
        point_confidence=conf,
    )
    result = evaluate_midline([{0: midline}])
    d = result.to_dict()
    assert "p10_confidence" in d
    assert "p50_confidence" in d
    assert "p90_confidence" in d
    assert isinstance(d["p50_confidence"], float)


def test_midline_metrics_backward_compat_without_percentiles() -> None:
    """MidlineMetrics can be constructed without percentile fields."""
    m = MidlineMetrics(
        mean_confidence=0.9,
        std_confidence=0.1,
        completeness=1.0,
        temporal_smoothness=0.0,
        total_midlines=10,
    )
    assert m.p10_confidence is None
    assert m.p50_confidence is None
    assert m.p90_confidence is None


# ---------------------------------------------------------------------------
# No engine imports
# ---------------------------------------------------------------------------


def test_no_engine_imports_in_midline_module() -> None:
    """midline.py must not import from aquapose.engine."""
    module_path = (
        Path(__file__).parents[3]
        / "src"
        / "aquapose"
        / "evaluation"
        / "stages"
        / "midline.py"
    )
    source = module_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("aquapose.engine"), (
                    f"Forbidden import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("aquapose.engine"), (
                f"Forbidden import from: {module}"
            )
