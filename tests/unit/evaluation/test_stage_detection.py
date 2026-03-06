"""Unit tests for DetectionMetrics and evaluate_detection in stages/detection.py."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aquapose.core.types.detection import Detection
from aquapose.evaluation.stages.detection import DetectionMetrics, evaluate_detection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(confidence: float = 0.9) -> Detection:
    """Create a minimal synthetic Detection."""
    return Detection(
        bbox=(0, 0, 10, 10),
        mask=None,
        area=100,
        confidence=confidence,
    )


def _make_frames(
    camera_counts: dict[str, list[int]],
    confidence: float = 0.9,
) -> list[dict[str, list[Detection]]]:
    """Build synthetic frames from per-camera count schedule.

    Args:
        camera_counts: Maps camera_id to list of per-frame detection counts.
        confidence: Confidence to assign to all synthetic detections.

    Returns:
        List of per-frame dicts mapping camera_id to list of Detection objects.
    """
    n_frames = max(len(v) for v in camera_counts.values())
    frames: list[dict[str, list[Detection]]] = []
    for f in range(n_frames):
        frame: dict[str, list[Detection]] = {}
        for cam_id, counts in camera_counts.items():
            if f < len(counts):
                frame[cam_id] = [_make_detection(confidence) for _ in range(counts[f])]
            else:
                frame[cam_id] = []
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_evaluate_detection_empty_returns_zeroes() -> None:
    """evaluate_detection([]) returns DetectionMetrics with all zeroes."""
    result = evaluate_detection([])
    assert isinstance(result, DetectionMetrics)
    assert result.total_detections == 0
    assert result.mean_confidence == 0.0
    assert result.std_confidence == 0.0
    assert result.mean_jitter == 0.0
    assert result.per_camera_counts == {}


# ---------------------------------------------------------------------------
# Known input: totals, confidence stats
# ---------------------------------------------------------------------------


def test_evaluate_detection_known_totals() -> None:
    """Known 3-frame 2-camera input produces correct total_detections."""
    # cam0: [3, 2, 1] = 6 detections; cam1: [2, 2, 2] = 6 detections → total 12
    frames = _make_frames({"cam0": [3, 2, 1], "cam1": [2, 2, 2]}, confidence=0.8)
    result = evaluate_detection(frames)
    assert result.total_detections == 12


def test_evaluate_detection_mean_confidence() -> None:
    """Known confidence value propagates to mean_confidence correctly."""
    frames = _make_frames({"cam0": [2, 2]}, confidence=0.5)
    result = evaluate_detection(frames)
    assert result.mean_confidence == pytest.approx(0.5)


def test_evaluate_detection_std_confidence_mixed() -> None:
    """std_confidence is non-zero when confidences differ across detections."""
    # 2 detections at 0.3 and 2 at 0.7 → std is non-zero
    d_low = _make_detection(confidence=0.3)
    d_high = _make_detection(confidence=0.7)
    frames = [{"cam0": [d_low, d_high], "cam1": [d_low, d_high]}]
    result = evaluate_detection(frames)
    assert result.std_confidence > 0.0


def test_evaluate_detection_std_confidence_uniform_is_zero() -> None:
    """std_confidence is 0.0 when all detections have the same confidence."""
    frames = _make_frames({"cam0": [3, 3], "cam1": [3, 3]}, confidence=0.9)
    result = evaluate_detection(frames)
    assert result.std_confidence == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Jitter metric
# ---------------------------------------------------------------------------


def test_evaluate_detection_stable_counts_zero_jitter() -> None:
    """Stable per-frame counts [5, 5, 5] → jitter == 0.0."""
    frames = _make_frames({"cam0": [5, 5, 5]})
    result = evaluate_detection(frames)
    assert result.mean_jitter == pytest.approx(0.0)


def test_evaluate_detection_flickering_counts_nonzero_jitter() -> None:
    """Flickering counts [5, 1, 5] → jitter == 2.0 (mean abs diff)."""
    # |5-1| = 4, |1-5| = 4 → mean = 4, but per camera mean of abs diff:
    # diffs = [|1-5|, |5-1|] = [4, 4] → np.mean([4, 4]) = 4 → mean across cameras = 4
    # Wait, with 3 frames: diffs = [|1-5|, |5-1|] = [4, 4], mean = 4
    # The plan says [5, 1, 5] → jitter 2.0 — that is mean(abs(diff)) = mean([4, 4]) = 4
    # Re-reading plan: "stable counts [5, 5, 5] per camera → jitter 0.0; flickering counts [5, 1, 5] → jitter 2.0"
    # diffs for [5, 1, 5] are [4, 4], mean = 4.0 ... but plan says 2.0
    # Possibly plan means a single-diff scenario: maybe they mean [5, 1] → |5-1|=4, mean=4?
    # Or maybe [5, 5, 5] vs [1, 5, 1]? Let me test what makes jitter=2.0:
    # [5, 5+2, 5] or similar. Actually for 2 frames: [5, 1] → diff=[4], mean=4
    # Perhaps the plan references two cameras: cam0 stable [5,5,5], cam1 flickering [5,1,5]
    # Then jitter = mean([0.0, 4.0]) = 2.0 — YES, that's the interpretation
    frames = _make_frames({"cam0": [5, 5, 5], "cam1": [5, 1, 5]})
    result = evaluate_detection(frames)
    assert result.mean_jitter == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Per-camera counts
# ---------------------------------------------------------------------------


def test_evaluate_detection_per_camera_counts() -> None:
    """per_camera_counts reflects total detections per camera."""
    frames = _make_frames({"cam0": [10], "cam1": [20]})
    result = evaluate_detection(frames)
    assert result.per_camera_counts["cam0"] == 10
    assert result.per_camera_counts["cam1"] == 20


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


def test_evaluate_detection_to_dict_python_native_types() -> None:
    """to_dict() returns only Python-native types (int, float, dict)."""
    frames = _make_frames({"cam0": [3, 2, 1], "cam1": [2, 2, 2]}, confidence=0.8)
    result = evaluate_detection(frames)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert isinstance(d["total_detections"], int)
    assert isinstance(d["mean_confidence"], float)
    assert isinstance(d["std_confidence"], float)
    assert isinstance(d["mean_jitter"], float)
    assert isinstance(d["per_camera_counts"], dict)
    # Ensure no numpy scalars
    for v in d["per_camera_counts"].values():
        assert isinstance(v, int), f"Expected int, got {type(v)}"


# ---------------------------------------------------------------------------
# No engine imports
# ---------------------------------------------------------------------------


def test_detection_py_has_no_engine_imports() -> None:
    """detection.py must have zero imports from aquapose.engine."""
    detection_py = (
        Path(__file__).parent.parent.parent.parent
        / "src"
        / "aquapose"
        / "evaluation"
        / "stages"
        / "detection.py"
    )
    tree = ast.parse(detection_py.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "aquapose.engine" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "aquapose.engine" not in module, (
                f"Forbidden import from {module!r} in detection.py"
            )
