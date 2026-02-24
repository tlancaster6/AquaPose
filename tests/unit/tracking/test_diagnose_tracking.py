"""Smoke tests for diagnose_tracking.py metric computation functions.

Tests are isolated from GPU and camera models — they verify pure metric
computation logic using mock GT matching data constructed in-process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load diagnose_tracking module via importlib (it's in scripts/, not a package)
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).parent.parent.parent.parent / "scripts" / "diagnose_tracking.py"
)


def _load_module() -> object:
    """Load diagnose_tracking.py as a module via importlib."""
    spec = importlib.util.spec_from_file_location(
        "diagnose_tracking", str(_SCRIPT_PATH)
    )
    assert spec is not None, f"Could not create spec for {_SCRIPT_PATH}"
    assert spec.loader is not None, "spec.loader is None"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_tracking"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_module()
TrackingMetrics = _mod.TrackingMetrics  # type: ignore[attr-defined]
compute_tracking_metrics = _mod.compute_tracking_metrics  # type: ignore[attr-defined]

import numpy as np  # noqa: E402 (import after module load is intentional)

# ---------------------------------------------------------------------------
# Helper to build mock data
# ---------------------------------------------------------------------------


def _make_mock_data(
    n_frames: int,
    n_gt_fish: int,
    track_assignments: dict[int, list[int | None]],
) -> tuple[
    list[dict[int, int | None]],
    list[np.ndarray],
    list[list[int]],
]:
    """Build mock gt_matching_per_frame, gt_positions, gt_fish_ids.

    Args:
        n_frames: Number of frames.
        n_gt_fish: Number of GT fish.
        track_assignments: Dict mapping track_id -> per-frame gt_fish_id
            assignments (None = unmatched).

    Returns:
        Tuple of (gt_matching_per_frame, gt_positions_per_frame, gt_fish_ids_per_frame).
    """
    # Build gt_matching_per_frame: per-frame dict[track_id -> gt_id | None]
    gt_matching_per_frame: list[dict[int, int | None]] = []
    for frame_idx in range(n_frames):
        frame_match: dict[int, int | None] = {}
        for track_id, assignments in track_assignments.items():
            if frame_idx < len(assignments):
                frame_match[track_id] = assignments[frame_idx]
        gt_matching_per_frame.append(frame_match)

    # GT positions: simple synthetic positions (not used in metric computation)
    gt_positions_per_frame: list[np.ndarray] = [
        np.zeros((n_gt_fish, 3), dtype=np.float32) for _ in range(n_frames)
    ]

    # All GT fish present in all frames
    gt_fish_ids_per_frame: list[list[int]] = [
        list(range(n_gt_fish)) for _ in range(n_frames)
    ]

    return gt_matching_per_frame, gt_positions_per_frame, gt_fish_ids_per_frame


# ---------------------------------------------------------------------------
# Test 1: Perfect tracking
# ---------------------------------------------------------------------------


def test_perfect_tracking_metrics() -> None:
    """Perfect 1:1 assignment yields MOTA=1.0, no ID switches, no fragmentation."""
    n_frames = 50
    n_gt_fish = 2

    # Track 0 always maps to GT fish 0; track 1 always maps to GT fish 1
    track_assignments: dict[int, list[int | None]] = {
        0: [0] * n_frames,
        1: [1] * n_frames,
    }

    gt_matching, gt_positions, gt_ids = _make_mock_data(
        n_frames, n_gt_fish, track_assignments
    )
    metrics = compute_tracking_metrics(gt_matching, gt_positions, gt_ids, n_gt_fish)

    assert metrics.mota == pytest.approx(1.0, abs=1e-6), (
        f"Expected MOTA=1.0, got {metrics.mota}"
    )
    assert metrics.id_switches == 0, (
        f"Expected 0 ID switches, got {metrics.id_switches}"
    )
    assert metrics.fragmentation == 0, (
        f"Expected 0 fragmentations, got {metrics.fragmentation}"
    )
    assert metrics.false_positives == 0, (
        f"Expected 0 false positives, got {metrics.false_positives}"
    )
    assert metrics.false_negatives == 0, (
        f"Expected 0 false negatives, got {metrics.false_negatives}"
    )


# ---------------------------------------------------------------------------
# Test 2: Complete miss — no GT fish ever matched
# ---------------------------------------------------------------------------


def test_complete_miss_metrics() -> None:
    """No GT matches: MOTA <= 0, FN = n_gt, FP = n_confirmed_tracks * n_frames."""
    n_frames = 30
    n_gt_fish = 2

    # Track 0 has no GT assignment (always None)
    track_assignments: dict[int, list[int | None]] = {
        0: [None] * n_frames,
    }

    gt_matching, gt_positions, gt_ids = _make_mock_data(
        n_frames, n_gt_fish, track_assignments
    )
    metrics = compute_tracking_metrics(gt_matching, gt_positions, gt_ids, n_gt_fish)

    assert metrics.mota <= 0.0, f"Expected MOTA <= 0, got {metrics.mota}"
    # All GT object-frames are misses
    expected_fn = n_frames * n_gt_fish
    assert metrics.false_negatives == expected_fn, (
        f"Expected FN={expected_fn}, got {metrics.false_negatives}"
    )
    # Track with None matches is a false positive each frame
    assert metrics.false_positives == n_frames, (
        f"Expected FP={n_frames}, got {metrics.false_positives}"
    )


# ---------------------------------------------------------------------------
# Test 3: ID switch detection
# ---------------------------------------------------------------------------


def test_id_switch_detection() -> None:
    """Track switching GT assignment triggers id_switches >= 1."""
    n_frames = 100
    n_gt_fish = 2

    # Track 0 maps to GT fish 0 for frames 0-49, then GT fish 1 for frames 50-99
    assign_0: list[int | None] = [*([0] * 50), *([1] * 50)]
    track_assignments: dict[int, list[int | None]] = {0: assign_0}

    gt_matching, gt_positions, gt_ids = _make_mock_data(
        n_frames, n_gt_fish, track_assignments
    )
    metrics = compute_tracking_metrics(gt_matching, gt_positions, gt_ids, n_gt_fish)

    assert metrics.id_switches >= 1, (
        f"Expected at least 1 ID switch, got {metrics.id_switches}"
    )


# ---------------------------------------------------------------------------
# Test 4: Fragmentation detection
# ---------------------------------------------------------------------------


def test_fragmentation_detection() -> None:
    """GT fish tracked, then lost, then re-tracked counts as a fragmentation event."""
    n_frames = 51
    n_gt_fish = 1

    # GT fish 0 covered frames 0-20, lost 21-30, covered again 31-50
    # Track 0: matches GT 0 for frames 0-20, None for 21-30, GT 0 again 31-50
    assignments: list[int | None] = [0] * 21 + [None] * 10 + [0] * 20
    track_assignments: dict[int, list[int | None]] = {0: assignments}

    gt_matching, gt_positions, gt_ids = _make_mock_data(
        n_frames, n_gt_fish, track_assignments
    )
    metrics = compute_tracking_metrics(gt_matching, gt_positions, gt_ids, n_gt_fish)

    assert metrics.fragmentation >= 1, (
        f"Expected at least 1 fragmentation, got {metrics.fragmentation}"
    )


# ---------------------------------------------------------------------------
# Test 5: TrackingMetrics dataclass completeness
# ---------------------------------------------------------------------------


def test_metrics_dataclass_fields() -> None:
    """TrackingMetrics has all required fields."""
    m = TrackingMetrics()

    required_fields = [
        "mota",
        "id_switches",
        "fragmentation",
        "true_positives",
        "false_negatives",
        "false_positives",
        "mostly_tracked",
        "mostly_lost",
        "mean_track_purity",
    ]

    for field_name in required_fields:
        assert hasattr(m, field_name), f"TrackingMetrics missing field: {field_name}"

    # Verify default values are sensible (no exceptions on access)
    assert isinstance(m.mota, float)
    assert isinstance(m.id_switches, int)
    assert isinstance(m.fragmentation, int)
    assert isinstance(m.true_positives, int)
    assert isinstance(m.false_negatives, int)
    assert isinstance(m.false_positives, int)
    assert isinstance(m.mostly_tracked, int)
    assert isinstance(m.mostly_lost, int)
    assert isinstance(m.mean_track_purity, float)
