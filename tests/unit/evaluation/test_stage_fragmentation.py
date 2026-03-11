"""Unit tests for FragmentationMetrics and evaluate_fragmentation()."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from aquapose.core.tracking.types import Tracklet2D
from aquapose.core.types.reconstruction import Midline3D

# ---------------------------------------------------------------------------
# Helper: build synthetic Midline3D
# ---------------------------------------------------------------------------


def _make_midline3d(
    fish_id: int = 0,
    frame_index: int = 0,
    mean_residual: float = 1.0,
    max_residual: float = 2.0,
) -> Midline3D:
    """Build a synthetic Midline3D with minimal valid arrays."""
    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=np.zeros((7, 3), dtype=np.float32),
        knots=np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], dtype=np.float32),
        degree=3,
        arc_length=0.2,
        half_widths=np.zeros(15, dtype=np.float32),
        n_cameras=3,
        mean_residual=mean_residual,
        max_residual=max_residual,
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_evaluate_fragmentation_empty_returns_zeroed() -> None:
    """evaluate_fragmentation([]) returns zeroed FragmentationMetrics."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation

    result = evaluate_fragmentation([], n_animals=9)
    assert result.total_gaps == 0
    assert result.mean_gap_duration == 0.0
    assert result.max_gap_duration == 0
    assert result.mean_continuity_ratio == 0.0
    assert result.per_fish_continuity == {}
    assert result.unique_fish_ids == 0
    assert result.expected_fish == 9
    assert result.track_births == 0
    assert result.track_deaths == 0
    assert result.mean_track_lifespan == 0.0
    assert result.median_track_lifespan == 0.0


def test_evaluate_fragmentation_all_none_entries() -> None:
    """evaluate_fragmentation with all-None entries returns zeroed metrics."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation

    result = evaluate_fragmentation([None, None, None], n_animals=3)
    assert result.total_gaps == 0
    assert result.unique_fish_ids == 0


# ---------------------------------------------------------------------------
# No gaps
# ---------------------------------------------------------------------------


def test_evaluate_fragmentation_no_gaps_continuity_1() -> None:
    """Two fish present in every frame: continuity_ratio=1.0 for each, gap_count=0."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation

    midlines_3d: list[dict[int, Midline3D] | None] = []
    for frame_idx in range(10):
        midlines_3d.append(
            {
                0: _make_midline3d(fish_id=0, frame_index=frame_idx),
                1: _make_midline3d(fish_id=1, frame_index=frame_idx),
            }
        )
    result = evaluate_fragmentation(midlines_3d, n_animals=2)
    assert result.total_gaps == 0
    assert result.mean_continuity_ratio == pytest.approx(1.0)
    assert result.per_fish_continuity[0] == pytest.approx(1.0)
    assert result.per_fish_continuity[1] == pytest.approx(1.0)
    assert result.unique_fish_ids == 2


# ---------------------------------------------------------------------------
# Known gaps
# ---------------------------------------------------------------------------


def test_evaluate_fragmentation_known_gaps() -> None:
    """Fish 0 present in frames 0,1,2,5,6 (gaps at 3,4). Fish 1 present all 7 frames."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation

    midlines_3d: list[dict[int, Midline3D] | None] = []
    for frame_idx in range(7):
        frame_dict: dict[int, Midline3D] = {}
        if frame_idx not in (3, 4):
            frame_dict[0] = _make_midline3d(fish_id=0, frame_index=frame_idx)
        frame_dict[1] = _make_midline3d(fish_id=1, frame_index=frame_idx)
        midlines_3d.append(frame_dict)

    result = evaluate_fragmentation(midlines_3d, n_animals=2)
    # Fish 0: frames {0,1,2,5,6}, span = 6-0+1 = 7, present = 5, continuity = 5/7
    assert result.per_fish_continuity[0] == pytest.approx(5.0 / 7.0)
    # Fish 1: frames {0..6}, span = 7, present = 7, continuity = 1.0
    assert result.per_fish_continuity[1] == pytest.approx(1.0)
    # Gaps for fish 0: one gap of duration 2 (frames 3,4)
    assert result.total_gaps == 1
    assert result.mean_gap_duration == pytest.approx(2.0)
    assert result.max_gap_duration == 2
    # Mean continuity = (5/7 + 1.0) / 2
    assert result.mean_continuity_ratio == pytest.approx((5.0 / 7.0 + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# Track births and deaths
# ---------------------------------------------------------------------------


def test_evaluate_fragmentation_births_and_deaths() -> None:
    """Fish appearing/disappearing mid-sequence: births and deaths counted."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation

    # Global frame range: 0..9
    # Fish 0: present in frames 0-9 (full span, no birth/death)
    # Fish 1: present in frames 3-9 (birth, no death)
    # Fish 2: present in frames 0-5 (no birth, death)
    midlines_3d: list[dict[int, Midline3D] | None] = []
    for frame_idx in range(10):
        frame_dict: dict[int, Midline3D] = {}
        frame_dict[0] = _make_midline3d(fish_id=0, frame_index=frame_idx)
        if frame_idx >= 3:
            frame_dict[1] = _make_midline3d(fish_id=1, frame_index=frame_idx)
        if frame_idx <= 5:
            frame_dict[2] = _make_midline3d(fish_id=2, frame_index=frame_idx)
        midlines_3d.append(frame_dict)

    result = evaluate_fragmentation(midlines_3d, n_animals=3)
    assert result.unique_fish_ids == 3
    assert result.track_births == 1  # fish 1 appears after global start
    assert result.track_deaths == 1  # fish 2 disappears before global end


# ---------------------------------------------------------------------------
# Single-frame tracks
# ---------------------------------------------------------------------------


def test_evaluate_fragmentation_single_frame_track() -> None:
    """A fish appearing in only one frame: continuity=1.0, no gaps."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation

    midlines_3d: list[dict[int, Midline3D] | None] = [
        {0: _make_midline3d(fish_id=0, frame_index=0)},
        None,
        None,
    ]
    result = evaluate_fragmentation(midlines_3d, n_animals=1)
    assert result.per_fish_continuity[0] == pytest.approx(1.0)
    assert result.total_gaps == 0


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


def test_to_dict_json_serializable() -> None:
    """FragmentationMetrics.to_dict() returns JSON-serializable dict."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation

    midlines_3d: list[dict[int, Midline3D] | None] = [
        {0: _make_midline3d(fish_id=0, frame_index=0)},
        {0: _make_midline3d(fish_id=0, frame_index=1)},
    ]
    result = evaluate_fragmentation(midlines_3d, n_animals=2)
    d = result.to_dict()
    json.dumps(d)  # must not raise
    assert isinstance(d, dict)
    assert "total_gaps" in d
    assert "per_fish_continuity" in d
    # int keys should be strings
    for k in d["per_fish_continuity"]:
        assert isinstance(k, str)


def test_to_dict_types_are_python_scalars() -> None:
    """to_dict() fields are plain Python types, not numpy types."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation

    result = evaluate_fragmentation([], n_animals=0)
    d = result.to_dict()
    assert isinstance(d["total_gaps"], int)
    assert isinstance(d["mean_gap_duration"], float)
    assert isinstance(d["max_gap_duration"], int)
    assert isinstance(d["mean_continuity_ratio"], float)
    assert isinstance(d["unique_fish_ids"], int)
    assert isinstance(d["track_births"], int)
    assert isinstance(d["track_deaths"], int)
    assert isinstance(d["mean_track_lifespan"], float)
    assert isinstance(d["median_track_lifespan"], float)


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------


def test_fragmentation_metrics_is_frozen() -> None:
    """FragmentationMetrics is a frozen dataclass."""
    import dataclasses

    from aquapose.evaluation.stages.fragmentation import FragmentationMetrics

    assert dataclasses.is_dataclass(FragmentationMetrics)
    m = FragmentationMetrics(
        total_gaps=0,
        mean_gap_duration=0.0,
        max_gap_duration=0,
        mean_continuity_ratio=0.0,
        per_fish_continuity={},
        unique_fish_ids=0,
        expected_fish=0,
        track_births=0,
        track_deaths=0,
        mean_track_lifespan=0.0,
        median_track_lifespan=0.0,
    )
    with pytest.raises(AttributeError):
        m.total_gaps = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# evaluate_fragmentation_2d tests
# ---------------------------------------------------------------------------


def _make_tracklet2d(
    track_id: int,
    frames: list[int],
    camera_id: str = "cam0",
) -> Tracklet2D:
    """Build a synthetic Tracklet2D for testing."""
    n = len(frames)
    return Tracklet2D(
        camera_id=camera_id,
        track_id=track_id,
        frames=tuple(frames),
        centroids=tuple((float(i), float(i)) for i in range(n)),
        bboxes=tuple((0.0, 0.0, 10.0, 10.0) for _ in range(n)),
        frame_status=tuple("detected" for _ in range(n)),
    )


def test_evaluate_fragmentation_2d_empty_returns_zero() -> None:
    """evaluate_fragmentation_2d([], n_animals=9) returns zeroed FragmentationMetrics."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation_2d

    result = evaluate_fragmentation_2d([], n_animals=9)
    assert result.total_gaps == 0
    assert result.mean_gap_duration == 0.0
    assert result.max_gap_duration == 0
    assert result.mean_continuity_ratio == 0.0
    assert result.per_fish_continuity == {}
    assert result.unique_fish_ids == 0
    assert result.expected_fish == 9
    assert result.track_births == 0
    assert result.track_deaths == 0
    assert result.mean_track_lifespan == 0.0
    assert result.median_track_lifespan == 0.0


def test_evaluate_fragmentation_2d_known_gaps() -> None:
    """Two tracklets: track 0 has gap at frames 3-4, track 1 is continuous."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation_2d

    t0 = _make_tracklet2d(track_id=0, frames=[0, 1, 2, 5, 6])
    t1 = _make_tracklet2d(track_id=1, frames=[0, 1, 2, 3, 4, 5, 6])

    result = evaluate_fragmentation_2d([t0, t1], n_animals=2)

    # Track 0: frames {0,1,2,5,6}, span = 6-0+1 = 7, present = 5, continuity = 5/7
    assert result.per_fish_continuity[0] == pytest.approx(5.0 / 7.0)
    # Track 1: frames {0..6}, continuity = 1.0
    assert result.per_fish_continuity[1] == pytest.approx(1.0)
    # One gap of duration 2 for track 0
    assert result.total_gaps == 1
    assert result.mean_gap_duration == pytest.approx(2.0)
    assert result.max_gap_duration == 2
    assert result.unique_fish_ids == 2
    assert result.expected_fish == 2
    # Births: track 0 starts at 0 (global first=0), no birth; track 1 starts at 0, no birth
    assert result.track_births == 0
    # Deaths: track 0 ends at 6 (global last=6), no death; track 1 ends at 6, no death
    assert result.track_deaths == 0


def test_evaluate_fragmentation_2d_single_track_all_frames() -> None:
    """Single tracklet spanning all frames: 0 gaps, continuity_ratio=1.0."""
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation_2d

    t0 = _make_tracklet2d(track_id=0, frames=list(range(10)))
    result = evaluate_fragmentation_2d([t0], n_animals=1)

    assert result.total_gaps == 0
    assert result.per_fish_continuity[0] == pytest.approx(1.0)
    assert result.mean_continuity_ratio == pytest.approx(1.0)
    assert result.unique_fish_ids == 1
    assert result.track_births == 0
    assert result.track_deaths == 0


def test_evaluate_fragmentation_2d_returns_frozen_fragmentation_metrics() -> None:
    """Return type is frozen FragmentationMetrics."""
    import dataclasses

    from aquapose.evaluation.stages.fragmentation import (
        FragmentationMetrics,
        evaluate_fragmentation_2d,
    )

    result = evaluate_fragmentation_2d([], n_animals=9)
    assert isinstance(result, FragmentationMetrics)
    assert dataclasses.is_dataclass(result)
    with pytest.raises(AttributeError):
        result.total_gaps = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# No engine imports
# ---------------------------------------------------------------------------


def test_no_engine_imports_in_fragmentation_module() -> None:
    """fragmentation.py must not import from aquapose.engine."""
    module_path = (
        Path(__file__).parents[3]
        / "src"
        / "aquapose"
        / "evaluation"
        / "stages"
        / "fragmentation.py"
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
