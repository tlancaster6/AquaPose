"""Unit tests for AssociationMetrics, evaluate_association(), and DEFAULT_GRID."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from aquapose.core.types.midline import Midline2D
from aquapose.core.types.reconstruction import MidlineSet
from aquapose.evaluation.stages.association import (
    DEFAULT_GRID,
    AssociationMetrics,
    evaluate_association,
)

# ---------------------------------------------------------------------------
# Helper: build synthetic MidlineSet
# ---------------------------------------------------------------------------


def _make_midline_set(fish_cameras: dict[int, list[str]]) -> MidlineSet:
    """Build a synthetic MidlineSet from a {fish_id: [camera_id, ...]} mapping.

    Args:
        fish_cameras: Maps fish_id to a list of camera IDs it appears in.

    Returns:
        A MidlineSet with minimal Midline2D stubs.
    """
    midline_set: MidlineSet = {}
    for fish_id, cam_ids in fish_cameras.items():
        cam_map: dict[str, Midline2D] = {}
        for cam_id in cam_ids:
            cam_map[cam_id] = Midline2D(
                points=np.zeros((10, 2), dtype=np.float32),
                half_widths=np.zeros(10, dtype=np.float32),
                fish_id=fish_id,
                camera_id=cam_id,
                frame_index=0,
                is_head_to_tail=False,
                point_confidence=None,
            )
        midline_set[fish_id] = cam_map
    return midline_set


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_evaluate_association_empty_returns_zeroed_metrics() -> None:
    """evaluate_association([]) returns AssociationMetrics with zeroed values."""
    result = evaluate_association([], n_animals=9)
    assert isinstance(result, AssociationMetrics)
    assert result.fish_yield_ratio == pytest.approx(0.0)
    assert result.singleton_rate == pytest.approx(0.0)
    assert result.camera_distribution == {}
    assert result.total_fish_observations == 0
    assert result.frames_evaluated == 0


# ---------------------------------------------------------------------------
# Yield ratio
# ---------------------------------------------------------------------------


def test_evaluate_association_yield_ratio_full() -> None:
    """2 frames each with 9 fish yields fish_yield_ratio=1.0 for n_animals=9."""
    # Each fish in 3 cameras (not singleton)
    frame = _make_midline_set(
        {fish_id: ["cam0", "cam1", "cam2"] for fish_id in range(9)}
    )
    result = evaluate_association([frame, frame], n_animals=9)
    # 18 observations / 2 frames = 9 fish/frame; 9/9 = 1.0
    assert result.fish_yield_ratio == pytest.approx(1.0)
    assert result.frames_evaluated == 2
    assert result.total_fish_observations == 18


def test_evaluate_association_yield_ratio_partial() -> None:
    """5 multi-view fish out of 10 expected yields fish_yield_ratio=0.5."""
    frame = _make_midline_set({fish_id: ["cam0", "cam1"] for fish_id in range(5)})
    result = evaluate_association([frame], n_animals=10)
    assert result.fish_yield_ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Singleton rate
# ---------------------------------------------------------------------------


def test_evaluate_association_singleton_rate() -> None:
    """5 singletons out of 18 total observations = singleton_rate 5/18."""
    # Frame 0: 5 singletons (1 cam), 4 multi-camera
    frame0: MidlineSet = {}
    for fish_id in range(5):
        frame0[fish_id] = {
            "cam0": Midline2D(
                points=np.zeros((10, 2), dtype=np.float32),
                half_widths=np.zeros(10, dtype=np.float32),
                fish_id=fish_id,
                camera_id="cam0",
                frame_index=0,
            )
        }
    for fish_id in range(5, 9):
        frame0[fish_id] = {
            "cam0": Midline2D(
                points=np.zeros((10, 2), dtype=np.float32),
                half_widths=np.zeros(10, dtype=np.float32),
                fish_id=fish_id,
                camera_id="cam0",
                frame_index=0,
            ),
            "cam1": Midline2D(
                points=np.zeros((10, 2), dtype=np.float32),
                half_widths=np.zeros(10, dtype=np.float32),
                fish_id=fish_id,
                camera_id="cam1",
                frame_index=0,
            ),
        }
    # Frame 1: 9 fish, no singletons (2 cams each)
    frame1: MidlineSet = {
        fish_id: {
            "cam0": Midline2D(
                points=np.zeros((10, 2), dtype=np.float32),
                half_widths=np.zeros(10, dtype=np.float32),
                fish_id=fish_id,
                camera_id="cam0",
                frame_index=1,
            ),
            "cam1": Midline2D(
                points=np.zeros((10, 2), dtype=np.float32),
                half_widths=np.zeros(10, dtype=np.float32),
                fish_id=fish_id,
                camera_id="cam1",
                frame_index=1,
            ),
        }
        for fish_id in range(9)
    }
    result = evaluate_association([frame0, frame1], n_animals=9)
    # 5 singletons out of 9+9=18 total observations
    assert result.total_fish_observations == 18
    assert result.singleton_rate == pytest.approx(5.0 / 18.0)


# ---------------------------------------------------------------------------
# Camera distribution
# ---------------------------------------------------------------------------


def test_evaluate_association_camera_distribution() -> None:
    """camera_distribution counts match expected {1: 5, 2: 8, 3: 5}."""
    # Build 2 frames totaling: 5 observations with 1 cam, 8 with 2 cams, 5 with 3 cams
    # Frame 0: 5 fish in 1 cam, 4 fish in 3 cams  → 5 x {1}, 4 x {3}
    # Frame 1: 8 fish in 2 cams, 1 fish in 3 cams → 8 x {2}, 1 x {3}
    frame0: MidlineSet = {}
    for fid in range(5):
        frame0[fid] = {
            "cam0": Midline2D(
                points=np.zeros((5, 2), dtype=np.float32),
                half_widths=np.zeros(5, dtype=np.float32),
                fish_id=fid,
                camera_id="cam0",
                frame_index=0,
            )
        }
    for fid in range(5, 9):
        frame0[fid] = {
            c: Midline2D(
                points=np.zeros((5, 2), dtype=np.float32),
                half_widths=np.zeros(5, dtype=np.float32),
                fish_id=fid,
                camera_id=c,
                frame_index=0,
            )
            for c in ["cam0", "cam1", "cam2"]
        }
    frame1: MidlineSet = {}
    for fid in range(8):
        frame1[fid] = {
            c: Midline2D(
                points=np.zeros((5, 2), dtype=np.float32),
                half_widths=np.zeros(5, dtype=np.float32),
                fish_id=fid,
                camera_id=c,
                frame_index=1,
            )
            for c in ["cam0", "cam1"]
        }
    frame1[8] = {
        c: Midline2D(
            points=np.zeros((5, 2), dtype=np.float32),
            half_widths=np.zeros(5, dtype=np.float32),
            fish_id=8,
            camera_id=c,
            frame_index=1,
        )
        for c in ["cam0", "cam1", "cam2"]
    }
    result = evaluate_association([frame0, frame1], n_animals=9)
    assert result.camera_distribution[1] == 5
    assert result.camera_distribution[2] == 8
    assert result.camera_distribution[3] == 5


# ---------------------------------------------------------------------------
# DEFAULT_GRID
# ---------------------------------------------------------------------------


def test_default_grid_has_exactly_five_keys() -> None:
    """DEFAULT_GRID has exactly 5 keys."""
    expected_keys = {
        "ray_distance_threshold",
        "score_min",
        "eviction_reproj_threshold",
        "leiden_resolution",
        "early_k",
    }
    assert set(DEFAULT_GRID.keys()) == expected_keys


def test_default_grid_ray_distance_threshold_values() -> None:
    """DEFAULT_GRID ray_distance_threshold has expected values."""
    assert DEFAULT_GRID["ray_distance_threshold"] == pytest.approx(
        [0.02, 0.04, 0.06, 0.10, 0.15]
    )


def test_default_grid_score_min_values() -> None:
    """DEFAULT_GRID score_min has expected values."""
    assert DEFAULT_GRID["score_min"] == pytest.approx([0.03, 0.08, 0.15, 0.20, 0.30])


def test_default_grid_eviction_reproj_threshold_values() -> None:
    """DEFAULT_GRID eviction_reproj_threshold has expected values."""
    assert DEFAULT_GRID["eviction_reproj_threshold"] == pytest.approx(
        [0.01, 0.03, 0.05, 0.08, 0.10]
    )


def test_default_grid_leiden_resolution_values() -> None:
    """DEFAULT_GRID leiden_resolution has expected values."""
    assert DEFAULT_GRID["leiden_resolution"] == pytest.approx([0.5, 1.0, 1.5, 2.0])


def test_default_grid_early_k_values() -> None:
    """DEFAULT_GRID early_k has expected values."""
    assert DEFAULT_GRID["early_k"] == pytest.approx([5.0, 10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


def test_to_dict_returns_json_serializable() -> None:
    """AssociationMetrics.to_dict() returns a JSON-serializable dict."""
    import json

    frame = _make_midline_set({0: ["cam0", "cam1"], 1: ["cam2"]})
    result = evaluate_association([frame], n_animals=5)
    d = result.to_dict()
    # Should not raise
    json.dumps(d)
    assert isinstance(d, dict)
    assert "fish_yield_ratio" in d
    assert "singleton_rate" in d
    assert "camera_distribution" in d


def test_to_dict_camera_distribution_str_keyed() -> None:
    """to_dict() converts camera_distribution keys to strings for JSON."""
    frame = _make_midline_set({0: ["cam0"], 1: ["cam0", "cam1"]})
    result = evaluate_association([frame], n_animals=2)
    d = result.to_dict()
    for key in d["camera_distribution"]:
        assert isinstance(key, str), f"Expected str key, got {type(key)}: {key!r}"


# ---------------------------------------------------------------------------
# No engine imports
# ---------------------------------------------------------------------------


def test_no_engine_imports_in_association_module() -> None:
    """association.py must not import from aquapose.engine."""
    module_path = (
        Path(__file__).parents[3]
        / "src"
        / "aquapose"
        / "evaluation"
        / "stages"
        / "association.py"
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
