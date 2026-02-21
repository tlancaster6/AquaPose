"""Unit tests for pipeline stage functions with mocked dependencies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from aquapose.pipeline.stages import run_tracking, run_triangulation
from aquapose.reconstruction.triangulation import (
    N_SAMPLE_POINTS,
    SPLINE_K,
    SPLINE_KNOTS,
    Midline3D,
    MidlineSet,
)
from aquapose.segmentation.detector import Detection
from aquapose.tracking.tracker import FishTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(x: int = 10, y: int = 20, w: int = 50, h: int = 80) -> Detection:
    """Create a minimal Detection fixture."""
    return Detection(bbox=(x, y, w, h), mask=None, area=w * h, confidence=1.0)


def _make_midline3d(fish_id: int) -> Midline3D:
    """Create a synthetic Midline3D for testing."""
    rng = np.random.default_rng(seed=fish_id)
    return Midline3D(
        fish_id=fish_id,
        frame_index=0,
        control_points=rng.random((7, 3)).astype(np.float32),
        knots=SPLINE_KNOTS.astype(np.float32),
        degree=SPLINE_K,
        arc_length=0.25 + fish_id * 0.05,
        half_widths=rng.random(N_SAMPLE_POINTS).astype(np.float32) * 0.01,
        n_cameras=3,
        mean_residual=1.0,
        max_residual=2.5,
        is_low_confidence=False,
    )


def _make_mock_models(camera_ids: list[str], n_detections: int = 1) -> dict:
    """Create minimal mock RefractiveProjectionModel objects.

    Configures cast_ray to return valid (origins, directions) tensors so that
    FishTracker can process detections without real projection geometry.
    """
    import torch

    mock_models = {}
    for cam_id in camera_ids:
        model = MagicMock()
        # cast_ray(pixels) -> (origins, directions) both shape (N, 3)
        origins = torch.zeros(n_detections, 3, dtype=torch.float32)
        directions = torch.zeros(n_detections, 3, dtype=torch.float32)
        directions[:, 2] = 1.0  # unit vectors pointing down
        model.cast_ray.return_value = (origins, directions)
        mock_models[cam_id] = model
    return mock_models


# ---------------------------------------------------------------------------
# Test 1: run_triangulation output structure
# ---------------------------------------------------------------------------


def test_run_triangulation() -> None:
    """run_triangulation calls triangulate_midlines per frame and aggregates results."""
    cameras = ["cam_a", "cam_b", "cam_c"]
    models = _make_mock_models(cameras)

    # Build synthetic MidlineSets for 3 frames with 2 fish each
    # (Midline2D contents don't matter — we mock triangulate_midlines)
    midline_sets: list[MidlineSet] = []
    for _frame_i in range(3):
        ms: MidlineSet = {}
        for fish_id in range(2):
            cam_midlines = {}
            for cam in cameras:
                m2d = MagicMock()
                cam_midlines[cam] = m2d
            ms[fish_id] = cam_midlines
        midline_sets.append(ms)

    # Mock triangulate_midlines to return 2 Midline3D objects per frame
    def _fake_triangulate(
        midline_set: MidlineSet,
        models_arg: dict,
        frame_index: int = 0,
        inlier_threshold: float = 15.0,
    ) -> dict[int, Midline3D]:
        return {fid: _make_midline3d(fid) for fid in midline_set}

    with patch(
        "aquapose.reconstruction.triangulation.triangulate_midlines",
        side_effect=_fake_triangulate,
    ):
        results = run_triangulation(midline_sets, models)

    # Verify structure
    assert len(results) == 3
    for _frame_i, frame_result in enumerate(results):
        assert isinstance(frame_result, dict)
        assert set(frame_result.keys()) == {0, 1}
        for fish_id, m3d in frame_result.items():
            assert isinstance(m3d, Midline3D)
            assert m3d.fish_id == fish_id
            assert m3d.control_points.shape == (7, 3)


# ---------------------------------------------------------------------------
# Test 2: run_tracking preserves tracker state
# ---------------------------------------------------------------------------


def test_run_tracking_preserves_tracker_state() -> None:
    """run_tracking does not re-create the tracker; frame_count advances correctly."""
    cameras = ["cam_a", "cam_b"]
    models = _make_mock_models(cameras)

    # Build 5 frames of empty detections
    n_frames = 5
    detections_per_frame: list[dict[str, list[Detection]]] = [
        {cam: [] for cam in cameras} for _ in range(n_frames)
    ]

    # Create tracker externally (should be the SAME object after run_tracking)
    tracker = FishTracker(expected_count=3)
    assert tracker.frame_count == 0

    tracks_per_frame = run_tracking(
        detections_per_frame=detections_per_frame,
        models=models,
        tracker=tracker,
    )

    # Tracker state should be mutated: frame_count advances to n_frames
    assert tracker.frame_count == n_frames

    # Output should have one list per frame
    assert len(tracks_per_frame) == n_frames

    # Each entry is a list of confirmed tracks (all empty with no detections)
    for confirmed in tracks_per_frame:
        assert isinstance(confirmed, list)


# ---------------------------------------------------------------------------
# Test 3: run_tracking with mock models (no actual projection)
# ---------------------------------------------------------------------------


def test_run_tracking_single_camera_detections() -> None:
    """run_tracking correctly processes single-camera detections via tracker."""
    cameras = ["cam_a"]
    models = _make_mock_models(cameras)

    # Provide a single detection in each frame for cam_a
    # FishTracker with single-camera detections may not confirm tracks in 1 frame,
    # but it should not crash and should return a list per frame.
    n_frames = 3
    det = _make_detection()
    detections_per_frame = [{cam: [det] for cam in cameras} for _ in range(n_frames)]

    tracker = FishTracker(expected_count=2)
    tracks_per_frame = run_tracking(
        detections_per_frame=detections_per_frame,
        models=models,
        tracker=tracker,
    )

    assert len(tracks_per_frame) == n_frames
    assert tracker.frame_count == n_frames
