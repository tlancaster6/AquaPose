"""Unit tests for EvalRunner with synthetic cache fixtures."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from aquapose.core.context import PipelineContext, StaleCacheError
from aquapose.core.types.detection import Detection
from aquapose.core.types.midline import Midline2D
from aquapose.core.types.reconstruction import Midline3D

# ---------------------------------------------------------------------------
# Helpers: build synthetic pipeline context objects
# ---------------------------------------------------------------------------

_N_FRAMES = 4
_CAM_IDS = ["cam0", "cam1"]
_N_ANIMALS = 2
_FISH_IDS = [0, 1]


def _make_detection(cam_id: str, frame_index: int) -> Detection:
    """Build a minimal synthetic Detection."""
    return Detection(
        bbox=(10.0, 20.0, 50.0, 80.0),
        confidence=0.9,
        area=4000.0,
        camera_id=cam_id,
        frame_index=frame_index,
    )


def _make_midline2d(fish_id: int, cam_id: str, frame_index: int) -> Midline2D:
    """Build a minimal synthetic Midline2D."""
    rng = np.random.default_rng(fish_id * 100 + frame_index)
    return Midline2D(
        points=rng.random((10, 2)).astype(np.float32),
        half_widths=rng.random(10).astype(np.float32),
        fish_id=fish_id,
        camera_id=cam_id,
        frame_index=frame_index,
        is_head_to_tail=True,
        point_confidence=np.ones(10, dtype=np.float32),
    )


def _make_midline3d(fish_id: int, frame_index: int) -> Midline3D:
    """Build a minimal synthetic Midline3D."""
    rng = np.random.default_rng(fish_id * 1000 + frame_index)
    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=rng.random((7, 3)).astype(np.float32),
        knots=np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], dtype=np.float32),
        degree=3,
        arc_length=0.2,
        half_widths=np.zeros(15, dtype=np.float32),
        n_cameras=2,
        mean_residual=1.5,
        max_residual=3.0,
        is_low_confidence=False,
        per_camera_residuals={"cam0": 1.2, "cam1": 1.8},
    )


def _make_tracklet2d(fish_id: int, cam_id: str, frames: tuple[int, ...]) -> Any:
    """Build a synthetic Tracklet2D-like object."""
    # Use a simple namespace to avoid importing the real Tracklet2D.
    # The real Tracklet2D has: track_id, camera_id, frames, centroids, frame_status.
    from aquapose.core.tracking.types import Tracklet2D

    centroids = tuple((float(i * 10), float(i * 10)) for i in range(len(frames)))
    frame_status = tuple("detected" for _ in frames)
    return Tracklet2D(
        track_id=fish_id,
        camera_id=cam_id,
        frames=frames,
        centroids=centroids,
        frame_status=frame_status,
    )


def _make_tracklet_group(fish_id: int, tracklets: tuple) -> Any:
    """Build a synthetic TrackletGroup."""
    from aquapose.core.association.types import TrackletGroup

    return TrackletGroup(
        fish_id=fish_id,
        tracklets=tracklets,
        confidence=0.95,
        consensus_centroids=None,
    )


def _make_annotated_detections(n_frames: int, cam_ids: list[str]) -> list:
    """Build synthetic annotated_detections list for midline stage.

    Returns list[dict[str, list[AnnotatedDetection]]] where each entry is
    a per-frame dict of camera_id -> list of AnnotatedDetection.
    Each AnnotatedDetection carries a Midline2D with fish_id=0 or 1.
    """
    from aquapose.core.midline.types import AnnotatedDetection
    from aquapose.core.types.detection import Detection

    frames = []
    for frame_idx in range(n_frames):
        frame_dict: dict[str, list] = {}
        for cam_id in cam_ids:
            ann_list = []
            for fish_id in _FISH_IDS:
                det = Detection(
                    bbox=(10.0 + fish_id * 5, 20.0, 50.0, 80.0),
                    confidence=0.9,
                    area=4000.0,
                    camera_id=cam_id,
                    frame_index=frame_idx,
                )
                midline = _make_midline2d(fish_id, cam_id, frame_idx)
                ann = AnnotatedDetection(
                    detection=det,
                    mask=None,
                    crop_region=None,
                    midline=midline,
                    camera_id=cam_id,
                    frame_index=frame_idx,
                )
                ann_list.append(ann)
            frame_dict[cam_id] = ann_list
        frames.append(frame_dict)
    return frames


def _write_cache(
    path: Path,
    ctx: PipelineContext,
    stage_name: str,
    run_id: str = "run_test_001",
) -> None:
    """Write a minimal cache envelope pickle file."""
    envelope = {
        "run_id": run_id,
        "timestamp": "2026-03-03T00:00:00Z",
        "stage_name": stage_name,
        "version_fingerprint": "abc123",
        "context": ctx,
    }
    path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))


def _write_config_yaml(path: Path, n_animals: int = _N_ANIMALS) -> None:
    """Write a minimal config.yaml with n_animals set."""
    content = f"n_animals: {n_animals}\n"
    path.write_text(content)


# ---------------------------------------------------------------------------
# Fixtures: run directory with various combinations of caches
# ---------------------------------------------------------------------------


def _make_run_dir_with_stages(
    tmp_path: Path,
    stages: list[str],
    n_frames: int = _N_FRAMES,
    write_config: bool = True,
) -> Path:
    """Create a run directory with the specified stage caches written."""
    run_dir = tmp_path / "run_001"
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True)

    if write_config:
        _write_config_yaml(run_dir / "config.yaml")

    # Build detection frames: list[dict[str, list[Detection]]]
    detection_frames = [
        {cam_id: [_make_detection(cam_id, fi)] for cam_id in _CAM_IDS}
        for fi in range(n_frames)
    ]

    # Build tracklets for tracking stage: dict[str, list[Tracklet2D]]
    all_frames_tuple = tuple(range(n_frames))
    tracks_2d = {
        cam_id: [
            _make_tracklet2d(fish_id, cam_id, all_frames_tuple) for fish_id in _FISH_IDS
        ]
        for cam_id in _CAM_IDS
    }

    # Build tracklet groups for association stage: list[TrackletGroup]
    tracklet_groups = [
        _make_tracklet_group(
            fish_id,
            tuple(
                _make_tracklet2d(fish_id, cam_id, all_frames_tuple)
                for cam_id in _CAM_IDS
            ),
        )
        for fish_id in _FISH_IDS
    ]

    # Build annotated_detections for midline stage
    annotated_detections = _make_annotated_detections(n_frames, _CAM_IDS)

    # Build midlines_3d for reconstruction stage: list[dict[int, Midline3D]]
    midlines_3d = [
        {fish_id: _make_midline3d(fish_id, fi) for fish_id in _FISH_IDS}
        for fi in range(n_frames)
    ]

    # Build progressive contexts (each stage adds its output field)
    detection_ctx = PipelineContext(
        frame_count=n_frames,
        camera_ids=_CAM_IDS,
        detections=detection_frames,
    )

    tracking_ctx = PipelineContext(
        frame_count=n_frames,
        camera_ids=_CAM_IDS,
        detections=detection_frames,
        tracks_2d=tracks_2d,
    )

    association_ctx = PipelineContext(
        frame_count=n_frames,
        camera_ids=_CAM_IDS,
        detections=detection_frames,
        tracks_2d=tracks_2d,
        tracklet_groups=tracklet_groups,
    )

    midline_ctx = PipelineContext(
        frame_count=n_frames,
        camera_ids=_CAM_IDS,
        detections=detection_frames,
        tracks_2d=tracks_2d,
        tracklet_groups=tracklet_groups,
        annotated_detections=annotated_detections,
    )

    reconstruction_ctx = PipelineContext(
        frame_count=n_frames,
        camera_ids=_CAM_IDS,
        detections=detection_frames,
        tracks_2d=tracks_2d,
        tracklet_groups=tracklet_groups,
        annotated_detections=annotated_detections,
        midlines_3d=midlines_3d,
    )

    stage_map = {
        "detection": ("DetectionStage", detection_ctx),
        "tracking": ("TrackingStage", tracking_ctx),
        "association": ("AssociationStage", association_ctx),
        "midline": ("MidlineStage", midline_ctx),
        "reconstruction": ("ReconstructionStage", reconstruction_ctx),
    }

    for stage_key in stages:
        stage_name, ctx = stage_map[stage_key]
        cache_path = diag_dir / f"{stage_key}_cache.pkl"
        _write_cache(cache_path, ctx, stage_name)

    return run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_run_dir_returns_all_none(tmp_path: Path) -> None:
    """EvalRunner on a dir with no caches returns EvalRunnerResult with all None metrics."""
    from aquapose.evaluation.runner import EvalRunner, EvalRunnerResult

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    (run_dir / "diagnostics").mkdir()

    runner = EvalRunner(run_dir)
    result = runner.run()

    assert isinstance(result, EvalRunnerResult)
    assert result.detection is None
    assert result.tracking is None
    assert result.association is None
    assert result.midline is None
    assert result.reconstruction is None
    assert result.stages_present == frozenset()
    assert result.frames_evaluated == 0
    assert result.frames_available == 0


def test_detection_only_cache(tmp_path: Path) -> None:
    """Run dir with only detection_cache.pkl -> result.detection is not None, others None."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_run_dir_with_stages(tmp_path, ["detection"])

    runner = EvalRunner(run_dir)
    result = runner.run()

    assert result.detection is not None
    assert result.tracking is None
    assert result.association is None
    assert result.midline is None
    assert result.reconstruction is None
    assert result.stages_present == frozenset({"detection"})
    assert result.frames_evaluated == _N_FRAMES
    assert result.frames_available == _N_FRAMES


def test_all_stages_present(tmp_path: Path) -> None:
    """Full run dir -> all 5 metrics populated, stages_present has 5 elements."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_run_dir_with_stages(
        tmp_path, ["detection", "tracking", "association", "midline", "reconstruction"]
    )

    runner = EvalRunner(run_dir)
    result = runner.run()

    assert result.detection is not None
    assert result.tracking is not None
    assert result.association is not None
    assert result.midline is not None
    assert result.reconstruction is not None
    assert result.stages_present == frozenset(
        {"detection", "tracking", "association", "midline", "reconstruction"}
    )
    assert len(result.stages_present) == 5


def test_n_frames_sampling(tmp_path: Path) -> None:
    """Verify frames_evaluated reflects sampled count when n_frames < frame_count."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_run_dir_with_stages(tmp_path, ["detection"], n_frames=10)

    runner = EvalRunner(run_dir)
    result = runner.run(n_frames=3)

    assert result.frames_available == 10
    assert result.frames_evaluated == 3


def test_to_dict_is_json_serializable(tmp_path: Path) -> None:
    """json.dumps(result.to_dict()) succeeds without error."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_run_dir_with_stages(
        tmp_path, ["detection", "tracking", "association", "midline", "reconstruction"]
    )

    runner = EvalRunner(run_dir)
    result = runner.run()

    d = result.to_dict()
    # Should not raise
    serialized = json.dumps(d)
    parsed = json.loads(serialized)
    assert "run_id" in parsed
    assert "stages" in parsed
    assert "frames_evaluated" in parsed
    assert "frames_available" in parsed


def test_missing_config_yaml_with_association_cache(tmp_path: Path) -> None:
    """Raises FileNotFoundError when config.yaml is missing and association cache is present."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_run_dir_with_stages(tmp_path, ["association"], write_config=False)

    runner = EvalRunner(run_dir)
    with pytest.raises(FileNotFoundError):
        runner.run()


def test_stale_cache_error_propagates(tmp_path: Path) -> None:
    """StaleCacheError from load_stage_cache propagates upward (not silently caught)."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_run_dir_with_stages(tmp_path, ["detection"])

    with mock.patch(
        "aquapose.evaluation.runner.load_stage_cache",
        side_effect=StaleCacheError("Cache is stale"),
    ):
        runner = EvalRunner(run_dir)
        with pytest.raises(StaleCacheError):
            runner.run()


def test_tracking_only_no_frame_count(tmp_path: Path) -> None:
    """Tracking stage evaluator is called with flat list from tracks_2d.values()."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_run_dir_with_stages(tmp_path, ["tracking"])

    runner = EvalRunner(run_dir)
    result = runner.run()

    assert result.tracking is not None
    assert result.tracking.track_count == len(_CAM_IDS) * len(_FISH_IDS)


def test_to_dict_absent_stages_omitted(tmp_path: Path) -> None:
    """to_dict stages dict only contains present stages, absent stages are omitted."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_run_dir_with_stages(tmp_path, ["detection"])

    runner = EvalRunner(run_dir)
    result = runner.run()
    d = result.to_dict()

    assert "detection" in d["stages"]
    assert "tracking" not in d["stages"]
    assert "association" not in d["stages"]
    assert "midline" not in d["stages"]
    assert "reconstruction" not in d["stages"]


def test_stages_present_sorted_list_in_to_dict(tmp_path: Path) -> None:
    """to_dict returns stages_present as a sorted list."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_run_dir_with_stages(tmp_path, ["detection", "tracking"])

    runner = EvalRunner(run_dir)
    result = runner.run()
    d = result.to_dict()

    stages_present = d["stages_present"]
    assert isinstance(stages_present, list)
    assert stages_present == sorted(stages_present)
