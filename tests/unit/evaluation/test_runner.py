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
        bbox=(10, 20, 50, 80),
        mask=None,
        area=4000,
        confidence=0.9,
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
    from aquapose.core.tracking.types import Tracklet2D

    centroids = tuple((float(i * 10), float(i * 10)) for i in range(len(frames)))
    bboxes = tuple(
        (float(i * 10), float(i * 10), 50.0, 80.0) for i in range(len(frames))
    )
    frame_status = tuple("detected" for _ in frames)
    return Tracklet2D(
        camera_id=cam_id,
        track_id=fish_id,
        frames=frames,
        centroids=centroids,
        bboxes=bboxes,
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


def _make_detections_with_keypoints(n_frames: int, cam_ids: list[str]) -> list:
    """Build synthetic detections list with keypoints on Detection objects (v3.7).

    Returns list[dict[str, list[Detection]]] where each Detection has
    keypoints and keypoint_conf populated directly.
    """
    from aquapose.core.types.detection import Detection

    frames = []
    for frame_idx in range(n_frames):
        frame_dict: dict[str, list] = {}
        for cam_id in cam_ids:
            det_list = []
            for fish_id in _FISH_IDS:
                rng = np.random.default_rng(fish_id * 100 + frame_idx)
                det = Detection(
                    bbox=(10 + fish_id * 5, 20, 50, 80),
                    mask=None,
                    area=4000,
                    confidence=0.9,
                    keypoints=rng.random((6, 2)).astype(np.float32),
                    keypoint_conf=np.ones(6, dtype=np.float32),
                )
                det_list.append(det)
            frame_dict[cam_id] = det_list
        frames.append(frame_dict)
    return frames


def _make_full_context(
    n_frames: int,
    run_id: str = "run_test_001",
    frame_offset: int = 0,
) -> PipelineContext:
    """Build a synthetic PipelineContext with all stages populated.

    Args:
        n_frames: Number of frames in this context.
        run_id: Run ID for the context (not stored on PipelineContext directly).
        frame_offset: Offset to add to frame indices (for multi-chunk contexts).

    Returns:
        PipelineContext with all fields populated.
    """
    all_frames_tuple = tuple(range(frame_offset, frame_offset + n_frames))
    detection_frames = [
        {cam_id: [_make_detection(cam_id, fi)] for cam_id in _CAM_IDS}
        for fi in range(n_frames)
    ]
    tracks_2d = {
        cam_id: [
            _make_tracklet2d(fish_id, cam_id, all_frames_tuple) for fish_id in _FISH_IDS
        ]
        for cam_id in _CAM_IDS
    }
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
    midlines_3d = [
        {fish_id: _make_midline3d(fish_id, fi) for fish_id in _FISH_IDS}
        for fi in range(n_frames)
    ]
    return PipelineContext(
        frame_count=n_frames,
        camera_ids=_CAM_IDS,
        detections=detection_frames,
        tracks_2d=tracks_2d,
        tracklet_groups=tracklet_groups,
        midlines_3d=midlines_3d,
    )


def _write_chunk_cache(
    chunk_dir: Path,
    ctx: PipelineContext,
    run_id: str = "run_test_001",
) -> None:
    """Write a chunk cache.pkl in the new per-chunk layout."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    cache_path = chunk_dir / "cache.pkl"
    envelope = {
        "run_id": run_id,
        "timestamp": "2026-03-03T00:00:00Z",
        "version_fingerprint": "abc123",
        "context": ctx,
    }
    cache_path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))


def _write_manifest(
    diag_dir: Path,
    chunks: list[dict],
    run_id: str = "run_test_001",
    total_frames: int | None = None,
) -> None:
    """Write a diagnostics/manifest.json."""
    manifest = {
        "run_id": run_id,
        "total_frames": total_frames,
        "chunk_size": _N_FRAMES,
        "version_fingerprint": "abc123",
        "chunks": chunks,
    }
    (diag_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _write_config_yaml(path: Path, n_animals: int = _N_ANIMALS) -> None:
    """Write a minimal config.yaml with n_animals set."""
    content = f"n_animals: {n_animals}\n"
    path.write_text(content)


# ---------------------------------------------------------------------------
# Helpers for legacy-style (old flat layout) test fixtures
# ---------------------------------------------------------------------------


def _write_legacy_cache(
    path: Path,
    ctx: PipelineContext,
    stage_name: str,
    run_id: str = "run_test_001",
) -> None:
    """Write a minimal cache envelope pickle file in the old per-stage flat layout."""
    envelope = {
        "run_id": run_id,
        "timestamp": "2026-03-03T00:00:00Z",
        "stage_name": stage_name,
        "version_fingerprint": "abc123",
        "context": ctx,
    }
    path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))


# ---------------------------------------------------------------------------
# New chunk layout test fixtures
# ---------------------------------------------------------------------------


def _make_chunk_run_dir(
    tmp_path: Path,
    n_chunks: int = 1,
    n_frames_per_chunk: int = _N_FRAMES,
    write_config: bool = True,
    write_manifest: bool = True,
) -> Path:
    """Create a run directory using the new per-chunk cache layout.

    Args:
        tmp_path: Base temp directory.
        n_chunks: Number of chunks to create.
        n_frames_per_chunk: Frames per chunk.
        write_config: Whether to write config.yaml.
        write_manifest: Whether to write manifest.json.

    Returns:
        Path to the run directory.
    """
    run_dir = tmp_path / "run_001"
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True)

    if write_config:
        _write_config_yaml(run_dir / "config.yaml")

    chunk_entries = []
    for chunk_idx in range(n_chunks):
        frame_offset = chunk_idx * n_frames_per_chunk
        ctx = _make_full_context(n_frames_per_chunk, frame_offset=frame_offset)
        chunk_dir = diag_dir / f"chunk_{chunk_idx:03d}"
        _write_chunk_cache(chunk_dir, ctx)
        chunk_entries.append(
            {
                "index": chunk_idx,
                "start_frame": frame_offset,
                "end_frame": frame_offset + n_frames_per_chunk,
                "stages_cached": [
                    "DetectionStage",
                    "PoseStage",
                    "TrackingStage",
                    "AssociationStage",
                    "ReconstructionStage",
                ],
            }
        )

    if write_manifest:
        _write_manifest(
            diag_dir,
            chunk_entries,
            total_frames=n_chunks * n_frames_per_chunk,
        )

    return run_dir


# ---------------------------------------------------------------------------
# Tests: single-chunk behavior (chunk_000 only)
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


def test_single_chunk_all_stages_present(tmp_path: Path) -> None:
    """Single-chunk run with chunk_000/cache.pkl -> all 5 metrics populated."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=1)

    runner = EvalRunner(run_dir)
    result = runner.run()

    assert result.detection is not None
    assert result.tracking is not None
    assert result.association is not None
    # midline evaluation removed in v3.7 (annotated_detections no longer produced)
    assert result.reconstruction is not None
    assert result.stages_present == frozenset(
        {"detection", "tracking", "association", "reconstruction"}
    )
    assert result.frames_available == _N_FRAMES
    assert result.frames_evaluated == _N_FRAMES


def test_single_chunk_n_frames_sampling(tmp_path: Path) -> None:
    """Single-chunk run respects n_frames sampling."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=1, n_frames_per_chunk=10)

    runner = EvalRunner(run_dir)
    result = runner.run(n_frames=3)

    assert result.frames_available == 10
    assert result.frames_evaluated == 3


def test_single_chunk_to_dict_is_json_serializable(tmp_path: Path) -> None:
    """json.dumps(result.to_dict()) succeeds without error on single-chunk run."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=1)

    runner = EvalRunner(run_dir)
    result = runner.run()

    d = result.to_dict()
    import json

    serialized = json.dumps(d)
    parsed = json.loads(serialized)
    assert "run_id" in parsed
    assert "stages" in parsed
    assert "frames_evaluated" in parsed
    assert "frames_available" in parsed


# ---------------------------------------------------------------------------
# Tests: manifest-based discovery
# ---------------------------------------------------------------------------


def test_manifest_based_discovery(tmp_path: Path) -> None:
    """EvalRunner reads manifest.json to find chunk directories."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=1)

    runner = EvalRunner(run_dir)
    result = runner.run()

    # Should succeed using manifest-driven discovery
    assert result.detection is not None
    assert result.frames_available == _N_FRAMES


def test_fallback_discovery_without_manifest(tmp_path: Path) -> None:
    """EvalRunner globs chunk_*/cache.pkl when manifest.json is absent."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=1, write_manifest=False)
    # Confirm no manifest was written
    assert not (run_dir / "diagnostics" / "manifest.json").exists()

    runner = EvalRunner(run_dir)
    result = runner.run()

    assert result.detection is not None
    assert result.frames_available == _N_FRAMES


# ---------------------------------------------------------------------------
# Tests: multi-chunk merging with frame offsets
# ---------------------------------------------------------------------------


def test_multi_chunk_merges_frame_counts(tmp_path: Path) -> None:
    """Multi-chunk run: frames_available equals sum of all chunk frame counts."""
    from aquapose.evaluation.runner import EvalRunner

    n_chunks = 3
    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=n_chunks)

    runner = EvalRunner(run_dir)
    result = runner.run()

    assert result.frames_available == n_chunks * _N_FRAMES


def test_multi_chunk_all_stages_present(tmp_path: Path) -> None:
    """Multi-chunk run: all stage metrics populated when all chunks have full contexts."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=2)

    runner = EvalRunner(run_dir)
    result = runner.run()

    assert result.detection is not None
    assert result.tracking is not None
    assert result.association is not None
    # midline evaluation removed in v3.7 (annotated_detections no longer produced)
    assert result.reconstruction is not None


def test_multi_chunk_stages_present_frozenset(tmp_path: Path) -> None:
    """Multi-chunk run: stages_present contains 4 stage keys (midline removed in v3.7)."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=2)

    runner = EvalRunner(run_dir)
    result = runner.run()

    assert result.stages_present == frozenset(
        {"detection", "tracking", "association", "reconstruction"}
    )


def test_multi_chunk_tracking_count(tmp_path: Path) -> None:
    """Multi-chunk run: tracking track_count reflects merged tracklets across chunks."""
    from aquapose.evaluation.runner import EvalRunner

    n_chunks = 2
    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=n_chunks)

    runner = EvalRunner(run_dir)
    result = runner.run()

    assert result.tracking is not None
    # Each chunk has 2 cameras * 2 fish = 4 tracklets -> 2 chunks = 8 total
    assert result.tracking.track_count == n_chunks * len(_CAM_IDS) * len(_FISH_IDS)


# ---------------------------------------------------------------------------
# Tests: backward compatibility with old per-stage flat layout
# ---------------------------------------------------------------------------


def test_missing_config_yaml_with_association_cache(tmp_path: Path) -> None:
    """Raises FileNotFoundError when config.yaml is missing and association cache present."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=1, write_config=False)

    runner = EvalRunner(run_dir)
    with pytest.raises(FileNotFoundError):
        runner.run()


def test_stale_cache_error_propagates(tmp_path: Path) -> None:
    """StaleCacheError from load_chunk_cache propagates upward."""
    from aquapose.evaluation.runner import EvalRunner

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=1)

    with mock.patch(
        "aquapose.evaluation.runner.load_chunk_cache",
        side_effect=StaleCacheError("Cache is stale"),
    ):
        runner = EvalRunner(run_dir)
        with pytest.raises(StaleCacheError):
            runner.run()


# ---------------------------------------------------------------------------
# Tests: load_run_context shared utility
# ---------------------------------------------------------------------------


def test_load_run_context_returns_merged_context_and_metadata(tmp_path: Path) -> None:
    """load_run_context returns a (PipelineContext, dict) tuple."""
    from aquapose.evaluation.runner import load_run_context

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=1)
    ctx, meta = load_run_context(run_dir)

    assert isinstance(ctx, PipelineContext)
    assert isinstance(meta, dict)
    assert ctx.frame_count == _N_FRAMES


def test_load_run_context_multi_chunk_frame_count(tmp_path: Path) -> None:
    """load_run_context merges chunk frame counts correctly."""
    from aquapose.evaluation.runner import load_run_context

    n_chunks = 3
    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=n_chunks)
    ctx, _meta = load_run_context(run_dir)

    assert ctx.frame_count == n_chunks * _N_FRAMES


def test_load_run_context_empty_dir_returns_none(tmp_path: Path) -> None:
    """load_run_context returns (None, {}) when no chunk caches are found."""
    from aquapose.evaluation.runner import load_run_context

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    (run_dir / "diagnostics").mkdir()

    ctx, meta = load_run_context(run_dir)

    assert ctx is None
    assert meta == {}


# ---------------------------------------------------------------------------
# Tests: to_dict correctness
# ---------------------------------------------------------------------------


def test_to_dict_absent_stages_omitted(tmp_path: Path) -> None:
    """to_dict stages dict only contains present stages, absent stages are omitted."""
    from aquapose.evaluation.runner import EvalRunner

    # Create a context with only detection data
    run_dir = tmp_path / "run_001"
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True)
    _write_config_yaml(run_dir / "config.yaml")

    # Write a chunk cache with only detection data
    detection_only_ctx = PipelineContext(
        frame_count=_N_FRAMES,
        camera_ids=_CAM_IDS,
        detections=[
            {cam_id: [_make_detection(cam_id, fi)] for cam_id in _CAM_IDS}
            for fi in range(_N_FRAMES)
        ],
    )
    _write_chunk_cache(diag_dir / "chunk_000", detection_only_ctx)
    _write_manifest(
        diag_dir,
        [
            {
                "index": 0,
                "start_frame": 0,
                "end_frame": _N_FRAMES,
                "stages_cached": ["DetectionStage"],
            }
        ],
    )

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

    run_dir = _make_chunk_run_dir(tmp_path, n_chunks=1)

    runner = EvalRunner(run_dir)
    result = runner.run()
    d = result.to_dict()

    stages_present = d["stages_present"]
    assert isinstance(stages_present, list)
    assert stages_present == sorted(stages_present)
