"""Unit tests for FrameSource protocol and VideoFrameSource implementation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from aquapose.core.types.frame_source import FrameSource, VideoFrameSource

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_video_frame_source_satisfies_protocol(tmp_path: Path) -> None:
    """VideoFrameSource is a FrameSource (runtime_checkable Protocol)."""
    vfs = _build_vfs(tmp_path)
    assert isinstance(vfs, FrameSource), (
        "VideoFrameSource must satisfy FrameSource Protocol"
    )


# ---------------------------------------------------------------------------
# camera_ids ordering
# ---------------------------------------------------------------------------


def test_camera_ids_returns_sorted(tmp_path: Path) -> None:
    """camera_ids returns camera IDs in sorted order."""
    vfs = _build_vfs(tmp_path, cam_ids=["cam3", "cam1", "cam2"])
    assert vfs.camera_ids == ["cam1", "cam2", "cam3"]


# ---------------------------------------------------------------------------
# max_frames truncation
# ---------------------------------------------------------------------------


def test_max_frames_truncation(tmp_path: Path) -> None:
    """VideoFrameSource with max_frames=2 yields at most 2 frames."""
    vfs = _build_vfs(tmp_path, cam_ids=["cam1"], n_frames=10, max_frames=2)
    with vfs:
        frames = list(vfs)
    assert len(frames) == 2


def test_max_frames_none_yields_all(tmp_path: Path) -> None:
    """VideoFrameSource with max_frames=None yields all available frames."""
    vfs = _build_vfs(tmp_path, cam_ids=["cam1"], n_frames=5, max_frames=None)
    with vfs:
        frames = list(vfs)
    assert len(frames) == 5


def test_len_capped_by_max_frames(tmp_path: Path) -> None:
    """__len__ is capped by max_frames when set."""
    vfs = _build_vfs(tmp_path, cam_ids=["cam1"], n_frames=10, max_frames=3)
    with vfs:
        assert len(vfs) == 3


def test_len_without_max_frames(tmp_path: Path) -> None:
    """__len__ returns full frame count when max_frames is not set."""
    vfs = _build_vfs(tmp_path, cam_ids=["cam1"], n_frames=7, max_frames=None)
    with vfs:
        assert len(vfs) == 7


# ---------------------------------------------------------------------------
# Frame yield structure
# ---------------------------------------------------------------------------


def test_iter_yields_frame_index_and_dict(tmp_path: Path) -> None:
    """__iter__ yields (frame_idx, dict[str, ndarray]) tuples."""
    vfs = _build_vfs(tmp_path, cam_ids=["cam1", "cam2"], n_frames=3)
    with vfs:
        results = list(vfs)
    assert len(results) == 3
    for i, (frame_idx, frames_dict) in enumerate(results):
        assert frame_idx == i
        assert isinstance(frames_dict, dict)
        assert set(frames_dict.keys()) == {"cam1", "cam2"}
        for frame in frames_dict.values():
            assert isinstance(frame, np.ndarray)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_video_dir_raises(tmp_path: Path) -> None:
    """VideoFrameSource raises FileNotFoundError for nonexistent video_dir."""
    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    with pytest.raises(FileNotFoundError, match="video_dir"):
        VideoFrameSource(
            video_dir=tmp_path / "nonexistent",
            calibration_path=calib_path,
        )


def test_no_videos_found_raises(tmp_path: Path) -> None:
    """VideoFrameSource raises ValueError when no video files found."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    mock_calib = MagicMock()
    mock_calib.cameras = {}

    with (
        patch(
            "aquapose.core.types.frame_source.load_calibration_data",
            return_value=mock_calib,
        ),
        pytest.raises(ValueError, match=r"No \.avi/\.mp4"),
    ):
        VideoFrameSource(video_dir=video_dir, calibration_path=calib_path)


# ---------------------------------------------------------------------------
# k_new property
# ---------------------------------------------------------------------------


def test_k_new_returns_dict_of_tensors(tmp_path: Path) -> None:
    """k_new returns dict mapping camera_id to K_new tensor."""
    import torch

    vfs = _build_vfs(tmp_path, cam_ids=["cam1", "cam2"])
    k_new = vfs.k_new
    assert isinstance(k_new, dict)
    assert set(k_new.keys()) == {"cam1", "cam2"}
    for v in k_new.values():
        assert isinstance(v, torch.Tensor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_video(
    path: Path, n_frames: int, width: int = 64, height: int = 48
) -> None:
    """Write a minimal synthetic AVI video with n_frames using cv2.VideoWriter."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    for _ in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _build_vfs(
    tmp_path: Path,
    cam_ids: list[str] | None = None,
    n_frames: int = 5,
    max_frames: int | None = None,
) -> VideoFrameSource:
    """Build a VideoFrameSource with synthetic video files and mocked calibration.

    Args:
        tmp_path: Temporary directory for test files.
        cam_ids: Camera IDs to create video files for.
        n_frames: Number of frames per synthetic video.
        max_frames: Optional max_frames cap.

    Returns:
        Constructed VideoFrameSource.
    """
    if cam_ids is None:
        cam_ids = ["cam1", "cam2"]

    video_dir = tmp_path / "videos"
    video_dir.mkdir(exist_ok=True)
    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    for cam_id in cam_ids:
        _make_synthetic_video(video_dir / f"{cam_id}-video.avi", n_frames)

    # Build mock calibration with proper numpy arrays for undistortion maps
    import torch

    mock_undist = MagicMock()
    mock_undist.K_new = torch.eye(3)
    mock_undist.map_x = np.zeros((48, 64), dtype=np.float32)
    mock_undist.map_y = np.zeros((48, 64), dtype=np.float32)

    mock_calib = MagicMock()
    mock_calib.cameras = {cam_id: MagicMock() for cam_id in cam_ids}

    with (
        patch(
            "aquapose.core.types.frame_source.load_calibration_data",
            return_value=mock_calib,
        ),
        patch(
            "aquapose.core.types.frame_source.compute_undistortion_maps",
            return_value=mock_undist,
        ),
    ):
        vfs = VideoFrameSource(
            video_dir=video_dir,
            calibration_path=calib_path,
            max_frames=max_frames,
        )

    return vfs
