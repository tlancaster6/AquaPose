"""Unit tests for VideoSet multi-camera video reader."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from aquapose.calibration.loader import UndistortionMaps
from aquapose.io.video import VideoSet


def _write_synthetic_video(path: Path, n_frames: int, color: tuple[int, ...]) -> None:
    """Write a synthetic video with solid-color frames (640x480, 30fps)."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (640, 480))
    frame = np.full((480, 640, 3), color, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


@pytest.fixture
def two_camera_videos(tmp_path: Path) -> dict[str, Path]:
    """Create 2 synthetic 5-frame videos."""
    paths = {
        "cam_a": tmp_path / "cam_a.avi",
        "cam_b": tmp_path / "cam_b.avi",
    }
    _write_synthetic_video(paths["cam_a"], 5, (255, 0, 0))
    _write_synthetic_video(paths["cam_b"], 5, (0, 255, 0))
    return paths


@pytest.fixture
def unequal_videos(tmp_path: Path) -> dict[str, Path]:
    """Create 2 videos with different frame counts (3 and 7)."""
    paths = {
        "cam_a": tmp_path / "cam_a.avi",
        "cam_b": tmp_path / "cam_b.avi",
    }
    _write_synthetic_video(paths["cam_a"], 3, (255, 0, 0))
    _write_synthetic_video(paths["cam_b"], 7, (0, 255, 0))
    return paths


class TestVideoSetIteration:
    def test_iterates_all_frames(self, two_camera_videos: dict[str, Path]) -> None:
        vs = VideoSet(two_camera_videos)
        with vs:
            frames_collected = list(vs)

        assert len(frames_collected) == 5
        for idx, (frame_idx, frame_dict) in enumerate(frames_collected):
            assert frame_idx == idx
            assert set(frame_dict.keys()) == {"cam_a", "cam_b"}
            for frame in frame_dict.values():
                assert frame.shape == (480, 640, 3)
                assert frame.dtype == np.uint8

    def test_frame_colors_correct(self, two_camera_videos: dict[str, Path]) -> None:
        vs = VideoSet(two_camera_videos)
        with vs:
            _, frames = next(iter(vs))

        # cam_a is blue (BGR), cam_b is green (BGR)
        assert frames["cam_a"][0, 0, 0] > 200  # B channel high
        assert frames["cam_b"][0, 0, 1] > 200  # G channel high


class TestVideoSetLen:
    def test_returns_min_frame_count(self, unequal_videos: dict[str, Path]) -> None:
        with VideoSet(unequal_videos) as vs:
            assert len(vs) == 3

    def test_equal_lengths(self, two_camera_videos: dict[str, Path]) -> None:
        with VideoSet(two_camera_videos) as vs:
            assert len(vs) == 5


class TestVideoSetRandomAccess:
    def test_read_frame_returns_correct_frame(
        self, two_camera_videos: dict[str, Path]
    ) -> None:
        with VideoSet(two_camera_videos) as vs:
            frames = vs.read_frame(2)

        assert set(frames.keys()) == {"cam_a", "cam_b"}
        for frame in frames.values():
            assert frame.shape == (480, 640, 3)

    def test_read_frame_out_of_range_raises(
        self, two_camera_videos: dict[str, Path]
    ) -> None:
        with VideoSet(two_camera_videos) as vs, pytest.raises(IndexError):
            vs.read_frame(999)

    def test_read_frame_negative_raises(
        self, two_camera_videos: dict[str, Path]
    ) -> None:
        with VideoSet(two_camera_videos) as vs, pytest.raises(IndexError):
            vs.read_frame(-1)


class TestVideoSetUndistortion:
    @pytest.fixture
    def identity_undist_maps(self) -> dict[str, UndistortionMaps]:
        """Maps that shift pixels by 10px in x — visibly different from raw."""
        h, w = 480, 640
        # Create a remap that shifts all pixels 10px to the right
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1)) - 10.0
        map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        K_new = torch.eye(3, dtype=torch.float32)
        K_new[0, 0] = 500.0
        K_new[1, 1] = 500.0
        K_new[0, 2] = 320.0
        K_new[1, 2] = 240.0
        maps = UndistortionMaps(K_new=K_new, map_x=map_x, map_y=map_y)
        return {"cam_a": maps, "cam_b": maps}

    def test_undistortion_applied(
        self,
        two_camera_videos: dict[str, Path],
        identity_undist_maps: dict[str, UndistortionMaps],
    ) -> None:
        """Frames with undistortion should differ from raw frames."""
        with VideoSet(two_camera_videos) as raw_vs:
            _, raw_frames = next(iter(raw_vs))

        with VideoSet(two_camera_videos, undistortion=identity_undist_maps) as und_vs:
            _, und_frames = next(iter(und_vs))

        # The shifted remap should produce different pixel values
        for cam in ("cam_a", "cam_b"):
            assert not np.array_equal(raw_frames[cam], und_frames[cam])


class TestVideoSetKNew:
    def test_k_new_returns_tensors(self, two_camera_videos: dict[str, Path]) -> None:
        K_new = torch.eye(3, dtype=torch.float32) * 500.0
        maps = UndistortionMaps(
            K_new=K_new,
            map_x=np.zeros((480, 640), dtype=np.float32),
            map_y=np.zeros((480, 640), dtype=np.float32),
        )
        vs = VideoSet(two_camera_videos, undistortion={"cam_a": maps, "cam_b": maps})
        result = vs.k_new
        assert set(result.keys()) == {"cam_a", "cam_b"}
        assert torch.equal(result["cam_a"], K_new)

    def test_k_new_raises_without_undistortion(
        self, two_camera_videos: dict[str, Path]
    ) -> None:
        vs = VideoSet(two_camera_videos)
        with pytest.raises(ValueError, match="not active"):
            vs.k_new  # noqa: B018


class TestVideoSetErrors:
    def test_empty_camera_map_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            VideoSet({})

    def test_missing_video_raises_on_enter(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "nonexistent.avi"
        vs = VideoSet({"cam": bad_path})
        with pytest.raises(RuntimeError, match="Cannot open video"):
            vs.__enter__()
