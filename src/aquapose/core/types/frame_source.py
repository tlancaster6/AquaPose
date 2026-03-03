"""FrameSource protocol and VideoFrameSource concrete implementation.

Defines the canonical interface for multi-camera frame providers used by
DetectionStage, MidlineStage, and future chunk-windowed sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import cv2
import numpy as np
import torch

from aquapose.calibration.loader import (
    UndistortionMaps,
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
from aquapose.io.discovery import discover_camera_videos

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["FrameSource", "VideoFrameSource"]

logger = logging.getLogger(__name__)


@runtime_checkable
class FrameSource(Protocol):
    """Protocol for multi-camera frame providers.

    Implementations yield ``(frame_idx, {cam_id: frame_bgr})`` tuples and
    support random access via :meth:`read_frame`. Context manager protocol
    opens/releases underlying I/O handles.

    All implementations must be usable as::

        with source:
            for frame_idx, frames in source:
                ...
    """

    @property
    def camera_ids(self) -> list[str]:
        """Sorted list of camera identifiers."""
        ...

    def __len__(self) -> int:
        """Total number of frames available (capped by max_frames if set)."""
        ...

    def __iter__(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Yield ``(frame_idx, {cam_id: frame_bgr})`` for each frame."""
        ...

    def __enter__(self) -> FrameSource:
        """Open underlying I/O resources."""
        ...

    def __exit__(self, *exc: Any) -> None:
        """Release underlying I/O resources."""
        ...

    def read_frame(self, idx: int) -> dict[str, np.ndarray]:
        """Read a specific frame by index from all cameras.

        Args:
            idx: Zero-based frame index.

        Returns:
            Dictionary mapping camera_id to BGR uint8 frame array.
        """
        ...


class VideoFrameSource:
    """Multi-camera video reader with video discovery, calibration, and undistortion.

        Concrete implementation of :class:`FrameSource` that discovers video files,
        loads calibration data, computes undistortion maps, and yields synchronized
        undistorted frames. Absorbs all responsibilities previously split between
    the stage constructors.

        Args:
            video_dir: Directory containing per-camera video files (``*.avi`` or
                ``*.mp4``). Camera ID is derived from ``stem.split("-")[0]``.
            calibration_path: Path to the AquaCal calibration JSON file.
            max_frames: If set, iteration stops after this many frames and
                ``__len__`` returns at most this value.

        Raises:
            FileNotFoundError: If *video_dir* does not exist.
            ValueError: If no ``.avi``/``.mp4`` files are found in *video_dir*, or
                no cameras have both video and calibration data.
    """

    def __init__(
        self,
        video_dir: str | Path,
        calibration_path: str | Path,
        max_frames: int | None = None,
    ) -> None:
        self._video_dir = Path(video_dir)
        self._calibration_path = Path(calibration_path)
        self._max_frames = max_frames

        # Validate paths eagerly
        if not self._video_dir.exists():
            raise FileNotFoundError(f"video_dir does not exist: {self._video_dir}")

        # Discover camera videos
        video_paths = discover_camera_videos(self._video_dir)

        if not video_paths:
            raise ValueError(f"No .avi/.mp4 files found in {self._video_dir}")

        logger.info(
            "VideoFrameSource: found %d cameras: %s",
            len(video_paths),
            sorted(video_paths),
        )

        # Load calibration and compute undistortion maps
        calib = load_calibration_data(self._calibration_path)
        undist_maps: dict[str, UndistortionMaps] = {}
        for cam_id in video_paths:
            if cam_id not in calib.cameras:
                logger.warning("Camera %r not in calibration; skipping", cam_id)
                continue
            undist_maps[cam_id] = compute_undistortion_maps(calib.cameras[cam_id])

        # Only keep cameras with both video and calibration
        self._video_paths: dict[str, Path] = {
            cam_id: p for cam_id, p in video_paths.items() if cam_id in undist_maps
        }
        self._undist_maps: dict[str, UndistortionMaps] = undist_maps

        if not self._video_paths:
            raise ValueError("No cameras matched between video_dir and calibration.")

        self._camera_ids: list[str] = sorted(self._video_paths.keys())
        self._captures: dict[str, cv2.VideoCapture] = {}
        self._frame_count: int = 0

    @property
    def camera_ids(self) -> list[str]:
        """Sorted list of camera identifiers."""
        return list(self._camera_ids)

    @property
    def k_new(self) -> dict[str, torch.Tensor]:
        """Updated intrinsic matrices for undistorted images.

        Returns:
            Dictionary mapping camera_id to ``K_new`` tensor of shape (3, 3).
        """
        return {cam_id: maps.K_new for cam_id, maps in self._undist_maps.items()}

    def __enter__(self) -> VideoFrameSource:
        """Open all video captures and compute frame count."""
        frame_counts: list[int] = []
        for cam_id in self._camera_ids:
            path = self._video_paths[cam_id]
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                for c in self._captures.values():
                    c.release()
                self._captures.clear()
                raise RuntimeError(f"Cannot open video: {path}")
            self._captures[cam_id] = cap
            frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        raw_count = min(frame_counts) if frame_counts else 0
        if self._max_frames is not None:
            self._frame_count = min(raw_count, self._max_frames)
        else:
            self._frame_count = raw_count
        return self

    def __exit__(self, *exc: Any) -> None:
        """Release all video captures."""
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()

    def __len__(self) -> int:
        """Total number of frames (capped by max_frames when set)."""
        return self._frame_count

    def __iter__(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Yield ``(frame_idx, {cam_id: undistorted_frame})`` until EOF or max_frames."""
        frame_idx = 0
        while True:
            if self._max_frames is not None and frame_idx >= self._max_frames:
                break

            frames: dict[str, np.ndarray] = {}
            any_eof = False
            for cam_id in self._camera_ids:
                ret, frame = self._captures[cam_id].read()
                if not ret:
                    any_eof = True
                    break
                if cam_id in self._undist_maps:
                    frame = undistort_image(frame, self._undist_maps[cam_id])
                frames[cam_id] = frame

            if any_eof:
                break

            yield frame_idx, frames
            frame_idx += 1

    def read_frame(self, idx: int) -> dict[str, np.ndarray]:
        """Read a specific frame by index from all cameras.

        Seeks each capture to the requested position and reads one frame.

        Args:
            idx: Zero-based frame index.

        Returns:
            Dictionary mapping camera_id to BGR uint8 frame array.

        Raises:
            IndexError: If ``idx`` is out of range.
            RuntimeError: If a capture fails to read.
        """
        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._frame_count})")

        frames: dict[str, np.ndarray] = {}
        for cam_id in self._camera_ids:
            cap = self._captures[cam_id]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from camera {cam_id}")
            if cam_id in self._undist_maps:
                frame = undistort_image(frame, self._undist_maps[cam_id])
            frames[cam_id] = frame

        return frames
