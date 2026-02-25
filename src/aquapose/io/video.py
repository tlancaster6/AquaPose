"""Synchronized multi-camera video reader with optional undistortion."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from aquapose.calibration.loader import (
    UndistortionMaps,
    compute_undistortion_maps,
    undistort_image,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from aquapose.calibration.loader import CalibrationData


class VideoSet:
    """Synchronized multi-camera video reader with optional undistortion.

    Provides context-managed, iterable access to multi-camera video frames.
    When undistortion is enabled, frames are automatically remapped and the
    corrected intrinsic matrix (``K_new``) is exposed for downstream geometry.

    Args:
        camera_map: Mapping from camera_id to video file path.
        undistortion: Undistortion source. Can be:
            - ``CalibrationData``: maps are computed internally per camera.
            - ``dict[str, UndistortionMaps]``: precomputed maps used directly.
            - ``None``: no undistortion (raw frames).

    Raises:
        ValueError: If ``camera_map`` is empty.
    """

    def __init__(
        self,
        camera_map: dict[str, Path],
        undistortion: CalibrationData | dict[str, UndistortionMaps] | None = None,
    ) -> None:
        if not camera_map:
            raise ValueError("camera_map must not be empty")

        self._camera_map = dict(camera_map)
        self._camera_ids = sorted(camera_map.keys())
        self._captures: dict[str, cv2.VideoCapture] = {}
        self._frame_count: int = 0

        # Resolve undistortion maps
        self._undist_maps: dict[str, UndistortionMaps] | None = None
        if undistortion is not None:
            if isinstance(undistortion, dict):
                self._undist_maps = undistortion
            else:
                # CalibrationData — compute maps for cameras we have videos for
                self._undist_maps = {}
                for cam_id in self._camera_ids:
                    if cam_id in undistortion.cameras:
                        self._undist_maps[cam_id] = compute_undistortion_maps(
                            undistortion.cameras[cam_id]
                        )

    @property
    def camera_ids(self) -> list[str]:
        """Sorted list of camera identifiers."""
        return list(self._camera_ids)

    @property
    def k_new(self) -> dict[str, torch.Tensor]:
        """Updated intrinsic matrices for undistorted images.

        Returns:
            Dictionary mapping camera_id to ``K_new`` tensor of shape (3, 3).

        Raises:
            ValueError: If undistortion is not active.
        """
        if self._undist_maps is None:
            raise ValueError("Undistortion is not active; K_new is unavailable")
        return {cam_id: maps.K_new for cam_id, maps in self._undist_maps.items()}

    def __enter__(self) -> VideoSet:
        """Open all video captures and compute frame count."""
        frame_counts: list[int] = []
        for cam_id in self._camera_ids:
            path = self._camera_map[cam_id]
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                # Release any already-opened captures
                for c in self._captures.values():
                    c.release()
                self._captures.clear()
                raise RuntimeError(f"Cannot open video: {path}")
            self._captures[cam_id] = cap
            frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        self._frame_count = min(frame_counts) if frame_counts else 0
        return self

    def __exit__(self, *exc: object) -> None:
        """Release all video captures."""
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()

    def __len__(self) -> int:
        """Minimum frame count across all cameras."""
        return self._frame_count

    def __iter__(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Yield ``(frame_idx, {cam_id: frame_bgr})`` until first EOF.

        Frames are undistorted when undistortion is active.
        """
        frame_idx = 0
        while True:
            frames: dict[str, np.ndarray] = {}
            any_eof = False
            for cam_id in self._camera_ids:
                ret, frame = self._captures[cam_id].read()
                if not ret:
                    any_eof = True
                    break
                if self._undist_maps is not None and cam_id in self._undist_maps:
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
            if self._undist_maps is not None and cam_id in self._undist_maps:
                frame = undistort_image(frame, self._undist_maps[cam_id])
            frames[cam_id] = frame

        return frames
