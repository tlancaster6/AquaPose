"""DetectionStage — Stage 1 of the 5-stage AquaPose pipeline.

Reads video frames across all cameras, runs YOLO detection per frame,
and populates :class:`~aquapose.core.context.PipelineContext` with
per-frame per-camera detection results.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from aquapose.core.context import PipelineContext
from aquapose.core.detection.backends import get_backend
from aquapose.core.detection.types import Detection

__all__ = ["DetectionStage"]

logger = logging.getLogger(__name__)


class DetectionStage:
    """Stage 1: Reads videos, runs per-camera YOLO detection, populates context.

    The detection backend is created eagerly at construction time. A missing
    weights file raises :class:`FileNotFoundError` immediately.

    Camera discovery:
    - Glob ``*.avi`` and ``*.mp4`` in *video_dir*
    - Camera ID = ``stem.split("-")[0]``
    - All cameras in the input directory are processed (no internal filtering)

    Calibration data is loaded at construction and undistortion maps are
    computed per camera for use with :class:`~aquapose.io.video.VideoSet`.

    Args:
        video_dir: Directory containing per-camera video files.
        calibration_path: Path to the AquaCal calibration JSON file.
        detector_kind: Backend kind — currently only ``"yolo"`` is supported.
        stop_frame: If set, stop processing after this many frames.
        **detector_kwargs: Additional kwargs forwarded to the detector
            constructor (e.g. ``model_path``, ``conf_threshold``).

    Raises:
        FileNotFoundError: If *video_dir*, *calibration_path*, or detector
            weights do not exist.
        ValueError: If no valid camera videos are found, or *detector_kind*
            is not recognized.
    """

    def __init__(
        self,
        video_dir: str | Path,
        calibration_path: str | Path,
        detector_kind: str = "yolo",
        stop_frame: int | None = None,
        **detector_kwargs: Any,
    ) -> None:
        from aquapose.calibration.loader import (
            compute_undistortion_maps,
            load_calibration_data,
        )

        self._video_dir = Path(video_dir)
        self._calibration_path = Path(calibration_path)
        self._stop_frame = stop_frame

        # Validate paths eagerly
        if not self._video_dir.exists():
            raise FileNotFoundError(f"video_dir does not exist: {self._video_dir}")
        if not self._calibration_path.exists():
            raise FileNotFoundError(
                f"calibration_path does not exist: {self._calibration_path}"
            )

        # Discover camera videos
        video_paths: dict[str, Path] = {}
        for suffix in ("*.avi", "*.mp4"):
            for p in self._video_dir.glob(suffix):
                camera_id = p.stem.split("-")[0]
                video_paths[camera_id] = p

        if not video_paths:
            raise ValueError(f"No .avi/.mp4 files found in {self._video_dir}")

        logger.info(
            "DetectionStage: found %d cameras: %s",
            len(video_paths),
            sorted(video_paths),
        )

        # Load calibration and compute undistortion maps
        calib = load_calibration_data(self._calibration_path)
        undist_maps = {}
        for cam_id in video_paths:
            if cam_id not in calib.cameras:
                logger.warning("Camera %r not in calibration; skipping", cam_id)
                continue
            undist_maps[cam_id] = compute_undistortion_maps(calib.cameras[cam_id])

        # Only keep cameras that have both video and calibration
        self._video_paths: dict[str, Path] = {
            cam_id: p for cam_id, p in video_paths.items() if cam_id in undist_maps
        }
        self._undist_maps = undist_maps

        if not self._video_paths:
            raise ValueError("No cameras matched between video_dir and calibration.")

        # Eagerly load the detector backend (fail-fast on missing weights)
        self._detector = get_backend(detector_kind, **detector_kwargs)

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run detection across all cameras for all frames.

        Opens :class:`~aquapose.io.video.VideoSet`, iterates frames, and runs
        the detector on each camera's frame. Results are stored in *context*.

        Args:
            context: Accumulated pipeline state. This stage writes
                ``detections``, ``frame_count``, and ``camera_ids``.

        Returns:
            The same *context* object with detection fields populated.
        """
        from aquapose.io.video import VideoSet

        t0 = time.perf_counter()

        video_set = VideoSet(self._video_paths, undistortion=self._undist_maps)
        camera_ids = video_set.camera_ids

        detections_per_frame: list[dict[str, list[Detection]]] = []

        with video_set:
            for frame_idx, frames in video_set:
                if self._stop_frame is not None and frame_idx >= self._stop_frame:
                    break

                frame_dets: dict[str, list[Detection]] = {}
                for cam_id in camera_ids:
                    frame_dets[cam_id] = self._detector.detect(frames[cam_id])

                detections_per_frame.append(frame_dets)

        elapsed = time.perf_counter() - t0
        logger.info(
            "DetectionStage.run: %d frames, %d cameras, %.2fs",
            len(detections_per_frame),
            len(camera_ids),
            elapsed,
        )

        context.detections = detections_per_frame
        context.frame_count = len(detections_per_frame)
        context.camera_ids = camera_ids

        return context
