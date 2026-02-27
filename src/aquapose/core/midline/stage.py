"""MidlineStage — Stage 2 of the 5-stage AquaPose pipeline.

Reads detection bounding boxes from Stage 1, crops and segments each detection
via U-Net, then extracts 15-point 2D midlines with half-widths via
skeletonization + BFS pruning. Populates PipelineContext.annotated_detections.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from aquapose.core.context import PipelineContext
from aquapose.core.midline.backends import get_backend

__all__ = ["MidlineStage"]

logger = logging.getLogger(__name__)


class MidlineStage:
    """Stage 2: Segments detections and extracts 2D midlines, populates context.

    Runs after DetectionStage (Stage 1). For each frame, for each camera, crops
    each detection and runs the configured backend to produce binary masks and
    15-point 2D midlines. All detections are annotated regardless of tracking
    state (unlike v1.0 which only processed tracked fish).

    The backend is created eagerly at construction time. A missing weights file
    raises :class:`FileNotFoundError` immediately.

    Camera and video discovery:
    - Glob ``*.avi`` and ``*.mp4`` in *video_dir*
    - Camera ID = ``stem.split("-")[0]``
    - All cameras in the input directory are processed (no internal filtering)

    Calibration data is loaded at construction for undistortion map computation.

    Args:
        video_dir: Directory containing per-camera video files.
        calibration_path: Path to the AquaCal calibration JSON file.
        weights_path: Path to U-Net model weights file. Raises FileNotFoundError
            if the path does not exist (None uses pretrained ImageNet encoder).
        confidence_threshold: Minimum confidence for mask acceptance.
        backend: Backend kind — "segment_then_extract" (default) or
            "direct_pose" (raises NotImplementedError).
        device: PyTorch device string (e.g. "cuda", "cpu").
        n_points: Number of midline points per detection.
        min_area: Minimum mask area (pixels) to attempt midline extraction.

    Raises:
        FileNotFoundError: If *video_dir*, *calibration_path*, or U-Net weights
            do not exist.
        ValueError: If no valid camera videos are found.
        NotImplementedError: If *backend* is ``"direct_pose"``.

    """

    def __init__(
        self,
        video_dir: str | Path,
        calibration_path: str | Path,
        weights_path: str | None = None,
        confidence_threshold: float = 0.5,
        backend: str = "segment_then_extract",
        device: str = "cuda",
        n_points: int = 15,
        min_area: int = 300,
    ) -> None:
        from aquapose.calibration.loader import (
            compute_undistortion_maps,
            load_calibration_data,
        )

        self._video_dir = Path(video_dir)
        self._calibration_path = Path(calibration_path)

        # Validate paths eagerly
        if not self._video_dir.exists():
            raise FileNotFoundError(f"video_dir does not exist: {self._video_dir}")
        if not self._calibration_path.exists():
            raise FileNotFoundError(
                f"calibration_path does not exist: {self._calibration_path}",
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
            "MidlineStage: found %d cameras: %s",
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

        # Only keep cameras with both video and calibration
        self._video_paths: dict[str, Path] = {
            cam_id: p for cam_id, p in video_paths.items() if cam_id in undist_maps
        }
        self._undist_maps = undist_maps

        if not self._video_paths:
            raise ValueError("No cameras matched between video_dir and calibration.")

        # Eagerly create backend (fail-fast on missing weights, unsupported kind)
        self._backend = get_backend(
            backend,
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            n_points=n_points,
            min_area=min_area,
            device=device,
        )

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run midline extraction across all cameras for all frames.

        Reads ``context.detections`` and ``context.camera_ids`` set by Stage 1.
        Re-opens videos to re-read frames (same pattern as v1.0: raw frames
        are not stored between stages to keep memory bounded).

        Populates ``context.annotated_detections`` with per-frame per-camera
        AnnotatedDetection lists — one entry per original Detection from Stage 1.

        Args:
            context: Accumulated pipeline state from prior stages. Must have
                ``detections`` and ``camera_ids`` populated by Stage 1.

        Returns:
            The same *context* object with ``annotated_detections`` populated.

        Raises:
            ValueError: If ``context.detections`` or ``context.camera_ids`` is
                None (Stage 1 has not yet run).

        """
        from aquapose.io.video import VideoSet

        detections = context.get("detections")
        camera_ids = context.get("camera_ids")

        t0 = time.perf_counter()

        video_set = VideoSet(self._video_paths, undistortion=self._undist_maps)
        annotated_per_frame: list[dict[str, list]] = []

        with video_set:
            for frame_idx, frames in video_set:
                if frame_idx >= len(detections):  # type: ignore[arg-type]
                    break

                frame_dets = detections[frame_idx]  # type: ignore[index]
                annotated = self._backend.process_frame(  # type: ignore[union-attr]
                    frame_idx=frame_idx,
                    frame_dets=frame_dets,
                    frames=frames,
                    camera_ids=camera_ids,  # type: ignore[arg-type]
                )
                annotated_per_frame.append(annotated)

        elapsed = time.perf_counter() - t0
        logger.info(
            "MidlineStage.run: %d frames, %d cameras, %.2fs",
            len(annotated_per_frame),
            len(camera_ids),  # type: ignore[arg-type]
            elapsed,
        )

        context.annotated_detections = annotated_per_frame
        return context
