"""DetectionStage — Stage 1 of the 5-stage AquaPose pipeline.

Reads video frames across all cameras via an injected FrameSource, runs YOLO
detection per frame, and populates :class:`~aquapose.core.context.PipelineContext`
with per-frame per-camera detection results.

When a ``detection_batch_frames`` limit is configured (default: 0 = no limit),
all camera frames for a timestep are submitted in a single batched
``detect_batch()`` call, wrapped by :func:`predict_with_oom_retry` for
automatic batch-size halving on CUDA OOM.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aquapose.core.context import PipelineContext
from aquapose.core.detection.backends import get_backend
from aquapose.core.inference import BatchState, predict_with_oom_retry
from aquapose.core.types.detection import Detection

if TYPE_CHECKING:
    from aquapose.core.types.frame_source import FrameSource

__all__ = ["DetectionStage"]

logger = logging.getLogger(__name__)


class DetectionStage:
    """Stage 1: Reads frames from a FrameSource, runs batched YOLO detection.

    The detection backend is created eagerly at construction time. A missing
    weights file raises :class:`FileNotFoundError` immediately.

    Frame I/O is delegated to the injected ``frame_source``, which handles
    video discovery, calibration loading, and undistortion. The stage only
    concerns itself with running the detector on the provided frames.

    All camera frames for a given timestep are collected and submitted to
    ``detect_batch()`` in a single call (wrapped by
    :func:`~aquapose.core.inference.predict_with_oom_retry`).  Missing cameras
    receive an empty detection list.

    Args:
        frame_source: Multi-camera frame provider (e.g. VideoFrameSource).
            Must satisfy the FrameSource protocol.
        detector_kind: Backend kind — currently ``"yolo"`` or ``"yolo_obb"``.
        detection_batch_frames: Maximum frames per batch call. ``0`` means
            no limit (all cameras in one call).
        **detector_kwargs: Additional kwargs forwarded to the detector
            constructor (e.g. ``weights_path``, ``conf_threshold``).

    Raises:
        FileNotFoundError: If detector weights do not exist.
        ValueError: If *detector_kind* is not recognized.

    """

    def __init__(
        self,
        frame_source: FrameSource,
        detector_kind: str = "yolo",
        detection_batch_frames: int = 0,
        **detector_kwargs: Any,
    ) -> None:
        self._frame_source = frame_source
        self._batch_size = detection_batch_frames
        self._batch_state = BatchState()

        # Eagerly load the detector backend (fail-fast on missing weights)
        self._detector = get_backend(detector_kind, **detector_kwargs)

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run batched detection across all cameras for all frames.

        Opens the frame source as a context manager, iterates frames, collects
        all available camera frames per timestep, and runs batched detection
        via :func:`predict_with_oom_retry`. Results are stored in *context*.

        Args:
            context: Accumulated pipeline state. This stage writes
                ``detections``, ``frame_count``, and ``camera_ids``.

        Returns:
            The same *context* object with detection fields populated.

        """
        t0 = time.perf_counter()

        camera_ids = self._frame_source.camera_ids
        detections_per_frame: list[dict[str, list[Detection]]] = []

        with self._frame_source:
            for _frame_idx, frames in self._frame_source:
                # Collect available camera frames in deterministic order
                ordered_items: list[tuple[str, Any]] = []
                missing_cams: list[str] = []
                for cam_id in camera_ids:
                    if cam_id in frames:
                        ordered_items.append((cam_id, frames[cam_id]))
                    else:
                        logger.warning(
                            "Camera %s: frame missing, skipping detection",
                            cam_id,
                        )
                        missing_cams.append(cam_id)

                # Batched detection via OOM retry
                if ordered_items:
                    frame_list = [f for _, f in ordered_items]
                    batch_results = predict_with_oom_retry(
                        self._detector.detect_batch,
                        frame_list,
                        self._batch_size,
                        self._batch_state,
                    )
                else:
                    batch_results = []

                # Map results back to camera IDs
                frame_dets: dict[str, list[Detection]] = {}
                for i, (cam_id, _) in enumerate(ordered_items):
                    frame_dets[cam_id] = batch_results[i]
                for cam_id in missing_cams:
                    frame_dets[cam_id] = []

                detections_per_frame.append(frame_dets)

        # Log OOM recommendation if batch size was reduced
        if self._batch_state.oom_occurred:
            logger.info(
                "Detection batch size was reduced to %d due to CUDA OOM. "
                "Consider setting detection.detection_batch_frames=%d in config.",
                self._batch_state.effective_batch_size,
                self._batch_state.effective_batch_size,
            )

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
