"""DetectionStage — Stage 1 of the 5-stage AquaPose pipeline.

Reads video frames across all cameras via an injected FrameSource, runs YOLO
detection per frame, and populates :class:`~aquapose.core.context.PipelineContext`
with per-frame per-camera detection results.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aquapose.core.context import PipelineContext
from aquapose.core.detection.backends import get_backend
from aquapose.core.types.detection import Detection

if TYPE_CHECKING:
    from aquapose.core.types.frame_source import FrameSource

__all__ = ["DetectionStage"]

logger = logging.getLogger(__name__)


class DetectionStage:
    """Stage 1: Reads frames from a FrameSource, runs per-camera YOLO detection.

    The detection backend is created eagerly at construction time. A missing
    weights file raises :class:`FileNotFoundError` immediately.

    Frame I/O is delegated to the injected ``frame_source``, which handles
    video discovery, calibration loading, and undistortion. The stage only
    concerns itself with running the detector on the provided frames.

    Args:
        frame_source: Multi-camera frame provider (e.g. VideoFrameSource).
            Must satisfy the FrameSource protocol.
        detector_kind: Backend kind — currently ``"yolo"`` or ``"yolo_obb"``.
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
        **detector_kwargs: Any,
    ) -> None:
        self._frame_source = frame_source

        # Eagerly load the detector backend (fail-fast on missing weights)
        self._detector = get_backend(detector_kind, **detector_kwargs)

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run detection across all cameras for all frames.

        Opens the frame source as a context manager, iterates frames, and runs
        the detector on each camera's frame. Results are stored in *context*.

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
                frame_dets: dict[str, list[Detection]] = {}
                for cam_id in camera_ids:
                    if cam_id not in frames:
                        logger.warning(
                            "Camera %s: frame missing, skipping detection",
                            cam_id,
                        )
                        frame_dets[cam_id] = []
                        continue
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
