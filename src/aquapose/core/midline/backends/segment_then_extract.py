"""Segment-then-extract backend stub for the Midline stage.

No-op stub pending Phase 37 YOLO-seg model integration. Returns
AnnotatedDetection(midline=None) for every detection. The backend
registers successfully under the existing backend name so the pipeline
remains runnable; actual segmentation and midline extraction will be
wired in Phase 37.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aquapose.core.midline.types import AnnotatedDetection
from aquapose.segmentation.detector import Detection

__all__ = ["SegmentThenExtractBackend"]

logger = logging.getLogger(__name__)


class SegmentThenExtractBackend:
    """Segment-then-extract midline backend (no-op stub).

    Instantiates without loading any model. All detections receive
    AnnotatedDetection(midline=None) until YOLO-seg is wired in Phase 37.

    Args:
        **kwargs: Accepted and ignored for API compatibility with get_backend()
            keyword forwarding (e.g. weights_path, confidence_threshold,
            n_points, min_area, device).
    """

    def __init__(self, **kwargs: Any) -> None:
        logger.warning(
            "SegmentThenExtractBackend: no segmentation model loaded — all midlines "
            "will be None until YOLO-seg is wired in Phase 37."
        )

    def process_frame(
        self,
        frame_idx: int,
        frame_dets: dict[str, list[Detection]],
        frames: dict[str, np.ndarray],
        camera_ids: list[str],
    ) -> dict[str, list[AnnotatedDetection]]:
        """Return AnnotatedDetection(midline=None) for every detection.

        Args:
            frame_idx: Current frame index (used for AnnotatedDetection metadata).
            frame_dets: Per-camera detection lists from Stage 1.
            frames: Per-camera undistorted frame images (BGR uint8). Unused.
            camera_ids: Active camera identifiers.

        Returns:
            Per-camera list of AnnotatedDetection objects, one per input
            detection. All midline fields are None.
        """
        annotated: dict[str, list[AnnotatedDetection]] = {}

        for cam_id in camera_ids:
            cam_dets = frame_dets.get(cam_id, [])
            annotated[cam_id] = [
                AnnotatedDetection(
                    detection=det,
                    mask=None,
                    crop_region=None,
                    midline=None,
                    camera_id=cam_id,
                    frame_index=frame_idx,
                )
                for det in cam_dets
            ]

        return annotated
