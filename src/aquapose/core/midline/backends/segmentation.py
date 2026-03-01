"""Segmentation backend for the Midline stage.

No-op stub pending Phase 37 YOLO-seg model integration. Returns
AnnotatedDetection(midline=None) for every detection. The backend
registers successfully under the ``"segmentation"`` backend name so the
pipeline remains runnable; actual segmentation and midline extraction will
be wired in Phase 37.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aquapose.core.midline.types import AnnotatedDetection
from aquapose.segmentation.detector import Detection

__all__ = ["SegmentationBackend"]

logger = logging.getLogger(__name__)


class SegmentationBackend:
    """YOLO-seg segmentation backend for the Midline stage (no-op stub).

    Instantiates without loading any model. All detections receive
    AnnotatedDetection(midline=None) until YOLO-seg is wired in Phase 37.

    Args:
        weights_path: Path to YOLO-seg model weights file. Stored for Phase 37
            implementation; currently ignored.
        confidence_threshold: Minimum confidence for mask acceptance. Stored
            for Phase 37 implementation; currently ignored.
        n_points: Number of midline points to produce per detection. Stored
            for Phase 37 implementation; currently ignored.
        min_area: Minimum mask area (pixels) to attempt midline extraction.
            Stored for Phase 37 implementation; currently ignored.
        device: PyTorch device string. Stored for Phase 37 implementation;
            currently ignored.
        crop_size: Output crop size ``(width, height)`` in pixels. Stored for
            Phase 37 implementation; currently ignored.
        **kwargs: Additional kwargs accepted for API compatibility with
            get_backend() forwarding.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        confidence_threshold: float = 0.5,
        n_points: int = 15,
        min_area: int = 300,
        device: str = "cuda",
        crop_size: tuple[int, int] = (128, 64),
        **kwargs: Any,
    ) -> None:
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.n_points = n_points
        self.min_area = min_area
        self.device = device
        self.crop_size = crop_size
        logger.warning(
            "SegmentationBackend: no segmentation model loaded — all midlines "
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
