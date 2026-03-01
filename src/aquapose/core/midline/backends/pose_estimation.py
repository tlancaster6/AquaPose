"""Pose estimation backend for the Midline stage.

No-op stub pending Phase 37 YOLO-pose model integration. Returns
AnnotatedDetection(midline=None) for every detection. The backend
registers successfully under the ``"pose_estimation"`` backend name so the
pipeline remains runnable; actual keypoint regression and midline fitting
will be wired in Phase 37.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aquapose.core.midline.types import AnnotatedDetection
from aquapose.segmentation.detector import Detection

__all__ = ["PoseEstimationBackend"]

logger = logging.getLogger(__name__)


class PoseEstimationBackend:
    """YOLO-pose keypoint regression backend for the Midline stage (no-op stub).

    Instantiates without loading any model. All detections receive
    AnnotatedDetection(midline=None) until YOLO-pose is wired in Phase 37.

    Args:
        weights_path: Path to YOLO-pose model weights file. Stored for Phase 37
            implementation; currently ignored.
        device: PyTorch device string. Stored for Phase 37 implementation;
            currently ignored.
        n_points: Number of midline points to produce per detection. Stored
            for Phase 37 implementation; currently ignored.
        n_keypoints: Number of anatomical keypoints. Stored for Phase 37
            implementation; currently ignored.
        keypoint_t_values: Per-keypoint arc-fraction values in [0, 1]. Stored
            for Phase 37 implementation; currently ignored.
        confidence_floor: Minimum per-keypoint confidence to treat as visible.
            Stored for Phase 37 implementation; currently ignored.
        min_observed_keypoints: Minimum visible keypoints required to fit the
            spline. Stored for Phase 37 implementation; currently ignored.
        crop_size: Output crop size ``(width, height)`` in pixels. Stored for
            Phase 37 implementation; currently ignored.
        **kwargs: Additional kwargs accepted for API compatibility with
            get_backend() forwarding.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cuda",
        n_points: int = 15,
        n_keypoints: int = 6,
        keypoint_t_values: list[float] | None = None,
        confidence_floor: float = 0.3,
        min_observed_keypoints: int = 3,
        crop_size: tuple[int, int] = (128, 64),
        **kwargs: Any,
    ) -> None:
        self.weights_path = weights_path
        self.device = device
        self.n_points = n_points
        self.n_keypoints = n_keypoints
        self.keypoint_t_values = keypoint_t_values
        self.confidence_floor = confidence_floor
        self.min_observed_keypoints = min_observed_keypoints
        self.crop_size = crop_size
        logger.warning(
            "PoseEstimationBackend: no pose model loaded — all midlines will be None "
            "until YOLO-pose is wired in Phase 37."
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
