"""Direct pose estimation backend stub for the Midline stage.

This is a planned alternative to the segment-then-extract backend that will
directly estimate fish midlines from detection bounding boxes using a pose
estimation model (e.g. key-point regression), bypassing U-Net segmentation.

Currently raises NotImplementedError to prove the backend registry pattern
while reserving the interface for future implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["DirectPoseBackend"]


class DirectPoseBackend:
    """Planned direct pose estimation backend (not yet implemented).

    Future implementation: accepts detection bounding boxes and directly
    regresses 15 midline key-points without an intermediate binary mask.
    This backend is expected to be faster than segment-then-extract at
    inference time but requires a separate key-point training stage.

    Args:
        **kwargs: Accepted but ignored — future parameters for key-point
            model weights path, device, etc.

    Raises:
        NotImplementedError: Always, at construction time.
    """

    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError(
            "DirectPoseBackend is a planned alternative backend for the Midline stage "
            "and has not yet been implemented. Use backend='segment_then_extract' "
            "(the default) instead."
        )

    def process_frame(
        self,
        frame_idx: int,
        frame_dets: dict[str, list],
        frames: dict[str, np.ndarray],
        camera_ids: list[str],
    ) -> dict[str, list]:
        """Process a single frame (not implemented).

        Args:
            frame_idx: Frame index.
            frame_dets: Per-camera detection lists.
            frames: Per-camera frame images.
            camera_ids: Active camera identifiers.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "DirectPoseBackend is not yet implemented. "
            "Use backend='segment_then_extract' instead."
        )
