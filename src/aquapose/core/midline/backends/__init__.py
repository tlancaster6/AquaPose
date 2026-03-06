"""Backend registry for the Midline stage.

Provides a factory function that resolves midline backend kind strings to
configured backend instances.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> object:
    """Create a midline backend by kind name.

    Args:
        kind: Backend identifier. Supported values:

            - ``"segmentation"`` (default): YOLO-seg mask inference with
              skeletonization to extract Midline2D objects.
            - ``"pose_estimation"``: YOLO-pose keypoint inference with
              spline interpolation to extract Midline2D objects.

        **kwargs: Forwarded to the backend constructor.

    Returns:
        A configured backend instance with a
        ``process_frame(frame_idx, frame_dets, frames, camera_ids)`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
    """
    if kind == "segmentation":
        from aquapose.core.midline.backends.segmentation import (
            SegmentationBackend,
        )

        return SegmentationBackend(**kwargs)

    if kind == "pose_estimation":
        from aquapose.core.midline.backends.pose_estimation import PoseEstimationBackend

        return PoseEstimationBackend(**kwargs)

    raise ValueError(
        f"Unknown midline backend kind: {kind!r}. "
        f"Supported kinds: ['segmentation', 'pose_estimation']"
    )
