"""Backend registry for the Midline stage.

Provides a factory function that resolves midline backend kind strings to
configured backend instances. Both "segment_then_extract" and "direct_pose"
are fully implemented.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> object:
    """Create a midline backend by kind name.

    Args:
        kind: Backend identifier. Supported values:

            - ``"segment_then_extract"`` (default): U-Net segmentation then
              skeletonization + BFS midline extraction. Accepted kwargs:
              ``weights_path``, ``confidence_threshold``, ``n_points``,
              ``min_area``, ``device``.
            - ``"direct_pose"``: Keypoint regression backend. Uses a
              _PoseModel (U-Net encoder + regression head) to directly predict
              anatomical keypoints, fits a CubicSpline, and resamples to
              n_points. Accepted kwargs: ``weights_path``, ``device``,
              ``n_points``, ``n_keypoints``, ``keypoint_t_values``,
              ``confidence_floor``, ``min_observed_keypoints``, ``crop_size``.

        **kwargs: Forwarded to the backend constructor.

    Returns:
        A configured backend instance with a
        ``process_frame(frame_idx, frame_dets, frames, camera_ids)`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
        FileNotFoundError: If required weights files do not exist.
    """
    if kind == "segment_then_extract":
        from aquapose.core.midline.backends.segment_then_extract import (
            SegmentThenExtractBackend,
        )

        return SegmentThenExtractBackend(**kwargs)

    if kind == "direct_pose":
        from aquapose.core.midline.backends.direct_pose import DirectPoseBackend

        return DirectPoseBackend(**kwargs)

    raise ValueError(
        f"Unknown midline backend kind: {kind!r}. "
        f"Supported kinds: ['segment_then_extract', 'direct_pose']"
    )
