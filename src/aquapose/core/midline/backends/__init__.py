"""Backend registry for the Midline stage.

Provides a factory function that resolves midline backend kind strings to
configured backend instances. Currently "segment_then_extract" is fully
implemented; "direct_pose" is a stub raising NotImplementedError.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> object:
    """Create a midline backend by kind name.

    Args:
        kind: Backend identifier. Supported values:

            - ``"segment_then_extract"`` (default): U-Net segmentation then
              skeletonization + BFS midline extraction.
            - ``"direct_pose"``: Planned key-point regression backend (stub).

        **kwargs: Forwarded to the backend constructor. For
            ``"segment_then_extract"``, accepted kwargs are: ``weights_path``,
            ``confidence_threshold``, ``n_points``, ``min_area``, ``device``.

    Returns:
        A configured backend instance with a
        ``process_frame(frame_idx, frame_dets, frames, camera_ids)`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
        NotImplementedError: If *kind* is ``"direct_pose"`` (stub).
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
