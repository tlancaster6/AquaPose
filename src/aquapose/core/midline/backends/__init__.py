"""Backend registry for the Midline stage.

Provides a factory function that resolves midline backend kind strings to
configured backend instances. Both backends are currently no-op stubs
pending Phase 37 YOLO model integration.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> object:
    """Create a midline backend by kind name.

    Both backends are currently no-op stubs that return midline=None for all
    detections. YOLO-seg and YOLO-pose models will be wired into these backends
    in Phase 37.

    Args:
        kind: Backend identifier. Supported values:

            - ``"segment_then_extract"`` (default): No-op stub awaiting
              YOLO-seg integration in Phase 37. All kwargs are accepted and
              ignored.
            - ``"direct_pose"``: No-op stub awaiting YOLO-pose integration
              in Phase 37. All kwargs are accepted and ignored.

        **kwargs: Forwarded to the backend constructor (currently ignored by
            both stubs).

    Returns:
        A configured backend instance with a
        ``process_frame(frame_idx, frame_dets, frames, camera_ids)`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
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
