"""Synthetic frame generation utilities for video-free visualization."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

__all__ = ["synthetic_frame_iter"]


def synthetic_frame_iter(
    camera_ids: list[str],
    frame_sizes: dict[str, tuple[int, int]],
    n_frames: int,
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """Yield ``(frame_idx, {cam_id: black_frame})`` for *n_frames* frames.

    Provides the same iteration interface as
    :class:`~aquapose.io.video.VideoSet` but with black (zero-filled) BGR
    frames, enabling visualization observers to render overlays without
    real video files.

    Args:
        camera_ids: Ordered list of camera identifiers to include.
        frame_sizes: Mapping of camera_id to ``(width, height)`` in pixels.
            Cameras missing from this dict are skipped.
        n_frames: Total number of frames to yield.

    Yields:
        Tuple of ``(frame_idx, frames_dict)`` where *frames_dict* maps each
        camera_id to a fresh black ``(H, W, 3)`` uint8 array.
    """
    # Pre-allocate template frames (one per camera).
    templates = {
        cam_id: np.zeros((h, w, 3), dtype=np.uint8)
        for cam_id in camera_ids
        if cam_id in frame_sizes
        for w, h in [frame_sizes[cam_id]]
    }

    for frame_idx in range(n_frames):
        # Copy so each frame is independently mutable by the caller.
        yield frame_idx, {cid: f.copy() for cid, f in templates.items()}
