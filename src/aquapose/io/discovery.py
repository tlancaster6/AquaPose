"""Shared utility for discovering camera-video mappings from a directory.

Provides a single canonical implementation of camera-video discovery used
by both DetectionStage and MidlineStage to avoid duplicated glob logic.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["discover_camera_videos"]


def discover_camera_videos(video_dir: str | Path) -> dict[str, Path]:
    """Discover camera-video mappings from a directory of video files.

    Globs for ``*.avi`` and ``*.mp4`` files and extracts the camera ID from
    each filename stem using ``stem.split("-")[0]``.  Files with the same
    camera ID are deduplicated — the last glob match wins.

    Args:
        video_dir: Directory to search for video files.

    Returns:
        Dict mapping camera_id to video Path.  Empty dict if no videos found.
    """
    video_paths: dict[str, Path] = {}
    for suffix in ("*.avi", "*.mp4"):
        for p in Path(video_dir).glob(suffix):
            camera_id = p.stem.split("-")[0]
            video_paths[camera_id] = p
    return video_paths
