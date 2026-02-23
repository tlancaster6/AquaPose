"""Stub functions for future synthetic Detection and FishTrack generation.

These stubs provide placeholders for future expansion of the synthetic data
module. When implemented, they will generate synthetic detection and tracking
data to enable unit testing of the full pipeline (stages 1-4) without real
video data.
"""

from __future__ import annotations


def generate_synthetic_detections(*args: object, **kwargs: object) -> object:
    """Generate synthetic Detection objects for testing midline extraction.

    When implemented, this function will produce synthetic Detection objects
    (bounding boxes and masks) corresponding to given FishConfig objects
    projected through a set of camera models. This would enable testing of
    the full Stage 1-4 pipeline (detection, segmentation, tracking, midline
    extraction) without real video data.

    Args:
        *args: Placeholder positional arguments.
        **kwargs: Placeholder keyword arguments.

    Returns:
        Not implemented.

    Raises:
        NotImplementedError: Always raised. This stub is not yet implemented.
    """
    raise NotImplementedError(
        "Stub: synthetic Detection generation for testing midline extraction"
    )


def generate_synthetic_tracks(*args: object, **kwargs: object) -> object:
    """Generate synthetic FishTrack objects for testing midline extraction.

    When implemented, this function will produce synthetic FishTrack objects
    representing tracked fish across multiple frames. Each track would include
    camera_detections mapped to synthetic Detection objects produced by
    generate_synthetic_detections. This would enable end-to-end testing of the
    tracking and midline extraction stages without real video data.

    Args:
        *args: Placeholder positional arguments.
        **kwargs: Placeholder keyword arguments.

    Returns:
        Not implemented.

    Raises:
        NotImplementedError: Always raised. This stub is not yet implemented.
    """
    raise NotImplementedError(
        "Stub: synthetic FishTrack generation for testing midline extraction with tracking"
    )
