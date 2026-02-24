"""Compatibility shims for legacy synthetic detection and tracking interfaces.

``generate_synthetic_detections`` now delegates to
``generate_detection_dataset`` in ``detection.py``.

``generate_synthetic_tracks`` remains a stub — track generation is a tracker
output, not an input, so it is not part of the synthetic data generation
pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.synthetic.detection import SyntheticDataset
    from aquapose.synthetic.trajectory import TrajectoryResult


def generate_synthetic_detections(
    trajectory: TrajectoryResult,
    models: dict[str, RefractiveProjectionModel],
    **kwargs: Any,
) -> SyntheticDataset:
    """Generate synthetic Detection objects from a trajectory.

    Delegates to :func:`~aquapose.synthetic.detection.generate_detection_dataset`.
    See that function for full parameter documentation.

    Args:
        trajectory: Multi-fish 3D trajectory from generate_trajectories().
        models: Dict mapping camera ID to RefractiveProjectionModel.
        **kwargs: Forwarded to generate_detection_dataset (noise_config,
            det_config, random_seed).

    Returns:
        SyntheticDataset with per-frame, per-camera Detection objects.
    """
    from aquapose.synthetic.detection import generate_detection_dataset

    return generate_detection_dataset(trajectory, models, **kwargs)


def generate_synthetic_tracks(*args: object, **kwargs: object) -> object:
    """Generate synthetic FishTrack objects for testing midline extraction.

    Track generation is a tracker output, not a synthetic input — FishTrack
    objects are produced by FishTracker.update() consuming detection streams.
    This stub is retained for API compatibility but is not implemented.

    Args:
        *args: Placeholder positional arguments.
        **kwargs: Placeholder keyword arguments.

    Returns:
        Not implemented.

    Raises:
        NotImplementedError: Always raised. Use FishTracker.update() with
            a SyntheticDataset to generate tracks.
    """
    raise NotImplementedError(
        "generate_synthetic_tracks is not implemented. "
        "Use FishTracker.update() with a SyntheticDataset to generate tracks."
    )
