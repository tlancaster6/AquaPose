"""Pure-function evaluator for the tracking stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aquapose.core.tracking.types import Tracklet2D


@dataclass(frozen=True)
class TrackingMetrics:
    """Aggregated metrics for the tracking stage.

    Attributes:
        track_count: Number of tracklets across all cameras.
        length_median: Median track length (number of frames).
        length_mean: Mean track length.
        length_min: Minimum track length.
        length_max: Maximum track length.
        coast_frequency: Overall fraction of (tracklet, frame) pairs that are
            coasted (i.e., position interpolated, not directly detected).
        detection_coverage: Fraction of (tracklet, frame) pairs that are
            directly detected (1.0 - coast_frequency).
    """

    track_count: int
    length_median: float
    length_mean: float
    length_min: int
    length_max: int
    coast_frequency: float
    detection_coverage: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict of all metric fields.

        Returns:
            Dict with Python-native types (int, float).
        """
        return {
            "track_count": int(self.track_count),
            "length_median": float(self.length_median),
            "length_mean": float(self.length_mean),
            "length_min": int(self.length_min),
            "length_max": int(self.length_max),
            "coast_frequency": float(self.coast_frequency),
            "detection_coverage": float(self.detection_coverage),
        }


def evaluate_tracking(tracklets: list[Tracklet2D]) -> TrackingMetrics:
    """Compute tracking-stage metrics from a list of Tracklet2D objects.

    Args:
        tracklets: List of Tracklet2D objects from all cameras, representing all
            tracks produced by the 2D tracker across the evaluated sequence.

    Returns:
        TrackingMetrics with aggregated counts, length statistics, coast
        frequency, and detection coverage.
    """
    if not tracklets:
        return TrackingMetrics(
            track_count=0,
            length_median=0.0,
            length_mean=0.0,
            length_min=0,
            length_max=0,
            coast_frequency=0.0,
            detection_coverage=0.0,
        )

    track_count = len(tracklets)
    lengths = [len(t.frames) for t in tracklets]
    lengths_arr = np.array(lengths, dtype=float)

    length_median = float(np.median(lengths_arr))
    length_mean = float(np.mean(lengths_arr))
    length_min = int(min(lengths))
    length_max = int(max(lengths))

    # Count coasted frames and total frames across all tracklets
    total_frames = sum(len(t.frames) for t in tracklets)
    total_coasted = sum(
        sum(1 for s in t.frame_status if s == "coasted") for t in tracklets
    )

    if total_frames == 0:
        coast_frequency = 0.0
        detection_coverage = 0.0
    else:
        coast_frequency = float(total_coasted / total_frames)
        detection_coverage = 1.0 - coast_frequency

    return TrackingMetrics(
        track_count=track_count,
        length_median=length_median,
        length_mean=length_mean,
        length_min=length_min,
        length_max=length_max,
        coast_frequency=coast_frequency,
        detection_coverage=detection_coverage,
    )
