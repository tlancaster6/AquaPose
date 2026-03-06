"""Pure-function evaluator for the midline stage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from aquapose.core.types.midline import Midline2D


@dataclass(frozen=True)
class MidlineMetrics:
    """Metrics for the midline extraction stage.

    Attributes:
        mean_confidence: Mean per-point confidence across all midlines.
            Points with point_confidence=None are treated as 1.0.
        std_confidence: Standard deviation of per-point confidence.
        completeness: Fraction of midline points with confidence > 0.
            Points with point_confidence=None count as complete (1.0).
        temporal_smoothness: Mean L2 distance between consecutive-frame
            midline centroids per fish, averaged across all fish.
            Lower values indicate smoother motion. 0.0 when fewer than
            two frames are available for any fish.
        total_midlines: Total number of Midline2D objects evaluated.
    """

    mean_confidence: float
    std_confidence: float
    completeness: float
    temporal_smoothness: float
    total_midlines: int
    p10_confidence: float | None = None
    p50_confidence: float | None = None
    p90_confidence: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict representation.

        Returns:
            Dict with all fields cast to plain Python scalars.
        """
        return {
            "mean_confidence": float(self.mean_confidence),
            "std_confidence": float(self.std_confidence),
            "completeness": float(self.completeness),
            "temporal_smoothness": float(self.temporal_smoothness),
            "total_midlines": int(self.total_midlines),
            "p10_confidence": float(self.p10_confidence)
            if self.p10_confidence is not None
            else None,
            "p50_confidence": float(self.p50_confidence)
            if self.p50_confidence is not None
            else None,
            "p90_confidence": float(self.p90_confidence)
            if self.p90_confidence is not None
            else None,
        }


def evaluate_midline(frames: list[dict[int, Midline2D]]) -> MidlineMetrics:
    """Evaluate midline extraction quality from a sequence of per-frame dicts.

    Computes confidence statistics, midline completeness, and temporal
    smoothness from a list of per-frame mappings of fish_id to Midline2D.

    Args:
        frames: One dict per evaluated frame, mapping fish_id to Midline2D.
            List order determines frame order for temporal smoothness.

    Returns:
        MidlineMetrics with confidence stats, completeness, and smoothness.
    """
    if not frames:
        return MidlineMetrics(
            mean_confidence=0.0,
            std_confidence=0.0,
            completeness=0.0,
            temporal_smoothness=0.0,
            total_midlines=0,
        )

    all_confidences: list[float] = []
    total_midlines = 0

    # For temporal smoothness: fish_id -> list of centroids in frame order
    fish_centroids: dict[int, list[np.ndarray]] = defaultdict(list)

    for frame_dict in frames:
        for fish_id, midline in frame_dict.items():
            total_midlines += 1
            n = midline.points.shape[0]

            # Resolve confidence array (None -> all 1.0s)
            if midline.point_confidence is not None:
                conf = midline.point_confidence.astype(float)
            else:
                conf = np.ones(n, dtype=float)

            all_confidences.extend(conf.tolist())

            # Centroid for temporal smoothness
            centroid = midline.points.mean(axis=0).astype(float)
            fish_centroids[fish_id].append(centroid)

    conf_array = np.array(all_confidences, dtype=float)
    mean_confidence = float(np.mean(conf_array)) if len(conf_array) > 0 else 0.0
    std_confidence = float(np.std(conf_array)) if len(conf_array) > 0 else 0.0
    completeness = float(np.mean(conf_array > 0)) if len(conf_array) > 0 else 0.0

    # Temporal smoothness: mean L2 between consecutive centroids per fish
    per_fish_smoothness: list[float] = []
    for _fish_id, centroids in fish_centroids.items():
        if len(centroids) < 2:
            per_fish_smoothness.append(0.0)
            continue
        distances = [
            float(np.linalg.norm(centroids[i + 1] - centroids[i]))
            for i in range(len(centroids) - 1)
        ]
        per_fish_smoothness.append(float(np.mean(distances)))

    temporal_smoothness = (
        float(np.mean(per_fish_smoothness)) if per_fish_smoothness else 0.0
    )

    # Compute confidence percentiles (EVAL-02)
    if len(conf_array) > 0:
        conf_pcts = np.percentile(conf_array, [10, 50, 90])
        p10_conf: float | None = float(conf_pcts[0])
        p50_conf: float | None = float(conf_pcts[1])
        p90_conf: float | None = float(conf_pcts[2])
    else:
        p10_conf = None
        p50_conf = None
        p90_conf = None

    return MidlineMetrics(
        mean_confidence=mean_confidence,
        std_confidence=std_confidence,
        completeness=completeness,
        temporal_smoothness=temporal_smoothness,
        total_midlines=total_midlines,
        p10_confidence=p10_conf,
        p50_confidence=p50_conf,
        p90_confidence=p90_conf,
    )


__all__ = ["MidlineMetrics", "evaluate_midline"]
