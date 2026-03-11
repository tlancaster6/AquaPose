"""Pure-function evaluator for the association stage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from aquapose.core.types.reconstruction import MidlineSet

DEFAULT_GRID: dict[str, list[float]] = {
    "ray_distance_threshold": [0.01, 0.02, 0.03],
    "score_min": [0.03, 0.15, 0.30],
    "keypoint_confidence_floor": [0.2, 0.3, 0.4],
    "eviction_reproj_threshold": [0.02, 0.03, 0.05],
    "leiden_resolution": [0.5, 1.0, 2.0],
    "early_k": [5.0, 10.0, 30.0],
}


@dataclass(frozen=True)
class AssociationMetrics:
    """Metrics for the cross-view association stage.

    Attributes:
        fish_yield_ratio: Mean fish observed per frame divided by n_animals.
            1.0 means all fish were observed every frame on average.
        singleton_rate: Fraction of fish-frame observations seen by only one
            camera. High singleton rate limits reconstruction coverage.
        camera_distribution: Maps n_cameras to the count of observations
            with that many camera views. Keys are the number of cameras.
        total_fish_observations: Total (fish, frame) observations across all
            evaluated frames.
        frames_evaluated: Number of frames evaluated.
    """

    fish_yield_ratio: float
    singleton_rate: float
    camera_distribution: dict[int, int]
    total_fish_observations: int
    frames_evaluated: int
    p50_camera_count: float | None = None
    p90_camera_count: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict representation.

        Returns:
            Dict with all fields; camera_distribution keys are converted to
            strings for JSON compatibility.
        """
        return {
            "fish_yield_ratio": float(self.fish_yield_ratio),
            "singleton_rate": float(self.singleton_rate),
            "camera_distribution": {
                str(k): int(v) for k, v in self.camera_distribution.items()
            },
            "total_fish_observations": int(self.total_fish_observations),
            "frames_evaluated": int(self.frames_evaluated),
            "p50_camera_count": float(self.p50_camera_count)
            if self.p50_camera_count is not None
            else None,
            "p90_camera_count": float(self.p90_camera_count)
            if self.p90_camera_count is not None
            else None,
        }


def evaluate_association(
    midline_sets: list[MidlineSet],
    n_animals: int,
) -> AssociationMetrics:
    """Evaluate cross-view association quality from a sequence of MidlineSets.

    Computes fish yield ratio, singleton rate, and camera coverage distribution
    from a list of per-frame MidlineSets. Logic migrated from
    tune_association.py::_compute_association_metrics().

    Args:
        midline_sets: One MidlineSet per evaluated frame, where each maps
            fish_id -> camera_id -> Midline2D.
        n_animals: Expected number of fish in the scene.

    Returns:
        AssociationMetrics with yield, singleton, and distribution stats.
    """
    camera_distribution: dict[int, int] = defaultdict(int)
    total_observations = 0
    singleton_count = 0
    multi_view_count = 0
    all_cam_counts: list[int] = []

    for midline_set in midline_sets:
        for _fish_id, cam_map in midline_set.items():
            n_cams = len(cam_map)
            camera_distribution[n_cams] += 1
            total_observations += 1
            all_cam_counts.append(n_cams)
            if n_cams == 1:
                singleton_count += 1
            else:
                multi_view_count += 1

    frames = len(midline_sets)
    singleton_rate = singleton_count / max(total_observations, 1)
    fish_per_frame = multi_view_count / max(frames, 1)
    yield_ratio = fish_per_frame / max(n_animals, 1)

    # Compute camera count percentiles (EVAL-03)
    if len(all_cam_counts) > 0:
        cam_pcts = np.percentile(all_cam_counts, [50, 90])
        p50_cam: float | None = float(cam_pcts[0])
        p90_cam: float | None = float(cam_pcts[1])
    else:
        p50_cam = None
        p90_cam = None

    # Convert defaultdict to plain dict for frozen dataclass storage
    return AssociationMetrics(
        fish_yield_ratio=float(yield_ratio),
        singleton_rate=float(singleton_rate),
        camera_distribution=dict(camera_distribution),
        total_fish_observations=total_observations,
        frames_evaluated=frames,
        p50_camera_count=p50_cam,
        p90_camera_count=p90_cam,
    )


__all__ = ["DEFAULT_GRID", "AssociationMetrics", "evaluate_association"]
