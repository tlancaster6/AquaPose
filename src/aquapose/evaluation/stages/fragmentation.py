"""Pure-function evaluator for 3D track fragmentation analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aquapose.core.types.reconstruction import Midline3D


@dataclass(frozen=True)
class FragmentationMetrics:
    """Metrics for 3D track fragmentation analysis.

    Frame-level metrics measure gaps within individual fish tracks.
    Track-level metrics measure fish identity births/deaths and lifespan.

    Attributes:
        total_gaps: Total number of frame-level gaps across all fish.
        mean_gap_duration: Mean gap duration in frames across all gaps.
        max_gap_duration: Maximum gap duration in frames across all gaps.
        mean_continuity_ratio: Mean continuity ratio across all fish.
        per_fish_continuity: Per-fish continuity ratio mapping fish_id to
            ratio of frames present to total frames in that fish's lifespan.
        unique_fish_ids: Number of unique fish IDs observed.
        expected_fish: Expected number of fish (n_animals parameter).
        track_births: Number of fish whose first appearance is after the
            global first frame.
        track_deaths: Number of fish whose last appearance is before the
            global last frame.
        mean_track_lifespan: Mean track lifespan in frames across all fish.
        median_track_lifespan: Median track lifespan in frames across all fish.
    """

    total_gaps: int
    mean_gap_duration: float
    max_gap_duration: int
    mean_continuity_ratio: float
    per_fish_continuity: dict[int, float]
    unique_fish_ids: int
    expected_fish: int
    track_births: int
    track_deaths: int
    mean_track_lifespan: float
    median_track_lifespan: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict representation.

        Returns:
            Dict with all fields. Integer keys in per_fish_continuity are
            converted to strings for JSON compatibility.
        """
        return {
            "total_gaps": int(self.total_gaps),
            "mean_gap_duration": float(self.mean_gap_duration),
            "max_gap_duration": int(self.max_gap_duration),
            "mean_continuity_ratio": float(self.mean_continuity_ratio),
            "per_fish_continuity": {
                str(k): float(v) for k, v in self.per_fish_continuity.items()
            },
            "unique_fish_ids": int(self.unique_fish_ids),
            "expected_fish": int(self.expected_fish),
            "track_births": int(self.track_births),
            "track_deaths": int(self.track_deaths),
            "mean_track_lifespan": float(self.mean_track_lifespan),
            "median_track_lifespan": float(self.median_track_lifespan),
        }


def evaluate_fragmentation(
    midlines_3d: list[dict[int, Midline3D] | None],
    n_animals: int,
) -> FragmentationMetrics:
    """Evaluate 3D track fragmentation from reconstruction frame results.

    Analyses frame-level gaps within fish tracks and track-level
    births/deaths across the sequence.

    Args:
        midlines_3d: Per-frame list of fish_id to Midline3D mappings.
            None entries indicate frames with no reconstruction data.
            Uses ``m3d.frame_index`` for global frame indexing (not list
            position).
        n_animals: Expected number of fish in the scene.

    Returns:
        FragmentationMetrics with gap counts, continuity ratios, and
        birth/death statistics.
    """
    # Build fish_id -> set of global frame indices
    fish_frames: dict[int, set[int]] = {}
    for entry in midlines_3d:
        if entry is None:
            continue
        for fish_id, m3d in entry.items():
            if fish_id not in fish_frames:
                fish_frames[fish_id] = set()
            fish_frames[fish_id].add(m3d.frame_index)

    if not fish_frames:
        return FragmentationMetrics(
            total_gaps=0,
            mean_gap_duration=0.0,
            max_gap_duration=0,
            mean_continuity_ratio=0.0,
            per_fish_continuity={},
            unique_fish_ids=0,
            expected_fish=n_animals,
            track_births=0,
            track_deaths=0,
            mean_track_lifespan=0.0,
            median_track_lifespan=0.0,
        )

    # Determine global frame range
    all_frames: set[int] = set()
    for frames in fish_frames.values():
        all_frames.update(frames)
    global_first = min(all_frames)
    global_last = max(all_frames)

    # Per-fish gap analysis
    total_gaps = 0
    all_gap_durations: list[int] = []
    per_fish_continuity: dict[int, float] = {}
    lifespans: list[int] = []
    track_births = 0
    track_deaths = 0

    for fish_id, frames in fish_frames.items():
        sorted_frames = sorted(frames)
        min_frame = sorted_frames[0]
        max_frame = sorted_frames[-1]
        span = max_frame - min_frame + 1
        continuity = len(frames) / span if span > 0 else 1.0
        per_fish_continuity[fish_id] = continuity
        lifespans.append(span)

        # Count gaps: consecutive frame indices that differ by > 1
        for i in range(len(sorted_frames) - 1):
            gap = sorted_frames[i + 1] - sorted_frames[i] - 1
            if gap > 0:
                total_gaps += 1
                all_gap_durations.append(gap)

        # Track births and deaths
        if min_frame > global_first:
            track_births += 1
        if max_frame < global_last:
            track_deaths += 1

    mean_gap_duration = float(np.mean(all_gap_durations)) if all_gap_durations else 0.0
    max_gap_duration = max(all_gap_durations) if all_gap_durations else 0
    mean_continuity = float(np.mean(list(per_fish_continuity.values())))
    mean_lifespan = float(np.mean(lifespans)) if lifespans else 0.0
    median_lifespan = float(np.median(lifespans)) if lifespans else 0.0

    return FragmentationMetrics(
        total_gaps=total_gaps,
        mean_gap_duration=mean_gap_duration,
        max_gap_duration=max_gap_duration,
        mean_continuity_ratio=mean_continuity,
        per_fish_continuity=per_fish_continuity,
        unique_fish_ids=len(fish_frames),
        expected_fish=n_animals,
        track_births=track_births,
        track_deaths=track_deaths,
        mean_track_lifespan=mean_lifespan,
        median_track_lifespan=median_lifespan,
    )


__all__ = ["FragmentationMetrics", "evaluate_fragmentation"]
