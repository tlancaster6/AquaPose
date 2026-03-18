"""Stitch-quality evaluator comparing pre- and post-stitch fragmentation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aquapose.core.types.reconstruction import Midline3D
from aquapose.evaluation.stages.fragmentation import (
    FragmentationMetrics,
    evaluate_fragmentation,
)
from aquapose.io.midline_writer import read_midline3d_results


@dataclass(frozen=True)
class StitchingMetrics:
    """Comparative metrics for identity stitching quality.

    Wraps pre-stitch and post-stitch ``FragmentationMetrics`` with derived
    comparison fields and stitched-data coverage statistics.

    Attributes:
        source_file: Name of the stitched H5 file evaluated.
        pre_stitch: Fragmentation metrics from ``midlines.h5``.
        post_stitch: Fragmentation metrics from ``midlines_stitched.h5``.
        id_reduction: Number of IDs collapsed by stitching.
        id_ratio: Ratio of post-stitch IDs to expected fish count (1.0 = perfect).
        continuity_improvement: Post-stitch minus pre-stitch mean continuity.
        gap_reduction: Number of gaps removed by stitching.
        birth_reduction: Number of births removed by stitching.
        death_reduction: Number of deaths removed by stitching.
        full_coverage_ratio: Fraction of frames with exactly ``n_animals`` fish.
        fish_count_distribution: Histogram of per-frame fish count in stitched
            data, mapping count to number of frames.
        per_fish_summary: Per stitched fish ID summary with span, continuity,
            and gap count.
    """

    source_file: str
    pre_stitch: FragmentationMetrics
    post_stitch: FragmentationMetrics
    id_reduction: int
    id_ratio: float
    continuity_improvement: float
    gap_reduction: int
    birth_reduction: int
    death_reduction: int
    full_coverage_ratio: float
    fish_count_distribution: dict[int, int]
    per_fish_summary: dict[int, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict representation.

        Returns:
            Dict with all scalar fields plus nested pre/post stitch dicts,
            fish_count_distribution (string keys), and per_fish_summary
            (string keys).
        """
        return {
            "source_file": self.source_file,
            "pre_stitch": self.pre_stitch.to_dict(),
            "post_stitch": self.post_stitch.to_dict(),
            "id_reduction": int(self.id_reduction),
            "id_ratio": float(self.id_ratio),
            "continuity_improvement": float(self.continuity_improvement),
            "gap_reduction": int(self.gap_reduction),
            "birth_reduction": int(self.birth_reduction),
            "death_reduction": int(self.death_reduction),
            "full_coverage_ratio": float(self.full_coverage_ratio),
            "fish_count_distribution": {
                str(k): int(v) for k, v in sorted(self.fish_count_distribution.items())
            },
            "per_fish_summary": {
                str(k): v for k, v in sorted(self.per_fish_summary.items())
            },
        }


def _h5_to_frame_list(
    path: Path,
) -> list[dict[int, Midline3D] | None]:
    """Convert an H5 midlines file to the frame-list format used by evaluate_fragmentation.

    Args:
        path: Path to a midlines HDF5 file.

    Returns:
        Per-frame list of ``{fish_id: Midline3D}`` dicts. Frames are ordered
        by ``frame_index`` from the H5.  Only ``frame_index`` is populated on
        each ``Midline3D`` (sufficient for fragmentation analysis).
    """
    data: dict[str, Any] = read_midline3d_results(path)
    frame_indices: np.ndarray = data["frame_index"]  # (N,)
    fish_ids: np.ndarray = data["fish_id"]  # (N, max_fish)

    result: list[dict[int, Midline3D] | None] = []
    for row_idx in range(len(frame_indices)):
        fi = int(frame_indices[row_idx])
        row_fish = fish_ids[row_idx]
        entry: dict[int, Midline3D] = {}
        for slot_idx in range(len(row_fish)):
            fid = int(row_fish[slot_idx])
            if fid == -1:
                continue
            entry[fid] = Midline3D(
                fish_id=fid,
                frame_index=fi,
                half_widths=np.empty(0),
                n_cameras=0,
                mean_residual=0.0,
                max_residual=0.0,
            )
        result.append(entry if entry else None)
    return result


def evaluate_stitching(
    run_dir: Path,
    n_animals: int,
) -> StitchingMetrics | None:
    """Evaluate stitch quality by comparing pre- and post-stitch H5 files.

    Auto-detects ``midlines_stitched.h5`` in ``run_dir``. If absent, returns
    ``None``.  Reads both ``midlines.h5`` and ``midlines_stitched.h5``, runs
    ``evaluate_fragmentation`` on each, and computes comparative metrics.

    Args:
        run_dir: Pipeline run directory containing H5 files.
        n_animals: Expected number of fish in the scene.

    Returns:
        StitchingMetrics if ``midlines_stitched.h5`` exists, else ``None``.
    """
    stitched_path = run_dir / "midlines_stitched.h5"
    raw_path = run_dir / "midlines.h5"

    if not stitched_path.exists():
        return None
    if not raw_path.exists():
        return None

    pre_frames = _h5_to_frame_list(raw_path)
    post_frames = _h5_to_frame_list(stitched_path)

    pre = evaluate_fragmentation(pre_frames, n_animals)
    post = evaluate_fragmentation(post_frames, n_animals)

    # Full coverage: fraction of frames with exactly n_animals fish
    n_total = len(post_frames)
    n_full = sum(
        1 for entry in post_frames if entry is not None and len(entry) == n_animals
    )
    full_coverage_ratio = n_full / n_total if n_total > 0 else 0.0

    # Fish count distribution: how many fish per frame
    fish_counts: Counter[int] = Counter()
    for entry in post_frames:
        fish_counts[len(entry) if entry is not None else 0] += 1

    # Per-fish summary from post-stitch fragmentation
    per_fish_summary: dict[int, dict[str, object]] = {}
    # Recompute per-fish gaps from the frame list
    fish_frame_sets: dict[int, set[int]] = {}
    for entry in post_frames:
        if entry is None:
            continue
        for fid, m3d in entry.items():
            if fid not in fish_frame_sets:
                fish_frame_sets[fid] = set()
            fish_frame_sets[fid].add(m3d.frame_index)

    for fid in sorted(fish_frame_sets):
        frames = sorted(fish_frame_sets[fid])
        span = frames[-1] - frames[0] + 1
        continuity = post.per_fish_continuity.get(fid, 0.0)
        gaps = sum(1 for i in range(len(frames) - 1) if frames[i + 1] - frames[i] > 1)
        per_fish_summary[fid] = {
            "span": span,
            "continuity": float(continuity),
            "gaps": gaps,
        }

    return StitchingMetrics(
        source_file="midlines_stitched.h5",
        pre_stitch=pre,
        post_stitch=post,
        id_reduction=pre.unique_fish_ids - post.unique_fish_ids,
        id_ratio=post.unique_fish_ids / n_animals if n_animals > 0 else 0.0,
        continuity_improvement=post.mean_continuity_ratio - pre.mean_continuity_ratio,
        gap_reduction=pre.total_gaps - post.total_gaps,
        birth_reduction=pre.track_births - post.track_births,
        death_reduction=pre.track_deaths - post.track_deaths,
        full_coverage_ratio=full_coverage_ratio,
        fish_count_distribution=dict(fish_counts),
        per_fish_summary=per_fish_summary,
    )


__all__ = [
    "StitchingMetrics",
    "evaluate_stitching",
]
