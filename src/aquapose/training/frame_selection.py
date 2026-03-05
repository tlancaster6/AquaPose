"""Frame selection utilities for training dataset assembly.

Provides temporal subsampling, empty-frame filtering, curvature computation,
and diversity-based sampling to build balanced training sets from pseudo-labels.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import numpy as np


class DiversitySampleResult(NamedTuple):
    """Result of diversity sampling with per-frame bin assignments."""

    frame_indices: list[int]
    frame_bins: dict[int, int]
    """Mapping of frame index to dominant curvature bin across fish."""


def temporal_subsample(frame_indices: list[int], step: int) -> list[int]:
    """Select every ``step``-th frame from a sorted list of frame indices.

    Args:
        frame_indices: Frame indices to subsample (need not be sorted).
        step: Subsample stride. ``step=1`` returns all frames.

    Returns:
        Sorted list of selected frame indices.
    """
    sorted_indices = sorted(frame_indices)
    return sorted_indices[::step]


def filter_empty_frames(
    frame_indices: list[int],
    midlines_3d: list[dict],
) -> list[int]:
    """Remove frames that have no 3D reconstructions.

    Args:
        frame_indices: Candidate frame indices.
        midlines_3d: Frame-indexed list of ``{fish_id: Midline3D}`` dicts.

    Returns:
        Subset of *frame_indices* where the corresponding entry in
        *midlines_3d* exists and is a non-empty dict.
    """
    result: list[int] = []
    for idx in frame_indices:
        if idx < 0 or idx >= len(midlines_3d):
            continue
        if midlines_3d[idx]:
            result.append(idx)
    return result


def compute_curvature(control_points: np.ndarray) -> float:
    """Compute mean absolute curvature from B-spline control points.

    Uses finite differences of tangent vectors:
    ``T[i] = cp[i+1] - cp[i]``, then
    ``k[i] = |T[i+1] - T[i]| / (0.5 * (|T[i]| + |T[i+1]|))``.

    Args:
        control_points: Shape ``(7, 3)`` array of B-spline control points.

    Returns:
        Mean absolute curvature (scalar). Near zero for straight lines.
    """
    tangents = np.diff(control_points, axis=0)  # (6, 3)
    tangent_norms = np.linalg.norm(tangents, axis=1)  # (6,)

    dt = np.diff(tangents, axis=0)  # (5, 3)
    dt_norms = np.linalg.norm(dt, axis=1)  # (5,)

    # Average adjacent tangent magnitudes for normalization
    avg_norms = 0.5 * (tangent_norms[:-1] + tangent_norms[1:])  # (5,)

    # Avoid division by zero for degenerate cases
    safe_norms = np.where(avg_norms > 1e-12, avg_norms, 1.0)
    curvatures = dt_norms / safe_norms

    return float(np.mean(curvatures))


def diversity_sample(
    midlines_3d: list[dict],
    frame_indices: list[int],
    n_bins: int = 5,
    max_per_bin: int | None = None,
    seed: int = 42,
) -> DiversitySampleResult:
    """Select frames covering diverse fish body curvatures via K-means binning.

    For each fish across all candidate frames, computes curvature from the
    control points, assigns each observation to one of *n_bins* clusters,
    and samples up to *max_per_bin* frames per cluster.

    Args:
        midlines_3d: Frame-indexed list of ``{fish_id: Midline3D}`` dicts.
        frame_indices: Candidate frame indices (must be valid indices into
            *midlines_3d* with non-empty entries).
        n_bins: Number of curvature clusters for K-means.
        max_per_bin: Maximum observations to keep per cluster. ``None``
            keeps all observations (no subsampling).
        seed: Random seed for reproducibility.

    Returns:
        A :class:`DiversitySampleResult` with ``frame_indices`` (sorted list
        of selected frame indices, union across all fish) and ``frame_bins``
        (mapping of frame index to dominant curvature bin).
    """
    from scipy.cluster.vq import kmeans2

    # Collect (fish_id, frame_idx, curvature) triples
    triples: list[tuple[int, int, float]] = []
    for fidx in frame_indices:
        if fidx < 0 or fidx >= len(midlines_3d):
            continue
        frame_data = midlines_3d[fidx]
        for fish_id, midline in frame_data.items():
            curv = compute_curvature(midline.control_points)
            triples.append((fish_id, fidx, curv))

    if not triples:
        return DiversitySampleResult([], {})

    curvatures = np.array([t[2] for t in triples], dtype=np.float64)

    # Handle edge case: fewer observations than bins
    effective_bins = min(n_bins, len(curvatures))

    rng = np.random.default_rng(seed)
    # kmeans2 needs at least as many data points as clusters
    _, labels = kmeans2(
        curvatures.reshape(-1, 1),
        effective_bins,
        minit="points",
        seed=rng,  # pyright: ignore[reportCallIssue]
    )

    # Build per-frame bin assignments (dominant bin across fish)
    frame_label_lists: dict[int, list[int]] = {}
    for i, (_, fidx, _) in enumerate(triples):
        frame_label_lists.setdefault(fidx, []).append(int(labels[i]))

    frame_bins: dict[int, int] = {
        fidx: Counter(bins).most_common(1)[0][0]
        for fidx, bins in frame_label_lists.items()
    }

    # If max_per_bin is None, return all frame indices that have data
    if max_per_bin is None:
        all_frames = sorted({t[1] for t in triples})
        return DiversitySampleResult(all_frames, {f: frame_bins[f] for f in all_frames})

    # Sample max_per_bin from each cluster
    selected_frames: set[int] = set()
    for bin_id in range(effective_bins):
        bin_mask = labels == bin_id
        bin_indices = np.nonzero(bin_mask)[0]
        if len(bin_indices) <= max_per_bin:
            chosen = bin_indices
        else:
            chosen = rng.choice(bin_indices, size=max_per_bin, replace=False)
        for ci in chosen:
            selected_frames.add(triples[ci][1])

    sorted_frames = sorted(selected_frames)
    return DiversitySampleResult(
        sorted_frames, {f: frame_bins[f] for f in sorted_frames}
    )
