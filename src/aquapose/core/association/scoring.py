"""Pairwise cross-camera tracklet affinity scoring using ray-ray geometry.

Implements SPECSEED Steps 0-1: camera overlap graph filtering and pairwise
scoring with ray-ray closest-point distance, early termination, and
aggregation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

if TYPE_CHECKING:
    from aquapose.calibration.luts import ForwardLUT, InverseLUT
    from aquapose.core.tracking.types import Tracklet2D

logger = logging.getLogger(__name__)

__all__ = [
    "AssociationConfigLike",
    "ray_ray_closest_point",
    "ray_ray_closest_point_batch",
    "score_all_pairs",
    "score_tracklet_pair",
]


# ---------------------------------------------------------------------------
# Config protocol (IB-003: core/ must not import engine/)
# ---------------------------------------------------------------------------


@runtime_checkable
class AssociationConfigLike(Protocol):
    """Structural protocol for association scoring configuration.

    Any object with these attributes can be passed to scoring functions.
    Satisfied by ``AssociationConfig`` from ``engine.config`` without an
    explicit import, preserving the core -> engine import boundary (IB-003).

    Attributes:
        ray_distance_threshold: Max ray-ray distance (m) for inlier classification.
        score_min: Minimum affinity score to keep an edge.
        t_min: Minimum shared frames to attempt scoring.
        t_saturate: Overlap reliability saturation frame count.
        early_k: Frames checked for early termination.
        min_shared_voxels: Camera pair adjacency voxel threshold.
    """

    ray_distance_threshold: float
    score_min: float
    t_min: int
    t_saturate: int
    early_k: int
    min_shared_voxels: int


# ---------------------------------------------------------------------------
# Ray-ray closest point (pure numpy)
# ---------------------------------------------------------------------------


def ray_ray_closest_point(
    origin_a: np.ndarray,
    dir_a: np.ndarray,
    origin_b: np.ndarray,
    dir_b: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute the closest point of approach between two 3D rays.

    Uses the analytic formula for two skew lines: given rays
    ``p + t*d`` and ``q + s*e``, solves the 2x2 linear system from
    the cross-product formulation.

    Args:
        origin_a: Origin of ray A, shape (3,).
        dir_a: Unit direction of ray A, shape (3,).
        origin_b: Origin of ray B, shape (3,).
        dir_b: Unit direction of ray B, shape (3,).

    Returns:
        Tuple of (distance, midpoint) where distance is the closest
        approach distance and midpoint is the average of the two
        closest points, shape (3,).
    """
    w0 = origin_a - origin_b
    a = float(np.dot(dir_a, dir_a))
    b = float(np.dot(dir_a, dir_b))
    c = float(np.dot(dir_b, dir_b))
    d = float(np.dot(dir_a, w0))
    e = float(np.dot(dir_b, w0))

    denom = a * c - b * b

    # Near-parallel rays: return large distance
    if abs(denom) < 1e-12:
        # Project origin_b onto ray A to get closest point on A
        t_a = 0.0
        s_b = e / c if abs(c) > 1e-12 else 0.0
        pt_a = origin_a + t_a * dir_a
        pt_b = origin_b + s_b * dir_b
        dist = float(np.linalg.norm(pt_a - pt_b))
        midpoint = (pt_a + pt_b) / 2.0
        return dist, midpoint

    t_a = (b * e - c * d) / denom
    s_b = (a * e - b * d) / denom

    pt_a = origin_a + t_a * dir_a
    pt_b = origin_b + s_b * dir_b

    dist = float(np.linalg.norm(pt_a - pt_b))
    midpoint = (pt_a + pt_b) / 2.0

    return dist, midpoint


def ray_ray_closest_point_batch(
    origins_a: np.ndarray,
    dirs_a: np.ndarray,
    origins_b: np.ndarray,
    dirs_b: np.ndarray,
) -> np.ndarray:
    """Compute closest-point distances for N ray pairs simultaneously.

    Pure-NumPy vectorized version of ``ray_ray_closest_point()`` that
    computes ray-ray closest-point distances for N ray pairs in a single
    call using element-wise broadcasting.  Returns distances only (no
    midpoints) since callers never use midpoints.

    All inputs are cast to float64 internally to match the scalar path's
    float64 arithmetic (which promotes via ``float(np.dot(...))``).

    Args:
        origins_a: Origins of rays A, shape ``(N, 3)``.
        dirs_a: Unit directions of rays A, shape ``(N, 3)``.
        origins_b: Origins of rays B, shape ``(N, 3)``.
        dirs_b: Unit directions of rays B, shape ``(N, 3)``.

    Returns:
        Array of closest-approach distances, shape ``(N,)``, dtype float64.
    """
    origins_a = origins_a.astype(np.float64)
    dirs_a = dirs_a.astype(np.float64)
    origins_b = origins_b.astype(np.float64)
    dirs_b = dirs_b.astype(np.float64)

    w0 = origins_a - origins_b  # (N, 3)
    a = (dirs_a * dirs_a).sum(axis=1)  # (N,)
    b = (dirs_a * dirs_b).sum(axis=1)  # (N,)
    c = (dirs_b * dirs_b).sum(axis=1)  # (N,)
    d = (dirs_a * w0).sum(axis=1)  # (N,)
    e = (dirs_b * w0).sum(axis=1)  # (N,)

    denom = a * c - b * b  # (N,)
    parallel_mask = np.abs(denom) < 1e-12  # (N,) bool

    # Near-parallel fallback: t_a=0, s_b = e/c if abs(c) > 1e-12 else 0.0
    safe_c = np.where(np.abs(c) > 1e-12, c, 1.0)
    s_b_parallel = np.where(np.abs(c) > 1e-12, e / safe_c, 0.0)
    pt_a_parallel = origins_a  # t_a = 0
    pt_b_parallel = origins_b + s_b_parallel[:, None] * dirs_b
    dist_parallel = np.linalg.norm(pt_a_parallel - pt_b_parallel, axis=1)  # (N,)

    # General case: avoid division by zero with safe_denom
    safe_denom = np.where(parallel_mask, 1.0, denom)
    t_a = (b * e - c * d) / safe_denom
    s_b = (a * e - b * d) / safe_denom
    pt_a = origins_a + t_a[:, None] * dirs_a
    pt_b = origins_b + s_b[:, None] * dirs_b
    dist_general = np.linalg.norm(pt_a - pt_b, axis=1)  # (N,)

    return np.where(parallel_mask, dist_parallel, dist_general)


# ---------------------------------------------------------------------------
# Single-pair scoring (SPECSEED Step 1)
# ---------------------------------------------------------------------------


def score_tracklet_pair(
    tracklet_a: Tracklet2D,
    tracklet_b: Tracklet2D,
    forward_luts: dict[str, ForwardLUT],
    config: AssociationConfigLike,
    *,
    frame_count: int | None = None,
) -> float:
    """Score a single cross-camera tracklet pair using a soft linear kernel.

    Implements SPECSEED Step 1: ray-ray distance aggregation with a soft
    linear kernel ``1 - dist / threshold``, early termination, and overlap
    reliability weighting.

    Args:
        tracklet_a: First tracklet (from camera A).
        tracklet_b: Second tracklet (from camera B).
        forward_luts: Per-camera ForwardLUT dict.
        config: Scoring configuration.

    Returns:
        Affinity score in [0, 1]. Zero if insufficient overlap or early
        termination triggered.
    """
    cam_a = tracklet_a.camera_id
    cam_b = tracklet_b.camera_id

    # Build frame -> index mappings for fast lookup
    frames_a = {f: i for i, f in enumerate(tracklet_a.frames)}
    frames_b = {f: i for i, f in enumerate(tracklet_b.frames)}

    shared_frames = sorted(set(tracklet_a.frames) & set(tracklet_b.frames))
    t_shared = len(shared_frames)

    if t_shared < config.t_min:
        return 0.0

    lut_a = forward_luts[cam_a]
    lut_b = forward_luts[cam_b]

    score_sum = 0.0

    for frame_idx_in_shared, frame in enumerate(shared_frames):
        idx_a = frames_a[frame]
        idx_b = frames_b[frame]

        # Back-project centroids to rays
        centroid_a = tracklet_a.centroids[idx_a]
        centroid_b = tracklet_b.centroids[idx_b]

        pix_a = torch.tensor([[centroid_a[0], centroid_a[1]]], dtype=torch.float32)
        pix_b = torch.tensor([[centroid_b[0], centroid_b[1]]], dtype=torch.float32)

        origins_a, dirs_a = lut_a.cast_ray(pix_a)
        origins_b, dirs_b = lut_b.cast_ray(pix_b)

        # Convert to numpy for ray_ray_closest_point
        o_a = origins_a[0].cpu().numpy()
        d_a = dirs_a[0].cpu().numpy()
        o_b = origins_b[0].cpu().numpy()
        d_b = dirs_b[0].cpu().numpy()

        dist, _ = ray_ray_closest_point(o_a, d_a, o_b, d_b)

        if dist < config.ray_distance_threshold:
            score_sum += 1.0 - (dist / config.ray_distance_threshold)

        # Early termination check
        if frame_idx_in_shared == config.early_k - 1 and score_sum == 0.0:
            return 0.0

    # Soft inlier fraction
    f = score_sum / t_shared

    # Overlap reliability — cap t_saturate at actual run length for short runs
    effective_saturate = (
        min(config.t_saturate, frame_count) if frame_count else config.t_saturate
    )
    w = min(t_shared, effective_saturate) / effective_saturate

    # Combined score
    score = f * w
    return score


# ---------------------------------------------------------------------------
# Score all pairs (SPECSEED Steps 0-1)
# ---------------------------------------------------------------------------

# Type alias for tracklet keys
TrackletKey = tuple[str, int]


def score_all_pairs(
    tracks_2d: dict[str, list[Tracklet2D]],
    forward_luts: dict[str, ForwardLUT],
    inverse_lut: InverseLUT,
    config: AssociationConfigLike,
    *,
    frame_count: int | None = None,
) -> dict[tuple[TrackletKey, TrackletKey], float]:
    """Score all cross-camera tracklet pairs using camera adjacency filtering.

    Implements SPECSEED Steps 0-1: uses ``camera_overlap_graph()`` to restrict
    scoring to adjacent camera pairs, then calls ``score_tracklet_pair()`` for
    each combination.

    Args:
        tracks_2d: Per-camera tracklet lists (from PipelineContext).
        forward_luts: Per-camera ForwardLUT dict.
        inverse_lut: InverseLUT for overlap graph computation.
        config: Scoring configuration.

    Returns:
        Dictionary mapping ``((cam_a, track_a), (cam_b, track_b))`` to
        affinity score, filtered by ``config.score_min``.
    """
    from aquapose.calibration.luts import camera_overlap_graph

    t0 = time.perf_counter()

    overlap = camera_overlap_graph(inverse_lut, config.min_shared_voxels)

    scored: dict[tuple[TrackletKey, TrackletKey], float] = {}
    total_pairs = 0
    edges_kept = 0

    for cam_a, cam_b in overlap:
        tracklets_a = tracks_2d.get(cam_a, [])
        tracklets_b = tracks_2d.get(cam_b, [])

        for ta in tracklets_a:
            for tb in tracklets_b:
                total_pairs += 1
                s = score_tracklet_pair(
                    ta,
                    tb,
                    forward_luts,
                    config,
                    frame_count=frame_count,
                )
                if s >= config.score_min:
                    key_a: TrackletKey = (ta.camera_id, ta.track_id)
                    key_b: TrackletKey = (tb.camera_id, tb.track_id)
                    scored[(key_a, key_b)] = s
                    edges_kept += 1

    elapsed = time.perf_counter() - t0
    logger.info(
        "Scored %d pairs, kept %d edges (%.1fs)",
        total_pairs,
        edges_kept,
        elapsed,
    )

    return scored
