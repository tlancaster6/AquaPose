"""Pairwise cross-camera tracklet affinity scoring using ray-ray geometry.

Implements SPECSEED Steps 0-1: camera overlap graph filtering and pairwise
scoring with ray-ray closest-point distance, early termination, and
aggregation.  Uses multi-keypoint rays (not centroids) for richer affinity.
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
        keypoint_confidence_floor: Minimum keypoint confidence for scoring
            participation. Keypoints below this on either tracklet are excluded
            from the frame's distance computation.
        aggregation_method: Method to aggregate per-keypoint distances within a
            frame. Currently only ``"mean"`` is supported.
        use_multi_keypoint_scoring: Toggle scoring method. ``True`` uses
            multi-keypoint ray distances (v3.8). ``False`` uses single centroid
            rays (v3.7 baseline).
    """

    ray_distance_threshold: float
    score_min: float
    t_min: int
    t_saturate: int
    early_k: int
    min_shared_voxels: int
    keypoint_confidence_floor: float
    aggregation_method: str
    use_multi_keypoint_scoring: bool


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
# Single-pair scoring (SPECSEED Step 1) — multi-keypoint
# ---------------------------------------------------------------------------


def _score_pair_centroid_only(
    tracklet_a: Tracklet2D,
    tracklet_b: Tracklet2D,
    forward_luts: dict[str, ForwardLUT],
    config: AssociationConfigLike,
    *,
    frame_count: int | None = None,
) -> float:
    """Score a cross-camera tracklet pair using single centroid rays (v3.7 baseline).

    Casts one ray per tracklet per frame from the tracklet centroid pixel,
    computes the ray-ray closest-point distance, and applies the same soft
    linear kernel as the multi-keypoint path.  Serves as a v3.7 baseline for
    apples-to-apples comparison when ``use_multi_keypoint_scoring=False``.

    Args:
        tracklet_a: First tracklet (from camera A).
        tracklet_b: Second tracklet (from camera B).
        forward_luts: Per-camera ForwardLUT dict.
        config: Scoring configuration.
        frame_count: Total frames in chunk (for overlap reliability).

    Returns:
        Affinity score in [0, 1]. Zero if insufficient overlap or early
        termination triggered.
    """
    cam_a = tracklet_a.camera_id
    cam_b = tracklet_b.camera_id

    frames_a = {f: i for i, f in enumerate(tracklet_a.frames)}
    frames_b = {f: i for i, f in enumerate(tracklet_b.frames)}

    shared_frames = sorted(set(tracklet_a.frames) & set(tracklet_b.frames))
    t_shared = len(shared_frames)

    if t_shared < config.t_min:
        return 0.0

    lut_a = forward_luts[cam_a]
    lut_b = forward_luts[cam_b]

    # Gather centroid pixels for shared frames
    idx_a = [frames_a[f] for f in shared_frames]
    idx_b = [frames_b[f] for f in shared_frames]

    cents_a = np.array(
        [tracklet_a.centroids[i] for i in idx_a], dtype=np.float64
    )  # (N, 2)
    cents_b = np.array(
        [tracklet_b.centroids[i] for i in idx_b], dtype=np.float64
    )  # (N, 2)

    # Determine early-termination split
    early_k = config.early_k
    if t_shared <= early_k:
        early_idx = slice(None)
        remaining_slice: slice | None = None
    else:
        early_idx = slice(0, early_k)
        remaining_slice = slice(early_k, None)

    def _score_centroid_batch(pix_a: np.ndarray, pix_b: np.ndarray) -> float:
        """Score a batch of centroid pairs, returns sum of soft contributions."""
        pix_a_t = torch.tensor(pix_a, dtype=torch.float32)
        pix_b_t = torch.tensor(pix_b, dtype=torch.float32)

        origins_a, dirs_a = lut_a.cast_ray(pix_a_t)
        origins_b, dirs_b = lut_b.cast_ray(pix_b_t)

        oa = origins_a.cpu().numpy().astype(np.float64)
        da = dirs_a.cpu().numpy().astype(np.float64)
        ob = origins_b.cpu().numpy().astype(np.float64)
        db = dirs_b.cpu().numpy().astype(np.float64)

        dists = ray_ray_closest_point_batch(oa, da, ob, db)  # (N,)
        contributions = np.where(
            dists < config.ray_distance_threshold,
            1.0 - dists / config.ray_distance_threshold,
            0.0,
        )
        return float(contributions.sum())

    # Phase 1: score early frames
    early_sum = _score_centroid_batch(cents_a[early_idx], cents_b[early_idx])

    # Early termination: no inliers in first early_k frames
    if t_shared >= early_k and early_sum == 0.0:
        return 0.0

    # Phase 2: remaining frames
    total_sum = early_sum
    if remaining_slice is not None:
        total_sum += _score_centroid_batch(
            cents_a[remaining_slice], cents_b[remaining_slice]
        )

    # Soft inlier fraction
    f = total_sum / t_shared

    # Overlap reliability
    effective_saturate = (
        min(config.t_saturate, frame_count) if frame_count else config.t_saturate
    )
    w = min(t_shared, effective_saturate) / effective_saturate

    return f * w


def score_tracklet_pair(
    tracklet_a: Tracklet2D,
    tracklet_b: Tracklet2D,
    forward_luts: dict[str, ForwardLUT],
    config: AssociationConfigLike,
    *,
    frame_count: int | None = None,
) -> float:
    """Score a cross-camera tracklet pair using multi-keypoint ray distances.

    Casts rays from all confident keypoints per detection per frame, computes
    matched keypoint ray-ray distances (nose-to-nose, head-to-head, etc.),
    aggregates per-frame via arithmetic mean, and applies a soft linear kernel.

    When ``config.use_multi_keypoint_scoring`` is ``False``, falls back to the
    v3.7 centroid-only path (single ray per frame) for baseline comparison.

    Args:
        tracklet_a: First tracklet (from camera A).
        tracklet_b: Second tracklet (from camera B).
        forward_luts: Per-camera ForwardLUT dict.
        config: Scoring configuration.
        frame_count: Total frames in chunk (for overlap reliability).

    Returns:
        Affinity score in [0, 1]. Zero if insufficient overlap, missing
        keypoints, or early termination triggered.
    """
    # v3.7 baseline: centroid-only scoring (toggle before keypoints check)
    if not config.use_multi_keypoint_scoring:
        return _score_pair_centroid_only(
            tracklet_a, tracklet_b, forward_luts, config, frame_count=frame_count
        )

    # No centroid fallback: keypoints required
    if tracklet_a.keypoints is None or tracklet_b.keypoints is None:
        return 0.0

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

    # Determine early-termination split point
    early_k = config.early_k
    if t_shared <= early_k:
        early_frames = shared_frames
        remaining_frames: list[int] = []
    else:
        early_frames = shared_frames[:early_k]
        remaining_frames = shared_frames[early_k:]

    # --- Phase 1: score early frames ---
    score_sum, n_skipped = _batch_score_frames_kpt(
        early_frames, frames_a, frames_b, tracklet_a, tracklet_b, lut_a, lut_b, config
    )

    # Early termination: if no inliers after early_k frames, bail out
    if t_shared >= early_k and score_sum == 0.0:
        return 0.0

    # --- Phase 2: score remaining frames (if any) ---
    total_skipped = n_skipped
    if remaining_frames:
        rem_score, rem_skipped = _batch_score_frames_kpt(
            remaining_frames,
            frames_a,
            frames_b,
            tracklet_a,
            tracklet_b,
            lut_a,
            lut_b,
            config,
        )
        score_sum += rem_score
        total_skipped += rem_skipped

    # Effective t_shared excludes frames where no keypoints were valid
    effective_t_shared = t_shared - total_skipped
    if effective_t_shared <= 0:
        return 0.0

    # Soft inlier fraction
    f = score_sum / effective_t_shared

    # Overlap reliability — cap t_saturate at actual run length for short runs
    effective_saturate = (
        min(config.t_saturate, frame_count) if frame_count else config.t_saturate
    )
    w = min(t_shared, effective_saturate) / effective_saturate

    # Combined score
    score = f * w
    return score


def _batch_score_frames_kpt(
    batch_frames: list[int],
    frames_a: dict[int, int],
    frames_b: dict[int, int],
    tracklet_a: Tracklet2D,
    tracklet_b: Tracklet2D,
    lut_a: ForwardLUT,
    lut_b: ForwardLUT,
    config: AssociationConfigLike,
) -> tuple[float, int]:
    """Score a batch of shared frames using multi-keypoint ray-ray distances.

    Extracts keypoints for both tracklets, builds a confidence intersection
    mask, flattens valid keypoints into a single ``cast_ray`` call per camera,
    computes matched distances, aggregates per-frame via mean, and applies the
    soft linear kernel.

    Args:
        batch_frames: Frame indices to score.
        frames_a: Frame-to-index mapping for tracklet A.
        frames_b: Frame-to-index mapping for tracklet B.
        tracklet_a: First tracklet (from camera A).
        tracklet_b: Second tracklet (from camera B).
        lut_a: ForwardLUT for camera A.
        lut_b: ForwardLUT for camera B.
        config: Scoring configuration.

    Returns:
        Tuple of ``(score_sum, n_skipped)`` where ``score_sum`` is the sum of
        soft-kernel contributions and ``n_skipped`` is the count of frames
        excluded because no keypoints passed the confidence intersection.
    """
    n_frames = len(batch_frames)
    idx_a = [frames_a[f] for f in batch_frames]
    idx_b = [frames_b[f] for f in batch_frames]

    # Extract keypoints and confidences for shared frames
    assert tracklet_a.keypoints is not None  # caller checks
    assert tracklet_b.keypoints is not None
    assert tracklet_a.keypoint_conf is not None
    assert tracklet_b.keypoint_conf is not None

    kpts_a = tracklet_a.keypoints[idx_a]  # (N, K, 2)
    kpts_b = tracklet_b.keypoints[idx_b]  # (N, K, 2)
    conf_a = tracklet_a.keypoint_conf[idx_a]  # (N, K)
    conf_b = tracklet_b.keypoint_conf[idx_b]  # (N, K)

    # Intersection mask: both tracklets must have confident keypoint
    floor = config.keypoint_confidence_floor
    valid = (conf_a >= floor) & (conf_b >= floor)  # (N, K) bool

    # Count valid keypoints per frame
    n_valid = valid.sum(axis=1)  # (N,)
    active_mask = n_valid > 0  # (N,) bool
    n_skipped = int((~active_mask).sum())

    if not active_mask.any():
        return 0.0, n_frames

    # Flatten valid keypoints for batched ray casting
    pixels_a = kpts_a[valid].astype(np.float64)  # (M, 2)
    pixels_b = kpts_b[valid].astype(np.float64)  # (M, 2)

    # Single cast_ray call per camera
    pix_a_t = torch.tensor(pixels_a, dtype=torch.float32)
    pix_b_t = torch.tensor(pixels_b, dtype=torch.float32)

    origins_a, dirs_a = lut_a.cast_ray(pix_a_t)
    origins_b, dirs_b = lut_b.cast_ray(pix_b_t)

    # Convert to numpy float64
    oa = origins_a.cpu().numpy().astype(np.float64)
    da = dirs_a.cpu().numpy().astype(np.float64)
    ob = origins_b.cpu().numpy().astype(np.float64)
    db = dirs_b.cpu().numpy().astype(np.float64)

    # Vectorized ray-ray distances for all valid keypoint pairs
    dists = ray_ray_closest_point_batch(oa, da, ob, db)  # (M,)

    # Scatter back to per-frame mean distances using reduceat
    active_n_valid = n_valid[active_mask]  # (A,) where A = active frame count
    offsets = np.concatenate([[0], np.cumsum(active_n_valid[:-1])])
    frame_sums = np.add.reduceat(dists, offsets)  # (A,)
    mean_dists = frame_sums / active_n_valid  # (A,)

    # Soft kernel applied AFTER per-frame mean distance
    contributions = np.where(
        mean_dists < config.ray_distance_threshold,
        1.0 - mean_dists / config.ray_distance_threshold,
        0.0,
    )

    return float(contributions.sum()), n_skipped


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
