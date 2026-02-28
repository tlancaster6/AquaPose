"""Pairwise cross-camera tracklet affinity scoring using ray-ray geometry.

Implements SPECSEED Steps 0-1: camera overlap graph filtering and pairwise
scoring with ray-ray closest-point distance, ghost-point penalties, early
termination, and aggregation.
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
        ghost_pixel_threshold: Max pixel distance for ghost "supporting" check.
        min_shared_voxels: Camera pair adjacency voxel threshold.
    """

    ray_distance_threshold: float
    score_min: float
    t_min: int
    t_saturate: int
    early_k: int
    ghost_pixel_threshold: float
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


# ---------------------------------------------------------------------------
# Single-pair scoring (SPECSEED Step 1)
# ---------------------------------------------------------------------------


def score_tracklet_pair(
    tracklet_a: Tracklet2D,
    tracklet_b: Tracklet2D,
    forward_luts: dict[str, ForwardLUT],
    inverse_lut: InverseLUT,
    detections: list[dict[str, list[tuple[float, float]]]],
    config: AssociationConfigLike,
    *,
    frame_count: int | None = None,
) -> float:
    """Score a single cross-camera tracklet pair.

    Implements SPECSEED Step 1: ray-ray distance aggregation with ghost-point
    penalties, early termination, and overlap reliability weighting.

    Args:
        tracklet_a: First tracklet (from camera A).
        tracklet_b: Second tracklet (from camera B).
        forward_luts: Per-camera ForwardLUT dict.
        inverse_lut: InverseLUT for ghost-point lookups.
        detections: Per-frame per-camera detection centroids. Indexed by
            frame_idx; each entry maps camera_id to list of (u, v) centroids.
        config: Scoring configuration.

    Returns:
        Affinity score in [0, 1]. Zero if insufficient overlap or early
        termination triggered.
    """
    from aquapose.calibration.luts import ghost_point_lookup

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

    inlier_count = 0
    ghost_ratios: list[float] = []
    scoring_cameras = {cam_a, cam_b}

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

        dist, midpoint = ray_ray_closest_point(o_a, d_a, o_b, d_b)

        if dist < config.ray_distance_threshold:
            inlier_count += 1

            # Ghost penalty for this inlier frame
            mid_tensor = torch.tensor(midpoint.reshape(1, 3), dtype=torch.float32)
            visibility = ghost_point_lookup(inverse_lut, mid_tensor)
            visible_cams = visibility[0]  # list[(cam_id, u, v)]

            # Exclude the two scoring cameras
            other_cams = [
                (cid, u, v) for cid, u, v in visible_cams if cid not in scoring_cameras
            ]
            n_visible_other = len(other_cams)

            if n_visible_other > 0:
                n_negative = 0
                for cid, exp_u, exp_v in other_cams:
                    # Check if any detection in this camera for this frame
                    # is within ghost_pixel_threshold
                    supporting = False
                    if frame < len(detections) and cid in detections[frame]:
                        for det_u, det_v in detections[frame][cid]:
                            pixel_dist = (
                                (det_u - exp_u) ** 2 + (det_v - exp_v) ** 2
                            ) ** 0.5
                            if pixel_dist < config.ghost_pixel_threshold:
                                supporting = True
                                break
                    if not supporting:
                        n_negative += 1
                ghost_ratios.append(n_negative / n_visible_other)
            else:
                ghost_ratios.append(0.0)

        # Early termination check
        if frame_idx_in_shared == config.early_k - 1 and inlier_count == 0:
            return 0.0

    # Inlier fraction
    f = inlier_count / t_shared

    # Mean ghost ratio
    mean_ghost = float(np.mean(ghost_ratios)) if ghost_ratios else 0.0

    # Overlap reliability — cap t_saturate at actual run length for short runs
    effective_saturate = (
        min(config.t_saturate, frame_count) if frame_count else config.t_saturate
    )
    w = min(t_shared, effective_saturate) / effective_saturate

    # Combined score
    score = f * (1.0 - mean_ghost) * w
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
    detections: list[dict[str, list[tuple[float, float]]]],
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
        inverse_lut: InverseLUT for ghost-point lookups and overlap graph.
        detections: Per-frame per-camera detection centroids.
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
                    inverse_lut,
                    detections,
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
