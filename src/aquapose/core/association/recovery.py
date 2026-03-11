"""Singleton recovery for post-validation association.

After group validation (Phase 90) splits/evicts tracklets, many become singletons.
This module gives singletons a second chance to join existing multi-tracklet groups
by scoring them via per-keypoint ray-to-3D residuals, greedy best-first whole
assignment, and a binary split-assign sweep for swap recovery.

No imports from validation.py or refinement.py — module is intentionally standalone.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

from aquapose.core.association.scoring import ray_ray_closest_point
from aquapose.core.association.types import TrackletGroup

if TYPE_CHECKING:
    from aquapose.calibration.luts import ForwardLUT
    from aquapose.core.tracking.types import Tracklet2D

logger = logging.getLogger(__name__)

__all__ = ["RecoveryConfigLike", "recover_singletons"]


# ---------------------------------------------------------------------------
# Config protocol (IB-003: core/ must not import engine/)
# ---------------------------------------------------------------------------


@runtime_checkable
class RecoveryConfigLike(Protocol):
    """Structural protocol for singleton recovery configuration.

    Any object with these attributes can be passed to recovery functions.
    Satisfied by ``AssociationConfig`` from ``engine.config`` without an
    explicit import, preserving the core -> engine import boundary (IB-003).

    Attributes:
        recovery_enabled: Toggle to skip singleton recovery entirely.
        recovery_residual_threshold: Maximum mean ray-to-3D residual (metres)
            for a singleton to be assigned to a group.
        recovery_min_shared_frames: Minimum shared frames between singleton and
            group for scoring to be attempted.
        recovery_min_segment_length: Minimum frames per segment for the binary
            split-assign sweep. Singletons shorter than
            2 * recovery_min_segment_length skip the sweep.
        keypoint_confidence_floor: Minimum keypoint confidence for a keypoint
            to participate in residual computation.
    """

    recovery_enabled: bool
    recovery_residual_threshold: float
    recovery_min_shared_frames: int
    recovery_min_segment_length: int
    keypoint_confidence_floor: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recover_singletons(
    groups: list[TrackletGroup],
    forward_luts: dict[str, ForwardLUT],
    config: RecoveryConfigLike,
) -> list[TrackletGroup]:
    """Give singletons a second chance to join existing multi-tracklet groups.

    Partitions groups into multi-tracklet groups and singletons, scores each
    singleton against all multi-tracklet groups using per-keypoint ray-to-3D
    residuals, assigns greedily (best residual first), attempts split-assign
    for remaining singletons, and reassigns unique fish_ids to any singletons
    that remain unmatched.

    Args:
        groups: TrackletGroup list from validation.
        forward_luts: Per-camera ForwardLUT dict for ray back-projection.
        config: Recovery configuration thresholds.

    Returns:
        Updated list of TrackletGroups with recovered singletons absorbed
        into their best-matching groups where possible.
    """
    if not config.recovery_enabled:
        return groups

    multi_groups, singletons = _partition_groups(groups)

    if not singletons:
        return groups

    # Compute next available track_id to avoid collisions for split segments
    next_track_id = _compute_next_track_id(groups)

    # -----------------------------------------------------------------------
    # Pass 1: Whole-tracklet greedy assignment
    # -----------------------------------------------------------------------
    # Score all (singleton, group) pairs, sort by residual ascending, assign
    # best match first. Each singleton and each slot is consumed at most once.

    scored_pairs: list[tuple[float, int, int]] = []  # (residual, s_idx, g_idx)

    for s_idx, singleton in enumerate(singletons):
        singleton_tracklet: Tracklet2D = singleton.tracklets[0]
        for g_idx, group in enumerate(multi_groups):
            if _has_camera_overlap(singleton_tracklet, group):
                continue
            score = _score_singleton_against_group(
                singleton_tracklet, group, forward_luts, config
            )
            if score is not None and score < config.recovery_residual_threshold:
                scored_pairs.append((score, s_idx, g_idx))

    # Sort ascending (best residual first)
    scored_pairs.sort(key=lambda x: x[0])

    assigned_singletons: set[int] = set()
    group_assignments: dict[int, list[Tracklet2D]] = {}  # g_idx -> new tracklets

    for _residual, s_idx, g_idx in scored_pairs:
        if s_idx in assigned_singletons:
            continue
        # Re-check overlap against updated group (with any already-absorbed tracklets)
        singleton_tracklet = singletons[s_idx].tracklets[0]
        if _g_idx_has_camera_overlap_with(
            singleton_tracklet, multi_groups[g_idx], group_assignments.get(g_idx, [])
        ):
            continue
        assigned_singletons.add(s_idx)
        group_assignments.setdefault(g_idx, []).append(singleton_tracklet)

    # -----------------------------------------------------------------------
    # Pass 2: Split-assign sweep for remaining singletons
    # -----------------------------------------------------------------------

    remaining_singletons = [
        (s_idx, singletons[s_idx])
        for s_idx in range(len(singletons))
        if s_idx not in assigned_singletons
    ]

    for s_idx, singleton in remaining_singletons:
        singleton_tracklet = singleton.tracklets[0]
        split_result = _attempt_split_assign(
            singleton_tracklet,
            multi_groups,
            group_assignments,
            forward_luts,
            config,
            next_track_id,
        )
        if split_result is not None:
            (before_seg, before_g_idx, after_seg, after_g_idx), used_ids = split_result
            next_track_id += used_ids
            assigned_singletons.add(s_idx)
            group_assignments.setdefault(before_g_idx, []).append(before_seg)
            group_assignments.setdefault(after_g_idx, []).append(after_seg)

    # -----------------------------------------------------------------------
    # Reassemble final group list
    # -----------------------------------------------------------------------
    updated_multi: list[TrackletGroup] = []
    for g_idx, group in enumerate(multi_groups):
        new_tracklets = group_assignments.get(g_idx, [])
        if new_tracklets:
            updated = TrackletGroup(
                fish_id=group.fish_id,
                tracklets=group.tracklets + tuple(new_tracklets),
                confidence=group.confidence,
                per_frame_confidence=None,  # invalidated by new member
                consensus_centroids=None,  # invalidated by new member
            )
            updated_multi.append(updated)
        else:
            updated_multi.append(group)

    remaining_singleton_groups = [
        singletons[s_idx]
        for s_idx in range(len(singletons))
        if s_idx not in assigned_singletons
    ]

    # Assign unique fish_ids to remaining singletons
    max_id = max((g.fish_id for g in updated_multi), default=-1)
    for i, sg in enumerate(remaining_singleton_groups):
        remaining_singleton_groups[i] = TrackletGroup(
            fish_id=max(max_id + 1 + i, 0),
            tracklets=sg.tracklets,
            confidence=sg.confidence,
            per_frame_confidence=sg.per_frame_confidence,
            consensus_centroids=sg.consensus_centroids,
        )

    n_assigned = len(assigned_singletons)
    if n_assigned > 0:
        logger.info(
            "Recovery: assigned %d/%d singletons to groups",
            n_assigned,
            len(singletons),
        )

    return updated_multi + remaining_singleton_groups


# ---------------------------------------------------------------------------
# Partitioning helpers
# ---------------------------------------------------------------------------


def _partition_groups(
    groups: list[TrackletGroup],
) -> tuple[list[TrackletGroup], list[TrackletGroup]]:
    """Split groups into multi-tracklet groups and singletons.

    Args:
        groups: All tracklet groups.

    Returns:
        Tuple of (multi_groups, singletons) where singletons have exactly
        one tracklet and multi_groups have more than one.
    """
    multi = [g for g in groups if len(g.tracklets) > 1]
    singletons = [g for g in groups if len(g.tracklets) == 1]
    return multi, singletons


def _compute_next_track_id(groups: list[TrackletGroup]) -> int:
    """Compute the next available track_id across all groups.

    Args:
        groups: All tracklet groups.

    Returns:
        Integer one greater than the maximum existing track_id.
    """
    max_id = -1
    for group in groups:
        for t in group.tracklets:
            if t.track_id > max_id:
                max_id = t.track_id
    return max_id + 1


# ---------------------------------------------------------------------------
# Overlap check
# ---------------------------------------------------------------------------


def _has_camera_overlap(singleton: Tracklet2D, group: TrackletGroup) -> bool:
    """Check if singleton's camera has detected-frame overlap with any group tracklet.

    Only ``frame_status == "detected"`` frames count as overlap. Coasted frames
    (Kalman predictions) do not block assignment.

    Args:
        singleton: The singleton tracklet being tested.
        group: The candidate group.

    Returns:
        True if the singleton's camera appears in the group with overlapping
        detected frames, False otherwise.
    """
    cam = singleton.camera_id
    det_s = {
        f
        for f, s in zip(singleton.frames, singleton.frame_status, strict=True)
        if s == "detected"
    }
    if not det_s:
        return False

    for t in group.tracklets:
        if t.camera_id != cam:
            continue
        det_t = {
            f for f, s in zip(t.frames, t.frame_status, strict=True) if s == "detected"
        }
        if det_s & det_t:
            return True
    return False


def _g_idx_has_camera_overlap_with(
    singleton: Tracklet2D,
    group: TrackletGroup,
    already_added: list[Tracklet2D],
) -> bool:
    """Check camera overlap against group plus any already-absorbed tracklets.

    Used during greedy assignment to prevent double-booking a camera slot.

    Args:
        singleton: The singleton tracklet being tested.
        group: The candidate group (original tracklets only).
        already_added: Tracklets already absorbed into this group during the
            current recovery pass.

    Returns:
        True if there is a detected-frame overlap conflict.
    """
    cam = singleton.camera_id
    det_s = {
        f
        for f, s in zip(singleton.frames, singleton.frame_status, strict=True)
        if s == "detected"
    }
    if not det_s:
        return False

    all_tracklets = list(group.tracklets) + already_added
    for t in all_tracklets:
        if t.camera_id != cam:
            continue
        det_t = {
            f for f, s in zip(t.frames, t.frame_status, strict=True) if s == "detected"
        }
        if det_s & det_t:
            return True
    return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_singleton_against_group(
    singleton: Tracklet2D,
    group: TrackletGroup,
    forward_luts: dict[str, ForwardLUT],
    config: RecoveryConfigLike,
) -> float | None:
    """Return mean per-keypoint ray-to-3D residual between singleton and group.

    Finds shared frames between the singleton and group's frame union,
    triangulates per-keypoint 3D positions from group member tracklets on-demand
    for those shared frames, casts rays for the singleton's confident keypoints,
    and computes the mean perpendicular distance.

    Falls back to centroid-only scoring if the singleton has no keypoints.

    Args:
        singleton: The singleton tracklet to score.
        group: The candidate multi-tracklet group.
        forward_luts: Per-camera ForwardLUT dict.
        config: Recovery configuration.

    Returns:
        Mean residual in metres, or None if insufficient overlap or no valid
        data could be extracted.
    """
    # Build frame union from group
    group_frames: set[int] = set()
    for t in group.tracklets:
        group_frames.update(t.frames)

    shared = sorted(set(singleton.frames) & group_frames)
    if len(shared) < config.recovery_min_shared_frames:
        return None

    if singleton.camera_id not in forward_luts:
        return None

    lut_singleton = forward_luts[singleton.camera_id]
    singleton_frame_map = {f: i for i, f in enumerate(singleton.frames)}

    use_keypoints = (
        singleton.keypoints is not None and singleton.keypoint_conf is not None
    )

    all_residuals: list[float] = []

    if use_keypoints:
        assert singleton.keypoints is not None
        assert singleton.keypoint_conf is not None
        n_kpts = singleton.keypoints.shape[1]

        for frame in shared:
            idx_s = singleton_frame_map[frame]
            kpts_s = singleton.keypoints[idx_s]  # (K, 2)
            conf_s = singleton.keypoint_conf[idx_s]  # (K,)

            # Triangulate group keypoints for this frame on-demand
            group_3d = _triangulate_group_keypoints_for_frame(
                frame,
                group.tracklets,
                forward_luts,
                config.keypoint_confidence_floor,
                n_kpts,
            )
            if group_3d is None:
                continue

            # For each confident singleton keypoint, compute ray-to-3D distance
            for k in range(n_kpts):
                if conf_s[k] < config.keypoint_confidence_floor:
                    continue
                pt_3d = group_3d[k]
                if pt_3d is None:
                    continue
                pix = torch.tensor(
                    [[float(kpts_s[k, 0]), float(kpts_s[k, 1])]],
                    dtype=torch.float32,
                )
                origins, dirs = lut_singleton.cast_ray(pix)
                origin = origins[0].cpu().numpy().astype(np.float64)
                direction = dirs[0].cpu().numpy().astype(np.float64)

                dist = _point_to_ray_distance(pt_3d, origin, direction)
                all_residuals.append(dist)

    else:
        # Centroid-only fallback
        for frame in shared:
            idx_s = singleton_frame_map[frame]
            centroid_s = singleton.centroids[idx_s]

            pix = torch.tensor(
                [[float(centroid_s[0]), float(centroid_s[1])]],
                dtype=torch.float32,
            )
            origins, dirs = lut_singleton.cast_ray(pix)
            origin = origins[0].cpu().numpy().astype(np.float64)
            direction = dirs[0].cpu().numpy().astype(np.float64)

            # Get group consensus centroid for this frame via centroid rays
            group_centroid = _triangulate_group_centroid_for_frame(
                frame, group.tracklets, forward_luts
            )
            if group_centroid is None:
                continue

            dist = _point_to_ray_distance(group_centroid, origin, direction)
            all_residuals.append(dist)

    if not all_residuals:
        return None

    return float(np.mean(all_residuals))


def _triangulate_group_keypoints_for_frame(
    frame: int,
    group_tracklets: tuple,
    forward_luts: dict[str, ForwardLUT],
    keypoint_confidence_floor: float,
    n_keypoints: int,
) -> list[np.ndarray | None] | None:
    """Triangulate per-keypoint 3D positions for one frame from group tracklets.

    For each keypoint k, collects rays from tracklets that have a confident
    keypoint k in this frame, computes pairwise ray-ray midpoints via
    ray_ray_closest_point, and averages them to obtain a consensus 3D position.

    Args:
        frame: The frame index to triangulate.
        group_tracklets: Tuple of Tracklet2D objects in the group.
        forward_luts: Per-camera ForwardLUT dict.
        keypoint_confidence_floor: Minimum confidence for keypoint inclusion.
        n_keypoints: Number of keypoints per tracklet.

    Returns:
        List of length n_keypoints where each entry is a (3,) array or None
        if fewer than 2 cameras have a confident keypoint for that index.
        Returns None if fewer than 2 cameras can contribute anything.
    """
    # Build per-tracklet frame index maps
    tracklet_frame_maps = [
        {f: i for i, f in enumerate(t.frames)} for t in group_tracklets
    ]

    # Check how many cameras are available for this frame
    cameras_in_frame = sum(
        1
        for t, fm in zip(group_tracklets, tracklet_frame_maps, strict=True)
        if frame in fm
    )
    if cameras_in_frame < 2:
        return None

    result: list[np.ndarray | None] = [None] * n_keypoints

    for k in range(n_keypoints):
        rays: list[tuple[np.ndarray, np.ndarray]] = []

        for t, fm in zip(group_tracklets, tracklet_frame_maps, strict=True):
            if frame not in fm:
                continue
            if t.keypoints is None or t.keypoint_conf is None:
                continue
            if t.camera_id not in forward_luts:
                continue

            idx = fm[frame]
            if t.keypoint_conf[idx, k] < keypoint_confidence_floor:
                continue

            kpt = t.keypoints[idx, k]  # (2,)
            pix = torch.tensor(
                [[float(kpt[0]), float(kpt[1])]],
                dtype=torch.float32,
            )
            lut = forward_luts[t.camera_id]
            origins, dirs = lut.cast_ray(pix)
            o = origins[0].cpu().numpy().astype(np.float64)
            d = dirs[0].cpu().numpy().astype(np.float64)
            rays.append((o, d))

        if len(rays) < 2:
            result[k] = None
            continue

        midpoints: list[np.ndarray] = []
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                _dist, midpoint = ray_ray_closest_point(
                    rays[i][0], rays[i][1], rays[j][0], rays[j][1]
                )
                midpoints.append(midpoint)

        result[k] = np.mean(midpoints, axis=0)

    return result


def _triangulate_group_centroid_for_frame(
    frame: int,
    group_tracklets: tuple,
    forward_luts: dict[str, ForwardLUT],
) -> np.ndarray | None:
    """Triangulate a consensus centroid from group tracklets for one frame.

    Uses centroid-based rays (fallback path for keypoints=None singletons).

    Args:
        frame: The frame index to triangulate.
        group_tracklets: Tuple of Tracklet2D objects in the group.
        forward_luts: Per-camera ForwardLUT dict.

    Returns:
        Consensus centroid as (3,) array, or None if fewer than 2 cameras
        contribute a ray for this frame.
    """
    rays: list[tuple[np.ndarray, np.ndarray]] = []

    for t in group_tracklets:
        frame_map = {f: i for i, f in enumerate(t.frames)}
        if frame not in frame_map:
            continue
        if t.camera_id not in forward_luts:
            continue
        idx = frame_map[frame]
        centroid = t.centroids[idx]
        pix = torch.tensor(
            [[float(centroid[0]), float(centroid[1])]],
            dtype=torch.float32,
        )
        lut = forward_luts[t.camera_id]
        origins, dirs = lut.cast_ray(pix)
        o = origins[0].cpu().numpy().astype(np.float64)
        d = dirs[0].cpu().numpy().astype(np.float64)
        rays.append((o, d))

    if len(rays) < 2:
        return None

    midpoints: list[np.ndarray] = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            _dist, midpoint = ray_ray_closest_point(
                rays[i][0], rays[i][1], rays[j][0], rays[j][1]
            )
            midpoints.append(midpoint)

    return np.mean(midpoints, axis=0)


# ---------------------------------------------------------------------------
# Split-assign
# ---------------------------------------------------------------------------


def _attempt_split_assign(
    singleton: Tracklet2D,
    multi_groups: list[TrackletGroup],
    existing_assignments: dict[int, list[Tracklet2D]],
    forward_luts: dict[str, ForwardLUT],
    config: RecoveryConfigLike,
    next_track_id: int,
) -> tuple[tuple[Tracklet2D, int, Tracklet2D, int], int] | None:
    """Attempt to split a singleton and assign each segment to different groups.

    Sweeps all valid split points, scores each segment pair against all groups,
    and picks the split minimising total residual where both segments match
    DIFFERENT groups below the threshold.

    Args:
        singleton: The singleton tracklet to potentially split.
        multi_groups: List of multi-tracklet groups to score against.
        existing_assignments: Map from group index to tracklets already absorbed
            in this recovery pass (used for overlap checking).
        forward_luts: Per-camera ForwardLUT dict.
        config: Recovery configuration.
        next_track_id: Starting track_id for the two new segment tracklets.

    Returns:
        Tuple of ((before_segment, before_group_idx, after_segment, after_group_idx),
        num_ids_used) if a valid split was found, otherwise None.
    """
    n_frames = len(singleton.frames)
    min_seg = config.recovery_min_segment_length

    if n_frames < 2 * min_seg:
        return None

    best: tuple[float, int, int, int] | None = None  # (total_res, split_idx, bg, ag)

    for split_idx in range(min_seg, n_frames - min_seg + 1):
        before = _slice_tracklet(singleton, 0, split_idx, next_track_id)
        after = _slice_tracklet(singleton, split_idx, n_frames, next_track_id + 1)

        # Find best-scoring group for each segment
        best_before: tuple[float, int] | None = None  # (residual, g_idx)
        best_after: tuple[float, int] | None = None

        for g_idx, group in enumerate(multi_groups):
            combined = list(group.tracklets) + existing_assignments.get(g_idx, [])

            # Check overlap for before segment
            if not _has_camera_overlap_with_list(before, combined):
                score_b = _score_singleton_against_group(
                    before, group, forward_luts, config
                )
                if (
                    score_b is not None
                    and score_b < config.recovery_residual_threshold
                    and (best_before is None or score_b < best_before[0])
                ):
                    best_before = (score_b, g_idx)

            # Check overlap for after segment
            if not _has_camera_overlap_with_list(after, combined):
                score_a = _score_singleton_against_group(
                    after, group, forward_luts, config
                )
                if (
                    score_a is not None
                    and score_a < config.recovery_residual_threshold
                    and (best_after is None or score_a < best_after[0])
                ):
                    best_after = (score_a, g_idx)

        # Both segments must match DIFFERENT groups
        if best_before is None or best_after is None:
            continue
        if best_before[1] == best_after[1]:
            continue

        total_residual = best_before[0] + best_after[0]
        if best is None or total_residual < best[0]:
            best = (total_residual, split_idx, best_before[1], best_after[1])

    if best is None:
        return None

    _, split_idx, before_g_idx, after_g_idx = best
    before_seg = _slice_tracklet(singleton, 0, split_idx, next_track_id)
    after_seg = _slice_tracklet(singleton, split_idx, n_frames, next_track_id + 1)

    return (before_seg, before_g_idx, after_seg, after_g_idx), 2


def _has_camera_overlap_with_list(
    tracklet: Tracklet2D, others: list[Tracklet2D]
) -> bool:
    """Check detected-frame overlap between a tracklet and a list of tracklets.

    Args:
        tracklet: Tracklet to check.
        others: List of candidate tracklets to check against.

    Returns:
        True if any tracklet in others shares the same camera_id with overlapping
        detected frames.
    """
    cam = tracklet.camera_id
    det_t = {
        f
        for f, s in zip(tracklet.frames, tracklet.frame_status, strict=True)
        if s == "detected"
    }
    if not det_t:
        return False

    for other in others:
        if other.camera_id != cam:
            continue
        det_o = {
            f
            for f, s in zip(other.frames, other.frame_status, strict=True)
            if s == "detected"
        }
        if det_t & det_o:
            return True
    return False


# ---------------------------------------------------------------------------
# Tracklet slicing
# ---------------------------------------------------------------------------


def _slice_tracklet(
    tracklet: Tracklet2D, start: int, end: int, new_id: int
) -> Tracklet2D:
    """Slice a Tracklet2D into a sub-segment [start:end].

    Args:
        tracklet: Original tracklet to slice.
        start: Start index (inclusive).
        end: End index (exclusive).
        new_id: Track ID for the new segment.

    Returns:
        New Tracklet2D instance with sliced fields.
    """
    from aquapose.core.tracking.types import Tracklet2D as Tracklet2DType

    kpts = tracklet.keypoints[start:end] if tracklet.keypoints is not None else None
    kconf = (
        tracklet.keypoint_conf[start:end]
        if tracklet.keypoint_conf is not None
        else None
    )
    return Tracklet2DType(
        camera_id=tracklet.camera_id,
        track_id=new_id,
        frames=tracklet.frames[start:end],
        centroids=tracklet.centroids[start:end],
        bboxes=tracklet.bboxes[start:end],
        frame_status=tracklet.frame_status[start:end],
        keypoints=kpts,
        keypoint_conf=kconf,
    )


# ---------------------------------------------------------------------------
# Geometric utilities (standalone — do not import from refinement.py)
# ---------------------------------------------------------------------------


def _point_to_ray_distance(
    point: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> float:
    """Compute perpendicular distance from a 3D point to a ray.

    Computes the minimum distance from ``point`` to the infinite ray defined
    by ``origin + t * direction``.

    Args:
        point: 3D point, shape (3,).
        origin: Ray origin, shape (3,).
        direction: Unit direction of the ray, shape (3,).

    Returns:
        Perpendicular distance in the same units as the inputs.
    """
    w = point - origin
    t = float(np.dot(w, direction))
    closest = origin + t * direction
    return float(np.linalg.norm(point - closest))
