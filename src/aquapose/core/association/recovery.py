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

from aquapose.core.association.scoring import (
    ray_ray_closest_point,
)
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

        # Collect ALL confident singleton keypoint pixels across ALL shared frames
        all_pixels: list[np.ndarray] = []
        pixel_meta: list[tuple[int, int]] = []  # (frame, kpt_idx)
        for frame in shared:
            idx_s = singleton_frame_map[frame]
            conf_s = singleton.keypoint_conf[idx_s]
            kpts_s = singleton.keypoints[idx_s]
            for k in range(n_kpts):
                if conf_s[k] < config.keypoint_confidence_floor:
                    continue
                all_pixels.append(np.array([float(kpts_s[k, 0]), float(kpts_s[k, 1])]))
                pixel_meta.append((frame, k))

        if not all_pixels:
            return None

        # Batch cast_ray for ALL singleton pixels at once
        pixels_batch = np.array(all_pixels, dtype=np.float64)
        pix_t = torch.tensor(pixels_batch, dtype=torch.float32)
        s_origins, s_dirs = lut_singleton.cast_ray(pix_t)
        s_o = s_origins.cpu().numpy().astype(np.float64)
        s_d = s_dirs.cpu().numpy().astype(np.float64)

        # Cache group 3D triangulations per frame
        frame_3d_cache: dict[int, list[np.ndarray | None] | None] = {}
        for i, (frame, k) in enumerate(pixel_meta):
            if frame not in frame_3d_cache:
                frame_3d_cache[frame] = _triangulate_group_keypoints_for_frame(
                    frame,
                    group.tracklets,
                    forward_luts,
                    config.keypoint_confidence_floor,
                    n_kpts,
                )
            group_3d = frame_3d_cache[frame]
            if group_3d is None:
                continue
            pt_3d = group_3d[k]
            if pt_3d is None:
                continue
            dist = _point_to_ray_distance(pt_3d, s_o[i], s_d[i])
            all_residuals.append(dist)

    else:
        # Centroid-only fallback — batch all singleton centroid pixels
        centroid_pixels: list[np.ndarray] = []
        centroid_frames: list[int] = []
        for frame in shared:
            idx_s = singleton_frame_map[frame]
            centroid_s = singleton.centroids[idx_s]
            centroid_pixels.append(
                np.array([float(centroid_s[0]), float(centroid_s[1])])
            )
            centroid_frames.append(frame)

        if not centroid_pixels:
            return None

        pixels_batch = np.array(centroid_pixels, dtype=np.float64)
        pix_t = torch.tensor(pixels_batch, dtype=torch.float32)
        s_origins, s_dirs = lut_singleton.cast_ray(pix_t)
        s_o = s_origins.cpu().numpy().astype(np.float64)
        s_d = s_dirs.cpu().numpy().astype(np.float64)

        for i, frame in enumerate(centroid_frames):
            group_centroid = _triangulate_group_centroid_for_frame(
                frame, group.tracklets, forward_luts
            )
            if group_centroid is None:
                continue
            dist = _point_to_ray_distance(group_centroid, s_o[i], s_d[i])
            all_residuals.append(dist)

    if not all_residuals:
        return None

    return float(np.mean(all_residuals))


def _compute_per_frame_residuals(
    singleton: Tracklet2D,
    group: TrackletGroup,
    forward_luts: dict[str, ForwardLUT],
    config: RecoveryConfigLike,
) -> np.ndarray | None:
    """Compute per-frame residuals for singleton vs group.

    Same computation as ``_score_singleton_against_group`` but returns
    per-singleton-frame residuals instead of a single mean. Frames without
    valid data get np.nan.

    Args:
        singleton: The singleton tracklet to score.
        group: The candidate multi-tracklet group.
        forward_luts: Per-camera ForwardLUT dict.
        config: Recovery configuration.

    Returns:
        Array of shape (n_singleton_frames,) with per-frame mean residuals,
        or None if no shared frames or camera missing.
    """
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
    n_frames = len(singleton.frames)
    residuals = np.full(n_frames, np.nan)

    use_keypoints = (
        singleton.keypoints is not None and singleton.keypoint_conf is not None
    )

    if use_keypoints:
        assert singleton.keypoints is not None
        assert singleton.keypoint_conf is not None
        n_kpts = singleton.keypoints.shape[1]

        # Batch all confident singleton keypoint pixels
        all_pixels: list[np.ndarray] = []
        pixel_meta: list[tuple[int, int, int]] = []  # (frame, kpt_idx, frame_idx)
        for frame in shared:
            idx_s = singleton_frame_map[frame]
            conf_s = singleton.keypoint_conf[idx_s]
            kpts_s = singleton.keypoints[idx_s]
            for k in range(n_kpts):
                if conf_s[k] < config.keypoint_confidence_floor:
                    continue
                all_pixels.append(np.array([float(kpts_s[k, 0]), float(kpts_s[k, 1])]))
                pixel_meta.append((frame, k, idx_s))

        if not all_pixels:
            return None

        pixels_batch = np.array(all_pixels, dtype=np.float64)
        pix_t = torch.tensor(pixels_batch, dtype=torch.float32)
        s_origins, s_dirs = lut_singleton.cast_ray(pix_t)
        s_o = s_origins.cpu().numpy().astype(np.float64)
        s_d = s_dirs.cpu().numpy().astype(np.float64)

        frame_3d_cache: dict[int, list[np.ndarray | None] | None] = {}
        # Collect per-frame residual lists
        frame_dists: dict[int, list[float]] = {}
        for i, (frame, k, idx_s) in enumerate(pixel_meta):
            if frame not in frame_3d_cache:
                frame_3d_cache[frame] = _triangulate_group_keypoints_for_frame(
                    frame,
                    group.tracklets,
                    forward_luts,
                    config.keypoint_confidence_floor,
                    n_kpts,
                )
            group_3d = frame_3d_cache[frame]
            if group_3d is None:
                continue
            pt_3d = group_3d[k]
            if pt_3d is None:
                continue
            dist = _point_to_ray_distance(pt_3d, s_o[i], s_d[i])
            frame_dists.setdefault(idx_s, []).append(dist)

        for idx_s, dists in frame_dists.items():
            residuals[idx_s] = float(np.mean(dists))

    else:
        # Centroid-only fallback
        centroid_pixels: list[np.ndarray] = []
        centroid_meta: list[tuple[int, int]] = []  # (frame, idx_s)
        for frame in shared:
            idx_s = singleton_frame_map[frame]
            centroid_s = singleton.centroids[idx_s]
            centroid_pixels.append(
                np.array([float(centroid_s[0]), float(centroid_s[1])])
            )
            centroid_meta.append((frame, idx_s))

        if not centroid_pixels:
            return None

        pixels_batch = np.array(centroid_pixels, dtype=np.float64)
        pix_t = torch.tensor(pixels_batch, dtype=torch.float32)
        s_origins, s_dirs = lut_singleton.cast_ray(pix_t)
        s_o = s_origins.cpu().numpy().astype(np.float64)
        s_d = s_dirs.cpu().numpy().astype(np.float64)

        for i, (frame, idx_s) in enumerate(centroid_meta):
            group_centroid = _triangulate_group_centroid_for_frame(
                frame, group.tracklets, forward_luts
            )
            if group_centroid is None:
                continue
            dist = _point_to_ray_distance(group_centroid, s_o[i], s_d[i])
            residuals[idx_s] = dist

    if np.all(np.isnan(residuals)):
        return None

    return residuals


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

    # Batch cast_ray per camera: collect (camera, keypoint_idx, pixel) tuples
    cam_kpt_entries: dict[str, list[tuple[int, np.ndarray]]] = {}
    for t, fm in zip(group_tracklets, tracklet_frame_maps, strict=True):
        if frame not in fm:
            continue
        if t.keypoints is None or t.keypoint_conf is None:
            continue
        if t.camera_id not in forward_luts:
            continue
        idx = fm[frame]
        for k in range(n_keypoints):
            if t.keypoint_conf[idx, k] < keypoint_confidence_floor:
                continue
            cam_kpt_entries.setdefault(t.camera_id, []).append((k, t.keypoints[idx, k]))

    # Batch cast_ray per camera
    kpt_rays: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}  # k -> [(o, d)]
    for cam_id, entries in cam_kpt_entries.items():
        pixels = np.array([e[1] for e in entries], dtype=np.float64)
        pix_t = torch.tensor(pixels, dtype=torch.float32)
        origins, dirs = forward_luts[cam_id].cast_ray(pix_t)
        origins_np = origins.cpu().numpy().astype(np.float64)
        dirs_np = dirs.cpu().numpy().astype(np.float64)
        for i, (k, _) in enumerate(entries):
            kpt_rays.setdefault(k, []).append((origins_np[i], dirs_np[i]))

    result: list[np.ndarray | None] = [None] * n_keypoints

    for k in range(n_keypoints):
        rays = kpt_rays.get(k, [])
        if len(rays) < 2:
            result[k] = None
            continue

        # Pairwise midpoints — small count (C(cameras,2)), needs midpoints
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
    # Collect centroid pixels grouped by camera for batch cast_ray
    cam_centroids: dict[str, list[np.ndarray]] = {}
    for t in group_tracklets:
        frame_map = {f: i for i, f in enumerate(t.frames)}
        if frame not in frame_map:
            continue
        if t.camera_id not in forward_luts:
            continue
        idx = frame_map[frame]
        centroid = t.centroids[idx]
        cam_centroids.setdefault(t.camera_id, []).append(
            np.array([float(centroid[0]), float(centroid[1])])
        )

    rays: list[tuple[np.ndarray, np.ndarray]] = []
    for cam_id, centroids in cam_centroids.items():
        pixels = np.array(centroids, dtype=np.float64)
        pix_t = torch.tensor(pixels, dtype=torch.float32)
        origins, dirs = forward_luts[cam_id].cast_ray(pix_t)
        origins_np = origins.cpu().numpy().astype(np.float64)
        dirs_np = dirs.cpu().numpy().astype(np.float64)
        for i in range(len(centroids)):
            rays.append((origins_np[i], dirs_np[i]))

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

    n_groups = len(multi_groups)

    # Phase 1: Precompute per-frame residuals for all groups
    residuals = np.full((n_groups, n_frames), np.nan)
    for g_idx, group in enumerate(multi_groups):
        per_frame = _compute_per_frame_residuals(singleton, group, forward_luts, config)
        if per_frame is not None:
            residuals[g_idx] = per_frame

    # Phase 2: Precompute per-group detected frame sets for overlap checking
    singleton_det_mask = np.array([s == "detected" for s in singleton.frame_status])
    singleton_frame_arr = np.array(singleton.frames)
    cam = singleton.camera_id

    group_det_frames: list[set[int]] = []
    for g_idx, group in enumerate(multi_groups):
        det: set[int] = set()
        all_tracklets = list(group.tracklets) + existing_assignments.get(g_idx, [])
        for t in all_tracklets:
            if t.camera_id != cam:
                continue
            det.update(
                f
                for f, s in zip(t.frames, t.frame_status, strict=True)
                if s == "detected"
            )
        group_det_frames.append(det)

    # Phase 3: Sweep split points using precomputed data
    best: tuple[float, int, int, int] | None = None  # (total_res, split_idx, bg, ag)
    threshold = config.recovery_residual_threshold

    for split_idx in range(min_seg, n_frames - min_seg + 1):
        # Compute mean residuals for before/after segments via numpy slicing
        with np.errstate(all="ignore"):
            before_means = np.nanmean(residuals[:, :split_idx], axis=1)
            after_means = np.nanmean(residuals[:, split_idx:], axis=1)

        # Compute detected frame sets for overlap checking
        before_det = set(
            singleton_frame_arr[:split_idx][singleton_det_mask[:split_idx]]
        )
        after_det = set(singleton_frame_arr[split_idx:][singleton_det_mask[split_idx:]])

        # Invalidate groups with camera overlap
        for g_idx in range(n_groups):
            g_det = group_det_frames[g_idx]
            if before_det and (before_det & g_det):
                before_means[g_idx] = np.inf
            if after_det and (after_det & g_det):
                after_means[g_idx] = np.inf

        # Apply threshold
        before_valid = before_means < threshold
        after_valid = after_means < threshold

        if not np.any(before_valid) or not np.any(after_valid):
            continue

        # Find best groups (must be DIFFERENT)
        best_before_g = int(np.argmin(np.where(before_valid, before_means, np.inf)))
        best_after_g = int(np.argmin(np.where(after_valid, after_means, np.inf)))

        if best_before_g == best_after_g:
            # Try second-best for one of them
            after_means_masked = after_means.copy()
            after_means_masked[best_before_g] = np.inf
            if np.any(after_means_masked < threshold):
                best_after_g = int(
                    np.argmin(
                        np.where(
                            after_means_masked < threshold, after_means_masked, np.inf
                        )
                    )
                )
            else:
                before_means_masked = before_means.copy()
                before_means_masked[best_after_g] = np.inf
                if np.any(before_means_masked < threshold):
                    best_before_g = int(
                        np.argmin(
                            np.where(
                                before_means_masked < threshold,
                                before_means_masked,
                                np.inf,
                            )
                        )
                    )
                else:
                    continue

        if best_before_g == best_after_g:
            continue

        total = before_means[best_before_g] + after_means[best_after_g]
        if best is None or total < best[0]:
            best = (total, split_idx, best_before_g, best_after_g)

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


def _point_to_ray_distance_batch(
    points: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Batch perpendicular distance from N 3D points to N rays.

    Args:
        points: 3D points, shape (N, 3).
        origins: Ray origins, shape (N, 3).
        directions: Unit directions of rays, shape (N, 3).

    Returns:
        Perpendicular distances, shape (N,).
    """
    w = points - origins  # (N, 3)
    t = np.sum(w * directions, axis=1, keepdims=True)  # (N, 1)
    closest = origins + t * directions  # (N, 3)
    return np.linalg.norm(points - closest, axis=1)  # (N,)
