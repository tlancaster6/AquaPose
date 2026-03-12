"""Multi-keypoint group validation with changepoint detection.

Validates tracklet membership in association clusters by computing per-tracklet
residual series from multi-keypoint ray distances. Detects temporal ID swaps
via threshold + run classifier changepoint detection and applies a keep/split/evict
decision tree. Replaces ``refinement.py`` with richer multi-keypoint analysis.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

from aquapose.core.association.scoring import (
    ray_ray_closest_point,
    ray_ray_closest_point_batch,
)
from aquapose.core.association.types import TrackletGroup

if TYPE_CHECKING:
    from aquapose.calibration.luts import ForwardLUT
    from aquapose.core.tracking.types import Tracklet2D

logger = logging.getLogger(__name__)

__all__ = ["ValidationConfigLike", "validate_groups"]


# ---------------------------------------------------------------------------
# Config protocol (IB-003: core/ must not import engine/)
# ---------------------------------------------------------------------------


@runtime_checkable
class ValidationConfigLike(Protocol):
    """Structural protocol for group validation configuration.

    Satisfied by ``AssociationConfig`` from ``engine.config`` without
    an explicit import, preserving the core -> engine import boundary.

    Attributes:
        eviction_reproj_threshold: Maximum ray-ray distance (metres) for
            a frame to be classified as consistent. Also used as the
            confidence normalisation denominator.
        min_cameras_validate: Minimum unique cameras in a group for
            validation to run. Groups below this threshold are returned
            unchanged.
        validation_enabled: Toggle to skip validation entirely.
        min_segment_length: Minimum frames per segment after a split.
            Both the consistent and inconsistent segments must meet this
            threshold for a split to be accepted.
        keypoint_confidence_floor: Minimum keypoint confidence for a
            keypoint to participate in residual computation.
    """

    eviction_reproj_threshold: float
    min_cameras_validate: int
    validation_enabled: bool
    min_segment_length: int
    keypoint_confidence_floor: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_groups(
    groups: list[TrackletGroup],
    forward_luts: dict[str, ForwardLUT],
    config: ValidationConfigLike,
) -> list[TrackletGroup]:
    """Validate groups via multi-keypoint residuals, split/evict bad tracklets.

    For each group with enough cameras, computes per-tracklet residual series
    from multi-keypoint ray distances, applies a keep/split/evict decision tree,
    and recomputes ``per_frame_confidence`` and ``consensus_centroids`` for
    cleaned groups.

    Args:
        groups: TrackletGroup list from clustering.
        forward_luts: Per-camera ForwardLUT dict for ray back-projection.
        config: Validation configuration thresholds.

    Returns:
        Validated list of TrackletGroups. Split/evicted tracklets become
        singleton groups. Groups below ``min_cameras_validate`` are returned
        unchanged.
    """
    if not config.validation_enabled:
        return groups

    # Compute next available track_id to avoid collisions after splits
    next_track_id = _compute_next_track_id(groups)

    validated: list[TrackletGroup] = []
    new_singletons: list[TrackletGroup] = []
    total_splits = 0
    total_evictions = 0

    for group in groups:
        tracklets: tuple[Tracklet2D, ...] = group.tracklets
        cam_ids = {t.camera_id for t in tracklets}
        n_cameras = len(cam_ids)

        if n_cameras < config.min_cameras_validate:
            validated.append(group)
            continue

        # Check all cameras have LUTs
        missing_luts = cam_ids - set(forward_luts.keys())
        if missing_luts:
            logger.warning(
                "Fish %d: missing ForwardLUTs for %s, skipping validation",
                group.fish_id,
                missing_luts,
            )
            validated.append(group)
            continue

        kept_tracklets: list[Tracklet2D] = []
        evicted_tracklets: list[Tracklet2D] = []
        split_singletons: list[Tracklet2D] = []

        for target in tracklets:
            others = [t for t in tracklets if t is not target]

            valid_frames, residuals = _compute_tracklet_residuals(
                target, others, forward_luts, config
            )

            if not residuals:
                # Not enough data to evaluate — keep as-is
                kept_tracklets.append(target)
                continue

            action, split_idx, consistent_is_before = _classify_tracklet(
                residuals,
                valid_frames,
                config.eviction_reproj_threshold,
                config.min_segment_length,
            )

            if action == "keep":
                kept_tracklets.append(target)
            elif action == "split":
                assert split_idx is not None
                # Map split_idx (index into valid_frames) back to tracklet index
                split_frame = valid_frames[split_idx]
                tracklet_split_idx = _find_tracklet_frame_index(target, split_frame)
                before, after = _split_tracklet_at(
                    target, tracklet_split_idx, next_track_id, next_track_id + 1
                )
                next_track_id += 2
                if consistent_is_before:
                    kept_tracklets.append(before)
                    split_singletons.append(after)
                else:
                    kept_tracklets.append(after)
                    split_singletons.append(before)
                total_splits += 1
            else:  # evict
                evicted_tracklets.append(target)
                total_evictions += 1

        # Check thin group dissolution: if only 1 camera remaining, dissolve
        kept_cam_ids = {t.camera_id for t in kept_tracklets}
        if len(kept_cam_ids) <= 1 and kept_tracklets:
            # Dissolve to singletons
            for t in kept_tracklets:
                evicted_tracklets.append(t)
            kept_tracklets = []

        # Create singleton groups for evicted and split-off tracklets
        for t in evicted_tracklets:
            new_singletons.append(
                TrackletGroup(
                    fish_id=-1,
                    tracklets=(t,),
                    confidence=0.1,
                    per_frame_confidence=None,
                    consensus_centroids=None,
                )
            )
        for t in split_singletons:
            new_singletons.append(
                TrackletGroup(
                    fish_id=-1,
                    tracklets=(t,),
                    confidence=0.1,
                    per_frame_confidence=None,
                    consensus_centroids=None,
                )
            )

        if not kept_tracklets:
            # All tracklets evicted — don't keep an empty group
            continue

        # Recompute consensus and confidence for cleaned group
        all_frames: set[int] = set()
        for t in kept_tracklets:
            all_frames.update(t.frames)
        frame_list = sorted(all_frames)

        kept_tuple = tuple(kept_tracklets)
        precomputed = _precompute_centroid_rays(frame_list, kept_tuple, forward_luts)
        frame_consensus = _compute_frame_consensus(
            frame_list, kept_tuple, forward_luts, precomputed_rays=precomputed
        )
        per_frame_conf = _compute_per_frame_confidence(
            frame_list,
            kept_tuple,
            frame_consensus,
            forward_luts,
            config.eviction_reproj_threshold,
            precomputed_rays=precomputed,
        )

        mean_conf = float(np.mean(per_frame_conf)) if per_frame_conf else None

        validated.append(
            TrackletGroup(
                fish_id=group.fish_id,
                tracklets=tuple(kept_tracklets),
                confidence=mean_conf,
                per_frame_confidence=tuple(per_frame_conf),
                consensus_centroids=tuple(
                    (f, frame_consensus.get(f)) for f in frame_list
                ),
            )
        )

    # Assign unique fish_ids to new singletons
    max_id = max((g.fish_id for g in validated), default=-1)
    for i, singleton in enumerate(new_singletons):
        new_singletons[i] = TrackletGroup(
            fish_id=max_id + 1 + i,
            tracklets=singleton.tracklets,
            confidence=singleton.confidence,
            per_frame_confidence=singleton.per_frame_confidence,
            consensus_centroids=singleton.consensus_centroids,
        )

    validated.extend(new_singletons)

    if total_splits > 0 or total_evictions > 0:
        logger.info(
            "Validation: %d splits, %d evictions from %d groups",
            total_splits,
            total_evictions,
            len(groups),
        )

    return validated


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------


def _compute_tracklet_residuals(
    target: Tracklet2D,
    others: list[Tracklet2D],
    forward_luts: dict[str, ForwardLUT],
    config: ValidationConfigLike,
) -> tuple[list[int], list[float]]:
    """Compute per-frame residual series for a target tracklet vs others.

    For each frame where the target is present, computes the mean ray-ray
    distance between the target's confident keypoints and the corresponding
    keypoints on other tracklets. Falls back to centroid-only residuals if
    keypoints are unavailable.

    Args:
        target: The tracklet being evaluated.
        others: Other tracklets in the group (excluding target).
        forward_luts: Per-camera ForwardLUT dict.
        config: Validation configuration.

    Returns:
        Tuple of (valid_frames, residuals) where valid_frames is the list
        of frame indices with valid residuals and residuals is the
        corresponding mean ray-ray distance per frame.
    """
    target_frame_map = {f: i for i, f in enumerate(target.frames)}
    other_frame_maps = [{f: i for i, f in enumerate(t.frames)} for t in others]

    valid_frames: list[int] = []
    residuals: list[float] = []

    use_keypoints = target.keypoints is not None and target.keypoint_conf is not None

    for frame in target.frames:
        idx_target = target_frame_map[frame]

        if use_keypoints:
            frame_residual = _compute_frame_residual_keypoints(
                target,
                idx_target,
                others,
                other_frame_maps,
                frame,
                forward_luts,
                config,
            )
        else:
            frame_residual = _compute_frame_residual_centroid(
                target,
                idx_target,
                others,
                other_frame_maps,
                frame,
                forward_luts,
            )

        if frame_residual is not None:
            valid_frames.append(frame)
            residuals.append(frame_residual)

    return valid_frames, residuals


def _compute_frame_residual_keypoints(
    target: Tracklet2D,
    idx_target: int,
    others: list[Tracklet2D],
    other_frame_maps: list[dict[int, int]],
    frame: int,
    forward_luts: dict[str, ForwardLUT],
    config: ValidationConfigLike,
) -> float | None:
    """Compute multi-keypoint residual for a single frame.

    Returns None if fewer than 2 cameras have confident keypoints.
    """
    assert target.keypoints is not None
    assert target.keypoint_conf is not None

    floor = config.keypoint_confidence_floor
    target_kpts = target.keypoints[idx_target]  # (K, 2)
    target_conf = target.keypoint_conf[idx_target]  # (K,)

    all_distances: list[float] = []

    for other, fm in zip(others, other_frame_maps, strict=True):
        if frame not in fm:
            continue
        if other.keypoints is None or other.keypoint_conf is None:
            continue

        idx_other = fm[frame]
        other_kpts = other.keypoints[idx_other]  # (K, 2)
        other_conf = other.keypoint_conf[idx_other]  # (K,)

        # Find keypoints confident in both target and other
        valid_mask = (target_conf >= floor) & (other_conf >= floor)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        # Cast rays for valid keypoints
        target_pixels = target_kpts[valid_indices].astype(np.float64)
        other_pixels = other_kpts[valid_indices].astype(np.float64)

        lut_target = forward_luts[target.camera_id]
        lut_other = forward_luts[other.camera_id]

        t_origins, t_dirs = lut_target.cast_ray(
            torch.tensor(target_pixels, dtype=torch.float32)
        )
        o_origins, o_dirs = lut_other.cast_ray(
            torch.tensor(other_pixels, dtype=torch.float32)
        )

        t_o = t_origins.cpu().numpy().astype(np.float64)
        t_d = t_dirs.cpu().numpy().astype(np.float64)
        o_o = o_origins.cpu().numpy().astype(np.float64)
        o_d = o_dirs.cpu().numpy().astype(np.float64)

        # Compute per-keypoint ray-ray distances (vectorized)
        dists = ray_ray_closest_point_batch(t_o, t_d, o_o, o_d)
        all_distances.extend(dists.tolist())

    if not all_distances:
        return None

    return float(np.mean(all_distances))


def _compute_frame_residual_centroid(
    target: Tracklet2D,
    idx_target: int,
    others: list[Tracklet2D],
    other_frame_maps: list[dict[int, int]],
    frame: int,
    forward_luts: dict[str, ForwardLUT],
) -> float | None:
    """Compute centroid-only residual for a single frame (fallback).

    Returns None if fewer than 1 other camera has this frame.
    """
    centroid_target = target.centroids[idx_target]
    lut_target = forward_luts[target.camera_id]
    pix_t = torch.tensor(
        [[float(centroid_target[0]), float(centroid_target[1])]],
        dtype=torch.float32,
    )
    t_origins, t_dirs = lut_target.cast_ray(pix_t)
    t_o = t_origins[0].cpu().numpy().astype(np.float64)
    t_d = t_dirs[0].cpu().numpy().astype(np.float64)

    distances: list[float] = []

    for other, fm in zip(others, other_frame_maps, strict=True):
        if frame not in fm:
            continue
        idx_other = fm[frame]
        centroid_other = other.centroids[idx_other]
        lut_other = forward_luts[other.camera_id]
        pix_o = torch.tensor(
            [[float(centroid_other[0]), float(centroid_other[1])]],
            dtype=torch.float32,
        )
        o_origins, o_dirs = lut_other.cast_ray(pix_o)
        o_o = o_origins[0].cpu().numpy().astype(np.float64)
        o_d = o_dirs[0].cpu().numpy().astype(np.float64)

        dist = ray_ray_closest_point_batch(
            t_o.reshape(1, 3),
            t_d.reshape(1, 3),
            o_o.reshape(1, 3),
            o_d.reshape(1, 3),
        )[0]
        distances.append(float(dist))

    if not distances:
        return None

    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# Changepoint detection
# ---------------------------------------------------------------------------


def _find_changepoint_by_run(
    residuals: list[float],
    frames: list[int],
    threshold: float,
    min_segment_length: int,
) -> tuple[int, bool] | None:
    """Find split index using longest-consistent-run heuristic.

    Classifies each frame as consistent (residual < threshold) or
    inconsistent, finds the longest contiguous consistent run, and
    returns the split index if both segments meet the minimum length.

    Args:
        residuals: Per-frame residual values.
        frames: Corresponding frame indices.
        threshold: Consistency threshold in metres.
        min_segment_length: Minimum frames per segment after split.

    Returns:
        Tuple of ``(split_idx, consistent_is_before)`` where *split_idx*
        is an index into ``residuals``/``frames`` and *consistent_is_before*
        indicates whether the consistent segment is ``[:split_idx]`` (True)
        or ``[split_idx:]`` (False).  Returns None if no valid split exists.
    """
    n = len(residuals)
    if n < 2 * min_segment_length:
        return None

    consistent = np.array(residuals) < threshold

    # Find run boundaries using diff on a padded array
    padded = np.concatenate([[False], consistent, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]  # run start indices
    ends = np.where(padded[:-1] & ~padded[1:])[0]  # run end indices (exclusive)

    if len(starts) == 0:
        return None  # no consistent frames at all

    lengths = ends - starts
    best_run_idx = int(np.argmax(lengths))
    best_end = int(ends[best_run_idx])  # exclusive

    best_length = int(lengths[best_run_idx])
    remaining_length = n - best_length

    # Both segments must meet min_segment_length
    if best_length < min_segment_length:
        return None
    if remaining_length < min_segment_length:
        return None

    # Return (split_index, consistent_is_before).
    # If the consistent run starts at 0, consistent = [:best_end].
    # Otherwise, consistent = [best_start:].
    best_start = int(starts[best_run_idx])
    if best_start == 0:
        return best_end, True
    return best_start, False


def _classify_tracklet(
    residuals: list[float],
    frames: list[int],
    threshold: float,
    min_segment_length: int,
) -> tuple[str, int | None, bool]:
    """Classify a tracklet as keep, split, or evict.

    Decision tree:
    1. If >50% frames are consistent (residual < threshold): keep
    2. If a changepoint is found with valid segments: split
    3. Otherwise: evict

    Args:
        residuals: Per-frame residual values.
        frames: Corresponding frame indices.
        threshold: Consistency threshold in metres.
        min_segment_length: Minimum frames per segment after split.

    Returns:
        Tuple of ``(action, split_idx, consistent_is_before)`` where
        *action* is ``"keep"``, ``"split"``, or ``"evict"``;
        *split_idx* is the index into frames/residuals at which to split
        (only for ``"split"``); and *consistent_is_before* indicates
        whether ``[:split_idx]`` is the consistent segment.
    """
    if not residuals:
        return "keep", None, True

    n_consistent = sum(1 for r in residuals if r < threshold)
    fraction_consistent = n_consistent / len(residuals)

    if fraction_consistent > 0.5:
        return "keep", None, True

    # Check for a changepoint
    result = _find_changepoint_by_run(residuals, frames, threshold, min_segment_length)
    if result is not None:
        split_idx, consistent_is_before = result
        return "split", split_idx, consistent_is_before

    # Uniformly inconsistent — evict
    return "evict", None, True


# ---------------------------------------------------------------------------
# Tracklet splitting
# ---------------------------------------------------------------------------


def _split_tracklet_at(
    tracklet: Tracklet2D,
    split_idx: int,
    id_before: int,
    id_after: int,
) -> tuple[Tracklet2D, Tracklet2D]:
    """Split a Tracklet2D at split_idx into two new tracklets.

    The ``before`` segment (frames[:split_idx]) gets ``id_before`` and the
    ``after`` segment (frames[split_idx:]) gets ``id_after``.

    Args:
        tracklet: Original tracklet to split.
        split_idx: Index at which to split (exclusive for before, inclusive
            for after).
        id_before: Track ID for the before segment.
        id_after: Track ID for the after segment.

    Returns:
        Tuple of (before, after) Tracklet2D instances.
    """
    from aquapose.core.tracking.types import Tracklet2D

    before = Tracklet2D(
        camera_id=tracklet.camera_id,
        track_id=id_before,
        frames=tracklet.frames[:split_idx],
        centroids=tracklet.centroids[:split_idx],
        bboxes=tracklet.bboxes[:split_idx],
        frame_status=tracklet.frame_status[:split_idx],
        keypoints=(
            tracklet.keypoints[:split_idx] if tracklet.keypoints is not None else None
        ),
        keypoint_conf=(
            tracklet.keypoint_conf[:split_idx]
            if tracklet.keypoint_conf is not None
            else None
        ),
    )
    after = Tracklet2D(
        camera_id=tracklet.camera_id,
        track_id=id_after,
        frames=tracklet.frames[split_idx:],
        centroids=tracklet.centroids[split_idx:],
        bboxes=tracklet.bboxes[split_idx:],
        frame_status=tracklet.frame_status[split_idx:],
        keypoints=(
            tracklet.keypoints[split_idx:] if tracklet.keypoints is not None else None
        ),
        keypoint_conf=(
            tracklet.keypoint_conf[split_idx:]
            if tracklet.keypoint_conf is not None
            else None
        ),
    )
    return before, after


def _find_tracklet_frame_index(tracklet: Tracklet2D, frame: int) -> int:
    """Find the index of a frame in a tracklet's frames tuple.

    Args:
        tracklet: Tracklet to search.
        frame: Frame number to find.

    Returns:
        Index into tracklet.frames.

    Raises:
        ValueError: If frame is not in tracklet.frames.
    """
    for i, f in enumerate(tracklet.frames):
        if f == frame:
            return i
    msg = f"Frame {frame} not in tracklet {tracklet.camera_id}:{tracklet.track_id}"
    raise ValueError(msg)


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
# Consensus and confidence (copied from refinement.py)
# ---------------------------------------------------------------------------

# Type alias for precomputed rays: tracklet_index -> {frame -> (origin, dir)}
_PrecomputedRays = dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]


def _precompute_centroid_rays(
    frame_list: list[int],
    tracklets: tuple[Tracklet2D, ...] | tuple,
    forward_luts: dict[str, ForwardLUT],
) -> _PrecomputedRays:
    """Batch cast_ray calls per camera across all frames.

    Groups centroid pixels by camera and makes one cast_ray call per camera
    instead of one per (frame, tracklet). Returns a nested dict indexed by
    tracklet index and frame number.

    Args:
        frame_list: Sorted list of frame indices to process.
        tracklets: Tracklets in the cluster.
        forward_luts: Per-camera ForwardLUTs.

    Returns:
        Dict mapping tracklet index -> {frame -> (origin_3d, dir_3d)}.
    """
    frame_set = set(frame_list)
    # Collect per-camera: list of (tracklet_idx, frame, pixel)
    cam_entries: dict[str, list[tuple[int, int, np.ndarray]]] = {}
    for t_idx, t in enumerate(tracklets):
        cam = t.camera_id
        for i, f in enumerate(t.frames):
            if f not in frame_set:
                continue
            centroid = t.centroids[i]
            cam_entries.setdefault(cam, []).append(
                (t_idx, f, np.array([float(centroid[0]), float(centroid[1])]))
            )

    result: _PrecomputedRays = {}

    for cam_id, entries in cam_entries.items():
        pixels = np.array([e[2] for e in entries], dtype=np.float64)
        pix_t = torch.tensor(pixels, dtype=torch.float32)
        origins, dirs = forward_luts[cam_id].cast_ray(pix_t)
        origins_np = origins.cpu().numpy().astype(np.float64)
        dirs_np = dirs.cpu().numpy().astype(np.float64)
        for idx, (t_idx, frame, _) in enumerate(entries):
            result.setdefault(t_idx, {})[frame] = (origins_np[idx], dirs_np[idx])

    return result


def _compute_frame_consensus(
    frame_list: list[int],
    tracklets: tuple[Tracklet2D, ...] | tuple,
    forward_luts: dict[str, ForwardLUT],
    precomputed_rays: _PrecomputedRays | None = None,
) -> dict[int, np.ndarray | None]:
    """Compute per-frame consensus 3D point from tracklet ray intersections.

    Uses centroid-based rays for downstream compatibility with
    ``consensus_centroids`` consumers.

    Args:
        frame_list: Sorted list of frame indices to process.
        tracklets: Tracklets in the cluster.
        forward_luts: Per-camera ForwardLUTs.
        precomputed_rays: Optional precomputed rays from
            ``_precompute_centroid_rays``. If None, rays are computed inline.

    Returns:
        Dict mapping frame index to consensus 3D point (shape (3,)),
        or None if fewer than 2 rays available for that frame.
    """
    if precomputed_rays is None:
        precomputed_rays = _precompute_centroid_rays(
            frame_list, tracklets, forward_luts
        )

    consensus: dict[int, np.ndarray | None] = {}

    for frame in frame_list:
        rays: list[tuple[np.ndarray, np.ndarray]] = []

        for t_idx in range(len(tracklets)):
            t_rays = precomputed_rays.get(t_idx)
            if t_rays is not None and frame in t_rays:
                rays.append(t_rays[frame])

        if len(rays) < 2:
            consensus[frame] = None
            continue

        # Pairwise midpoints — small count (C(cameras,2) ~ 3-15), needs midpoints
        # so scalar ray_ray_closest_point is appropriate here
        midpoints: list[np.ndarray] = []
        distances: list[float] = []
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                dist, midpoint = ray_ray_closest_point(
                    rays[i][0], rays[i][1], rays[j][0], rays[j][1]
                )
                midpoints.append(midpoint)
                distances.append(dist)

        if distances:
            n_keep = max(1, len(distances) // 2)
            sorted_idx = np.argsort(distances)
            best_midpoints = [midpoints[i] for i in sorted_idx[:n_keep]]
            consensus[frame] = np.mean(best_midpoints, axis=0)
        else:
            consensus[frame] = None

    return consensus


def _compute_per_frame_confidence(
    frame_list: list[int],
    tracklets: tuple[Tracklet2D, ...] | tuple,
    frame_consensus: dict[int, np.ndarray | None],
    forward_luts: dict[str, ForwardLUT],
    threshold: float,
    precomputed_rays: _PrecomputedRays | None = None,
) -> list[float]:
    """Compute per-frame confidence from ray convergence quality.

    Confidence = 1.0 - (mean_pairwise_distance / threshold), clamped [0, 1].
    Frames without consensus get confidence 0.0.

    Args:
        frame_list: Sorted frame indices.
        tracklets: Tracklets in the cluster.
        frame_consensus: Per-frame consensus 3D points.
        forward_luts: Per-camera ForwardLUTs.
        threshold: Eviction threshold in metres.
        precomputed_rays: Optional precomputed rays from
            ``_precompute_centroid_rays``. If None, rays are computed inline.

    Returns:
        List of confidence values, one per frame in frame_list.
    """
    if precomputed_rays is None:
        precomputed_rays = _precompute_centroid_rays(
            frame_list, tracklets, forward_luts
        )

    confidences: list[float] = []

    for frame in frame_list:
        consensus_pt = frame_consensus.get(frame)
        if consensus_pt is None:
            confidences.append(0.0)
            continue

        rays: list[tuple[np.ndarray, np.ndarray]] = []
        for t_idx in range(len(tracklets)):
            t_rays = precomputed_rays.get(t_idx)
            if t_rays is not None and frame in t_rays:
                rays.append(t_rays[frame])

        if len(rays) < 2:
            confidences.append(0.0)
            continue

        # Use batch for pairwise distances (no midpoints needed)
        oa = np.array([r[0] for r in rays])  # (R, 3)
        da = np.array([r[1] for r in rays])  # (R, 3)
        ii, jj = np.triu_indices(len(rays), k=1)
        pair_dists = ray_ray_closest_point_batch(oa[ii], da[ii], oa[jj], da[jj])
        mean_dist = float(pair_dists.mean())
        conf = max(0.0, min(1.0, 1.0 - mean_dist / threshold))
        confidences.append(conf)

    return confidences
