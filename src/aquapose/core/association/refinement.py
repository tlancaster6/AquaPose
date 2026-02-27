"""3D triangulation refinement for cross-camera identity clusters.

Validates tracklet membership in association clusters by back-projecting
centroids to rays via ForwardLUTs and computing per-frame 3D consensus
positions. Tracklets with consistently high ray-to-consensus distance are
evicted to singleton groups.
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

__all__ = ["RefinementConfigLike", "refine_clusters"]


# ---------------------------------------------------------------------------
# Config protocol (IB-003: core/ must not import engine/)
# ---------------------------------------------------------------------------


@runtime_checkable
class RefinementConfigLike(Protocol):
    """Structural protocol for cluster refinement configuration.

    Satisfied by ``AssociationConfig`` from ``engine.config`` without
    an explicit import, preserving the core -> engine import boundary.

    Attributes:
        eviction_reproj_threshold: Maximum median ray-to-consensus distance
            (metres) for a tracklet to remain in its cluster.
        min_cameras_refine: Minimum cameras in a cluster for 3D refinement.
        refinement_enabled: Toggle to skip refinement entirely.
    """

    eviction_reproj_threshold: float
    min_cameras_refine: int
    refinement_enabled: bool


# ---------------------------------------------------------------------------
# Core refinement logic
# ---------------------------------------------------------------------------


def refine_clusters(
    groups: list[TrackletGroup],
    forward_luts: dict[str, ForwardLUT],
    config: RefinementConfigLike,
) -> list[TrackletGroup]:
    """Refine association clusters via per-frame 3D triangulation error.

    For each cluster with enough cameras, back-projects tracklet centroids
    to rays, computes per-frame consensus 3D positions, and evicts tracklets
    whose median ray-to-consensus distance exceeds the threshold.

    Args:
        groups: TrackletGroup list from clustering/merging.
        forward_luts: Per-camera ForwardLUT dict for ray back-projection.
        config: Refinement configuration thresholds.

    Returns:
        Refined list of TrackletGroups. Evicted tracklets become singleton
        groups with low confidence. Groups below min_cameras_refine are
        returned unchanged.
    """
    if not config.refinement_enabled:
        return groups

    refined: list[TrackletGroup] = []
    evicted_singletons: list[TrackletGroup] = []

    for group in groups:
        tracklets: tuple[Tracklet2D, ...] = group.tracklets
        # Count unique cameras
        cam_ids = {t.camera_id for t in tracklets}
        n_cameras = len(cam_ids)

        if n_cameras < config.min_cameras_refine:
            # Skip refinement, keep as-is
            refined.append(group)
            continue

        # Check all cameras have LUTs
        missing_luts = cam_ids - set(forward_luts.keys())
        if missing_luts:
            logger.warning(
                "Fish %d: missing ForwardLUTs for %s, skipping refinement",
                group.fish_id,
                missing_luts,
            )
            refined.append(group)
            continue

        # Build union of all frames
        all_frames: set[int] = set()
        for t in tracklets:
            all_frames.update(t.frames)
        frame_list = sorted(all_frames)

        # Per-frame triangulation
        frame_consensus = _compute_frame_consensus(frame_list, tracklets, forward_luts)

        # Per-tracklet median distance to consensus
        tracklet_distances = _compute_tracklet_distances(
            tracklets, frame_list, frame_consensus, forward_luts
        )

        # Evict tracklets exceeding threshold
        kept_tracklets: list[Tracklet2D] = []
        evicted_tracklets: list[Tracklet2D] = []

        for t, median_dist in zip(tracklets, tracklet_distances, strict=True):
            if median_dist > config.eviction_reproj_threshold:
                evicted_tracklets.append(t)
            else:
                kept_tracklets.append(t)

        # Create singleton groups for evicted tracklets
        for t in evicted_tracklets:
            evicted_singletons.append(
                TrackletGroup(
                    fish_id=-1,  # Will be reassigned
                    tracklets=(t,),
                    confidence=0.1,
                    per_frame_confidence=None,
                )
            )

        if not kept_tracklets:
            # All tracklets evicted -- should not normally happen
            logger.warning(
                "Fish %d: all tracklets evicted, keeping original group",
                group.fish_id,
            )
            refined.append(group)
            continue

        # Re-triangulate cleaned cluster for updated confidence
        cleaned_consensus = _compute_frame_consensus(
            frame_list, tuple(kept_tracklets), forward_luts
        )
        per_frame_conf = _compute_per_frame_confidence(
            frame_list,
            tuple(kept_tracklets),
            cleaned_consensus,
            forward_luts,
            config.eviction_reproj_threshold,
        )

        mean_conf = float(np.mean(per_frame_conf)) if per_frame_conf else None

        refined.append(
            TrackletGroup(
                fish_id=group.fish_id,
                tracklets=tuple(kept_tracklets),
                confidence=mean_conf,
                per_frame_confidence=tuple(per_frame_conf),
            )
        )

    # Assign unique IDs to evicted singletons
    max_id = max((g.fish_id for g in refined), default=-1)
    for i, singleton in enumerate(evicted_singletons):
        evicted_singletons[i] = TrackletGroup(
            fish_id=max_id + 1 + i,
            tracklets=singleton.tracklets,
            confidence=singleton.confidence,
            per_frame_confidence=singleton.per_frame_confidence,
        )

    refined.extend(evicted_singletons)

    n_evicted = len(evicted_singletons)
    if n_evicted > 0:
        logger.info(
            "Refinement: %d tracklets evicted from %d groups",
            n_evicted,
            len(groups),
        )

    return refined


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_frame_consensus(
    frame_list: list[int],
    tracklets: tuple[Tracklet2D, ...] | tuple,
    forward_luts: dict[str, ForwardLUT],
) -> dict[int, np.ndarray | None]:
    """Compute per-frame consensus 3D point from tracklet ray intersections.

    Args:
        frame_list: Sorted list of frame indices to process.
        tracklets: Tracklets in the cluster.
        forward_luts: Per-camera ForwardLUTs.

    Returns:
        Dict mapping frame index to consensus 3D point (shape (3,)),
        or None if fewer than 2 rays available for that frame.
    """
    # Build per-tracklet frame->index lookup
    tracklet_frame_maps: list[dict[int, int]] = []
    for t in tracklets:
        tracklet_frame_maps.append({f: i for i, f in enumerate(t.frames)})

    consensus: dict[int, np.ndarray | None] = {}

    for frame in frame_list:
        # Collect rays from tracklets that have this frame
        rays: list[tuple[np.ndarray, np.ndarray]] = []

        for t, frame_map in zip(tracklets, tracklet_frame_maps, strict=True):
            if frame not in frame_map:
                continue
            idx = frame_map[frame]
            centroid = t.centroids[idx]
            lut = forward_luts[t.camera_id]
            pix = torch.tensor(
                [[float(centroid[0]), float(centroid[1])]], dtype=torch.float32
            )
            origins, dirs = lut.cast_ray(pix)
            o = origins[0].cpu().numpy().astype(np.float64)
            d = dirs[0].cpu().numpy().astype(np.float64)
            rays.append((o, d))

        if len(rays) < 2:
            consensus[frame] = None
            continue

        # Compute all pairwise closest points and distances
        midpoints: list[np.ndarray] = []
        distances: list[float] = []
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                dist, midpoint = ray_ray_closest_point(
                    rays[i][0], rays[i][1], rays[j][0], rays[j][1]
                )
                midpoints.append(midpoint)
                distances.append(dist)

        # Use only the tightest pairs for robust consensus.
        # Sort by distance and keep the best 50% (at least 1 pair).
        if distances:
            n_keep = max(1, len(distances) // 2)
            sorted_idx = np.argsort(distances)
            best_midpoints = [midpoints[i] for i in sorted_idx[:n_keep]]
            consensus[frame] = np.mean(best_midpoints, axis=0)
        else:
            consensus[frame] = None

    return consensus


def _compute_tracklet_distances(
    tracklets: tuple[Tracklet2D, ...] | tuple,
    frame_list: list[int],
    frame_consensus: dict[int, np.ndarray | None],
    forward_luts: dict[str, ForwardLUT],
) -> list[float]:
    """Compute per-tracklet median ray-to-consensus distance.

    For each tracklet, for each frame where it has a centroid AND a
    consensus point exists, compute the distance from the tracklet's ray
    to the consensus 3D point. Return the median of these distances.

    Args:
        tracklets: Tracklets in the cluster.
        frame_list: Sorted frame indices.
        frame_consensus: Per-frame consensus 3D points.
        forward_luts: Per-camera ForwardLUTs.

    Returns:
        List of median distances, one per tracklet. ``inf`` if the tracklet
        has no frames with valid consensus.
    """
    distances: list[float] = []

    for t in tracklets:
        frame_map = {f: i for i, f in enumerate(t.frames)}
        dists_for_tracklet: list[float] = []

        for frame in frame_list:
            if frame not in frame_map:
                continue
            consensus_pt = frame_consensus.get(frame)
            if consensus_pt is None:
                continue

            idx = frame_map[frame]
            centroid = t.centroids[idx]
            lut = forward_luts[t.camera_id]
            pix = torch.tensor(
                [[float(centroid[0]), float(centroid[1])]], dtype=torch.float32
            )
            origins, dirs = lut.cast_ray(pix)
            o = origins[0].cpu().numpy().astype(np.float64)
            d = dirs[0].cpu().numpy().astype(np.float64)

            # Distance from ray to consensus point
            dist = _point_to_ray_distance(consensus_pt, o, d)
            dists_for_tracklet.append(dist)

        if dists_for_tracklet:
            distances.append(float(np.median(dists_for_tracklet)))
        else:
            distances.append(float("inf"))

    return distances


def _compute_per_frame_confidence(
    frame_list: list[int],
    tracklets: tuple[Tracklet2D, ...] | tuple,
    frame_consensus: dict[int, np.ndarray | None],
    forward_luts: dict[str, ForwardLUT],
    threshold: float,
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

    Returns:
        List of confidence values, one per frame in frame_list.
    """
    tracklet_frame_maps = [{f: i for i, f in enumerate(t.frames)} for t in tracklets]
    confidences: list[float] = []

    for frame in frame_list:
        consensus_pt = frame_consensus.get(frame)
        if consensus_pt is None:
            confidences.append(0.0)
            continue

        # Collect rays for this frame
        rays: list[tuple[np.ndarray, np.ndarray]] = []
        for t, frame_map in zip(tracklets, tracklet_frame_maps, strict=True):
            if frame not in frame_map:
                continue
            idx = frame_map[frame]
            centroid = t.centroids[idx]
            lut = forward_luts[t.camera_id]
            pix = torch.tensor(
                [[float(centroid[0]), float(centroid[1])]], dtype=torch.float32
            )
            origins, dirs = lut.cast_ray(pix)
            o = origins[0].cpu().numpy().astype(np.float64)
            d = dirs[0].cpu().numpy().astype(np.float64)
            rays.append((o, d))

        if len(rays) < 2:
            confidences.append(0.0)
            continue

        # Mean pairwise distance
        pair_dists: list[float] = []
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                dist, _ = ray_ray_closest_point(
                    rays[i][0], rays[i][1], rays[j][0], rays[j][1]
                )
                pair_dists.append(dist)

        mean_dist = float(np.mean(pair_dists))
        conf = max(0.0, min(1.0, 1.0 - mean_dist / threshold))
        confidences.append(conf)

    return confidences


def _point_to_ray_distance(
    point: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> float:
    """Compute the perpendicular distance from a 3D point to a ray.

    Args:
        point: 3D point, shape (3,).
        origin: Ray origin, shape (3,).
        direction: Unit ray direction, shape (3,).

    Returns:
        Perpendicular distance in metres.
    """
    w = point - origin
    t = float(np.dot(w, direction))
    closest = origin + t * direction
    return float(np.linalg.norm(point - closest))
