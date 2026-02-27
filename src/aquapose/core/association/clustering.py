"""Leiden-based tracklet clustering with must-not-link constraints and fragment merging.

Implements SPECSEED Steps 2-4: graph construction from scored edges, Leiden
community detection, must-not-link enforcement for same-camera conflicts, and
same-camera fragment merging with interpolated gap frames.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aquapose.core.tracking.types import Tracklet2D

from aquapose.core.association.types import TrackletGroup

logger = logging.getLogger(__name__)

__all__ = [
    "ClusteringConfigLike",
    "build_must_not_link",
    "cluster_tracklets",
    "merge_fragments",
]


# ---------------------------------------------------------------------------
# Config protocol (IB-003: core/ must not import engine/)
# ---------------------------------------------------------------------------

TrackletKey = tuple[str, int]


@runtime_checkable
class ClusteringConfigLike(Protocol):
    """Structural protocol for clustering configuration.

    Satisfied by ``AssociationConfig`` from ``engine.config`` without import.

    Attributes:
        score_min: Minimum affinity score for an edge.
        expected_fish_count: Expected number of fish clusters.
        leiden_resolution: Leiden resolution parameter.
        max_merge_gap: Maximum frame gap for fragment merging.
    """

    score_min: float
    expected_fish_count: int
    leiden_resolution: float
    max_merge_gap: int


# ---------------------------------------------------------------------------
# Must-not-link constraints (SPECSEED Step 3.1)
# ---------------------------------------------------------------------------


def build_must_not_link(
    tracks_2d: dict[str, list[Tracklet2D]],
) -> set[frozenset[TrackletKey]]:
    """Build must-not-link constraint set from same-camera detection overlaps.

    Two tracklets from the same camera are must-not-link if they share any
    frames where both have ``frame_status == "detected"``. Coasted-only
    overlap is NOT a constraint (those are fragment merge candidates).

    Args:
        tracks_2d: Per-camera tracklet lists.

    Returns:
        Set of frozen pairs of tracklet keys ``(camera_id, track_id)``.
    """
    constraints: set[frozenset[TrackletKey]] = set()

    for cam_id, tracklets in tracks_2d.items():
        n = len(tracklets)
        for i in range(n):
            for j in range(i + 1, n):
                ta = tracklets[i]
                tb = tracklets[j]

                # Build detected frame sets
                det_a = {
                    f
                    for f, s in zip(ta.frames, ta.frame_status, strict=False)
                    if s == "detected"
                }
                det_b = {
                    f
                    for f, s in zip(tb.frames, tb.frame_status, strict=False)
                    if s == "detected"
                }

                if det_a & det_b:
                    key_a: TrackletKey = (cam_id, ta.track_id)
                    key_b: TrackletKey = (cam_id, tb.track_id)
                    constraints.add(frozenset({key_a, key_b}))

    return constraints


# ---------------------------------------------------------------------------
# Leiden clustering (SPECSEED Steps 2-3)
# ---------------------------------------------------------------------------


def cluster_tracklets(
    scores: dict[tuple[TrackletKey, TrackletKey], float],
    tracks_2d: dict[str, list[Tracklet2D]],
    must_not_link: set[frozenset[TrackletKey]],
    config: ClusteringConfigLike,
) -> list[TrackletGroup]:
    """Cluster tracklets into fish identity groups via Leiden algorithm.

    Builds a weighted undirected graph from scored edges, splits into
    connected components, runs Leiden on each component, enforces
    must-not-link constraints, and packages results as TrackletGroup objects.

    Args:
        scores: Weighted edges from ``score_all_pairs()``.
        tracks_2d: Per-camera tracklets (for building the tracklet lookup).
        must_not_link: Same-camera conflict constraints.
        config: Clustering configuration.

    Returns:
        List of TrackletGroup objects, one per fish cluster.
    """
    import igraph as ig
    import leidenalg

    # Build tracklet lookup: key -> Tracklet2D
    tracklet_lookup: dict[TrackletKey, Tracklet2D] = {}
    for cam_id, tracklets in tracks_2d.items():
        for t in tracklets:
            tracklet_lookup[(cam_id, t.track_id)] = t

    # Collect all tracklet keys (nodes)
    all_keys: set[TrackletKey] = set(tracklet_lookup.keys())

    if not all_keys:
        return []

    # Map keys to integer indices
    key_list = sorted(all_keys)
    key_to_idx = {k: i for i, k in enumerate(key_list)}

    # Build igraph graph
    g = ig.Graph(n=len(key_list))
    edges: list[tuple[int, int]] = []
    weights: list[float] = []

    for (ka, kb), w in scores.items():
        if ka in key_to_idx and kb in key_to_idx:
            edges.append((key_to_idx[ka], key_to_idx[kb]))
            weights.append(w)

    g.add_edges(edges)
    g.es["weight"] = weights

    # Connected components
    components = g.connected_components()

    groups: list[TrackletGroup] = []
    fish_id = 0

    for component_indices in components:
        if len(component_indices) == 1:
            # Singleton — no clustering needed
            key = key_list[component_indices[0]]
            t = tracklet_lookup[key]
            groups.append(
                TrackletGroup(fish_id=fish_id, tracklets=(t,), confidence=0.0)
            )
            fish_id += 1
            continue

        # Subgraph for this component
        subgraph = g.subgraph(component_indices)
        sub_key_list = [key_list[i] for i in component_indices]

        # Leiden clustering
        partition = leidenalg.find_partition(
            subgraph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=config.leiden_resolution,
        )

        # Build cluster membership
        clusters: dict[int, list[int]] = {}
        for node_idx, cluster_id in enumerate(partition.membership):
            clusters.setdefault(cluster_id, []).append(node_idx)

        # Must-not-link enforcement
        for _cluster_id, members in list(clusters.items()):
            member_keys = [sub_key_list[m] for m in members]

            # Check for violations
            violations_found = True
            max_attempts = 3
            attempt = 0

            while violations_found and attempt < max_attempts:
                violations_found = False
                attempt += 1

                for i_m in range(len(member_keys)):
                    for j_m in range(i_m + 1, len(member_keys)):
                        pair = frozenset({member_keys[i_m], member_keys[j_m]})
                        if pair in must_not_link:
                            violations_found = True
                            # Evict the tracklet with lower total affinity
                            k_i = member_keys[i_m]
                            k_j = member_keys[j_m]
                            aff_i = sum(
                                scores.get((k_i, ok), 0.0) + scores.get((ok, k_i), 0.0)
                                for ok in member_keys
                                if ok != k_i
                            )
                            aff_j = sum(
                                scores.get((k_j, ok), 0.0) + scores.get((ok, k_j), 0.0)
                                for ok in member_keys
                                if ok != k_j
                            )
                            evict = k_j if aff_j <= aff_i else k_i

                            # Remove from cluster, create singleton
                            member_keys = [mk for mk in member_keys if mk != evict]
                            evict_t = tracklet_lookup[evict]
                            groups.append(
                                TrackletGroup(
                                    fish_id=fish_id,
                                    tracklets=(evict_t,),
                                    confidence=0.0,
                                )
                            )
                            fish_id += 1
                            break
                    if violations_found:
                        break

            # Build group from remaining members
            if member_keys:
                member_tracklets = tuple(tracklet_lookup[k] for k in member_keys)

                # Confidence: mean internal edge weight
                internal_weights: list[float] = []
                for i_m in range(len(member_keys)):
                    for j_m in range(i_m + 1, len(member_keys)):
                        w = scores.get(
                            (member_keys[i_m], member_keys[j_m]), 0.0
                        ) + scores.get((member_keys[j_m], member_keys[i_m]), 0.0)
                        if w > 0:
                            internal_weights.append(w)

                confidence = (
                    sum(internal_weights) / len(internal_weights)
                    if internal_weights
                    else 0.0
                )

                groups.append(
                    TrackletGroup(
                        fish_id=fish_id,
                        tracklets=member_tracklets,
                        confidence=confidence,
                    )
                )
                fish_id += 1

    # Diagnostic warning
    if len(groups) != config.expected_fish_count:
        logger.warning(
            "Cluster count %d != expected fish count %d",
            len(groups),
            config.expected_fish_count,
        )

    return groups


# ---------------------------------------------------------------------------
# Fragment merging (SPECSEED Step 4)
# ---------------------------------------------------------------------------


def merge_fragments(
    groups: list[TrackletGroup],
    config: ClusteringConfigLike,
) -> list[TrackletGroup]:
    """Merge same-camera non-overlapping fragments within each cluster.

    For each TrackletGroup, identifies same-camera tracklets that can be
    merged (non-overlapping, within ``max_merge_gap`` frames). Gap frames
    are filled with linearly interpolated centroids tagged as
    ``"interpolated"``.

    Args:
        groups: TrackletGroup list from ``cluster_tracklets()``.
        config: Clustering configuration with max_merge_gap.

    Returns:
        Updated TrackletGroup list with merged fragments.
    """
    merged_groups: list[TrackletGroup] = []

    for group in groups:
        tracklets_by_cam: dict[str, list[Tracklet2D]] = {}
        for t in group.tracklets:
            tracklets_by_cam.setdefault(t.camera_id, []).append(t)

        new_tracklets: list[Tracklet2D] = []

        for _cam_id, cam_tracklets in tracklets_by_cam.items():
            # Sort by first frame
            cam_tracklets = sorted(cam_tracklets, key=lambda t: t.frames[0])

            merged_cam = _merge_cam_fragments(cam_tracklets, config.max_merge_gap)
            new_tracklets.extend(merged_cam)

        merged_groups.append(
            TrackletGroup(
                fish_id=group.fish_id,
                tracklets=tuple(new_tracklets),
                confidence=group.confidence,
            )
        )

    return merged_groups


def _merge_cam_fragments(
    tracklets: list[Tracklet2D],
    max_merge_gap: int,
) -> list[Tracklet2D]:
    """Merge a sorted list of same-camera tracklets where possible.

    Args:
        tracklets: Same-camera tracklets sorted by first frame.
        max_merge_gap: Maximum gap in frames to allow merging.

    Returns:
        Merged tracklet list (may be shorter than input).
    """
    if len(tracklets) <= 1:
        return list(tracklets)

    result: list[Tracklet2D] = []
    current = tracklets[0]

    for next_t in tracklets[1:]:
        merged = _try_merge_pair(current, next_t, max_merge_gap)
        if merged is not None:
            current = merged
        else:
            result.append(current)
            current = next_t

    result.append(current)
    return result


def _try_merge_pair(
    earlier: Tracklet2D,
    later: Tracklet2D,
    max_merge_gap: int,
) -> Tracklet2D | None:
    """Try to merge two same-camera tracklets.

    Returns merged Tracklet2D or None if merge is not possible.
    """
    from aquapose.core.tracking.types import Tracklet2D

    # Check for detection-backed overlap
    det_earlier = {
        f
        for f, s in zip(earlier.frames, earlier.frame_status, strict=False)
        if s == "detected"
    }
    det_later = {
        f
        for f, s in zip(later.frames, later.frame_status, strict=False)
        if s == "detected"
    }

    if det_earlier & det_later:
        logger.warning(
            "Detection-backed overlap between tracklets %s-%d and %s-%d; skipping merge",
            earlier.camera_id,
            earlier.track_id,
            later.camera_id,
            later.track_id,
        )
        return None

    # Get the last detected frame of earlier and first detected frame of later
    earlier_det_frames = sorted(det_earlier)
    later_det_frames = sorted(det_later)

    if not earlier_det_frames or not later_det_frames:
        # Can't merge without detected frames
        return None

    last_det_earlier = earlier_det_frames[-1]
    first_det_later = later_det_frames[0]

    gap = first_det_later - last_det_earlier - 1

    if gap > max_merge_gap:
        return None

    # Build merged tracklet:
    # 1. Keep all detected frames from both tracklets
    # 2. Discard coasted frames
    # 3. Interpolate the gap

    frames_list: list[int] = []
    centroids_list: list[tuple[float, float]] = []
    bboxes_list: list[tuple[float, float, float, float]] = []
    status_list: list[str] = []

    # Add detected frames from earlier
    for f, c, b, s in zip(
        earlier.frames,
        earlier.centroids,
        earlier.bboxes,
        earlier.frame_status,
        strict=False,
    ):
        if s == "detected":
            frames_list.append(f)
            centroids_list.append(c)
            bboxes_list.append(b)
            status_list.append("detected")

    # Get last detected centroid and bbox from earlier for interpolation
    last_centroid = centroids_list[-1]
    last_bbox = bboxes_list[-1]

    # Get first detected centroid and bbox from later for interpolation
    first_later_idx = later_det_frames[0]
    first_later_pos = None
    first_later_bbox_val = None
    for f, c, b, s in zip(
        later.frames,
        later.centroids,
        later.bboxes,
        later.frame_status,
        strict=False,
    ):
        if f == first_later_idx and s == "detected":
            first_later_pos = c
            first_later_bbox_val = b
            break

    if first_later_pos is None or first_later_bbox_val is None:
        return None

    # Interpolate gap frames
    if gap > 0:
        for g_idx in range(1, gap + 1):
            alpha = g_idx / (gap + 1)
            interp_u = last_centroid[0] + alpha * (
                first_later_pos[0] - last_centroid[0]
            )
            interp_v = last_centroid[1] + alpha * (
                first_later_pos[1] - last_centroid[1]
            )
            interp_bx = last_bbox[0] + alpha * (first_later_bbox_val[0] - last_bbox[0])
            interp_by = last_bbox[1] + alpha * (first_later_bbox_val[1] - last_bbox[1])
            interp_bw = last_bbox[2] + alpha * (first_later_bbox_val[2] - last_bbox[2])
            interp_bh = last_bbox[3] + alpha * (first_later_bbox_val[3] - last_bbox[3])

            frames_list.append(last_det_earlier + g_idx)
            centroids_list.append((interp_u, interp_v))
            bboxes_list.append((interp_bx, interp_by, interp_bw, interp_bh))
            status_list.append("interpolated")

    # Add detected frames from later
    for f, c, b, s in zip(
        later.frames,
        later.centroids,
        later.bboxes,
        later.frame_status,
        strict=False,
    ):
        if s == "detected":
            frames_list.append(f)
            centroids_list.append(c)
            bboxes_list.append(b)
            status_list.append("detected")

    return Tracklet2D(
        camera_id=earlier.camera_id,
        track_id=earlier.track_id,
        frames=tuple(frames_list),
        centroids=tuple(centroids_list),
        bboxes=tuple(bboxes_list),
        frame_status=tuple(status_list),
    )
