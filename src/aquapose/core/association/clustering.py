"""Leiden-based tracklet clustering with must-not-link constraints.

Implements SPECSEED Steps 2-3: graph construction from scored edges, Leiden
community detection, and must-not-link enforcement for same-camera conflicts.
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
    """

    score_min: float
    expected_fish_count: int
    leiden_resolution: float


# ---------------------------------------------------------------------------
# Must-not-link constraints (SPECSEED Step 3.1)
# ---------------------------------------------------------------------------


def build_must_not_link(
    tracks_2d: dict[str, list[Tracklet2D]],
) -> set[frozenset[TrackletKey]]:
    """Build must-not-link constraint set from same-camera frame overlaps.

    Two tracklets from the same camera are must-not-link if they share any
    frames, regardless of status. When the tracker has two tracks alive
    simultaneously (even if one is coasting), it has already determined
    they are different fish — the association stage must respect that.

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

                if set(ta.frames) & set(tb.frames):
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

    # Diagnostic warning (ignore singletons — orphan tracklets are expected)
    n_multi = sum(1 for g in groups if len(g.tracklets) > 1)
    n_single = len(groups) - n_multi
    if n_multi != config.expected_fish_count:
        logger.warning(
            "Non-singleton cluster count %d != expected fish count %d "
            "(%d singletons excluded)",
            n_multi,
            config.expected_fish_count,
            n_single,
        )

    return groups
