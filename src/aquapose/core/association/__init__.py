"""Association stage (Stage 3) for the AquaPose v2.1 pipeline.

Exports domain types, scoring functions, clustering functions, and the
AssociationStage that wires cross-camera tracklet association into the pipeline.
"""

from aquapose.core.association.clustering import (
    ClusteringConfigLike,
    build_must_not_link,
    cluster_tracklets,
    merge_fragments,
)
from aquapose.core.association.refinement import (
    RefinementConfigLike,
    refine_clusters,
)
from aquapose.core.association.scoring import (
    AssociationConfigLike,
    ray_ray_closest_point,
    score_all_pairs,
    score_tracklet_pair,
)
from aquapose.core.association.stage import AssociationStage
from aquapose.core.association.types import (
    AssociationBundle,
    HandoffState,
    TrackletGroup,
)

__all__ = [
    "AssociationBundle",
    "AssociationConfigLike",
    "AssociationStage",
    "ClusteringConfigLike",
    "HandoffState",
    "RefinementConfigLike",
    "TrackletGroup",
    "build_must_not_link",
    "cluster_tracklets",
    "merge_fragments",
    "ray_ray_closest_point",
    "refine_clusters",
    "score_all_pairs",
    "score_tracklet_pair",
]
