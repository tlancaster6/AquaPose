"""Association stage (Stage 3) for the AquaPose v2.1 pipeline.

Exports domain types, scoring functions, clustering functions, and the
AssociationStage that wires cross-camera tracklet association into the pipeline.
"""

from aquapose.core.association.clustering import (
    ClusteringConfigLike,
    build_must_not_link,
    cluster_tracklets,
)
from aquapose.core.association.scoring import (
    AssociationConfigLike,
    ray_ray_closest_point,
    ray_ray_closest_point_batch,
    score_all_pairs,
    score_tracklet_pair,
)
from aquapose.core.association.stage import AssociationStage
from aquapose.core.association.types import (
    AssociationBundle,
    HandoffState,
    TrackletGroup,
)
from aquapose.core.association.validation import (
    ValidationConfigLike,
    validate_groups,
)

__all__ = [
    "AssociationBundle",
    "AssociationConfigLike",
    "AssociationStage",
    "ClusteringConfigLike",
    "HandoffState",
    "TrackletGroup",
    "ValidationConfigLike",
    "build_must_not_link",
    "cluster_tracklets",
    "ray_ray_closest_point",
    "ray_ray_closest_point_batch",
    "score_all_pairs",
    "score_tracklet_pair",
    "validate_groups",
]
