"""Association stage (Stage 3) domain types and scoring for the AquaPose v2.1 pipeline.

Exports TrackletGroup, AssociationBundle, and pairwise scoring functions for
cross-camera tracklet affinity computation.
"""

from aquapose.core.association.scoring import (
    AssociationConfigLike,
    ray_ray_closest_point,
    score_all_pairs,
    score_tracklet_pair,
)
from aquapose.core.association.types import AssociationBundle, TrackletGroup

__all__ = [
    "AssociationBundle",
    "AssociationConfigLike",
    "TrackletGroup",
    "ray_ray_closest_point",
    "score_all_pairs",
    "score_tracklet_pair",
]
