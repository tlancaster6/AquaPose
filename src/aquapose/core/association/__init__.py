"""Association stage (Stage 3) for the AquaPose 5-stage pipeline.

Provides the AssociationStage class that groups fish detections across cameras
into cross-view bundles via RANSAC centroid clustering. Populates
PipelineContext.associated_bundles.
"""

from aquapose.core.association.stage import AssociationStage
from aquapose.core.association.types import AssociationBundle

__all__ = ["AssociationBundle", "AssociationStage"]
