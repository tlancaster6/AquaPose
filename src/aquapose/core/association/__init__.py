"""Association stage (Stage 3) domain types for the AquaPose v2.1 pipeline.

Exports TrackletGroup — the cross-camera identity cluster produced by Stage 3 — and
AssociationBundle for reconstruction compatibility until Phase 26.
"""

from aquapose.core.association.types import AssociationBundle, TrackletGroup

__all__ = ["AssociationBundle", "TrackletGroup"]
