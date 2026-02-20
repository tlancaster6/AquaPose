"""Parametric fish mesh model and differentiable rendering."""

from .builder import build_fish_mesh
from .cross_section import build_cross_section_verts
from .profiles import DEFAULT_CICHLID_PROFILE, CrossSectionProfile
from .spine import build_spine_frames
from .state import FishState

__all__ = [
    "DEFAULT_CICHLID_PROFILE",
    "CrossSectionProfile",
    "FishState",
    "build_cross_section_verts",
    "build_fish_mesh",
    "build_spine_frames",
]
