"""Cross-section profile definitions for the parametric fish mesh."""

from dataclasses import dataclass


@dataclass
class CrossSectionProfile:
    """Per-section shape profile for the parametric fish mesh.

    Attributes:
        section_positions: Fractional positions along the spine in [0, 1],
            monotonically increasing. Length N, where N is the number of
            cross-sections (5-8 recommended).
        heights: Height-to-body-length ratios at each section, shape (N,).
            Height is measured in the dorsoventral (normal) direction.
        widths: Width-to-body-length ratios at each section, shape (N,).
            Width is measured in the lateral (binormal) direction.
    """

    section_positions: list[float]
    heights: list[float]
    widths: list[float]


# Default cichlid profile with 7 sections.
# Sections are denser at head (0.0-0.25) and tail (0.75-1.0) to better capture
# the rapid taper at the snout and caudal peduncle.
# Heights and widths are fractions of body length s.
# Cichlid proportions: tapered snout, widest body at ~40%, narrow caudal peduncle.
# height > width reflects the typical cichlid body that is taller than it is wide.
DEFAULT_CICHLID_PROFILE = CrossSectionProfile(
    section_positions=[0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0],
    heights=[0.04, 0.10, 0.14, 0.13, 0.08, 0.05, 0.02],
    widths=[0.03, 0.07, 0.10, 0.09, 0.06, 0.03, 0.01],
)
