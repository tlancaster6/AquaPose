"""FishState dataclass holding the 5-parameter fish pose state vector."""

from dataclasses import dataclass

import torch


@dataclass
class FishState:
    """Fish pose state vector {p, ψ, θ, κ, s}.

    Attributes:
        p: 3D position (center of fish) in world frame, shape (3,), float32.
        psi: Yaw angle (rotation about world Z axis), radians, shape (), float32.
        theta: Pitch angle (nose-up/down tilt from XY plane), radians, shape (), float32.
            Positive values mean the nose tilts downward (+Z direction in AquaPose
            world frame where +Z points into water).
        kappa: Spine curvature (inverse radius of circular arc), shape (), float32.
            kappa=0 means a straight fish. Positive kappa bends toward the dorsal side.
        s: Uniform scale factor; s=0.15 means the fish body is 0.15 m long, shape (),
            float32. Cross-section positions are defined as fractions along [0, 1] and
            multiplied by s to yield world-frame arc lengths.
    """

    p: torch.Tensor  # (3,)
    psi: torch.Tensor  # scalar
    theta: torch.Tensor  # scalar
    kappa: torch.Tensor  # scalar
    s: torch.Tensor  # scalar
