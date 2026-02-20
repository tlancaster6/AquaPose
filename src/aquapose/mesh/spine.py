"""Circular arc spine generation with stable near-zero curvature handling."""

import torch


def build_spine_frames(
    psi: torch.Tensor,
    theta: torch.Tensor,
    kappa: torch.Tensor,
    s: torch.Tensor,
    section_positions: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate spine center points and local coordinate frames along a circular arc.

    The spine is a circular arc in the heading-dorsal plane. All operations are
    differentiable PyTorch ops — no numpy, no item(), no detach(). The arc is
    centered so that the midpoint (t=0.5) is at the origin; caller translates
    by state.p.

    Args:
        psi: Yaw angle in radians, scalar tensor.
        theta: Pitch angle in radians, scalar tensor.
        kappa: Spine curvature (1/radius), scalar tensor. kappa=0 means straight.
        s: Scale factor (body length in meters), scalar tensor.
        section_positions: List of N fractional positions along [0, 1].

    Returns:
        centers: (N, 3) — 3D positions of cross-section centers relative to origin.
            Caller adds state.p for world-frame positions.
        tangents: (N, 3) — Unit tangent vectors along spine at each section.
        normals: (N, 3) — Unit normal vectors (dorsoventral direction).
        binormals: (N, 3) — Unit binormal vectors (lateral direction).
    """
    eps = torch.tensor(1e-8, dtype=psi.dtype, device=psi.device)

    # Build heading vector from yaw and pitch.
    # AquaPose world: +Z down into water. Positive theta tilts nose downward.
    heading = torch.stack(
        [
            torch.cos(psi) * torch.cos(theta),
            torch.sin(psi) * torch.cos(theta),
            torch.sin(theta),
        ]
    )  # (3,)

    # Build dorsal vector via Gram-Schmidt against world-up [0, 0, 1].
    # Fallback to [1, 0, 0] when heading is nearly parallel to world-up.
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=psi.dtype, device=psi.device)
    fallback = torch.tensor([1.0, 0.0, 0.0], dtype=psi.dtype, device=psi.device)
    dot_with_up = (heading * world_up).sum()
    ref = torch.where(dot_with_up.abs() > 0.99, fallback, world_up)

    # Gram-Schmidt to get dorsal perpendicular to heading
    dorsal_raw = ref - (heading * ref).sum() * heading
    dorsal = dorsal_raw / (torch.linalg.norm(dorsal_raw) + eps)  # (3,)

    # Binormal = heading x dorsal (lateral direction)
    binormal = torch.linalg.cross(heading, dorsal)
    binormal = binormal / (torch.linalg.norm(binormal) + eps)  # (3,)

    # Build spine centers using a numerically stable arc formula.
    # For kappa != 0: x(t) = sin(kappa*t*s)/(kappa+eps) * heading
    #                         + (1-cos(kappa*t*s))/(kappa+eps) * dorsal
    # At kappa=0: sin(kappa*t*s)/(kappa+eps) ~ t*s (Taylor: sin(x)/x -> 1 as x->0)
    #             (1-cos(kappa*t*s))/(kappa+eps) -> 0
    # To make this numerically exact at kappa=0 we use sinc: sin(x)/x = sinc(x/pi)*pi
    # but PyTorch sinc expects x in units where sinc(x) = sin(pi*x)/(pi*x).
    # Simpler: divide by (kappa+eps), which is tiny but nonzero. At kappa=0, eps=1e-8,
    # sin(0)/(0+1e-8)=0 — WRONG (should be t*s). Fix: use the stable formula
    # sin(kappa*t*s) / kappa = t*s * sinc(kappa*t*s / pi) [torch.sinc takes x, computes sin(pi*x)/(pi*x)]
    # i.e., sin(kappa*t*s)/kappa = t*s * torch.sinc(kappa*t*s / pi)
    # Similarly (1-cos(kappa*t*s))/kappa: use limit, smooth with t*s*half-sinc identity.
    # For (1-cos(x))/x we can write: (1-cos(x))/x = sin(x/2)*2*sin(x/2)/(2*sin(x/2)*x/2...
    # Simplest stable form: use (kappa+eps) but then we must scale correctly.
    # Better: use sign_kappa * (|kappa| + eps) so that limit gives t*s correctly.
    # CLEANEST SOLUTION: use torch.where to branch on |kappa| < threshold, with
    # both branches computed without NaN (torch.where evaluates both branches but
    # we use the formula that is numerically stable in each regime).

    t_vals = torch.tensor(section_positions, dtype=psi.dtype, device=psi.device)  # (N,)

    arc_arg = kappa * t_vals * s  # (N,), = kappa * t * s

    # Stable sin(kappa*t*s) / kappa:
    # = t*s * sin(kappa*t*s) / (kappa*t*s)   [= t*s * sinc_unnorm(kappa*t*s)]
    # torch.sinc(x) = sin(pi*x)/(pi*x), so sinc_unnorm(x) = torch.sinc(x/pi)
    # This is smooth and correct at kappa=0 (sinc_unnorm(0)=1, so result = t*s).
    along = t_vals * s * torch.sinc(arc_arg / torch.pi)  # (N,)

    # Stable (1 - cos(kappa*t*s)) / kappa:
    # = kappa * t^2 * s^2 / 2 * sinc^2(kappa*t*s / (2*pi))  ... complex
    # Easiest: use (1-cos(x))/x = sin(x/2)*2*(sin(x/2)/x) = sin(x/2)^2 * 2/x
    # Or: divide by kappa with a sign-preserving eps:
    #   safe_kappa = kappa + sign(kappa)*eps (but sign(0)=0 so fails)
    # Best: use torch.where branching. |kappa| < 1e-4 → use 0 (first-order approx)
    # For |kappa| >= 1e-4 → use (1-cos(arc_arg))/kappa exactly.
    # The cross terms from torch.where evaluating the non-NaN branch are fine.
    kappa_safe = torch.where(
        kappa.abs() < 1e-4,
        torch.ones_like(kappa),  # dummy value to avoid division by ~zero
        kappa,
    )
    perp_arc = (1.0 - torch.cos(arc_arg)) / kappa_safe  # (N,), safe for |kappa|>=1e-4
    # Near zero: (1-cos(kappa*t*s))/kappa ≈ kappa*(t*s)^2/2 → 0 as kappa→0
    perp_zero = torch.zeros_like(t_vals)  # first-order approximation at kappa=0
    perp = torch.where(kappa.abs() < 1e-4, perp_zero, perp_arc)  # (N,)

    # Centers as outer product: along[:,None]*heading[None,:] + perp[:,None]*dorsal[None,:]
    centers_raw = (
        along[:, None] * heading[None, :] + perp[:, None] * dorsal[None, :]
    )  # (N, 3)

    # Center the spine so t=0.5 is at the origin.
    # Compute midpoint by interpolating at t=0.5.
    t_mid = torch.tensor(0.5, dtype=psi.dtype, device=psi.device)
    arc_mid = kappa * t_mid * s
    along_mid = t_mid * s * torch.sinc(arc_mid / torch.pi)
    perp_arc_mid = (1.0 - torch.cos(arc_mid)) / kappa_safe
    perp_mid = torch.where(kappa.abs() < 1e-4, torch.zeros_like(kappa), perp_arc_mid)
    midpoint = along_mid * heading + perp_mid * dorsal  # (3,)
    centers = centers_raw - midpoint[None, :]  # (N, 3)

    # Tangent vectors: derivative of position wrt arc-length parameter t.
    # d(along)/dt = s * cos(kappa*t*s)  [d/dt of t*s*sinc(kappa*t*s/pi) reduces to s*cos(kappa*t*s)]
    # d(perp)/dt  = s * sin(kappa*t*s)  [d/dt of (1-cos(kappa*t*s))/kappa = sin(kappa*t*s)*s]
    # These are exact and smooth everywhere including kappa=0 (cos(0)=1, sin(0)=0 → pure heading).
    tang_along = torch.cos(arc_arg) * s  # (N,)
    tang_perp = torch.sin(arc_arg) * s  # (N,)
    tangents_raw = (
        tang_along[:, None] * heading[None, :] + tang_perp[:, None] * dorsal[None, :]
    )  # (N, 3)
    tang_norms = torch.linalg.norm(tangents_raw, dim=-1, keepdim=True)
    tangents = tangents_raw / (tang_norms + eps)  # (N, 3)

    # Normals: perpendicular to tangent in the bending plane (heading-dorsal).
    # Rotate tangent 90° in the bending plane.
    norm_raw = (
        -tang_perp[:, None] * heading[None, :] + tang_along[:, None] * dorsal[None, :]
    )
    norm_norms = torch.linalg.norm(norm_raw, dim=-1, keepdim=True)
    normals = norm_raw / (norm_norms + eps)  # (N, 3)

    # Binormal is constant along a planar arc (out-of-plane direction).
    binormals = binormal[None, :].expand(len(section_positions), 3)  # (N, 3)

    return centers, tangents, normals, binormals
