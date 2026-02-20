"""Elliptical cross-section vertex generation for the parametric fish mesh."""

import math

import torch


def build_cross_section_verts(
    centers: torch.Tensor,
    normals: torch.Tensor,
    binormals: torch.Tensor,
    heights: torch.Tensor,
    widths: torch.Tensor,
    s: torch.Tensor,
    n_verts: int = 10,
) -> torch.Tensor:
    """Generate elliptical cross-section vertices in the world frame.

    For each cross-section, generates M evenly spaced vertices around an
    ellipse defined by height (dorsoventral) and width (lateral) radii.
    Heights and widths are multiplied by the scale factor s to convert from
    body-fraction proportions to world-frame dimensions in meters.

    Args:
        centers: Cross-section center positions, shape (N, 3).
        normals: Unit dorsoventral vectors at each section, shape (N, 3).
        binormals: Unit lateral vectors at each section, shape (N, 3).
        heights: Height-to-body-length ratios per section, shape (N,).
            May have requires_grad=True for free cross-section mode.
        widths: Width-to-body-length ratios per section, shape (N,).
            May have requires_grad=True for free cross-section mode.
        s: Uniform scale factor (body length in meters), scalar tensor.
        n_verts: Number of vertices per ellipse (M). Default 10.

    Returns:
        verts: Ellipse vertices in world frame, shape (N, M, 3).
    """
    angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        n_verts + 1,
        dtype=centers.dtype,
        device=centers.device,
    )[:-1]  # (M,)

    cos_a = torch.cos(angles)  # (M,)
    sin_a = torch.sin(angles)  # (M,)

    # Scale heights and widths to world frame: multiply by s
    # heights/widths: (N,); s: scalar
    h_world = heights * s  # (N,)
    w_world = widths * s  # (N,)

    # Ellipse vertices in world frame:
    # vertex[i, j] = center[i] + w_world[i]*cos(angle[j])*binormal[i]
    #                           + h_world[i]*sin(angle[j])*normal[i]
    # Use broadcasting: (N, 1, 1) * (1, M, 1) * (N, 1, 3)
    binormal_contrib = (
        w_world[:, None, None] * cos_a[None, :, None] * binormals[:, None, :]
    )  # (N, M, 3)
    normal_contrib = (
        h_world[:, None, None] * sin_a[None, :, None] * normals[:, None, :]
    )  # (N, M, 3)

    verts = centers[:, None, :] + binormal_contrib + normal_contrib  # (N, M, 3)
    return verts
