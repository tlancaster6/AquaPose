"""Mesh assembly: combines spine and cross-sections into a PyTorch3D Meshes object."""

import torch
from pytorch3d.structures import Meshes

from .cross_section import build_cross_section_verts
from .profiles import DEFAULT_CICHLID_PROFILE, CrossSectionProfile
from .spine import build_spine_frames
from .state import FishState

# Default mesh resolution (within the required ranges from CONTEXT.md)
M_VERTS: int = 10  # vertices per ellipse (8-12 range)


def _build_faces(n_sections: int, verts_per_section: int) -> torch.LongTensor:
    """Build triangle face indices for a swept surface mesh.

    Produces a watertight mesh with:
    - Tube body: 2 * (n_sections - 1) * verts_per_section triangles
    - Head cap fan: verts_per_section triangles
    - Tail cap fan: verts_per_section triangles

    Winding order: CCW when viewed from outside the mesh (outward-facing normals).
    Tube verts are indexed [0, n_sections*M). Head apex at n_sections*M,
    tail apex at n_sections*M + 1.

    Args:
        n_sections: Number of cross-sections along the spine.
        verts_per_section: Number of vertices per elliptical cross-section.

    Returns:
        faces: Triangle face indices, shape (F, 3), LongTensor.
    """
    M = verts_per_section
    faces: list[list[int]] = []

    # Tube body: connect adjacent section pairs with quads split into 2 triangles.
    # Section i has verts [i*M, ..., i*M + M-1].
    # Winding: for each quad (v0, v1, v2, v3) with v0,v1 on section i and v2,v3 on i+1,
    # two CCW triangles viewed from outside.
    for i in range(n_sections - 1):
        base = i * M
        for j in range(M):
            j_next = (j + 1) % M
            v0 = base + j
            v1 = base + j_next
            v2 = v0 + M
            v3 = v1 + M
            # CCW when viewed from outside: (v0, v2, v1) and (v1, v2, v3)
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # Head cap fan: apex at index n_sections * M (front of fish, section 0).
    apex_head = n_sections * M
    for j in range(M):
        j_next = (j + 1) % M
        # CCW: apex, then j_next, then j (fan from apex into section 0)
        faces.append([apex_head, j_next, j])

    # Tail cap fan: apex at index n_sections * M + 1 (back of fish, section N-1).
    apex_tail = n_sections * M + 1
    tail_base = (n_sections - 1) * M
    for j in range(M):
        j_next = (j + 1) % M
        # CCW: apex, then j, then j_next (fan from apex into last section)
        faces.append([apex_tail, tail_base + j, tail_base + j_next])

    return torch.tensor(faces, dtype=torch.long)  # type: ignore[return-value]


def _build_single_mesh_verts(
    state: FishState,
    profile: CrossSectionProfile,
    heights: torch.Tensor | None,
    widths: torch.Tensor | None,
    n_verts: int,
) -> torch.Tensor:
    """Build vertex tensor for a single fish mesh.

    Args:
        state: FishState pose parameters.
        profile: Cross-section profile providing section positions and shape ratios.
        heights: Optional (N,) override heights for free cross-section mode.
        widths: Optional (N,) override widths for free cross-section mode.
        n_verts: Vertices per ellipse.

    Returns:
        verts: (V, 3) float32 tensor, V = N*M + 2.
    """
    # 1. Build spine frames (centers relative to origin, to be translated by p)
    centers, tangents, normals, binormals = build_spine_frames(
        state.psi,
        state.theta,
        state.kappa,
        state.s,
        profile.section_positions,
    )

    # 2. Translate centers by state.p (world position)
    centers = centers + state.p[None, :]  # (N, 3)

    # 3. Determine heights/widths tensors
    if heights is None:
        heights_t = torch.tensor(
            profile.heights, dtype=state.s.dtype, device=state.s.device
        )
    else:
        heights_t = heights
    if widths is None:
        widths_t = torch.tensor(
            profile.widths, dtype=state.s.dtype, device=state.s.device
        )
    else:
        widths_t = widths

    # 4. Build cross-section vertices: (N, M, 3)
    section_verts = build_cross_section_verts(
        centers, normals, binormals, heights_t, widths_t, state.s, n_verts
    )  # (N, M, 3)

    n_sections = len(profile.section_positions)

    # 5. Flatten tube verts to (N*M, 3)
    tube_verts = section_verts.reshape(n_sections * n_verts, 3)  # (N*M, 3)

    # 6. Head apex: head section center slightly offset backward along head tangent.
    # Use a small fraction of the inter-section spacing as the apex offset.
    first_spacing = (
        profile.section_positions[1] - profile.section_positions[0]
    ) * state.s
    head_apex = centers[0] - tangents[0] * first_spacing * 0.5  # (3,)

    # 7. Tail apex: last section center offset forward along tail tangent.
    last_spacing = (
        profile.section_positions[-1] - profile.section_positions[-2]
    ) * state.s
    tail_apex = centers[-1] + tangents[-1] * last_spacing * 0.5  # (3,)

    # 8. Stack all verts: [tube, head_apex, tail_apex] → (N*M + 2, 3)
    verts = torch.cat([tube_verts, head_apex[None, :], tail_apex[None, :]], dim=0)
    return verts


def build_fish_mesh(
    states: list[FishState],
    profile: CrossSectionProfile | None = None,
    heights: torch.Tensor | None = None,
    widths: torch.Tensor | None = None,
    n_verts: int = M_VERTS,
) -> Meshes:
    """Build a differentiable parametric fish mesh from a batch of FishState objects.

    For each FishState, generates a swept elliptical cross-section mesh along a
    circular arc spine. All operations are differentiable — gradients flow from mesh
    vertex positions back through the state parameters (p, psi, theta, kappa, s).

    Args:
        states: List of FishState objects, one per fish in the batch.
        profile: Cross-section profile providing section positions and shape ratios.
            Defaults to DEFAULT_CICHLID_PROFILE.
        heights: Optional (N,) tensor of height ratios for free cross-section mode.
            If provided, used instead of profile.heights for all states in the batch.
            May have requires_grad=True to optimize cross-section shapes.
        widths: Optional (N,) tensor of width ratios for free cross-section mode.
            Same as heights but for lateral widths.
        n_verts: Number of vertices per elliptical cross-section. Default 10.

    Returns:
        Meshes: PyTorch3D Meshes object with batch size len(states). Vertex tensors
            are in the autograd graph — call mesh.verts_list()[i].sum().backward()
            to compute gradients w.r.t. state parameters.
    """
    if profile is None:
        profile = DEFAULT_CICHLID_PROFILE

    n_sections = len(profile.section_positions)
    # Precompute faces once — non-differentiable, reused across all batch members.
    faces = _build_faces(n_sections, n_verts)

    verts_list = []
    faces_list = []
    for state in states:
        verts = _build_single_mesh_verts(state, profile, heights, widths, n_verts)
        verts_list.append(verts)
        faces_list.append(faces)

    return Meshes(verts=verts_list, faces=faces_list)
