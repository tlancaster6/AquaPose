"""Tests for build_cross_section_verts: ellipse geometry, symmetry, gradient flow."""

import torch

from aquapose.mesh.cross_section import build_cross_section_verts

N = 5  # number of cross-sections for testing
M = 10  # vertices per ellipse


def _make_frames(n: int = N) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create simple aligned frames (X-axis heading) for testing."""
    centers = torch.zeros(n, 3)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3).clone()
    binormals = torch.tensor([[0.0, 1.0, 0.0]]).expand(n, 3).clone()
    return centers, normals, binormals


def test_output_shape():
    """Output shape is (N, M, 3)."""
    centers, normals, binormals = _make_frames()
    heights = torch.full((N,), 0.1)
    widths = torch.full((N,), 0.08)
    s = torch.tensor(0.15)
    verts = build_cross_section_verts(
        centers, normals, binormals, heights, widths, s, M
    )
    assert verts.shape == (N, M, 3)


def test_circular_section_equidistant():
    """When h=w, all vertices are equidistant from the center (circle)."""
    centers, normals, binormals = _make_frames(1)
    h = 0.1
    heights = torch.full((1,), h)
    widths = torch.full((1,), h)
    s = torch.tensor(1.0)  # s=1 so h_world = h
    verts = build_cross_section_verts(
        centers, normals, binormals, heights, widths, s, M
    )
    # verts[0]: (M, 3). Distances from center[0]=[0,0,0] should all be h*s
    dists = torch.linalg.norm(verts[0], dim=-1)  # (M,)
    assert torch.allclose(dists, torch.full((M,), h), atol=1e-5), (
        f"Circle: distances {dists} not all equal to {h}"
    )


def test_ellipse_aspect_ratio():
    """Section with h != w produces correct aspect ratio in extremal vertices."""
    centers, normals, binormals = _make_frames(1)
    h = 0.2
    w = 0.1
    heights = torch.tensor([h])
    widths = torch.tensor([w])
    s = torch.tensor(1.0)
    # Use enough vertices to land close to the axes
    M_dense = 1000
    verts = build_cross_section_verts(
        centers, normals, binormals, heights, widths, s, M_dense
    )
    dists = torch.linalg.norm(verts[0], dim=-1)  # (M_dense,)
    # Max distance ~= h (at normal axis at angle pi/2 and 3pi/2)
    # Min distance ~= w (at binormal axis at angle 0 and pi)
    assert torch.isclose(dists.max(), torch.tensor(h), atol=1e-3), (
        f"Max dist {dists.max()} not close to h={h}"
    )
    assert torch.isclose(dists.min(), torch.tensor(w), atol=1e-3), (
        f"Min dist {dists.min()} not close to w={w}"
    )


def test_left_right_symmetry():
    """Vertices are symmetric about the normal plane (dorsoventral axis).

    The ellipse is verts[j] = center + w*cos(angle[j])*binormal + h*sin(angle[j])*normal.
    For angle[j] = 2pi*j/M and angle[M-j] = 2pi - 2pi*j/M:
      cos(angle[M-j]) = cos(2pi*j/M) (same binormal component)
      sin(angle[M-j]) = -sin(2pi*j/M) (opposite normal component)
    So symmetry is about the binormal-normal plane: normal component flips, binormal stays.
    binormal=[0,1,0] → Y same. normal=[0,0,1] → Z flips.
    """
    centers, normals, binormals = _make_frames(1)
    h = 0.15
    w = 0.08
    heights = torch.tensor([h])
    widths = torch.tensor([w])
    s = torch.tensor(1.0)
    verts = build_cross_section_verts(
        centers, normals, binormals, heights, widths, s, M
    )
    v = verts[0]  # (M, 3)
    # binormal is [0, 1, 0] (Y-axis): cos flips symmetrically → Y[j] == Y[M-j]
    # normal is [0, 0, 1] (Z-axis): sin flips anti-symmetrically → Z[j] == -Z[M-j]
    for j in range(1, M // 2):
        z_j = v[j, 2]
        z_neg = v[M - j, 2]
        assert torch.isclose(z_j, -z_neg, atol=1e-5), (
            f"Z-symmetry broken: v[{j},2]={z_j}, v[{M - j},2]={z_neg}"
        )
        y_j = v[j, 1]
        y_neg = v[M - j, 1]
        assert torch.isclose(y_j, y_neg, atol=1e-5), (
            f"Y-symmetry broken: v[{j},1]={y_j}, v[{M - j},1]={y_neg}"
        )


def test_scale_factor_applied():
    """Heights and widths are multiplied by s in world frame."""
    centers, normals, binormals = _make_frames(1)
    h = 0.1
    heights = torch.tensor([h])
    widths = torch.tensor([h])
    s_val = 0.5
    s = torch.tensor(s_val)
    verts = build_cross_section_verts(
        centers, normals, binormals, heights, widths, s, M
    )
    dists = torch.linalg.norm(verts[0], dim=-1)
    expected = h * s_val
    assert torch.allclose(dists, torch.full((M,), expected), atol=1e-5)


def test_gradient_flows_through_heights():
    """Gradients flow from verts back to heights (free cross-section mode)."""
    centers, normals, binormals = _make_frames()
    heights = torch.full((N,), 0.1, requires_grad=True)
    widths = torch.full((N,), 0.08)
    s = torch.tensor(0.15)
    verts = build_cross_section_verts(
        centers, normals, binormals, heights, widths, s, M
    )
    verts.sum().backward()
    assert heights.grad is not None
    assert torch.all(heights.grad.isfinite())


def test_gradient_flows_through_widths():
    """Gradients flow from verts back to widths (free cross-section mode)."""
    centers, normals, binormals = _make_frames()
    heights = torch.full((N,), 0.1)
    widths = torch.full((N,), 0.08, requires_grad=True)
    s = torch.tensor(0.15)
    verts = build_cross_section_verts(
        centers, normals, binormals, heights, widths, s, M
    )
    verts.sum().backward()
    assert widths.grad is not None
    assert torch.all(widths.grad.isfinite())


def test_gradient_flows_through_s():
    """Gradients flow from verts back to s."""
    centers, normals, binormals = _make_frames()
    heights = torch.full((N,), 0.1)
    widths = torch.full((N,), 0.08)
    s = torch.tensor(0.15, requires_grad=True)
    verts = build_cross_section_verts(
        centers, normals, binormals, heights, widths, s, M
    )
    verts.sum().backward()
    assert s.grad is not None
    assert s.grad.isfinite()


def test_default_n_verts():
    """Default n_verts=10 gives 10 vertices per section."""
    centers, normals, binormals = _make_frames()
    heights = torch.full((N,), 0.1)
    widths = torch.full((N,), 0.08)
    s = torch.tensor(0.15)
    verts = build_cross_section_verts(centers, normals, binormals, heights, widths, s)
    assert verts.shape == (N, 10, 3)
