"""Tests for build_fish_mesh: watertight mesh, gradient flow, batch API."""
# ruff: noqa — dead module, pending deletion in Phase 20

import pytest

pytestmark = pytest.mark.skip(
    reason="mesh module is dead code, pending deletion in Phase 20"
)

import torch

from aquapose.mesh.builder import build_fish_mesh
from aquapose.mesh.profiles import DEFAULT_CICHLID_PROFILE
from aquapose.mesh.state import FishState

try:
    from pytorch3d.structures import Meshes
except ImportError:
    Meshes = None

N_SECTIONS = len(DEFAULT_CICHLID_PROFILE.section_positions)
M_VERTS = 10
EXPECTED_VERTS = N_SECTIONS * M_VERTS + 2  # tube verts + head apex + tail apex
EXPECTED_FACES = 2 * (N_SECTIONS - 1) * M_VERTS + 2 * M_VERTS  # tube + 2 caps


def make_state(
    p: list[float] | None = None,
    psi: float = 0.0,
    theta: float = 0.0,
    kappa: float = 1.0,
    s: float = 0.15,
    requires_grad: bool = False,
) -> FishState:
    """Create a FishState for testing."""
    p_tensor = torch.tensor(
        p if p is not None else [0.0, 0.0, 1.2],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    return FishState(
        p=p_tensor,
        psi=torch.tensor(psi, requires_grad=requires_grad),
        theta=torch.tensor(theta, requires_grad=requires_grad),
        kappa=torch.tensor(kappa, requires_grad=requires_grad),
        s=torch.tensor(s, requires_grad=requires_grad),
    )


def test_build_fish_mesh_returns_meshes():
    """Single FishState returns Meshes with 1 mesh, correct vertex/face counts."""
    state = make_state()
    mesh = build_fish_mesh([state])
    assert isinstance(mesh, Meshes)
    assert len(mesh) == 1
    verts = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    assert verts.shape == (EXPECTED_VERTS, 3), (
        f"Expected ({EXPECTED_VERTS}, 3), got {verts.shape}"
    )
    assert faces.shape == (EXPECTED_FACES, 3), (
        f"Expected ({EXPECTED_FACES}, 3), got {faces.shape}"
    )


def test_watertight_mesh():
    """All edges in the mesh are shared by exactly 2 faces (closed manifold)."""
    state = make_state()
    mesh = build_fish_mesh([state])
    verts = mesh.verts_list()[0]
    faces = mesh.faces_list()[0].numpy()
    V = verts.shape[0]

    # Count how many times each directed edge appears as the reverse edge.
    # For a watertight manifold, every undirected edge appears in exactly 2 faces.
    from collections import Counter

    edge_counts: Counter[tuple[int, int]] = Counter()
    for f in faces:
        v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
        # Add undirected edges (sorted so (a,b) == (b,a))
        for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
            edge_counts[tuple(sorted((a, b)))] += 1  # type: ignore[arg-type]

    non_manifold = {e: c for e, c in edge_counts.items() if c != 2}
    assert not non_manifold, (
        f"Non-manifold edges found (not exactly 2 faces each): "
        f"{len(non_manifold)} edges. Examples: {list(non_manifold.items())[:5]}"
    )
    _ = V  # suppress unused warning


def test_gradients_flow_through_position():
    """state.p.grad is not None after backward through verts.sum()."""
    p = torch.tensor([0.0, 0.0, 1.2], requires_grad=True)
    state = FishState(
        p=p,
        psi=torch.tensor(0.0),
        theta=torch.tensor(0.0),
        kappa=torch.tensor(0.5),
        s=torch.tensor(0.15),
    )
    mesh = build_fish_mesh([state])
    loss = mesh.verts_list()[0].sum()
    loss.backward()
    assert p.grad is not None
    assert torch.all(p.grad.isfinite())


def test_gradients_flow_through_psi():
    """Gradients flow from mesh verts back to psi (yaw)."""
    psi = torch.tensor(0.5, requires_grad=True)
    state = FishState(
        p=torch.zeros(3),
        psi=psi,
        theta=torch.tensor(0.0),
        kappa=torch.tensor(0.5),
        s=torch.tensor(0.15),
    )
    mesh = build_fish_mesh([state])
    mesh.verts_list()[0].sum().backward()
    assert psi.grad is not None
    assert psi.grad.isfinite()


def test_gradients_flow_through_theta():
    """Gradients flow from mesh verts back to theta (pitch)."""
    theta = torch.tensor(0.2, requires_grad=True)
    state = FishState(
        p=torch.zeros(3),
        psi=torch.tensor(0.0),
        theta=theta,
        kappa=torch.tensor(0.5),
        s=torch.tensor(0.15),
    )
    mesh = build_fish_mesh([state])
    mesh.verts_list()[0].sum().backward()
    assert theta.grad is not None
    assert theta.grad.isfinite()


def test_gradients_flow_through_kappa():
    """Gradients flow to kappa for both typical and near-zero curvature."""
    for kappa_val in [0.5, 1e-6]:
        kappa = torch.tensor(kappa_val, requires_grad=True)
        state = FishState(
            p=torch.zeros(3),
            psi=torch.tensor(0.0),
            theta=torch.tensor(0.0),
            kappa=kappa,
            s=torch.tensor(0.15),
        )
        mesh = build_fish_mesh([state])
        mesh.verts_list()[0].sum().backward()
        assert kappa.grad is not None, f"kappa={kappa_val}: grad is None"
        assert kappa.grad.isfinite(), f"kappa={kappa_val}: grad={kappa.grad} not finite"


def test_gradients_flow_through_scale():
    """Gradients flow from mesh verts back to s (scale)."""
    s = torch.tensor(0.15, requires_grad=True)
    state = FishState(
        p=torch.zeros(3),
        psi=torch.tensor(0.0),
        theta=torch.tensor(0.0),
        kappa=torch.tensor(0.5),
        s=s,
    )
    mesh = build_fish_mesh([state])
    mesh.verts_list()[0].sum().backward()
    assert s.grad is not None
    assert s.grad.isfinite()


def test_free_cross_section_gradients():
    """Heights and widths with requires_grad=True receive gradients."""
    state = make_state()
    heights = torch.tensor(
        DEFAULT_CICHLID_PROFILE.heights, dtype=torch.float32, requires_grad=True
    )
    widths = torch.tensor(
        DEFAULT_CICHLID_PROFILE.widths, dtype=torch.float32, requires_grad=True
    )
    mesh = build_fish_mesh([state], heights=heights, widths=widths)
    mesh.verts_list()[0].sum().backward()
    assert heights.grad is not None
    assert torch.all(heights.grad.isfinite())
    assert widths.grad is not None
    assert torch.all(widths.grad.isfinite())


def test_batch_build():
    """List of 3 FishStates returns Meshes with 3 meshes, correct per-mesh counts."""
    states = [make_state(psi=i * 0.5) for i in range(3)]
    mesh = build_fish_mesh(states)
    assert isinstance(mesh, Meshes)
    assert len(mesh) == 3
    for i, (v, f) in enumerate(zip(mesh.verts_list(), mesh.faces_list(), strict=True)):
        assert v.shape == (EXPECTED_VERTS, 3), (
            f"Mesh {i}: expected ({EXPECTED_VERTS}, 3), got {v.shape}"
        )
        assert f.shape == (EXPECTED_FACES, 3), (
            f"Mesh {i}: expected ({EXPECTED_FACES}, 3), got {f.shape}"
        )


def test_zero_curvature_no_nan():
    """kappa=0.0 exactly produces no NaN in verts or gradients."""
    kappa = torch.tensor(0.0, requires_grad=True)
    state = FishState(
        p=torch.zeros(3),
        psi=torch.tensor(0.3),
        theta=torch.tensor(0.1),
        kappa=kappa,
        s=torch.tensor(0.15),
    )
    mesh = build_fish_mesh([state])
    verts = mesh.verts_list()[0]
    assert torch.all(verts.isfinite()), "NaN in verts with kappa=0"
    verts.sum().backward()
    assert kappa.grad is not None
    assert kappa.grad.isfinite(), f"kappa.grad={kappa.grad} not finite with kappa=0"


def test_verts_require_grad():
    """Mesh verts maintain requires_grad when state params have requires_grad."""
    p = torch.tensor([0.0, 0.0, 1.2], requires_grad=True)
    state = FishState(
        p=p,
        psi=torch.tensor(0.0),
        theta=torch.tensor(0.0),
        kappa=torch.tensor(0.5),
        s=torch.tensor(0.15),
    )
    mesh = build_fish_mesh([state])
    assert mesh.verts_list()[0].requires_grad, "Mesh verts should require grad"
