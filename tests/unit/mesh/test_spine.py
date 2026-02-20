"""Tests for build_spine_frames: arc geometry, frame orthonormality, gradient flow."""

import math

import torch

from aquapose.mesh.profiles import DEFAULT_CICHLID_PROFILE
from aquapose.mesh.spine import build_spine_frames

POSITIONS = DEFAULT_CICHLID_PROFILE.section_positions
N = len(POSITIONS)


def _make_tensors(**kwargs: float) -> dict[str, torch.Tensor]:
    """Convert float keyword args to scalar float32 tensors."""
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in kwargs.items()}


def test_straight_fish_centers_along_heading():
    """kappa=0: centers lie along heading direction, spacing matches section_positions * s."""
    t = _make_tensors(psi=0.0, theta=0.0, kappa=0.0, s=0.2)
    centers, _tangents, _normals, _binormals = build_spine_frames(
        t["psi"], t["theta"], t["kappa"], t["s"], POSITIONS
    )
    # Heading is [1, 0, 0] for psi=0, theta=0
    # Centers should be collinear along X axis (after centering at midpoint)
    assert centers.shape == (N, 3)
    # All centers should have Y=0 and Z=0
    assert torch.allclose(centers[:, 1], torch.zeros(N), atol=1e-5)
    assert torch.allclose(centers[:, 2], torch.zeros(N), atol=1e-5)
    # Spacing along X should match (pos - 0.5) * s
    expected_x = (
        torch.tensor([p - 0.5 for p in POSITIONS], dtype=torch.float32) * t["s"]
    )
    assert torch.allclose(centers[:, 0], expected_x, atol=1e-5)


def test_curved_fish_radius():
    """kappa=5: centers trace arc with radius ~= 1/kappa."""
    kappa_val = 5.0
    t = _make_tensors(psi=0.0, theta=0.0, kappa=kappa_val, s=0.2)
    centers, _, _, _ = build_spine_frames(
        t["psi"], t["theta"], t["kappa"], t["s"], POSITIONS
    )
    assert centers.shape == (N, 3)
    # For a circular arc, all centers should be equidistant from the arc center.
    # Arc center is at (0, 1/kappa) in heading-dorsal frame (heading=X, dorsal=Z here
    # since psi=0, theta=0 → dorsal is world-up [0,0,1]).
    # But centers are centered at t=0.5; compute arc center relative to that.
    # Instead just verify: the curvature of the fitted circle matches kappa.
    # Simple check: distance from centers to a consistent arc center should be ~1/kappa.
    # With psi=0, theta=0: heading=[1,0,0], dorsal=[0,0,1]
    # Arc center (before centering) is at p_arc = (1/kappa) * dorsal = [0, 0, 1/kappa]
    # Midpoint center: arc_arg_mid = kappa*0.5*s=5*0.5*0.2=0.5
    # midpoint = [sin(0.5)/5, 0, (1-cos(0.5))/5]
    arc_arg_mid = kappa_val * 0.5 * 0.2
    mid_x = math.sin(arc_arg_mid) / kappa_val
    mid_z = (1 - math.cos(arc_arg_mid)) / kappa_val
    # Arc center in absolute frame: [0, 0, 1/kappa]
    # After centering by midpoint: arc_center_rel = [0 - mid_x, 0, 1/kappa - mid_z]
    arc_center_x = 0.0 - mid_x
    arc_center_z = 1.0 / kappa_val - mid_z
    arc_center = torch.tensor([arc_center_x, 0.0, arc_center_z])
    dists = torch.linalg.norm(centers - arc_center[None, :], dim=-1)  # (N,)
    expected_r = 1.0 / kappa_val
    assert torch.allclose(dists, torch.full((N,), expected_r), atol=1e-5), (
        f"Radii {dists} not close to 1/kappa={expected_r}"
    )


def test_tangent_unit_length():
    """Tangent vectors are unit length for both straight and curved fish."""
    for kappa_val in [0.0, 0.001, 2.0, 5.0]:
        t = _make_tensors(psi=0.5, theta=0.1, kappa=kappa_val, s=0.15)
        _, tangents, _, _ = build_spine_frames(
            t["psi"], t["theta"], t["kappa"], t["s"], POSITIONS
        )
        norms = torch.linalg.norm(tangents, dim=-1)
        assert torch.allclose(norms, torch.ones(N), atol=1e-5), (
            f"kappa={kappa_val}: tangent norms {norms} not unit"
        )


def test_near_zero_kappa_finite():
    """kappa=1e-6 (near zero) produces finite centers and frames (no NaN)."""
    t = _make_tensors(psi=0.3, theta=0.05, kappa=1e-6, s=0.15)
    centers, tangents, normals, binormals = build_spine_frames(
        t["psi"], t["theta"], t["kappa"], t["s"], POSITIONS
    )
    for name, tensor in [
        ("centers", centers),
        ("tangents", tangents),
        ("normals", normals),
        ("binormals", binormals),
    ]:
        assert torch.all(tensor.isfinite()), f"{name} has non-finite values: {tensor}"


def test_gradient_flows_through_psi():
    """Gradients flow from centers back to psi (yaw)."""
    psi = torch.tensor(0.3, requires_grad=True)
    theta = torch.tensor(0.0)
    kappa = torch.tensor(1.0)
    s = torch.tensor(0.15)
    centers, _, _, _ = build_spine_frames(psi, theta, kappa, s, POSITIONS)
    centers.sum().backward()
    assert psi.grad is not None
    assert psi.grad.isfinite()


def test_gradient_flows_through_theta():
    """Gradients flow from centers back to theta (pitch)."""
    psi = torch.tensor(0.0)
    theta = torch.tensor(0.2, requires_grad=True)
    kappa = torch.tensor(1.0)
    s = torch.tensor(0.15)
    centers, _, _, _ = build_spine_frames(psi, theta, kappa, s, POSITIONS)
    centers.sum().backward()
    assert theta.grad is not None
    assert theta.grad.isfinite()


def test_gradient_flows_through_kappa():
    """Gradients flow from centers back to kappa (curvature)."""
    psi = torch.tensor(0.0)
    theta = torch.tensor(0.0)
    kappa = torch.tensor(2.0, requires_grad=True)
    s = torch.tensor(0.15)
    centers, _, _, _ = build_spine_frames(psi, theta, kappa, s, POSITIONS)
    centers.sum().backward()
    assert kappa.grad is not None
    assert kappa.grad.isfinite()


def test_gradient_flows_through_s():
    """Gradients flow from centers back to s (scale)."""
    psi = torch.tensor(0.0)
    theta = torch.tensor(0.0)
    kappa = torch.tensor(1.0)
    s = torch.tensor(0.15, requires_grad=True)
    centers, _, _, _ = build_spine_frames(psi, theta, kappa, s, POSITIONS)
    centers.sum().backward()
    assert s.grad is not None
    assert s.grad.isfinite()


def test_gradient_flows_near_zero_kappa():
    """Gradients flow and are finite even at kappa=1e-6."""
    psi = torch.tensor(0.0)
    theta = torch.tensor(0.0)
    kappa = torch.tensor(1e-6, requires_grad=True)
    s = torch.tensor(0.15)
    centers, _, _, _ = build_spine_frames(psi, theta, kappa, s, POSITIONS)
    centers.sum().backward()
    assert kappa.grad is not None
    assert kappa.grad.isfinite(), f"kappa grad={kappa.grad} is not finite"


def test_frames_shape():
    """All returned tensors have shape (N, 3)."""
    t = _make_tensors(psi=0.0, theta=0.0, kappa=1.0, s=0.15)
    centers, tangents, normals, binormals = build_spine_frames(
        t["psi"], t["theta"], t["kappa"], t["s"], POSITIONS
    )
    for name, tensor in [
        ("centers", centers),
        ("tangents", tangents),
        ("normals", normals),
        ("binormals", binormals),
    ]:
        assert tensor.shape == (N, 3), f"{name} shape {tensor.shape} != ({N}, 3)"
