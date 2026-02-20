"""Tests for FishState dataclass."""

import torch

from aquapose.mesh.state import FishState


def make_state(requires_grad: bool = False) -> FishState:
    """Create a typical FishState for testing."""
    return FishState(
        p=torch.tensor([0.1, 0.2, 1.0], requires_grad=requires_grad),
        psi=torch.tensor(0.3, requires_grad=requires_grad),
        theta=torch.tensor(0.1, requires_grad=requires_grad),
        kappa=torch.tensor(2.0, requires_grad=requires_grad),
        s=torch.tensor(0.15, requires_grad=requires_grad),
    )


def test_construction():
    """FishState can be constructed from 5 tensors."""
    state = make_state()
    assert isinstance(state, FishState)


def test_field_types():
    """All fields are torch.Tensor instances."""
    state = make_state()
    assert isinstance(state.p, torch.Tensor)
    assert isinstance(state.psi, torch.Tensor)
    assert isinstance(state.theta, torch.Tensor)
    assert isinstance(state.kappa, torch.Tensor)
    assert isinstance(state.s, torch.Tensor)


def test_field_shapes():
    """Field shapes match the documented conventions."""
    state = make_state()
    assert state.p.shape == (3,), f"p shape {state.p.shape} != (3,)"
    assert state.psi.shape == (), f"psi shape {state.psi.shape} != ()"
    assert state.theta.shape == (), f"theta shape {state.theta.shape} != ()"
    assert state.kappa.shape == (), f"kappa shape {state.kappa.shape} != ()"
    assert state.s.shape == (), f"s shape {state.s.shape} != ()"


def test_requires_grad_preserved():
    """FishState preserves requires_grad on tensor fields."""
    state = make_state(requires_grad=True)
    assert state.p.requires_grad
    assert state.psi.requires_grad
    assert state.theta.requires_grad
    assert state.kappa.requires_grad
    assert state.s.requires_grad


def test_scalar_tensors_accepted():
    """Scalar fields accept both tensor() and tensor(0.0) form."""
    state = FishState(
        p=torch.zeros(3),
        psi=torch.tensor(0.0),
        theta=torch.zeros(()),
        kappa=torch.tensor(0.5),
        s=torch.tensor(0.2),
    )
    assert state.psi.ndim == 0
    assert state.theta.ndim == 0
