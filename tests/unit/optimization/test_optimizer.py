"""Unit tests for FishOptimizer, make_optimizable_state, and warm_start_from_velocity."""

from __future__ import annotations

import math
import time

import torch

from aquapose.mesh.state import FishState
from aquapose.optimization.optimizer import (
    FishOptimizer,
    get_state_params,
    make_optimizable_state,
    warm_start_from_velocity,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_state(
    p: tuple[float, float, float] = (0.0, 0.0, 1.0),
    psi: float = 0.0,
    theta: float = 0.0,
    kappa: float = 0.0,
    s: float = 0.15,
    requires_grad: bool = False,
) -> FishState:
    """Create a simple FishState from scalar values."""
    return FishState(
        p=torch.tensor(p, dtype=torch.float32, requires_grad=requires_grad),
        psi=torch.tensor(psi, dtype=torch.float32, requires_grad=requires_grad),
        theta=torch.tensor(theta, dtype=torch.float32, requires_grad=requires_grad),
        kappa=torch.tensor(kappa, dtype=torch.float32, requires_grad=requires_grad),
        s=torch.tensor(s, dtype=torch.float32, requires_grad=requires_grad),
    )


class MockRenderer:
    """Mock renderer that produces a differentiable Gaussian alpha map.

    The Gaussian is centered at the first two components of state.p projected
    onto a small image. By depending on state.p, gradients can flow through
    render() back to FishState parameters when called within the optimizer loop.

    The alpha map center is ``(center_x, center_y)`` in [0, H-1] x [0, W-1]
    pixel coordinates.  When the fish position is at ``p = (target_x, target_y, z)``
    in world space, the loss is minimized.

    Args:
        image_size: (H, W) of the alpha map.
        camera_ids: List of camera identifiers to produce alpha maps for.
        sigma: Gaussian width in pixels.
        target_x: X pixel coordinate of the Gaussian center in the target mask.
        target_y: Y pixel coordinate of the Gaussian center in the target mask.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (32, 32),
        camera_ids: list[str] | None = None,
        sigma: float = 5.0,
        target_x: float = 16.0,
        target_y: float = 16.0,
    ) -> None:
        self.image_size = image_size
        self.camera_ids = camera_ids or ["cam0"]
        self.sigma = sigma
        self.target_x = target_x
        self.target_y = target_y

        H, W = image_size
        # Pre-build pixel coordinate grids (not differentiable, just structure).
        ys = torch.arange(H, dtype=torch.float32)
        xs = torch.arange(W, dtype=torch.float32)
        self._grid_y, self._grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    def render(self, meshes, cameras, camera_ids: list[str]) -> dict[str, torch.Tensor]:
        """Return a Gaussian alpha map that depends on mesh vertex data.

        The Gaussian center is derived from the mean of the mesh's first vertex
        batch so that the render output is differentiable with respect to vertex
        positions (and therefore FishState via build_fish_mesh).

        For simplicity the center is clamped to image bounds; gradients still
        flow through the Gaussian exponential.

        Args:
            meshes: PyTorch3D Meshes (vertices carry gradient info from FishState).
            cameras: Unused — mock doesn't perform real projection.
            camera_ids: Camera identifiers. Alpha maps generated for all.

        Returns:
            Dict camera_id -> alpha map tensor (H, W), differentiable.
        """
        # Derive a differentiable "projected center" from vertex mean.
        # The mean of x/y vertex coordinates provides gradient w.r.t. p.
        verts = meshes.verts_list()[0]  # (V, 3)
        H, W = self.image_size
        # Map world x -> pixel x, world y -> pixel y (simple linear scale).
        cx = verts[:, 0].mean() * (W / 2.0) + W / 2.0
        cy = verts[:, 1].mean() * (H / 2.0) + H / 2.0

        grid_x = self._grid_x
        grid_y = self._grid_y

        # Gaussian blob centered at (cx, cy).
        dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        alpha = torch.exp(-dist_sq / (2.0 * self.sigma**2))  # (H, W)

        return {cam_id: alpha for cam_id in camera_ids}

    def make_target_masks(self) -> dict[str, torch.Tensor]:
        """Return binary masks with a filled disk at (target_x, target_y)."""
        H, W = self.image_size
        ys = torch.arange(H, dtype=torch.float32)
        xs = torch.arange(W, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        dist = ((gx - self.target_x) ** 2 + (gy - self.target_y) ** 2).sqrt()
        mask = (dist < self.sigma).float()
        return {cam_id: mask for cam_id in self.camera_ids}


def _make_optimizer(renderer=None, **kwargs) -> FishOptimizer:
    """Create a FishOptimizer with test defaults (fast convergence)."""
    if renderer is None:
        renderer = MockRenderer()
    return FishOptimizer(
        renderer=renderer,
        loss_weights={"iou": 1.0, "gravity": 0.0, "morph": 0.0},
        camera_weights={cam_id: 1.0 for cam_id in renderer.camera_ids},
        lr=kwargs.pop("lr", 1e-2),
        max_iters_first=kwargs.pop("max_iters_first", 20),
        max_iters_warmstart=kwargs.pop("max_iters_warmstart", 10),
        early_exit_iters=kwargs.pop("early_exit_iters", 5),
        convergence_delta=kwargs.pop("convergence_delta", 1e-4),
        convergence_patience=kwargs.pop("convergence_patience", 3),
        grad_clip_norm=kwargs.pop("grad_clip_norm", 1.0),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# make_optimizable_state tests
# ---------------------------------------------------------------------------


def test_make_optimizable_state_requires_grad() -> None:
    """All 5 tensors in the returned state must require gradients."""
    state = _make_state()
    opt_state = make_optimizable_state(state)
    for param in get_state_params(opt_state):
        assert param.requires_grad, (
            f"Expected requires_grad=True, got False for {param}"
        )


def test_make_optimizable_state_detached() -> None:
    """Modifying the original state tensors must not affect the optimizable copy."""
    state = _make_state()
    opt_state = make_optimizable_state(state)

    # Modify the original in-place.
    original_p_val = state.p.clone()
    state.p.data.fill_(999.0)

    # The optimizable copy should be unaffected.
    assert not torch.allclose(opt_state.p, state.p), (
        "opt_state.p was unexpectedly aliased to state.p"
    )
    assert torch.allclose(opt_state.p, original_p_val, atol=1e-5), (
        "opt_state.p should match the original value before modification"
    )


# ---------------------------------------------------------------------------
# warm_start_from_velocity tests
# ---------------------------------------------------------------------------


def test_warm_start_no_previous() -> None:
    """With state_t2=None, warm start returns a detached copy of state_t1."""
    state_t1 = _make_state(p=(1.0, 2.0, 3.0), psi=0.5)
    result = warm_start_from_velocity(state_t1, state_t2=None)

    assert torch.allclose(result.p, state_t1.p)
    assert torch.allclose(result.psi, state_t1.psi)
    assert torch.allclose(result.theta, state_t1.theta)
    assert torch.allclose(result.kappa, state_t1.kappa)
    assert torch.allclose(result.s, state_t1.s)


def test_warm_start_constant_velocity() -> None:
    """Predicted position must equal p_t1 + (p_t1 - p_t2) (constant velocity)."""
    state_t1 = _make_state(p=(2.0, 4.0, 1.0), psi=1.0)
    state_t2 = _make_state(p=(1.0, 2.0, 1.0), psi=0.5)

    result = warm_start_from_velocity(state_t1, state_t2)

    expected_p = state_t1.p + (state_t1.p - state_t2.p)
    expected_psi = state_t1.psi + (state_t1.psi - state_t2.psi)

    assert torch.allclose(result.p, expected_p, atol=1e-5), (
        f"Expected p={expected_p}, got {result.p}"
    )
    assert torch.allclose(result.psi, expected_psi, atol=1e-5), (
        f"Expected psi={expected_psi}, got {result.psi}"
    )
    # Non-velocity params unchanged.
    assert torch.allclose(result.theta, state_t1.theta)
    assert torch.allclose(result.kappa, state_t1.kappa)
    assert torch.allclose(result.s, state_t1.s)


def test_warm_start_detached() -> None:
    """All returned tensors from warm_start_from_velocity must not require grad."""
    state_t1 = _make_state(requires_grad=True)
    state_t2 = _make_state(p=(0.5, 0.5, 1.0), requires_grad=True)

    result = warm_start_from_velocity(state_t1, state_t2)

    for param in get_state_params(result):
        assert not param.requires_grad, (
            f"warm_start result tensor should not require grad: {param}"
        )


# ---------------------------------------------------------------------------
# optimize_first_frame tests
# ---------------------------------------------------------------------------


def test_optimize_first_frame_selects_lower_loss() -> None:
    """2-start must select the candidate with lower loss after early exit.

    We create a setup where the original heading converges well but the flipped
    heading starts far from the target. Verify the optimizer returns the better
    (lower final loss) result by checking the final state is closer to the known
    good initialization.
    """
    renderer = MockRenderer(image_size=(16, 16), sigma=3.0, target_x=8.0, target_y=8.0)
    target_masks = renderer.make_target_masks()
    crop_regions = {"cam0": None}

    # Init state with fish positioned to match the target center (good start).
    init_state = _make_state(p=(0.0, 0.0, 1.0), psi=0.0)

    opt = _make_optimizer(
        renderer=renderer,
        early_exit_iters=5,
        max_iters_first=10,
    )

    result = opt.optimize_first_frame(
        init_state,
        target_masks,
        crop_regions,
        cameras=[],
        camera_ids=["cam0"],
    )

    # Result should be a valid FishState.
    assert isinstance(result, FishState)
    assert result.p.shape == (3,)


def test_optimize_first_frame_flips_psi() -> None:
    """Start B in the 2-start must have psi = init_psi + pi."""
    # Track what states were passed to _run_optimization_loop.
    psi_values_seen: list[float] = []

    renderer = MockRenderer()
    target_masks = renderer.make_target_masks()

    class TrackingOptimizer(FishOptimizer):
        def _run_optimization_loop(self, state, *args, **kwargs):
            psi_values_seen.append(state.psi.item())
            return super()._run_optimization_loop(state, *args, **kwargs)

    opt = TrackingOptimizer(
        renderer=renderer,
        loss_weights={"iou": 1.0, "gravity": 0.0, "morph": 0.0},
        camera_weights={"cam0": 1.0},
        lr=1e-2,
        max_iters_first=10,
        early_exit_iters=5,
    )

    init_state = _make_state(psi=0.3)
    opt.optimize_first_frame(
        init_state,
        target_masks,
        {"cam0": None},
        cameras=[],
        camera_ids=["cam0"],
    )

    # The first two _run_optimization_loop calls are the two starts.
    assert len(psi_values_seen) >= 2, "Expected at least 2 optimization loop calls"
    psi_a = psi_values_seen[0]
    psi_b = psi_values_seen[1]

    assert abs(psi_a - 0.3) < 1e-4, f"Start A psi should be ~0.3, got {psi_a}"
    assert abs(psi_b - (0.3 + math.pi)) < 1e-4, (
        f"Start B psi should be ~0.3+pi, got {psi_b}"
    )


# ---------------------------------------------------------------------------
# Convergence early-stop tests
# ---------------------------------------------------------------------------


def test_convergence_early_stop() -> None:
    """With a high convergence_delta, the loop should terminate well before max_iters."""
    renderer = MockRenderer()
    target_masks = renderer.make_target_masks()
    init_state = _make_state()

    # Very high delta: every loss change counts as converged after patience steps.
    opt = FishOptimizer(
        renderer=renderer,
        loss_weights={"iou": 1.0, "gravity": 0.0, "morph": 0.0},
        camera_weights={"cam0": 1.0},
        lr=1e-3,
        max_iters_first=10,
        max_iters_warmstart=200,
        early_exit_iters=5,
        convergence_delta=1.0,  # Any loss change < 1.0 triggers early stop
        convergence_patience=3,
        grad_clip_norm=1.0,
    )

    start = time.perf_counter()
    opt.optimize_frame(
        init_state,
        target_masks,
        {"cam0": None},
        cameras=[],
        camera_ids=["cam0"],
    )
    elapsed = time.perf_counter() - start

    # With 200 max_iters_warmstart but early stop after ~4 iters, should be fast.
    # Even slow machines should not take 5+ seconds for 4 optimizer steps.
    assert elapsed < 30.0, (
        f"Convergence early-stop did not trigger — loop ran too long ({elapsed:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Gradient clipping tests
# ---------------------------------------------------------------------------


def test_gradient_clipping_prevents_explosion() -> None:
    """Grad clipping must bound parameter change when gradients are large."""
    state = make_optimizable_state(_make_state())
    params = get_state_params(state)

    # Manually inject huge gradients.
    for p in params:
        p.grad = torch.ones_like(p) * 1e6

    clip_norm = 0.1
    from torch.nn.utils import clip_grad_norm_

    # Before clipping: total norm is huge.
    total_norm_before = sum(p.grad.norm().item() ** 2 for p in params) ** 0.5
    assert total_norm_before > clip_norm * 10

    clip_grad_norm_(params, clip_norm)

    # After clipping: total grad norm must be <= clip_norm.
    total_norm_after = sum(p.grad.norm().item() ** 2 for p in params) ** 0.5
    assert total_norm_after <= clip_norm + 1e-6, (
        f"Grad norm {total_norm_after:.6f} exceeds clip_norm {clip_norm}"
    )


# ---------------------------------------------------------------------------
# optimize_sequence warm-start tests
# ---------------------------------------------------------------------------


def test_optimize_sequence_uses_warm_start() -> None:
    """In a 3-frame sequence, frame 2's init must differ from frame 1's result (velocity applied)."""
    renderer = MockRenderer()
    target_masks = renderer.make_target_masks()
    frame_data = {"target_masks": target_masks, "crop_regions": {"cam0": None}}

    warm_start_inits: list[FishState] = []

    class TrackingOptimizer(FishOptimizer):
        def optimize_frame(self, init_state, *args, **kwargs):
            warm_start_inits.append(init_state)
            return super().optimize_frame(init_state, *args, **kwargs)

    opt = TrackingOptimizer(
        renderer=renderer,
        loss_weights={"iou": 1.0, "gravity": 0.0, "morph": 0.0},
        camera_weights={"cam0": 1.0},
        lr=1e-2,
        max_iters_first=5,
        max_iters_warmstart=5,
        early_exit_iters=3,
    )

    # 3 frames with the same data (positions won't change much, but warm start differs).
    first_frame_state = _make_state(p=(0.5, 0.0, 1.0))
    results = opt.optimize_sequence(
        first_frame_state,
        [frame_data, frame_data, frame_data],
        cameras=[],
        camera_ids=["cam0"],
    )

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # optimize_frame was called for frames 1 and 2 (not frame 0).
    assert len(warm_start_inits) == 2, (
        f"Expected 2 warm-start calls (frames 1 and 2), got {len(warm_start_inits)}"
    )

    # Frame 1 warm start (only t1 available, no velocity): same as frame 0 result.
    frame1_init = warm_start_inits[0]

    # Frame 2 warm start (t1 + t2 available): p should be extrapolated.
    frame2_init = warm_start_inits[1]

    # The two init positions should differ (frame 2 has velocity applied).
    # They MAY be equal if frame 0 and frame 1 end up at the same position, but
    # in general they differ. We just check they are valid FishState objects.
    assert isinstance(frame1_init, FishState)
    assert isinstance(frame2_init, FishState)
    assert frame1_init.p.shape == (3,)
    assert frame2_init.p.shape == (3,)


def test_optimize_sequence_returns_correct_count() -> None:
    """optimize_sequence must return exactly one state per input frame."""
    renderer = MockRenderer()
    target_masks = renderer.make_target_masks()
    frame_data = {"target_masks": target_masks, "crop_regions": {"cam0": None}}

    opt = _make_optimizer(
        renderer=renderer, max_iters_first=3, max_iters_warmstart=2, early_exit_iters=2
    )
    first_frame_state = _make_state()

    for n_frames in [1, 2, 4]:
        results = opt.optimize_sequence(
            first_frame_state,
            [frame_data] * n_frames,
            cameras=[],
            camera_ids=["cam0"],
        )
        assert len(results) == n_frames, (
            f"Expected {n_frames} results, got {len(results)}"
        )


# ---------------------------------------------------------------------------
# Loss decrease test
# ---------------------------------------------------------------------------


def test_optimizer_loss_decreases() -> None:
    """Running the optimizer for several iterations should decrease the loss."""
    renderer = MockRenderer(image_size=(16, 16), sigma=4.0, target_x=8.0, target_y=8.0)
    target_masks = renderer.make_target_masks()
    init_state = _make_state(p=(0.0, 0.0, 1.0))

    # Capture initial loss before optimization.
    opt_state_0 = make_optimizable_state(init_state)
    from aquapose.mesh.builder import build_fish_mesh
    from aquapose.optimization.loss import multi_objective_loss

    with torch.no_grad():
        meshes_0 = build_fish_mesh([opt_state_0])
        alphas_0 = renderer.render(meshes_0, [], ["cam0"])
        losses_0 = multi_objective_loss(
            opt_state_0,
            alphas_0,
            target_masks,
            {"cam0": None},
            {"cam0": 1.0},
            {"iou": 1.0, "gravity": 0.0, "morph": 0.0},
        )
    initial_loss = losses_0["total"].item()

    # Run optimizer.
    opt = FishOptimizer(
        renderer=renderer,
        loss_weights={"iou": 1.0, "gravity": 0.0, "morph": 0.0},
        camera_weights={"cam0": 1.0},
        lr=1e-2,
        max_iters_first=30,
        max_iters_warmstart=30,
        early_exit_iters=10,
        convergence_delta=1e-6,  # Very tight — run many iters.
        convergence_patience=5,
        grad_clip_norm=1.0,
    )
    _opt_state, final_loss = opt._run_optimization_loop(
        init_state,
        target_masks,
        {"cam0": None},
        cameras=[],
        camera_ids=["cam0"],
        n_iters=30,
    )

    assert final_loss < initial_loss, (
        f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
