"""FishOptimizer: 2-start first-frame and warm-start per-frame analysis-by-synthesis."""

from __future__ import annotations

import torch
from torch.nn.utils import clip_grad_norm_

from aquapose.mesh.builder import build_fish_mesh
from aquapose.mesh.state import FishState

from .loss import multi_objective_loss


def make_optimizable_state(state: FishState, device: str = "cpu") -> FishState:
    """Clone a FishState into gradient-enabled tensors on the target device.

    All 5 tensors are detached from any existing computation graph, cloned,
    moved to ``device``, and marked as requiring gradients.

    Args:
        state: Source FishState. Tensors may or may not have gradients.
        device: Target device string (e.g. ``"cpu"``, ``"cuda:0"``).

    Returns:
        A new FishState with all tensors requiring gradients on ``device``.
    """
    return FishState(
        p=state.p.detach().clone().to(device).requires_grad_(True),
        psi=state.psi.detach().clone().to(device).requires_grad_(True),
        theta=state.theta.detach().clone().to(device).requires_grad_(True),
        kappa=state.kappa.detach().clone().to(device).requires_grad_(True),
        s=state.s.detach().clone().to(device).requires_grad_(True),
    )


def get_state_params(state: FishState) -> list[torch.Tensor]:
    """Return a list of the 5 optimisable FishState parameter tensors.

    Args:
        state: FishState with gradient-enabled tensors.

    Returns:
        ``[state.p, state.psi, state.theta, state.kappa, state.s]``
    """
    return [state.p, state.psi, state.theta, state.kappa, state.s]


def warm_start_from_velocity(
    state_t1: FishState,
    state_t2: FishState | None,
) -> FishState:
    """Predict the next-frame initial pose via constant-velocity extrapolation.

    Given the two most recent optimised states (``state_t1`` is more recent than
    ``state_t2``), extrapolates position and heading one step forward using a
    first-order constant-velocity model::

        p_pred   = p_t1   + (p_t1   - p_t2)
        psi_pred = psi_t1 + (psi_t1 - psi_t2)

    Curvature, pitch, and scale are copied unchanged from ``state_t1`` (no
    velocity model for these parameters).

    If ``state_t2`` is None (only one prior frame available), returns a detached
    copy of ``state_t1`` with no velocity applied.

    All returned tensors are detached from the prior computation graph so that
    no gradients leak between frames.

    Args:
        state_t1: Most recent optimised FishState (frame t-1).
        state_t2: Second most recent optimised FishState (frame t-2), or None
            if the sequence has fewer than two completed frames.

    Returns:
        FishState with detached tensors representing the predicted initialisation
        for the next frame.
    """
    if state_t2 is None:
        return FishState(
            p=state_t1.p.detach().clone(),
            psi=state_t1.psi.detach().clone(),
            theta=state_t1.theta.detach().clone(),
            kappa=state_t1.kappa.detach().clone(),
            s=state_t1.s.detach().clone(),
        )

    p_pred = (state_t1.p + (state_t1.p - state_t2.p)).detach().clone()
    psi_pred = (state_t1.psi + (state_t1.psi - state_t2.psi)).detach().clone()

    return FishState(
        p=p_pred,
        psi=psi_pred,
        theta=state_t1.theta.detach().clone(),
        kappa=state_t1.kappa.detach().clone(),
        s=state_t1.s.detach().clone(),
    )


class FishOptimizer:
    """Analysis-by-synthesis optimizer for a single fish across a video sequence.

    Drives the per-frame optimization loop: renders the fish mesh from a given
    FishState, computes the multi-objective loss against observed silhouette masks,
    and updates the state parameters via Adam with gradient clipping.

    Two initialization strategies are supported:

    - **2-start (first frame)**: Runs both the original orientation and a 180-degree
      heading flip for ``early_exit_iters`` iterations, selects the lower-loss
      candidate, then continues the winner for the remaining iterations.
    - **Warm-start (subsequent frames)**: Uses constant-velocity extrapolation
      from the two most-recent optimised states as the initial guess, then
      optimises for ``max_iters_warmstart`` iterations.

    Convergence early-stop: the loop terminates when the absolute loss change
    over the last ``convergence_patience + 1`` steps is below ``convergence_delta``.

    Args:
        renderer: A ``RefractiveSilhouetteRenderer`` instance ready to call
            ``renderer.render(meshes, cameras, camera_ids)``.
        loss_weights: Dict of scalar multipliers for loss terms, e.g.
            ``{"iou": 1.0, "gravity": 0.05, "morph": 0.2}``.
        camera_weights: Dict camera_id -> float weight (from
            ``compute_angular_diversity_weights``).
        lr: Base Adam learning rate. Position uses ``lr * 5``.
        max_iters_first: Total iteration budget for first-frame optimization.
        max_iters_warmstart: Iteration budget for warm-started subsequent frames.
        early_exit_iters: Number of iterations for each 2-start candidate before
            selecting the winner.
        convergence_delta: Minimum absolute loss change over ``convergence_patience``
            steps to continue iterating.
        convergence_patience: Number of consecutive steps below ``convergence_delta``
            needed to trigger early stop.
        grad_clip_norm: Maximum L2 norm for gradient clipping. Applied each step.
    """

    def __init__(
        self,
        renderer,
        loss_weights: dict[str, float],
        camera_weights: dict[str, float],
        lr: float = 1e-3,
        max_iters_first: int = 300,
        max_iters_warmstart: int = 50,
        early_exit_iters: int = 50,
        convergence_delta: float = 1e-4,
        convergence_patience: int = 3,
        grad_clip_norm: float = 1.0,
    ) -> None:
        self.renderer = renderer
        self.loss_weights = loss_weights
        self.camera_weights = camera_weights
        self.lr = lr
        self.max_iters_first = max_iters_first
        self.max_iters_warmstart = max_iters_warmstart
        self.early_exit_iters = early_exit_iters
        self.convergence_delta = convergence_delta
        self.convergence_patience = convergence_patience
        self.grad_clip_norm = grad_clip_norm

    def _run_optimization_loop(
        self,
        state: FishState,
        target_masks: dict[str, torch.Tensor],
        crop_regions: dict[str, tuple[int, int, int, int] | None],
        cameras: list,
        camera_ids: list[str],
        n_iters: int,
    ) -> tuple[FishState, float]:
        """Run Adam optimization on a FishState for ``n_iters`` iterations.

        Creates gradient-enabled tensors, sets up per-parameter Adam groups
        (position LR is 5x base LR), then iterates: zero_grad -> build_fish_mesh
        -> render -> loss -> backward -> clip_grad_norm -> step.

        Convergence early-stop: exits when the absolute loss change over
        ``convergence_patience + 1`` consecutive steps is below
        ``convergence_delta``.

        Args:
            state: Initial FishState (tensors need not have gradients yet).
            target_masks: Dict camera_id -> binary mask tensor, shape (H, W).
            crop_regions: Dict camera_id -> (y1, x1, y2, x2) or None.
            cameras: List of RefractiveProjectionModel instances.
            camera_ids: Camera identifier strings matching ``cameras``.
            n_iters: Maximum iterations to run.

        Returns:
            Tuple of (optimised FishState, final total loss value).
        """
        state = make_optimizable_state(state)

        param_groups = [
            {"params": [state.p], "lr": self.lr * 5},
            {
                "params": [state.psi, state.theta, state.kappa, state.s],
                "lr": self.lr,
            },
        ]
        optimizer = torch.optim.Adam(param_groups)

        loss_history: list[float] = []
        final_loss: float = float("inf")

        for _ in range(n_iters):
            optimizer.zero_grad()

            meshes = build_fish_mesh([state])
            alpha_maps = self.renderer.render(meshes, cameras, camera_ids)
            losses = multi_objective_loss(
                state,
                alpha_maps,
                target_masks,
                crop_regions,
                self.camera_weights,
                self.loss_weights,
            )

            losses["total"].backward()
            clip_grad_norm_(get_state_params(state), self.grad_clip_norm)
            optimizer.step()

            loss_val = losses["total"].item()
            loss_history.append(loss_val)
            final_loss = loss_val

            # Convergence early-stop: check last patience+1 values.
            if len(loss_history) >= self.convergence_patience + 1:
                window = loss_history[-(self.convergence_patience + 1) :]
                deltas = [abs(window[i] - window[i - 1]) for i in range(1, len(window))]
                if all(d < self.convergence_delta for d in deltas):
                    break

        return state, final_loss

    def optimize_first_frame(
        self,
        init_state: FishState,
        target_masks: dict[str, torch.Tensor],
        crop_regions: dict[str, tuple[int, int, int, int] | None],
        cameras: list,
        camera_ids: list[str],
    ) -> FishState:
        """Optimize the first frame with 2-start heading disambiguation.

        Runs two initializations for ``early_exit_iters`` each:

        - **Start A**: the original ``init_state``
        - **Start B**: same as A but with ``psi`` flipped by ``pi`` radians

        The candidate with the lower loss after the early-exit phase continues
        for the remaining ``max_iters_first - early_exit_iters`` iterations.

        Args:
            init_state: Initial pose estimate (e.g. from Phase 3 triangulation).
            target_masks: Dict camera_id -> binary mask tensor, shape (H, W).
            crop_regions: Dict camera_id -> (y1, x1, y2, x2) or None.
            cameras: List of RefractiveProjectionModel instances.
            camera_ids: Camera identifier strings.

        Returns:
            Optimised FishState for the first frame.
        """
        # Start A: original heading.
        state_a, loss_a = self._run_optimization_loop(
            init_state,
            target_masks,
            crop_regions,
            cameras,
            camera_ids,
            self.early_exit_iters,
        )

        # Start B: 180-degree heading flip.
        flipped_psi = init_state.psi.detach() + torch.pi
        init_b = FishState(
            p=init_state.p.detach().clone(),
            psi=flipped_psi,
            theta=init_state.theta.detach().clone(),
            kappa=init_state.kappa.detach().clone(),
            s=init_state.s.detach().clone(),
        )
        state_b, loss_b = self._run_optimization_loop(
            init_b,
            target_masks,
            crop_regions,
            cameras,
            camera_ids,
            self.early_exit_iters,
        )

        # Pick winner.
        winner = state_a if loss_a <= loss_b else state_b

        # Run remaining iterations from the winner.
        remaining = self.max_iters_first - self.early_exit_iters
        if remaining > 0:
            winner, _ = self._run_optimization_loop(
                winner,
                target_masks,
                crop_regions,
                cameras,
                camera_ids,
                remaining,
            )

        return winner

    def optimize_frame(
        self,
        init_state: FishState,
        target_masks: dict[str, torch.Tensor],
        crop_regions: dict[str, tuple[int, int, int, int] | None],
        cameras: list,
        camera_ids: list[str],
    ) -> FishState:
        """Optimize a single subsequent frame using a warm-started initial pose.

        Runs ``max_iters_warmstart`` iterations from ``init_state`` (which should
        be produced by ``warm_start_from_velocity`` for best performance).

        Args:
            init_state: Warm-started initial pose for the current frame.
            target_masks: Dict camera_id -> binary mask tensor, shape (H, W).
            crop_regions: Dict camera_id -> (y1, x1, y2, x2) or None.
            cameras: List of RefractiveProjectionModel instances.
            camera_ids: Camera identifier strings.

        Returns:
            Optimised FishState for the current frame.
        """
        state, _ = self._run_optimization_loop(
            init_state,
            target_masks,
            crop_regions,
            cameras,
            camera_ids,
            self.max_iters_warmstart,
        )
        return state

    def optimize_sequence(
        self,
        first_frame_state: FishState,
        frames_data: list[dict],
        cameras: list,
        camera_ids: list[str],
    ) -> list[FishState]:
        """Optimize a sequence of frames using 2-start + warm-start strategy.

        Frame 0 is optimized with ``optimize_first_frame`` (2-start heading
        disambiguation). Each subsequent frame uses ``warm_start_from_velocity``
        to extrapolate from the two most-recent results, then calls
        ``optimize_frame``.

        Args:
            first_frame_state: Initial pose estimate for the first frame.
            frames_data: List of dicts, one per frame. Each dict must have:
                - ``"target_masks"``: Dict camera_id -> mask tensor, shape (H, W).
                - ``"crop_regions"``: Dict camera_id -> (y1, x1, y2, x2) or None.
            cameras: List of RefractiveProjectionModel instances (shared across frames).
            camera_ids: Camera identifier strings.

        Returns:
            List of optimised FishState, one per frame, in sequence order.
        """
        results: list[FishState] = []

        for i, frame in enumerate(frames_data):
            target_masks = frame["target_masks"]
            crop_regions = frame["crop_regions"]

            if i == 0:
                state = self.optimize_first_frame(
                    first_frame_state,
                    target_masks,
                    crop_regions,
                    cameras,
                    camera_ids,
                )
            else:
                state_t2 = results[-2] if len(results) >= 2 else None
                init = warm_start_from_velocity(results[-1], state_t2)
                state = self.optimize_frame(
                    init,
                    target_masks,
                    crop_regions,
                    cameras,
                    camera_ids,
                )

            results.append(state)

        return results
