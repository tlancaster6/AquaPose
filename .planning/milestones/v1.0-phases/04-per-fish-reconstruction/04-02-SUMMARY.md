---
phase: 04-per-fish-reconstruction
plan: 02
subsystem: optimization
tags: [optimizer, adam, 2-start, warm-start, convergence, gradient-clipping, analysis-by-synthesis]

# Dependency graph
requires:
  - phase: 04-per-fish-reconstruction
    plan: 01
    provides: "RefractiveSilhouetteRenderer.render() and multi_objective_loss() — the inner loop called each optimizer iteration"
  - phase: 03-fish-mesh-model-and-3d-initialization
    provides: "build_fish_mesh(list[FishState]) -> Meshes, FishState dataclass"
provides:
  - "FishOptimizer: 2-start first-frame + warm-start subsequent-frame optimizer"
  - "make_optimizable_state: clone FishState with requires_grad=True"
  - "warm_start_from_velocity: constant-velocity p+psi extrapolation from last 2 frames"
affects:
  - 04-03 (sequence pipeline)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "2-start initialization: run both original psi and psi+pi for early_exit_iters, pick lower-loss winner, continue for remaining iters"
    - "Warm-start: p_pred = p_t1 + (p_t1 - p_t2), psi_pred similarly; theta/kappa/s copied unchanged"
    - "Per-parameter Adam groups: position lr=5x base, angular/morph params lr=1x base"
    - "Convergence early-stop: exit when all deltas in last patience+1 steps < convergence_delta"
    - "MockRenderer pattern: Gaussian blob alpha derived from mesh vertex mean — GPU-free, differentiable test harness"

key-files:
  created:
    - src/aquapose/optimization/optimizer.py
    - tests/unit/optimization/test_optimizer.py
  modified:
    - src/aquapose/optimization/__init__.py

key-decisions:
  - "Per-parameter Adam LR groups: p uses lr*5 (per RESEARCH.md Z-anisotropy note), angular/morph params use base lr"
  - "2-start early exit then continue winner: avoids running full budget on losing start, balances exploration vs. exploitation"
  - "MockRenderer Gaussian blob from vertex mean: avoids pytorch3d rendering overhead in unit tests while keeping real autograd graph through build_fish_mesh"

requirements-completed: [RECON-03, RECON-04]

# Metrics
duration: 6min
completed: 2026-02-21
---

# Phase 4 Plan 02: FishOptimizer with 2-Start and Warm-Start Summary

**FishOptimizer wrapping Adam optimization with 2-start heading disambiguation, warm-start constant-velocity prediction, and convergence early-stop — plus 12 unit tests with a GPU-free mock renderer**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-21T03:44:15Z
- **Completed:** 2026-02-21T03:50:05Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `make_optimizable_state`: clones all 5 FishState tensors with `requires_grad=True` on target device
- `get_state_params`: returns `[p, psi, theta, kappa, s]` list for optimizer/clip_grad_norm_ calls
- `warm_start_from_velocity`: constant-velocity extrapolation for p and psi; theta/kappa/s copied from most recent frame; all tensors detached
- `FishOptimizer._run_optimization_loop`: Adam with per-parameter LR groups (p: 5x base), clip_grad_norm_, convergence early-stop (patience-based)
- `FishOptimizer.optimize_first_frame`: 2-start (original + pi-flipped psi), early exit to pick winner, continue winner for remaining budget
- `FishOptimizer.optimize_frame`: single-call warm-start optimization
- `FishOptimizer.optimize_sequence`: chains first-frame + warm-start across arbitrary-length sequences
- 12 unit tests covering all optimizer behaviors; MockRenderer with differentiable Gaussian blob avoids GPU/pytorch3d overhead
- 282 total tests passing (no regressions)

## Task Commits

1. **Task 1: Implement FishOptimizer with 2-start and warm-start strategies** - `316868a` (feat)
2. **Task 2: Unit tests for optimizer strategies** - `6e25b10` (feat)

**Plan metadata:** _(final docs commit — see below)_

## Files Created/Modified

- `src/aquapose/optimization/optimizer.py` — FishOptimizer, make_optimizable_state, get_state_params, warm_start_from_velocity
- `src/aquapose/optimization/__init__.py` — Added FishOptimizer, make_optimizable_state, warm_start_from_velocity to public API
- `tests/unit/optimization/test_optimizer.py` — 12 tests: state utilities, warm-start velocity, 2-start selection, convergence early-stop, gradient clipping, sequence chaining, loss decrease

## Decisions Made

- **Per-parameter Adam LR groups**: The plan specified `[{"params": [state.p], "lr": lr * 5}, ...]`. This matches the RESEARCH.md note that Z/XY anisotropy (~132x) means position benefits from a larger step size relative to angular parameters. Implemented as planned.

- **2-start early exit then continue winner**: Running both starts for `early_exit_iters` then selecting the winner and continuing avoids wasting the full `max_iters_first` budget on the losing initialization. The remaining budget (`max_iters_first - early_exit_iters`) runs from the winner state.

- **MockRenderer Gaussian blob from vertex mean**: Tests need a differentiable renderer so gradients can actually flow through the optimizer loop. Using the mean of mesh vertex x/y coordinates scaled to pixel space provides a lightweight but real autograd graph through `build_fish_mesh`. This lets `test_optimizer_loss_decreases` verify actual gradient descent without pytorch3d rendering.

## Deviations from Plan

None — plan executed exactly as written. Auto-fix Rule 1 applied once for the ruff `RUF059` lint error (unused variable `opt_state` renamed to `_opt_state`). No behavioral changes.

## Self-Check

Files verified to exist:
- [x] src/aquapose/optimization/optimizer.py
- [x] src/aquapose/optimization/__init__.py
- [x] tests/unit/optimization/test_optimizer.py

Commits verified: 316868a, 6e25b10

## Self-Check: PASSED

---
*Phase: 04-per-fish-reconstruction*
*Completed: 2026-02-21*
