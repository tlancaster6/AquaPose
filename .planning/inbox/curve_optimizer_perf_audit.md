# Curve Optimizer Performance Audit

**Date**: 2026-02-23
**File**: `src/aquapose/reconstruction/curve_optimizer.py` (1,430 lines)
**Supporting**: `src/aquapose/calibration/projection.py` (258 lines)

## Context

The curve optimizer fits 3D B-spline midlines against 2D skeleton observations via L-BFGS. The closure (loss computation + backward) is the hot path — called multiple times per L-BFGS step, across coarse and fine stages, for every frame. With 9 fish × ~4-6 cameras × ~50 fine steps × 300+ frames, small inefficiencies compound significantly.

## Findings

### P0 — High ROI

#### 1. Redundant Spline Evaluation (4x repeated einsum per closure)

- **Where**: `closure_coarse` (L1025), `closure_fine` (L1148) — each calls `_data_loss`, `_length_penalty`, `_z_variance_penalty`, `_chord_arc_penalty`
- **Problem**: Each loss function independently computes `spline_pts = einsum("ek,nkd->ned", basis, ctrl_pts)`. That's 4 identical `(n_eval, K) @ (N_fish, K, 3)` matmuls per closure call. `_length_penalty` and `_chord_arc_penalty` also both compute segment lengths from the same points.
- **Fix**: Compute `spline_pts` once in the closure, pass to all loss functions. Also compute shared `seg_lengths` / `arc_lengths` once for the two functions that need them.
- **Effort**: Low (1-2 hours). Refactor loss function signatures.
- **ROI**: ~25% closure compute reduction.

#### 2. Sequential Per-Fish Projection in `_data_loss` (L324-377)

- **Where**: `_data_loss` — the hottest function
- **Problem**: Python `for i in range(n_fish)` loop, then `for cam_id` loop, calling `model.project()` per (fish, camera) pair. With 9 fish × ~5 cameras = ~45 separate `project()` calls per closure eval. Each `project()` runs 10 Newton-Raphson iterations.
- **Fix**: Batch per-camera — concatenate all fish's spline points for a given camera into one `(N_fish * n_eval, 3)` tensor, project once, scatter results back. Collapses ~45 calls to ~12 (one per camera).
- **Effort**: Medium (3-5 hours). Must handle scatter-back and variable valid masks.
- **ROI**: ~3-4x speedup on data loss. Dominant cost due to NR iterations inside `project()`.

#### 3. Redundant Convergence Monitoring Forward Pass (L1192-1218)

- **Where**: Fine stage adaptive stopping loop
- **Problem**: After every L-BFGS step, per-fish losses are recomputed by calling `_data_loss` individually for each non-frozen fish in a Python for-loop. This duplicates all work the closure just did.
- **Fix**: Cache per-fish losses from the closure (it computes them internally at L349-351 but averages and discards). Add `_last_per_fish_losses` list populated during `_data_loss`.
- **Effort**: Low-Medium (2-3 hours).
- **ROI**: Eliminates ~450 redundant `_data_loss` calls per frame (9 fish × ~50 steps).

### P1 — Medium ROI

#### 4. Over-Iterated Newton-Raphson in `project()` (projection.py L212)

- **Where**: `RefractiveProjectionModel.project()`
- **Problem**: Fixed 10 NR iterations regardless of convergence. NR for this geometry typically converges in 3-5 iterations. Half the iterations are wasted.
- **Fix**: Profile convergence, reduce to 5-6 fixed iterations (still autograd-safe).
- **Effort**: Low (1 hour).
- **ROI**: ~40% speedup on projection. Compounds with Finding 2.

#### 5. Output Assembly Python Loops (L1288-1383)

- **Where**: Post-optimization Midline3D construction
- **Problem**: Three sub-issues:
  - scipy BSpline + 1000-point eval per fish (could use existing torch basis matrix)
  - `for j in range(n_pts)` inner loop for residual computation (trivially vectorizable)
  - Per-point half-width conversion loop
- **Fix**: Replace scipy with torch einsum for arc-length; vectorize residuals with `np.linalg.norm(proj - obs, axis=1)`; vectorize half-width.
- **Effort**: Low-Medium (2-3 hours).
- **ROI**: Runs once per frame (not in optimization loop), but 300+ frames adds up. The point-loop at L1320-1328 is especially wasteful.

### P2 — Low ROI

#### 6. Chamfer `unsqueeze` Allocation (L264-272)

- **Where**: `_chamfer_distance_2d`, called ~45 times per closure
- **Problem**: `torch.cdist(proj.unsqueeze(0), obs.unsqueeze(0)).squeeze(0)` — unnecessary 3D tensor reshaping.
- **Fix**: Use `torch.cdist(proj, obs)` directly (PyTorch ≥2.0 supports 2D).
- **Effort**: Trivial (15 min).
- **ROI**: Marginal allocation savings.

#### 7. Warm-Start Fallback Check Sequential Eval (L1097-1128)

- **Where**: Pre-fine-stage warm-start validation
- **Problem**: Same pattern as Finding 3 — per-fish sequential `_data_loss` calls.
- **Fix**: Batch into single `_data_loss` call returning per-fish values (same refactor as Finding 3).
- **Effort**: Low (1 hour, piggybacks on Finding 3).
- **ROI**: N_fish forward passes per frame, only when warm-starts exist.

## Estimated Impact

| # | Finding | Effort | ROI |
|---|---------|--------|-----|
| 1 | Redundant spline eval (4x einsum) | Low | High |
| 2 | Sequential per-fish projection | Medium | High |
| 3 | Redundant convergence monitoring | Low-Med | High |
| 4 | Over-iterated Newton-Raphson | Low | Medium |
| 5 | Output assembly Python loops | Low-Med | Medium |
| 6 | Chamfer unsqueeze allocation | Trivial | Low |
| 7 | Warm-start check sequential eval | Low | Low-Med |

**P0 items (1-3) together**: estimated 60-80% reduction in per-frame optimization time (~3-5x speedup on the optimization loop).
