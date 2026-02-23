# Phase 9: Curve-Based Optimization as a Replacement for Triangulation - Research

**Researched:** 2026-02-22
**Domain:** Differentiable B-spline optimization, Chamfer distance, L-BFGS, PyTorch autograd
**Confidence:** HIGH (all key components verified against existing codebase)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Migration strategy:**
- Keep both old triangulation and new curve optimizer side-by-side during validation
- Code organization at Claude's discretion — prefer clean separation (e.g., new module alongside old)
- Wire both methods into `scripts/diagnose_pipeline.py` so user can compare by running the script with either method
- No need to write new diagnostics — just ensure both methods are accessible from the existing script
- After validation, old triangulation code will be deleted — design for eventual removal, don't over-invest in compatibility layers
- Breaking API changes are fine — prefer clean code

**Species priors & tuning:**
- Fish body length is 70–100mm (not 45mm as in proposal draft)
- All regularization weights (length, curvature, smoothness) exposed via a `CurveOptimizerConfig` dataclass
- Global species prior only for Phase 9 — no per-identity length refinement yet
- Per-identity length prior deferred to a future iteration if needed

**Performance targets:**
- Must be faster than current triangulation pipeline (~76s for 30 frames)
- CUDA GPU available — optimizer should leverage GPU for batched optimization
- Implement warm-start from previous frame's solution with cold-start fallback
- Implement adaptive early stopping (per-fish convergence masking) — important for 9-fish batches

### Claude's Discretion
- Curvature limit starting value
- Code organization (new module structure, naming)
- Coarse-to-fine stage count and control point counts
- L-BFGS hyperparameters (learning rate, max iterations per stage)
- Convergence threshold for early stopping
- How to handle the multi-start flip mitigation (if needed)

### Deferred Ideas (OUT OF SCOPE)
- Per-identity length prior (running average of past reconstructions per fish) — add if global prior proves insufficient
- Multi-start optimization for head-tail flip resolution — add if flip rate is too high with coarse-to-fine alone
- Velocity-based warm-start extrapolation — add if simple previous-frame copy doesn't converge fast enough
</user_constraints>

---

## Summary

Phase 9 replaces the interior of `triangulate_midlines()` in `src/aquapose/reconstruction/triangulation.py` with a correspondence-free B-spline optimizer. Instead of solving explicit point-to-point correspondences across cameras (the source of five interacting subsystems and multiple confirmed bugs), the new approach directly optimizes 3D cubic B-spline control points by minimizing chamfer distance between reprojected spline points and observed 2D skeleton points across all cameras, using `RefractiveProjectionModel.project()` for differentiable refractive reprojection.

The implementation relies entirely on existing project infrastructure: `RefractiveProjectionModel.project()` already uses fixed-iteration Newton-Raphson (10 iterations) for autograd compatibility — confirmed by reading the source. Chamfer distance is ~10 lines of `torch.cdist`. B-spline basis matrices are precomputed constants. L-BFGS is available in `torch.optim.LBFGS`. All fish (9-fish batch) are processed in a single `(N_fish, K, 3)` tensor on GPU with per-fish convergence masking for adaptive early stopping.

The new module lives alongside `triangulation.py` as `curve_optimizer.py` in the same `reconstruction/` package. Both methods are wired into `scripts/diagnose_pipeline.py` behind a `--method` flag. The public API signature of `triangulate_midlines()` is preserved but implementation is replaced; the old code is retained in place for side-by-side comparison until the new optimizer is validated.

**Primary recommendation:** Implement `CurveOptimizer` as a stateful class in `src/aquapose/reconstruction/curve_optimizer.py` that holds warm-start state across frames; expose `optimize_midlines(midline_set, models, frame_index)` as the drop-in replacement for `triangulate_midlines()`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | >=2.0 (project requirement) | Autograd, GPU tensors, L-BFGS optimizer, `torch.cdist` | Already in project; `project()` is PyTorch-native |
| `torch.optim.LBFGS` | built-in | Second-order quasi-Newton optimizer for smooth low-dim problems | Correct choice for ~21 params/fish, smooth differentiable loss |
| `scipy.interpolate` | >=1.11 (project requirement) | Knot insertion via `make_lsq_spline`, B-spline basis reference | Already in project; used by existing triangulation |
| `numpy` | >=1.24 (project requirement) | Basis matrix precomputation (constant, cached), arc-length | Already in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.nn.functional.huber_loss` | built-in | Per-camera chamfer aggregation with outlier downweighting | Applied after per-camera chamfer scalars are computed |
| `dataclasses.dataclass` | stdlib | `CurveOptimizerConfig` for exposed hyperparameters | Follow project pattern (Midline3D, FishTrack are dataclasses) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `torch.optim.LBFGS` | `torch.optim.Adam` | Adam is noisier on smooth deterministic problems; L-BFGS converges faster for low-dim smooth objective |
| Precomputed basis matrix multiply | De Boor algorithm | De Boor is iterative, harder to batch, no autograd benefit over matrix form |
| `torch.cdist` for chamfer | Manual nearest-neighbor loops | `torch.cdist` is GPU-batched; loops are O(N^2) per camera in Python |

**No new dependencies needed.** All required libraries are already in `pyproject.toml`.

---

## Architecture Patterns

### Recommended Project Structure

New file added to existing `reconstruction/` package:
```
src/aquapose/reconstruction/
├── __init__.py              # Add CurveOptimizer, optimize_midlines to exports
├── midline.py               # Unchanged (2D extraction)
├── triangulation.py         # Unchanged during validation; deleted post-validation
└── curve_optimizer.py       # NEW: CurveOptimizerConfig + CurveOptimizer class
tests/unit/
└── test_curve_optimizer.py  # NEW: unit tests for optimizer components
```

### Pattern 1: Stateful Optimizer Class

**What:** `CurveOptimizer` holds warm-start state (dict from `fish_id` to previous-frame control points) across calls. This is how it provides warm-start initialization without callers managing state.

**When to use:** Any time you need to carry optimizer state across sequential frames.

```python
# src/aquapose/reconstruction/curve_optimizer.py

@dataclass
class CurveOptimizerConfig:
    """Exposed hyperparameters for the curve-based optimizer."""
    nominal_length_m: float = 0.085        # 85mm midpoint of 70-100mm range
    length_tolerance: float = 0.30         # ±30% of nominal
    lambda_length: float = 1.0
    lambda_curvature: float = 0.5
    lambda_smoothness: float = 0.1
    max_bend_angle_deg: float = 30.0       # per adjacent control-point triplet
    n_coarse_ctrl: int = 4                 # K for coarse stage
    n_fine_ctrl: int = 7                   # K for fine stage (matches SPLINE_N_CTRL)
    n_eval_points: int = 20               # N for spline evaluation during loss
    lbfgs_lr: float = 0.1
    lbfgs_max_iter_coarse: int = 60
    lbfgs_max_iter_fine: int = 100
    convergence_delta: float = 1e-4        # per-fish loss delta for early stopping
    convergence_patience: int = 5          # consecutive steps below delta to freeze
    warm_start_loss_ratio: float = 2.0    # fall back to cold start if warm loss > ratio * prev loss


class CurveOptimizer:
    """Stateful curve-based 3D midline optimizer."""

    def __init__(self, config: CurveOptimizerConfig | None = None) -> None: ...

    def optimize_midlines(
        self,
        midline_set: MidlineSet,
        models: dict[str, RefractiveProjectionModel],
        frame_index: int = 0,
        fish_centroids: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, Midline3D]: ...
```

**Key design point:** `optimize_midlines()` matches the signature of `triangulate_midlines()` for drop-in substitution in `diagnose_pipeline.py`.

### Pattern 2: Batched GPU Optimization

**What:** All fish control points are batched into a single `(N_fish, K, 3)` tensor. L-BFGS operates on the full batch. Per-fish convergence masking zeros out gradients for converged fish without removing them from the batch (avoids dynamic graph issues).

**When to use:** When N_fish is small (9) and K is small (4 or 7); batching avoids Python loop overhead and GPU launch latency per fish.

```python
# Batched control points: (N_fish, K, 3), requires_grad=True
ctrl_pts = torch.zeros(n_fish, K, 3, device=device, requires_grad=True)

# Convergence mask: (N_fish,) bool, True = still optimizing
active = torch.ones(n_fish, dtype=torch.bool, device=device)

def closure():
    optimizer.zero_grad()
    loss = _compute_total_loss(ctrl_pts, midlines_per_fish, models, config)
    # Zero gradients for converged fish BEFORE backward
    # NOTE: use mask on loss per fish, not on gradients directly
    loss.backward()
    # After backward: zero grad for inactive fish
    with torch.no_grad():
        ctrl_pts.grad[~active] = 0.0
    return loss

optimizer.step(closure)
```

**Warning:** L-BFGS with `torch.optim.LBFGS` requires `closure` — the closure must call `zero_grad`, compute loss, call `backward`, and return the scalar loss. This is different from Adam.

### Pattern 3: B-Spline Basis Matrix (Precomputed)

**What:** For each K (n_coarse_ctrl=4, n_fine_ctrl=7), precompute the `(N, K)` basis matrix once at startup. Spline evaluation at N points is then `P = B @ C` where `C` is `(K, 3)`.

**Implementation approach:** The uniform cubic B-spline basis depends only on K and N, not on the data. Build using `scipy.interpolate.BSpline` with uniform knots evaluated at N points, or compute directly from the Cox-de Boor formula. Cache as a module-level dict keyed by `(N, K)`.

```python
import scipy.interpolate
import numpy as np
import torch

def _build_basis_matrix(n_eval: int, n_ctrl: int) -> torch.Tensor:
    """Precompute (n_eval, n_ctrl) B-spline basis matrix, uniform cubic."""
    # Uniform knot vector for cubic B-spline with n_ctrl control points
    degree = 3
    n_knots = n_ctrl + degree + 1
    # Clamped uniform knots: repeat endpoints degree+1 times
    t = np.concatenate([
        np.zeros(degree),
        np.linspace(0, 1, n_knots - 2 * degree),
        np.ones(degree),
    ])
    # Evaluate each basis function at n_eval points
    u = np.linspace(0, 1, n_eval)
    B = np.zeros((n_eval, n_ctrl), dtype=np.float32)
    for i in range(n_ctrl):
        c = np.zeros(n_ctrl)
        c[i] = 1.0
        spl = scipy.interpolate.BSpline(t, c, degree)
        B[:, i] = spl(u)
    return torch.from_numpy(B)  # (n_eval, n_ctrl)

# Module-level cache
_BASIS_CACHE: dict[tuple[int, int], torch.Tensor] = {}

def get_basis(n_eval: int, n_ctrl: int, device: torch.device) -> torch.Tensor:
    key = (n_eval, n_ctrl)
    if key not in _BASIS_CACHE:
        _BASIS_CACHE[key] = _build_basis_matrix(n_eval, n_ctrl)
    return _BASIS_CACHE[key].to(device)
```

**Confidence:** HIGH — this is standard B-spline theory; the clamped knot formula matches `SPLINE_KNOTS` used in existing triangulation for K=7.

### Pattern 4: Knot Insertion for Coarse-to-Fine

**What:** After coarse stage (K=4) converges, upsample control points to fine stage (K=7) using B-spline knot insertion — an exact linear map that preserves curve shape.

**Implementation:** This is a matrix multiply: `C_fine = T @ C_coarse` where `T` is the `(7, 4)` Oslo algorithm refinement matrix. Alternatively, use scipy to evaluate the coarse spline at the fine knot positions and re-fit. The exact approach is simpler and avoids a second fitting pass.

**Practical approach:** Evaluate the coarse spline at the fine control point parameter values using the basis matrix, then use those as the fine initial control points. This is not exact knot insertion but is numerically equivalent for initialization purposes and avoids implementing the Oslo algorithm:

```python
# After coarse stage: ctrl_coarse is (N_fish, 4, 3)
# Evaluate coarse spline at fine parameter positions to initialize fine ctrl
B_coarse = get_basis(n_fine_ctrl, n_coarse_ctrl, device)  # (7, 4)
ctrl_fine_init = B_coarse @ ctrl_coarse  # (N_fish, 7, 3) — approximate knot insertion
```

**Confidence:** MEDIUM — This is approximate (not exact knot insertion) but is standard practice for coarse-to-fine spline initialization. Exact Oslo algorithm adds ~30 lines of code for marginal accuracy gain at initialization only.

### Pattern 5: Chamfer Distance

**What:** For each reprojected spline point, find its nearest observed skeleton point; for each observed skeleton point, find its nearest reprojected point. Sum both directions.

```python
def chamfer_distance_2d(
    proj: torch.Tensor,   # (N_eval, 2) — reprojected spline points
    obs: torch.Tensor,    # (M, 2)      — observed skeleton points
) -> torch.Tensor:
    """Symmetric chamfer distance, scalar."""
    # (N_eval, M) pairwise distances
    dists = torch.cdist(proj.unsqueeze(0), obs.unsqueeze(0))[0]  # (N_eval, M)
    # Forward: each proj point to nearest obs
    fwd = dists.min(dim=1).values.mean()
    # Backward: each obs point to nearest proj
    bwd = dists.min(dim=0).values.mean()
    return fwd + bwd
```

**Key insight:** With N_eval=20 and M=15 (skeleton points), the distance matrix is tiny (20x15). `torch.cdist` handles this trivially on GPU.

### Anti-Patterns to Avoid

- **Per-fish Python loops in the optimization step:** The closure passed to L-BFGS runs many times per step. Any Python-level loop over fish inside the closure destroys GPU throughput. Batch everything.
- **Calling `project()` on CPU tensors:** `RefractiveProjectionModel` has a `.to(device)` method. Move all model tensors to GPU before optimization. The model stores internal tensors (K, R, t, C, normal) — all must be on the same device as the control points.
- **Using `torch.autograd.grad` inside the L-BFGS closure:** L-BFGS manages its own gradient accumulation. Always use `loss.backward()` inside the closure, then `return loss`.
- **NaN propagation from invalid projections:** `project()` returns NaN pixels for points above water surface. Filter `valid` mask before computing chamfer: only include valid projected points in loss.
- **Modifying `ctrl_pts` tensor in-place after `requires_grad=True`:** Use non-in-place ops. If you need to apply warm-start values, do it before calling `requires_grad_(True)`, or use `torch.no_grad()` context.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Quasi-Newton optimizer | Custom L-BFGS or gradient descent | `torch.optim.LBFGS` | Memory-efficient BFGS already handles line search, history management |
| Nearest-neighbor for chamfer | KD-tree or Python loops | `torch.cdist` | GPU-native pairwise distance; for small N (~20x15) it's exact and fast |
| Spline evaluation | Iterative de Boor per point | Precomputed basis matrix + `@` | One BLAS call, differentiable, cacheable |
| Autograd through Snell's law | Custom backward pass | Use `project()` as-is | It already uses fixed-count Newton-Raphson, which is autograd-compatible |

**Key insight:** `RefractiveProjectionModel.project()` already solves the autograd-through-iterative-solver problem via fixed 10-iteration Newton-Raphson. This is confirmed by reading the source. The proposal's concern about needing implicit differentiation is not needed — the existing implementation is already autograd-safe.

---

## Common Pitfalls

### Pitfall 1: L-BFGS Closure Requirements
**What goes wrong:** L-BFGS re-evaluates the loss multiple times per step (for line search). If the closure doesn't call `zero_grad()` before the forward pass, gradients accumulate across line search steps.
**Why it happens:** L-BFGS differs from first-order optimizers — it calls the closure multiple times.
**How to avoid:** Always include `optimizer.zero_grad()` as the first line inside the closure. Always return the scalar loss from the closure.
**Warning signs:** Loss oscillates or diverges even on a convex problem.

### Pitfall 2: RefractiveProjectionModel Device Mismatch
**What goes wrong:** `project()` fails with device mismatch if the model's internal tensors (K, R, t, C, normal) are on CPU but `ctrl_pts` is on CUDA.
**Why it happens:** `RefractiveProjectionModel` stores raw tensors, not `nn.Module` parameters.
**How to avoid:** Call `model.to("cuda")` on all models before optimization. The `.to()` method is implemented and moves all internal tensors.
**Warning signs:** `RuntimeError: Expected all tensors to be on the same device`.

### Pitfall 3: Chamfer Loss Scale Mismatch Across Cameras
**What goes wrong:** Cameras with many visible skeleton points contribute much larger chamfer loss than cameras with few points, unintentionally downweighting sparse views.
**Why it happens:** Chamfer is a sum/mean over points; more points = larger magnitude.
**How to avoid:** Use symmetric chamfer (mean, not sum) per camera, then Huber-aggregate across cameras. This normalizes by point count in each direction.

### Pitfall 4: Coarse Stage K=4 May Not Resolve Head-Tail Ambiguity Alone
**What goes wrong:** Head-tail flip is a symmetric global minimum. If the coarse initialization places control points equidistant between the two possible orientations, L-BFGS may get stuck at the saddle.
**Why it happens:** The coarse landscape is smoother but not guaranteed convex at the symmetry point.
**How to avoid:** Initialize the coarse spline using the 2D skeleton's principal axis from the reference camera to break symmetry. The cross-view identity centroid seed (from Phase 5) gives position; the reference camera's skeleton PCA gives initial orientation.
**Warning signs:** Arc length of optimized spline is half-expected (folded/compressed solution).

### Pitfall 5: L-BFGS History on GPU with Many Fish
**What goes wrong:** L-BFGS stores a history of `(s, y)` vector pairs (by default `history_size=100`). For N_fish=9 and K=7, the parameter vector is 9×7×3=189 floats — history is tiny. But if `history_size` is set too large, memory accumulates.
**Why it happens:** L-BFGS default `history_size` is 100 in PyTorch, which is fine for this problem size.
**How to avoid:** `history_size=10` is sufficient for ~21 parameters per fish. No action needed unless memory is a concern.

### Pitfall 6: `requires_grad=True` After Warm-Start Copy
**What goes wrong:** Copying previous-frame control points for warm-start and then calling `requires_grad_(True)` may silently discard gradients if done incorrectly.
**Why it happens:** `torch.Tensor.requires_grad_()` is in-place and safe, but `tensor.detach().clone().requires_grad_(True)` must be the pattern to avoid carrying computation graph from the previous optimization step.
**How to avoid:** Always `detach().clone().requires_grad_(True)` when copying warm-start values.

### Pitfall 7: Huber Loss Delta Scale
**What goes wrong:** `torch.nn.functional.huber_loss` has a `delta` parameter (default 1.0) that controls the quadratic-to-linear transition. In pixel units, chamfer distances can be 10-100px, making delta=1.0 effectively linear everywhere.
**Why it happens:** Huber delta must be set in the same units as the chamfer distance.
**How to avoid:** Set `delta` to roughly the expected inlier chamfer distance (e.g., 15-20px) so outlier cameras (residual > delta) transition to linear loss.

---

## Code Examples

### Verified: project() Is Autograd-Compatible

From `src/aquapose/calibration/projection.py` (lines 211-232):
```python
# Newton-Raphson iterations (fixed count for differentiability)
for _ in range(10):
    ...
    r_p = r_p - f / (f_prime + 1e-12)
    r_p = torch.clamp(r_p, min=0.0)
    r_p = torch.minimum(r_p, r_q)
```
The fixed 10-iteration count and use of standard PyTorch ops (no `torch.no_grad()`, no explicit `.detach()`) confirm full autograd compatibility. No custom backward pass needed.

**Confidence:** HIGH — verified by reading source.

### Verified: model.to(device) Is Implemented

From `src/aquapose/calibration/projection.py` (lines 92-107):
```python
def to(self, device: str | torch.device) -> RefractiveProjectionModel:
    self.K = self.K.to(device)
    self.K_inv = self.K_inv.to(device)
    self.R = self.R.to(device)
    self.t = self.t.to(device)
    self.C = self.C.to(device)
    self.normal = self.normal.to(device)
    return self
```
All model tensors moved correctly. Call `model.to("cuda")` before optimization.

**Confidence:** HIGH — verified by reading source.

### Verified: Midline3D Output Contract

From `src/aquapose/reconstruction/triangulation.py` (lines 48-86):
```python
@dataclass
class Midline3D:
    fish_id: int
    frame_index: int
    control_points: np.ndarray  # shape (7, 3), float32
    knots: np.ndarray           # shape (11,), float32
    degree: int                 # always 3
    arc_length: float
    half_widths: np.ndarray     # shape (N_SAMPLE_POINTS,), float32
    n_cameras: int
    mean_residual: float
    max_residual: float
    is_low_confidence: bool = False
    per_camera_residuals: dict[str, float] | None = None
```
The curve optimizer must output the same dataclass. `control_points` must be `np.ndarray` not `torch.Tensor` (existing HDF5 writer and overlay code expect numpy). Use `.detach().cpu().numpy()` at output time.

**Confidence:** HIGH — verified by reading source.

### Verified: existing SPLINE_KNOTS for K=7

From `src/aquapose/reconstruction/triangulation.py` (lines 27-31):
```python
SPLINE_KNOTS: np.ndarray = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
)
```
This is the clamped uniform cubic B-spline knot vector for K=7. The curve optimizer's fine stage must use the same knot vector so `Midline3D.knots` output is consistent with the HDF5 writer and downstream visualization code.

**Confidence:** HIGH — verified by reading source.

### L-BFGS Closure Pattern (PyTorch)

```python
# Standard L-BFGS usage in PyTorch
optimizer = torch.optim.LBFGS(
    [ctrl_pts],
    lr=config.lbfgs_lr,
    max_iter=config.lbfgs_max_iter_fine,
    history_size=10,
    line_search_fn="strong_wolfe",
)

def closure() -> torch.Tensor:
    optimizer.zero_grad()
    loss = _total_loss(ctrl_pts, ...)  # scalar
    loss.backward()
    return loss

optimizer.step(closure)
```

**Confidence:** HIGH — standard PyTorch L-BFGS API, unchanged since PyTorch 1.x.

### Warm-Start Pattern

```python
# _warm_starts: dict[int, torch.Tensor]  — fish_id -> (K, 3) control pts from prev frame

def _init_ctrl_pts(
    fish_ids: list[int],
    centroids: dict[int, torch.Tensor],
    warm_starts: dict[int, torch.Tensor],
    K: int,
    nominal_length: float,
    ref_orientations: dict[int, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Initialize (N_fish, K, 3) control points for optimization."""
    n_fish = len(fish_ids)
    ctrl = torch.zeros(n_fish, K, 3, device=device)
    for i, fid in enumerate(fish_ids):
        if fid in warm_starts:
            # Warm start: copy previous frame solution, detached
            prev = warm_starts[fid]  # (K_prev, 3) or (K, 3)
            # Handle K mismatch between coarse and fine stages via interpolation
            ctrl[i] = _upsample_ctrl_pts(prev, K, device)
        else:
            # Cold start: straight line at centroid, oriented along skeleton PCA
            ctrl[i] = _cold_start(centroids[fid], ref_orientations.get(fid), K, nominal_length, device)
    return ctrl.detach().clone().requires_grad_(True)
```

---

## Wiring Into diagnose_pipeline.py

The user wants both methods accessible from `scripts/diagnose_pipeline.py` via a flag. The recommended approach:

1. Add `--method {triangulation,curve}` argument to `diagnose_pipeline.py` (and `diagnose_triangulation.py` if needed)
2. In the triangulation stage, dispatch:
   ```python
   if args.method == "curve":
       from aquapose.reconstruction.curve_optimizer import CurveOptimizer
       optimizer = CurveOptimizer(config=CurveOptimizerConfig())
       midlines_3d = optimizer.optimize_midlines(midline_set, models, frame_index)
   else:
       from aquapose.reconstruction.triangulation import triangulate_midlines
       midlines_3d = triangulate_midlines(midline_set, models, frame_index)
   ```
3. The rest of the pipeline (half-width conversion, HDF5 output, visualization) consumes `dict[int, Midline3D]` unchanged in both cases.

**Confidence:** HIGH — verified by reading `diagnose_pipeline.py` and `diagnose_triangulation.py` structure.

---

## State of the Art

| Old Approach (current) | New Approach (Phase 9) | Impact |
|------------------------|------------------------|--------|
| RANSAC point-wise triangulation | Correspondence-free chamfer optimization | Eliminates 5 interacting subsystems |
| Greedy orientation alignment | Coarse-to-fine resolves flip implicitly | Removes O(N) orientation pre-pass |
| Epipolar correspondence snapping | Not needed (chamfer is correspondence-free) | Removes NaN-contamination source |
| `scipy.interpolate.make_lsq_spline` post-fit | Spline IS the optimization variable | Control points are directly optimized, not fit-to-points |
| CPU-only, per-fish serial | GPU-batched, all fish in parallel | Expected speedup over ~76s current pipeline |

---

## Open Questions

1. **Will `gradcheck` pass on `project()`?**
   - What we know: Fixed-iteration Newton-Raphson is autograd-compatible (all standard PyTorch ops, no `no_grad` blocks). The `torch.clamp` and `torch.minimum` non-in-place ops are differentiable.
   - What's unclear: Float32 precision through 10 NR iterations may cause numerical gradient errors in `gradcheck` (which uses float64). The existing code has `1e-12` epsilon guards for division stability.
   - Recommendation: Run `torch.autograd.gradcheck(project_fn, inputs, eps=1e-3)` with double precision inputs early in development. If it fails only due to float32/float64 mismatch, it is not a real issue (gradcheck is conservative). The fixed iteration count makes the computation graph well-defined.

2. **Head-tail flip rate without multi-start**
   - What we know: Coarse stage (K=4) with PCA-based orientation initialization breaks symmetry in most cases. Multi-camera geometry (3-4 views) provides additional disambiguation.
   - What's unclear: Whether coarse-to-fine alone achieves acceptable flip rate on real data.
   - Recommendation: Run diagnostic on 30 frames and measure arc length distribution. Flipped splines will show arc_length << nominal_length (compressed/folded). If flip rate > 5%, add the deferred multi-start as a follow-up.

3. **L-BFGS convergence speed vs Adam on this specific loss**
   - What we know: L-BFGS is theoretically superior for smooth, deterministic, low-dimensional objectives. The chamfer+regularization loss satisfies all these conditions.
   - What's unclear: Whether the Huber-aggregated per-camera chamfer has enough smoothness for strong_wolfe line search to work efficiently. Chamfer distance has non-differentiable kinks at point-switching events.
   - Recommendation: Use `line_search_fn="strong_wolfe"` (standard choice). If L-BFGS convergence is erratic due to chamfer non-smoothness, fall back to Adam with a fixed learning rate schedule. This is a low-risk concern — chamfer kinks are rare in practice when N_eval >> N_obs.

4. **GPU batch size: all 9 fish or fish-by-fish?**
   - What we know: N_fish=9, K=7, N_eval=20 gives 189 total parameters. This is a tiny problem for GPU. Batching all 9 is trivially fast.
   - What's unclear: Whether L-BFGS can efficiently handle batched loss where different fish have different camera counts.
   - Recommendation: Batch all 9 fish. Per-camera chamfer computation loops over cameras (C-level loop is fine; the bottleneck is GPU tensor ops inside).

---

## Sources

### Primary (HIGH confidence)
- `src/aquapose/calibration/projection.py` — verified autograd compatibility of `project()`, `to()` method implementation, Newton-Raphson loop structure
- `src/aquapose/reconstruction/triangulation.py` — verified `Midline3D` dataclass contract, `SPLINE_KNOTS` for K=7, `MidlineSet` type, `N_SAMPLE_POINTS=15`, `triangulate_midlines()` signature
- `src/aquapose/reconstruction/midline.py` — verified `Midline2D` structure, `points` shape (N, 2) float32 in full-frame pixels
- `pyproject.toml` — verified existing dependencies (torch, scipy, numpy all present)
- `scripts/diagnose_pipeline.py` — verified dispatch pattern for `--method` flag integration
- `.planning/inbox/curve_optimization_proposal.md` — primary design reference, all algorithmic decisions

### Secondary (MEDIUM confidence)
- PyTorch docs (training knowledge): `torch.optim.LBFGS` requires closure, `line_search_fn="strong_wolfe"` recommended, `history_size` default 100
- PyTorch docs (training knowledge): `torch.cdist` computes batched pairwise distance matrices, GPU-accelerated

### Tertiary (LOW confidence)
- B-spline knot insertion (Oslo algorithm): standard reference algorithm; approximate initialization via basis matrix evaluation is simpler and sufficient — LOW confidence on exact numerical equivalence, but HIGH confidence it works for initialization
- Fish maximum bend angle: 30° per adjacent control-point triplet is a starting estimate from general fish biomechanics literature; should be validated against observed data

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project, no new dependencies
- Architecture: HIGH — based on direct reading of existing code; drop-in contract verified
- Pitfalls: HIGH — L-BFGS closure pattern and device management from PyTorch conventions; NaN behavior verified from `project()` source
- Hyperparameters: LOW-MEDIUM — starting values are principled estimates, require empirical tuning

**Research date:** 2026-02-22
**Valid until:** Stable (no fast-moving dependencies; PyTorch LBFGS API unchanged since 2018)
