# Curve Optimizer — Detailed Technical Overview

> Source of truth: `src/aquapose/reconstruction/curve_optimizer.py` and its
> imports from `triangulation.py`, `projection.py`, and `midline.py`.

## 1. High-Level Summary

The curve optimizer (`CurveOptimizer`) replaces point-wise triangulation with
direct optimization of 3D B-spline control points against 2D skeleton
observations. Instead of establishing per-body-point correspondences across
cameras (orientation alignment, epipolar snapping, etc.), it minimizes a
chamfer distance between the reprojected 3D spline and observed 2D skeletons.
This is **correspondence-free**: the optimizer implicitly discovers which 3D
spline point maps to which 2D observation.

All fish in a frame are batched into a single `(N_fish, K, 3)` tensor and
optimized in parallel on GPU via L-BFGS.

---

## 2. Entry Point & Invocation

### From `diagnose_pipeline.py`

```python
# scripts/diagnose_pipeline.py lines 317-330
if args.method == "curve":
    from aquapose.reconstruction.curve_optimizer import (
        CurveOptimizer,
        CurveOptimizerConfig,
    )
    optimizer = CurveOptimizer(config=CurveOptimizerConfig(max_depth=2.0))
    for frame_idx, midline_set in enumerate(midline_sets):
        results = optimizer.optimize_midlines(
            midline_set, models, frame_index=frame_idx
        )
        midlines_3d.append(results)
```

Invoked via CLI: `python scripts/diagnose_pipeline.py --method curve`

The `CurveOptimizer` instance persists across frames to maintain warm-start
state (`_warm_starts`, `_prev_losses`).

### Convenience wrapper

`optimize_midlines()` (module-level function, line 1147) creates a fresh
`CurveOptimizer` and runs a single frame. Only suitable for one-off use;
multi-frame pipelines should use the class directly.

---

## 3. Input / Output Trace

### Input

| Name | Type | Source | Description |
|------|------|--------|-------------|
| `midline_set` | `MidlineSet` = `dict[int, dict[str, Midline2D]]` | Stage 4 (midline extraction) | fish_id → camera_id → `Midline2D` (15 points + half-widths in full-frame pixels) |
| `models` | `dict[str, RefractiveProjectionModel]` | Calibration loading | camera_id → refractive projection model (K, R, t, water_z, Snell's law) |
| `frame_index` | `int` | Loop counter | Embedded in output for identification |
| `fish_centroids` | `dict[int, torch.Tensor] \| None` | Optional | Per-fish 3D centroids for cold-start; if None, estimated from 2D obs |

### Internal Flow

```
midline_set (2D observations)
  │
  ├─ Filter NaN rows, move to device → midlines_per_fish[i][cam_id] = (N, 2) tensor
  │
  ├─ Estimate orientation per fish (PCA of best-camera skeleton) → ref_orientations
  │
  ├─ Estimate centroids (ray casting at 0.5m depth if not provided) → centroids
  │
  ├─ Triangulation-seeded cold start:
  │   ├─ Run triangulate_midlines() on CPU for fish lacking warm-starts
  │   ├─ Validate: majority of ctrl pts must be below water surface
  │   └─ Accepted seeds stored in _warm_starts as "tri-seed"
  │
  ├─ STAGE 1 — Coarse optimization (K=4):
  │   ├─ _init_ctrl_pts() → (N_fish, 4, 3), warm-start or cold-start
  │   ├─ L-BFGS.step(closure) with max_iter_coarse iterations
  │   └─ Loss = data + λ_length * length + λ_curv * curvature + λ_smooth * smoothness
  │
  ├─ STAGE 2 — Fine optimization (K=7):
  │   ├─ _upsample_ctrl_pts(coarse → fine) via basis matrix
  │   ├─ Warm-start fallback check (if loss > ratio * prev_loss → cold start)
  │   ├─ Manual L-BFGS loop (max_iter=1 per step) for adaptive early stopping
  │   └─ Per-fish convergence: freeze when |Δloss| < delta for patience steps
  │
  └─ OUTPUT ASSEMBLY:
      ├─ scipy BSpline from final ctrl pts
      ├─ Arc length via 1000-pt numerical integration
      ├─ Reprojection residuals (per-camera and aggregate)
      ├─ Half-widths via _pixel_half_width_to_metres()
      ├─ Confidence flag (n_cameras < 2 or mean_residual > 50.0)
      └─ Store warm-start + prev_loss for next frame
```

### Output

| Name | Type | Description |
|------|------|-------------|
| `results` | `dict[int, Midline3D]` | fish_id → 3D midline with 7 control points, knots, arc length, half-widths, residuals |

The `Midline3D` struct contains:
- `control_points`: shape (7, 3), float32 — B-spline control points
- `knots`: `[0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]` — clamped uniform
- `degree`: 3 (cubic)
- `arc_length`: total spline length in metres (1000-pt integration)
- `half_widths`: shape (15,), float32, world metres — uniform value from mean across cameras
- `n_cameras`: number of camera observations for this fish
- `mean_residual` / `max_residual`: reprojection error in pixels
- `is_low_confidence`: `True` when n_cameras < 2 or mean_residual > 50.0
- `per_camera_residuals`: dict of camera_id → mean pixel error

---

## 4. Loss Function Breakdown

### Total loss (closure, lines 821-843 / 927-945)

```
L_total = L_data + λ_length * L_length + λ_curv * L_curvature + λ_smooth * L_smoothness
```

### 4.1 Data loss: `_data_loss()` (line 227)

For each fish:
1. Evaluate spline: `P = B @ C` where B is `(n_eval, K)`, C is `(K, 3)` → `(n_eval, 3)`
2. For each camera with observations:
   - Project spline pts via `model.project(spline_pts)` → `(n_eval, 2)` pixels + validity mask
   - Filter NaN/invalid projections
   - Compute symmetric chamfer distance to observed 2D skeleton
3. Average chamfer distances across cameras → per-fish loss
4. Average across fish → scalar loss

**Depth penalty fallback** (line 300-318): If a fish has NO valid projections
in ANY camera (all points above water), the loss becomes `mean(clamp(-h_q, 0)) * 100.0`,
which pushes the spline back underwater via gradients.

### 4.2 Length penalty: `_length_penalty()` (line 327)

- Evaluates spline at n_eval points, sums segment lengths → arc_length per fish
- Tolerance band: `[nominal * (1-tol), nominal * (1+tol)]`
- Penalty = 0 inside band; quadratic outside: `(deviation²).mean()`

### 4.3 Curvature penalty: `_curvature_penalty()` (line 366)

- For each consecutive control-point triplet `(C_{i-1}, C_i, C_{i+1})`:
  - Compute bend angle via `acos(dot(v1_unit, v2_unit))`
  - 0° = straight, 180° = U-turn
- Penalty = `clamp(angle - max_angle, min=0)²` averaged over all joints/fish
- `cos_bend` clamped to `[-1+ε, 1-ε]` to keep acos gradient finite

### 4.4 Smoothness penalty: `_smoothness_penalty()` (line 419)

- Second-difference: `C_{i+1} - 2*C_i + C_{i-1}` for each fish
- Penalty = squared magnitude, summed over 3D, averaged over joints/fish

---

## 5. Tunable Parameters

All parameters are exposed in `CurveOptimizerConfig` (line 130).

| Parameter | Current Value | Type | Description |
|-----------|--------------|------|-------------|
| `nominal_length_m` | `0.085` | `float` | Nominal fish body length in metres (midpoint of 70-100mm species range) |
| `length_tolerance` | `0.30` | `float` | Fractional tolerance on nominal length (±30%); penalty is zero within `[0.0595, 0.1105]` m |
| `lambda_length` | `10.0` | `float` | Weight for arc-length penalty relative to data loss |
| `lambda_curvature` | `5.0` | `float` | Weight for bend-angle penalty relative to data loss |
| `lambda_smoothness` | `1.0` | `float` | Weight for second-difference smoothness penalty |
| `max_bend_angle_deg` | `30.0` | `float` | Maximum bend angle (degrees) per control-point triplet; from fish biomechanics literature |
| `n_coarse_ctrl` | `4` | `int` | Number of B-spline control points in coarse stage |
| `n_fine_ctrl` | `7` | `int` | Number of control points in fine stage (matches `Midline3D` contract: `SPLINE_N_CTRL=7`) |
| `n_eval_points` | `20` | `int` | Number of spline evaluation points for loss computation |
| `lbfgs_lr` | `1.0` | `float` | L-BFGS learning rate / step size |
| `lbfgs_max_iter_coarse` | `20` | `int` | Max L-BFGS iterations for coarse stage (passed directly to `torch.optim.LBFGS`) |
| `lbfgs_max_iter_fine` | `40` | `int` | Max outer loop iterations for fine stage (one L-BFGS step per iteration) |
| `lbfgs_history_size` | `10` | `int` | L-BFGS Hessian approximation history window |
| `convergence_delta` | `0.5` | `float` | Per-fish absolute loss delta threshold (pixels) for convergence detection |
| `convergence_patience` | `3` | `int` | Consecutive sub-delta steps before a fish is frozen |
| `warm_start_loss_ratio` | `2.0` | `float` | If warm-start loss > ratio × prev_loss, revert to cold start |
| `max_depth` | `None` | `float \| None` | Max depth below water (m) for triangulation seed validation; set to `2.0` in diagnose_pipeline |

### Constants not in config but potentially tunable

| Constant | Value | Location | Description |
|----------|-------|----------|-------------|
| `_MAX_BEND_ANGLE_DEG_DEFAULT` | `30.0` | curve_optimizer.py:42 | Module-level default for `max_bend_angle_deg` |
| `_N_COARSE_CTRL` | `4` | curve_optimizer.py:45 | Module-level default for coarse control points |
| `_N_FINE_CTRL` | `SPLINE_N_CTRL` (7) | curve_optimizer.py:46 | Module-level default for fine control points |
| `N_SAMPLE_POINTS` | `15` | triangulation.py:35 | Number of body points in Midline2D / Midline3D output |
| `SPLINE_K` | `3` | triangulation.py:28 | B-spline degree (cubic) |
| `SPLINE_N_CTRL` | `7` | triangulation.py:29 | Standard control point count for Midline3D |
| `SPLINE_KNOTS` | `[0,0,0,0,0.25,0.5,0.75,1,1,1,1]` | triangulation.py:30-32 | Clamped uniform knot vector |
| Depth penalty scale | `100.0` | curve_optimizer.py:308 | Hardcoded scale factor for above-water depth penalty |
| Cold-start depth estimate | `0.5` m | curve_optimizer.py:707 | Approximate depth for centroid estimation when no 3D centroid provided |
| Low-confidence threshold | `50.0` px mean residual or `< 2` cameras | curve_optimizer.py:1092 | Criteria for `is_low_confidence` flag |
| L-BFGS line search | `"strong_wolfe"` | curve_optimizer.py:818,921 | Line search strategy (strong Wolfe conditions) |

---

## 6. Key Functions Reference

### Public API

| Function | Line | Signature | Description |
|----------|------|-----------|-------------|
| `CurveOptimizer.__init__` | 609 | `(config: CurveOptimizerConfig \| None)` | Stores config, initializes `_warm_starts` and `_prev_losses` dicts |
| `CurveOptimizer.optimize_midlines` | 614 | `(midline_set, models, frame_index, fish_centroids?) → dict[int, Midline3D]` | Main entry point; runs full coarse→fine pipeline for one frame |
| `optimize_midlines` (module) | 1147 | `(midline_set, models, frame_index, config?, fish_centroids?)` | Convenience wrapper; creates fresh optimizer (no warm-start across frames) |
| `CurveOptimizerConfig` | 130 | `@dataclass` | All hyperparameters; see table above |

### Loss functions (private)

| Function | Line | Signature | Description |
|----------|------|-----------|-------------|
| `_data_loss` | 227 | `(ctrl_pts, basis, midlines_per_fish, models, config) → Tensor` | Mean chamfer distance (pixels) over fish and cameras; depth penalty fallback |
| `_length_penalty` | 327 | `(ctrl_pts, basis, config) → Tensor` | Quadratic penalty outside nominal length tolerance band |
| `_curvature_penalty` | 366 | `(ctrl_pts, config) → Tensor` | Quadratic penalty for bend angles exceeding `max_bend_angle_deg` |
| `_smoothness_penalty` | 419 | `(ctrl_pts) → Tensor` | Second-difference squared magnitude on control points |
| `_chamfer_distance_2d` | 191 | `(proj, obs) → Tensor` | Symmetric chamfer distance between two 2D point sets via `torch.cdist` |

### Initialization helpers (private)

| Function | Line | Signature | Description |
|----------|------|-----------|-------------|
| `_cold_start` | 444 | `(centroid, orientation, K, nominal_length, device) → Tensor (K,3)` | Straight-line spline centered at centroid, oriented along direction |
| `_estimate_orientation_from_skeleton` | 485 | `(obs_pts) → Tensor (3,)` | PCA principal axis of 2D skeleton (z=0) |
| `_init_ctrl_pts` | 512 | `(fish_ids, centroids, warm_starts, K, config, orientations, device) → Tensor (N,K,3)` | Batched init: warm-start (with optional K resample) or cold-start per fish |
| `_upsample_ctrl_pts` | 567 | `(coarse_ctrl, n_coarse, n_fine, device) → Tensor (N, n_fine, 3)` | Upsample via basis matrix evaluation (shape-preserving) |

### B-spline utilities

| Function | Line | Signature | Description |
|----------|------|-----------|-------------|
| `_build_basis_matrix` | 55 | `(n_eval, n_ctrl) → Tensor (n_eval, n_ctrl)` | Construct clamped uniform cubic B-spline basis via scipy |
| `get_basis` | 108 | `(n_eval, n_ctrl, device) → Tensor` | Cached basis matrix retrieval |

---

## 7. Initialization Strategy

### Priority order for each fish

1. **Warm-start** (from `_warm_starts[fish_id]`): Previous frame's optimized
   control points. If K differs from target, upsampled via basis matrix.
   Used for temporal coherence.

2. **Triangulation seed** (label `"tri-seed"`): For fish needing cold start,
   `triangulate_midlines()` is run on CPU. Seeds are validated: a majority of
   control points must be below `water_z`. Bad seeds are rejected and the fish
   falls back to option 3.

3. **Geometric cold start**: Straight-line spline of `nominal_length_m`,
   centered at the fish centroid, oriented along PCA principal axis of the
   best camera's 2D skeleton.

### Warm-start fallback

During fine stage initialization (line 876-907): for each fish with a previous
loss record, if the current fine-stage loss exceeds
`warm_start_loss_ratio × prev_loss`, that fish's control points are replaced
with a cold-start initialization. This catches tracking drift or identity
switches.

---

## 8. Coarse-to-Fine Strategy

### Coarse stage (K=4)

- Basis: `(n_eval_points=20, 4)` matrix
- Optimizer: Single `torch.optim.LBFGS` with `max_iter=20` (full batch)
- Purpose: Find approximate position, orientation, gross curvature with
  low-dimensional smooth landscape (~12 DoF per fish)
- Frozen fish: per-fish gradient zeroing for any converged fish
- No adaptive stopping in coarse stage (single L-BFGS `.step()` call)

### Transition: coarse → fine

- `_upsample_ctrl_pts()`: evaluates coarse spline at 7 parameter positions
  via `B_up @ coarse_ctrl` — curve shape is preserved

### Fine stage (K=7)

- Basis: `(20, 7)` matrix
- Optimizer: `torch.optim.LBFGS` with `max_iter=1` per outer loop step
- Outer loop: up to `lbfgs_max_iter_fine=40` steps
- **Adaptive early stopping**: per-fish loss delta tracked; fish frozen when
  `|Δloss| < convergence_delta (0.5 px)` for `convergence_patience (3)` consecutive steps
- Frozen fish gradients zeroed to prevent drift
- Per-fish losses logged at step 0 and every 10 steps

---

## 9. Edge-Case Guardrails

### Above-water spline detection

**Depth penalty fallback** (`_data_loss`, line 300-318): When all spline points
project as invalid (above water surface), the loss switches from chamfer to a
depth penalty: `mean(clamp(water_z - z, min=0)) * 100`. This provides gradients
pushing the spline underwater and prevents silent convergence at loss=0.

### Triangulation seed validation

**Bad seed rejection** (line 746-764): Triangulated seeds where fewer than half
the control points are below `water_z` are rejected. The fish falls back to
geometric cold start. Logged as warning.

### Warm-start anomaly detection

**Loss ratio check** (line 876-907): If warm-started fine loss >
`warm_start_loss_ratio × prev_loss`, control points are replaced with cold
start for that fish. Catches identity switches / tracking drift.

### NaN filtering

- 2D observation points with NaN are filtered before creating GPU tensors
  (line 662-663)
- Projected points: NaN/invalid filtered via `valid_mask = valid & ~isnan(proj_px)`
  (line 286-287)
- Cameras with no valid projections are skipped (line 288-289)

### Empty batch handling

- No fish with valid observations → return `{}` (line 674-675)
- No camera losses for a fish → depth penalty applied (line 300)
- Empty chamfer inputs → returns zero (line 208-209)

### Numerical stability

- `acos` argument clamped to `[-1+ε, 1+ε]` with `ε=1e-6` (line 403)
  to prevent gradient blowup at collinear/antiparallel control points
- Vector norms clamped to `min=1e-8` (line 393-394) to prevent division by zero
- `torch.clamp(1 - sin_t_sq, min=0)` in projection (projection.py:156)
  prevents sqrt of negative in Snell's law

### Device management

- Models moved to GPU for optimization, to CPU for triangulation seeding,
  back to GPU after (with exception safety: lines 786-793)
- All tensors explicitly moved to `device` during initialization

---

## 10. Differences from the Original Proposal

The implementation (`curve_optimizer.py`) differs from the original proposal
(`.planning/inbox/curve_optimization_proposal.md`) in several ways:

| Aspect | Proposal | Implementation |
|--------|----------|----------------|
| Per-camera aggregation | Huber (smooth-L1) loss | Simple mean of chamfer distances |
| Per-identity length prior | Self-refining per-fish median length | Global species prior only (`nominal_length_m ± tolerance`) |
| Velocity extrapolation for warm-start | Linear extrapolation of control points | Direct copy (no velocity model) |
| Multi-start for head-tail ambiguity | Both orientations in batch, keep lower loss | Not implemented; relies on coarse stage + warm-start |
| Coarse K | K=3-4 | K=4 fixed |

---

## 11. Dependencies

### From `triangulation.py` (imports at line 23-32)

- `N_SAMPLE_POINTS` (15) — output point count
- `SPLINE_K` (3) — B-spline degree
- `SPLINE_KNOTS` — knot vector for Midline3D
- `SPLINE_N_CTRL` (7) — fine stage control point count
- `Midline3D` — output data structure
- `MidlineSet` — input type alias
- `_pixel_half_width_to_metres` — half-width conversion
- `triangulate_midlines` — used for triangulation-seeded cold start

### From `projection.py`

- `RefractiveProjectionModel` — `.project()` for data loss, `.cast_ray()` for centroid estimation
- `.project()` returns `(pixels, valid)` tensors; 10-iteration Newton-Raphson, autograd-compatible
- `.cast_ray()` returns `(origins, directions)` with Snell's law refraction

### From `midline.py`

- `Midline2D` — input 2D skeleton data (points + half_widths)

### External

- `torch` — tensors, autograd, L-BFGS optimizer
- `scipy.interpolate.BSpline` — final arc-length computation and basis construction
- `numpy` — array operations in output assembly
