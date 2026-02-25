# Curve-Based 3D Midline Optimization (v2)

## Motivation

Direct triangulation requires solved point-to-point correspondences across cameras before it can reconstruct 3D. Establishing those correspondences on noisy 2D skeletons (148px median epipolar error from BFS ordering) requires orientation alignment, epipolar snapping, NaN filtering, collision resolution, and threshold tuning — five interacting subsystems, three confirmed bugs, two design flaws. Curve optimization sidesteps all of this by optimizing a 3D spline directly against observed 2D skeletons via reprojection.

## Approach

**State**: A 3D cubic B-spline parameterized as a `(K, 3)` tensor of control points, where K varies during optimization (see Coarse-to-Fine Progression). The final output uses K=7, matching `Midline3D`. Optionally include a scalar scale or per-point half-widths.

**Spline evaluation**: The spline is evaluated analytically via precomputed B-spline basis matrices. For a uniform cubic B-spline with K control points, evaluation at N parameter values `t ∈ [0, 1]` is a matrix multiply `P = B @ C`, where `B` is the `(N, K)` basis matrix (precomputed once per K) and `C` is the `(K, 3)` control point tensor. This avoids iterative de Boor evaluation, is trivially differentiable, and allows N to be tuned for accuracy vs. speed (N=15–20 is sufficient).

**Loss**: For each observing camera, reproject the spline into the image using `RefractiveProjectionModel.project()` (already torch-based), then compute chamfer distance to the observed 2D skeleton points. Chamfer distance = for each reprojected point, find the nearest observed skeleton point, and vice versa. This is correspondence-free — the optimizer discovers which 3D spline point maps to which 2D observation implicitly.

Per-camera chamfer distances are aggregated using a Huber (smooth-L1) loss rather than a simple sum. This downweights outlier cameras — e.g., a camera where the cross-view identity step mis-associated a skeleton — without fully discarding them. If a camera's chamfer distance exceeds a threshold, it contributes linearly rather than quadratically to the total loss, preventing a single bad view from distorting the reconstruction.

```
L_data    = sum over cameras of huber(chamfer_distance(reprojected_spline, observed_skeleton))
L_length  = length_penalty(spline, expected_length)
L_curv    = curvature_penalty(control_points, max_bend_angle)
L_smooth  = smoothness_penalty(control_points)

L_total   = L_data + λ_len * L_length + λ_curv * L_curv + λ_smooth * L_smooth
```

### Regularization

**Length constraint (global prior → per-identity refinement)**: Fish body length is approximately constant across frames. The length penalty has two tiers:

1. *Global species prior*: A hard-ish penalty outside ±30% of a nominal species length (e.g. 45mm). This is always active and prevents degenerate collapsed or elongated splines.
2. *Per-identity soft prior*: Once a fish identity has been successfully reconstructed in ≥5 frames, the median arc length of its own past reconstructions replaces the species prior center. This self-refining estimate tightens automatically over time. The penalty is Gaussian: `(arc_length - expected)² / σ²`, where σ narrows from the species tolerance to a per-identity tolerance as data accumulates.

On the first frame or for newly appearing fish, only the global prior is available. The per-identity prior is stored in a lightweight lookup keyed by the cross-view identity label.

**Curvature limits**: Real fish have a maximum bending angle between adjacent body segments. For each consecutive triplet of control points `(C_{i-1}, C_i, C_{i+1})`, compute the angle at `C_i` and penalize angles below a species-appropriate minimum (i.e., high curvature). This acts as a soft joint-angle limit. The penalty is zero when curvature is within the biological range and ramps up quadratically beyond it. This directly prevents the head-tail fold failure mode and physically implausible contortions.

**Smoothness penalty**: Curvature regularization on control points (second-difference penalty). This is a standard spline smoothness term and prevents high-frequency oscillations.

### Coarse-to-Fine Progression

Optimization proceeds in two stages with increasing control-point resolution:

1. **Coarse stage (K=3–4 control points, ~30% of total iterations)**: The spline has very few degrees of freedom, producing a smooth loss landscape with fewer local minima. The optimizer finds the approximate position, orientation, and gross body curvature. This stage resolves head-tail ambiguity in most cases because the coarse landscape is nearly convex.

2. **Fine stage (K=7 control points, remaining iterations)**: Control points are upsampled via B-spline knot insertion (an exact operation — the curve shape is preserved). The optimizer refines local body shape with the full degrees of freedom, starting from the coarse solution.

This decomposition means the fine-stage optimizer starts in the basin of attraction of the correct solution, dramatically reducing local minima risk.

### Initialization

**Frame 1 or new fish**: Use the RANSAC centroid seed from cross-view identity (already computed in Phase 5). Initialize a straight-line spline of nominal species length centered at the seed, oriented along the principal axis of the 2D skeleton in the reference camera.

**Subsequent frames (warm-start)**: Initialize from the previous frame's optimized spline. If velocity estimates are available from the tracker, apply a linear extrapolation to the control points before optimization. This warm-start dramatically reduces the number of iterations needed for convergence (typically 50–100 vs. 200–500 from cold start), since fish motion between frames is small relative to body length.

If the warm-started optimization converges to a loss significantly higher than the previous frame's final loss, fall back to a cold-start initialization for that fish to escape potential tracking drift.

### Optimizer & Convergence

**Optimizer**: L-BFGS on the control point positions. The problem is smooth, low-dimensional (~21 parameters per fish at K=7), and deterministic, making L-BFGS a better fit than Adam. All fish are batched into a single `(N_fish, K, 3)` tensor and optimized in parallel on GPU.

**Adaptive early stopping**: Not all fish require the same number of iterations. Within the batch, per-fish loss deltas are monitored. When a fish's loss delta falls below a threshold for several consecutive steps, it is masked out of gradient updates. Its control points are frozen, and it no longer contributes to computation. This typically halves the effective iteration count across the batch, since fish visible in 3–4 cameras with good initialization converge fast while occluded or ambiguous fish need more steps.

Final reprojection residual is reported as `Midline3D.mean_residual`. Per-camera residuals are also stored for downstream quality assessment — a single camera with disproportionately high residual may indicate a cross-view identity mis-association.

## What it replaces

The entire interior of `triangulate_midlines()`:
- `_align_midline_orientations` — eliminated (orientation resolved implicitly by optimizer, especially in coarse stage)
- `_refine_correspondences_epipolar` — eliminated (no correspondences needed)
- `_triangulate_body_point` — eliminated (no point-wise triangulation)
- `_fit_spline` — eliminated (spline IS the optimization variable)
- `_pixel_half_width_to_metres` — keep (half-width conversion still needed post-optimization)

The public API stays the same: `midline_set` in, `dict[int, Midline3D]` out.

## What it needs

1. **Differentiable refractive projection**: `RefractiveProjectionModel.project()` already uses torch tensors. Verify gradients flow through Snell's law computation. If the projection uses an iterative Snell's law solver (e.g., Newton's method), the converged solution likely supports implicit differentiation — the solution satisfies F(x)=0, so gradients can be computed via the implicit function theorem without differentiating through the iterations. Verify with `torch.autograd.gradcheck`; if it fails, implement a `torch.autograd.Function` with a custom backward pass using the implicit derivative.

2. **Chamfer distance**: ~10 lines of torch code. For each reprojected point, find min distance to any observed skeleton point, and vice versa. `torch.cdist` does the heavy lifting. Wrap in Huber loss for robust per-camera aggregation.

3. **B-spline basis matrices**: Precompute the `(N, K)` basis matrix for each K used (K=3, 4, 7). These are constant and cached. Spline evaluation becomes a single batched matrix multiply.

4. **Knot insertion for coarse-to-fine**: Standard B-spline knot insertion algorithm. Given K_coarse control points, produce K_fine control points that define the same curve. This is an exact linear map (a matrix multiply on the control points), readily available in scipy or implementable in ~20 lines.

5. **Initialization from centroid seeds + warm-start logic**: Straight-line spline at seed position for cold start. Previous-frame copy with optional velocity extrapolation for warm start. Fallback logic when warm-start loss is anomalously high.

6. **Per-identity length tracker**: A dictionary mapping fish identity → running list of arc lengths. Updated after each successful reconstruction. Provides the per-identity length prior center and tolerance.

7. **Convergence criteria**: Per-fish loss delta monitoring with masking for early stopping. Global stopping when all fish have converged or max iterations reached.

## What it doesn't solve

- **6 dark cameras**: Not a bug. In the 1-second data sample we often use for debugging, the fish do not enter the FOV of these cameras.
- **Cross-view identity**: Still need RANSAC centroid clustering to know which 2D skeletons belong to the same fish.
- **Half-width estimation**: Still need pixel-to-metres conversion post-optimization.

## Risk

- **Local minima**: Head-tail flip is a symmetric local minimum. Mitigated by: (1) coarse-to-fine progression, which resolves orientation at the coarse stage where the landscape is smoother; (2) multi-camera geometry (3–4 views from different angles break symmetry); (3) curvature limits, which penalize the folded configurations that make flipped solutions attractive. If flip rate is still too high after these mitigations, add multi-start (both orientations in the same batch, keep lower loss) — this doubles batch size but is parallel so near-free.
- **Speed**: Iterative optimization is slower per-fish than closed-form triangulation. But current pipeline spends most of its triangulation time on epipolar refinement (~50s of ~76s for 30 frames). Key speed levers: (1) batching all fish on GPU; (2) warm-start from previous frame reduces iterations ~2–4×; (3) adaptive early stopping halves effective iteration count; (4) analytical B-spline evaluation avoids per-point iterative de Boor. Net expectation: comparable or faster than current pipeline.
- **Differentiability**: If `RefractiveProjectionModel.project()` has non-differentiable steps, use implicit differentiation at the converged Snell's law solution (preferred) or numerical gradients as a fallback.

## Files to modify

- `src/aquapose/reconstruction/triangulation.py` — replace internals of `triangulate_midlines()`, keep public API
- Possibly `src/aquapose/calibration/projection.py` — ensure `project()` supports autograd

## Files to keep unchanged

- `src/aquapose/reconstruction/midline.py` — 2D midline extraction stays the same
- `src/aquapose/tracking/` — cross-view identity stays the same
- `scripts/diagnose_pipeline.py` — calls `triangulate_midlines()`, works with new implementation
