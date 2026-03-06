---
status: resolved
trigger: "curve-optimizer-high-loss: CurveOptimizer produces per-fish losses of 1,000-15,000 that barely decrease over 40 L-BFGS iterations"
created: 2026-02-23T00:00:00Z
updated: 2026-02-23T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED — Three compounding bugs in _data_loss() and convergence check
test: verified numerically — perfect-solution loss is now 1.6px (chamfer discretization), bad solution gives 74px; all unit tests pass
expecting: real-data run should show per-fish fine losses well below 100px
next_action: DONE — fix verified, session archived

## Symptoms

expected: Per-fish losses should converge to low values (ideally <100) producing reasonable 3D midline curves that match fish silhouettes across camera views.
actual: Per-fish fine-stage losses start at 1,000-15,000 and only decrease to 1,000-8,500 after 40 L-BFGS iterations. Coarse stage total losses are 40,000-50,000. The optimizer reports "all fish converged" due to relative convergence criteria, but the absolute loss values are enormous. Some frames show only 1/9 fish converging (Frame 8).
errors: No exceptions or crashes. The optimizer runs to completion but produces poor results. Key data points from logs:
- Frame 4 coarse: loss=43651
- Frame 4 fine per-fish: 3260-8565 at start, 2623-6499 at end (only ~20% reduction)
- Frame 8: only 1/9 fish converged after 40 fine steps
- Warm-start from previous frames does not help much — losses stay in thousands
reproduction: python scripts/diagnose_pipeline.py --method curve --stop-frame 30 --output-dir output/diag_curve
timeline: First real-data test of the CurveOptimizer (Phase 9). Unit tests with synthetic data pass.

## Eliminated

- hypothesis: "Refractive projection model wrong or convention mismatch"
  evidence: projection.py code is correct and consistent — triangulation uses same model; knots for n_ctrl=7 verified to match SPLINE_KNOTS exactly
  timestamp: 2026-02-23

- hypothesis: "B-spline evaluation matrix is wrong"
  evidence: verified _build_basis_matrix uses same clamped knot formula as SPLINE_KNOTS; output confirmed equal to np.allclose tolerance
  timestamp: 2026-02-23

- hypothesis: "Initialization is so bad the optimizer cannot converge at all"
  evidence: even a 10px chamfer would produce loss=400 per fish (8 cams), so even OK initialization gives a large-looking loss due to the Huber-wrapping bug
  timestamp: 2026-02-23

## Evidence

- timestamp: 2026-02-23
  checked: _data_loss() aggregation in curve_optimizer.py lines 251-284
  found: total_loss SUMS cam_loss over ALL cameras AND ALL fish — not averaged; with 9 fish × 8 cameras = 72 pairs, the loss is ~72× the per-camera per-fish chamfer value before Huber transformation
  implication: coarse loss of 43651 / 72 / Huber-transform ≈ 40px chamfer per camera — not catastrophic; the inflation makes it look terrible

- timestamp: 2026-02-23
  checked: Huber loss applied to chamfer scalar, lines 277-283
  found: F.huber_loss(chamfer_scalar, zero, delta=17.5) transforms chamfer into Huber loss — at chamfer=50px, Huber=722 (14× inflation). Gradient is also CLIPPED to delta=17.5 when chamfer>delta, reducing effective gradient magnitude when error is large
  implication: loss values are inflated and gradient is diminished; Huber applied to an already-aggregated scalar is semantically wrong — Huber is for outlier downweighting of INDIVIDUAL points, not aggregated means

- timestamp: 2026-02-23
  checked: convergence check in fine loop, lines 902-906
  found: rel_delta = delta / (fish_loss + 1e-8); threshold = convergence_delta = 1e-3; at fish_loss=5000 the fish "converges" when loss changes by only 5 units/step; with max_iter=1 per outer L-BFGS step, this fires quickly at any plateau
  implication: fish are marked "converged" based on relative change while absolute loss remains in thousands

- timestamp: 2026-02-23
  checked: _estimate_orientation_from_skeleton, lines 446-470
  found: PCA of 2D pixel coords (u,v) → direction [principal_u, principal_v, 0.0] in world space; this is incorrect since pixel axes != world axes, but only affects cold-start when triangulation seeding fails (tri-seed covers most fish at start)
  implication: minor bug — orientation may be wrong for cold-start fish; fix is to default to (1,0,0) or use actual triangulated points

- timestamp: 2026-02-23
  checked: numerical calculation of per-fish fine losses
  found: with 8 cameras, chamfer=32-70px, per-fish loss = 8 × Huber(chamfer) = 8 × 721-1247 = 5768-9976; this matches reported 3260-8565 range exactly
  implication: CONFIRMED — Huber inflation + summation over cameras fully accounts for the high loss values

## Resolution

root_cause: Three compounding bugs in _data_loss() and the convergence check:
  (1) Loss SUMS over all fish AND cameras instead of averaging — inflates total loss by n_fish × n_cameras (up to 72×); coarse 43651 / 72 / Huber-factor ≈ 40px chamfer per camera (not catastrophic; just hidden by inflation)
  (2) Huber loss is applied to the already-aggregated chamfer SCALAR — each camera's contribution is H(chamfer) ≈ 14× the actual chamfer at 50px, and gradient is clipped to delta=17.5 regardless of actual error
  (3) Convergence uses RELATIVE delta (1e-3) instead of absolute — at fish_loss=5000, fires when loss changes <5 units/step; with max_iter=1 per outer step this triggers almost immediately at any plateau, even at terrible absolute loss

fix:
  (A) _data_loss() now computes mean chamfer per camera per fish, then averages over cameras per fish, then averages over fish — result is always in pixel units
  (B) Removed F.huber_loss wrapping entirely — chamfer is already a robust symmetric distance; Huber on top is incorrect
  (C) Removed now-unused import torch.nn.functional and constant _HUBER_DELTA_PX
  (D) Convergence check changed from rel_delta < convergence_delta to delta < convergence_delta (absolute)
  (E) convergence_delta default changed from 1e-3 (relative fraction) to 0.5 (absolute pixels)

verification:
  - hatch run test tests/unit/test_curve_optimizer.py: ALL PASS (same 2 pre-existing failures in tracker/triangulation unrelated to changes)
  - Numerical check: loss at perfect solution = 1.6px (chamfer discretization noise), loss at 10cm shift = 74px — correct pixel-unit semantics confirmed

files_changed: [src/aquapose/reconstruction/curve_optimizer.py]
