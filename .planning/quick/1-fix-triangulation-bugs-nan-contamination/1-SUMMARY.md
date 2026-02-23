# Quick Task 1 Summary: Fix Triangulation Bugs

## Outcome
2 of 3 fixes applied successfully. Fix 3 (brute-force orientation) reverted after regression.

## Results

| Metric | Baseline | Fix 1 (NaN) | Fix 2 (Decouple) | Fix 3 (Reverted) |
|--------|----------|-------------|------------------|------------------|
| Mean residual | NaN | 13.67 px | **8.65 px** | 143 px (reverted) |
| Max residual | NaN | 696 px | 1827 px | - |
| Low-confidence | 82.0% | 81.3% | 80.9% | - |
| Frames with 3D | 26/30 | 26/30 | 26/30 | - |

## Commits
- `cd9b78b` — fix: add NaN check for observed points in residual computation
- `c1cd279` — fix: decouple snap_threshold from inlier_threshold in triangulation
- `7450d50` — fix: replace greedy orientation with brute-force (chord scoring) — REGRESSED
- `489d7a1` — fix: use multi-camera reprojection residual for orientation scoring — REGRESSED
- `4d3d07e` — revert: restore greedy orientation alignment

## Diagnostic Runs
- `output/fix0_baseline/` — pre-fix baseline (all NaN)
- `output/fix1_nan_check/` — after NaN fix (13.67px)
- `output/fix2_decouple_thresholds/` — after threshold decoupling (8.65px)
- `output/fix3_bruteforce_orient/` — chord-length brute-force (143px, reverted)
- `output/fix3b_ransac_orient/` — RANSAC brute-force (142px, reverted)

## Open Items
- Fix 3 needs a different approach: orientation runs before epipolar refinement on noisy raw correspondences (~148px median epipolar distance). Consider moving orientation after epipolar refinement, or a different scoring approach.
- `inlier_threshold` could be raised to 150-200px for better coverage (not done, user chose to defer).
- Some fish arc lengths still NaN — likely separate propagation path in spline fitting.
