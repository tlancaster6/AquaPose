---
phase: quick-5
plan: "01"
subsystem: reconstruction/curve_optimizer
tags: [investigation, regularization, spline, folding, B-spline]
dependency_graph:
  requires: []
  provides: [5-REPORT.md]
  affects: [curve_optimizer.py, CurveOptimizerConfig]
tech_stack:
  added: []
  patterns: [diagnostic-script, numerical-analysis]
key_files:
  created:
    - .planning/quick/5-improve-3d-spline-constraints-investigat/5-REPORT.md
  modified: []
decisions:
  - "Root cause: K=7 with max_bend=30 allows 150-degree total fold with zero penalty (each of 5 angles = 30 deg = exactly at threshold)"
  - "B-spline smoothing masks control-point folds: 180-degree CP fold -> 2.4-degree evaluated curve angle"
  - "lambda_curvature=5 gives curvature gradients only 2% of data loss gradients for K=7"
  - "Chord-arc ratio penalty is recommended as the highest-impact, lowest-effort fix"
metrics:
  duration: "20 min"
  completed: "2026-02-23"
  tasks_completed: 1
  files_created: 1
---

# Phase quick-5: Spline Folding Investigation Summary

**One-liner:** Numerical analysis proving K=7 spline allows 150-degree total fold with zero curvature penalty due to threshold distribution across 5 control points, B-spline smoothing masking, and lambda values 50x too small.

## What Was Done

Ran three diagnostic scripts against `curve_optimizer.py` to quantify why splines fold despite regularization:

1. **Penalty table analysis** (K=4 and K=7 at 0/30/60/90/120/150/180 degree folds): Showed that K=7 with `max_bend_angle_deg=30` produces exactly zero curvature penalty at 150-degree fold (5 interior angles x 30 deg/angle = 150 total = all at threshold).

2. **B-spline smoothing analysis**: Measured max bend angle on the evaluated curve (100 dense points). Found that a 180-degree control-point fold produces only 2.4–2.5 degrees max angle on the actual rendered curve. The curvature penalty measures control-point geometry, not curve geometry.

3. **Gradient comparison**: Estimated data loss gradient at ~1400 px/m (focal=1400, depth=0.5m) and measured curvature penalty gradient norms. For K=7 at 180-degree fold, curvature gradient norm is 29.6 — only **2.1% of data loss gradient**. The optimizer trivially absorbs the penalty to fit poor observations.

4. **Chord-arc ratio analysis**: Computed (1 - chord/arc)^2 for each fold angle. This penalty grows continuously from 0 (straight) to 0.25 (U-turn) and cannot be evaded by distributing angles below a threshold.

## Key Findings

- A 150-degree total fold in K=7 has **exactly zero curvature penalty** — the threshold is met but not exceeded at any individual control point
- B-spline smoothing means the curvature penalty penalizes a property not observable in the rendered curve
- `lambda_curvature=5` needs to be ~47-150 to compete with data loss gradients
- Smoothness penalty (second differences) is blind to smooth U-turns (small second differences, large total turning)
- The chord-arc ratio penalty (Rec 1) and dense-curve curvature (Rec 2) address these blindspots

## Recommendations Delivered

Six recommendations ranked by impact/effort:
1. **Add chord-arc penalty** (lambda=50) — highest impact, 1-2 hours
2. **Increase lambda_curvature from 5 to 50** — 5 minutes, 10x more resistance
3. **Evaluate curvature on dense curve** (reduce max_bend to 5 deg) — removes basis-smoothing blindspot
4. **Lower coarse max_bend to 15 deg** — 5 minutes, prevents coarse-stage folding
5. **Add monotonicity penalty** on tangent direction — directly prevents reversals
6. **Add total turning angle constraint** — complement to chord-arc (Rec 1 preferred)

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- Report file exists: `.planning/quick/5-improve-3d-spline-constraints-investigat/5-REPORT.md` — 357 lines (exceeds 50-line minimum)
- Numerical evidence tables included for both K=4 and K=7
- At least 3 concrete recommendations with impact/effort ranking (6 delivered)
- Commit exists: cedbbe1
