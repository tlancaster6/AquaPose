---
phase: quick-16
plan: 01
subsystem: scripts/association-tuning
tags: [association, tuning, sweep, grid-search]
dependency_graph:
  requires: []
  provides: [scripts/tune_association.py restructured with joint grid sweep]
  affects: [association parameter tuning workflow]
tech_stack:
  added: []
  patterns: [joint 2D grid sweep, carry-forward parameter locking]
key_files:
  created: []
  modified:
    - scripts/tune_association.py
decisions:
  - ray_distance_threshold and score_min swept jointly (7x8=56 combos) because they interact via soft scoring kernel
  - Best joint pair locked into carry_forward before eviction_reproj_threshold sequential sweep
  - Joint grid results included as top-N candidates to avoid filtering out non-winner combos
metrics:
  duration: ~5 minutes
  completed: "2026-03-03"
  tasks_completed: 1
  tasks_total: 1
  files_changed: 1
---

# Quick Task 16: Restructure tune_association.py with Joint Grid Sweep Summary

One-liner: Joint 2D grid sweep for ray_distance_threshold x score_min (7x8=56 combos) with widened overnight ranges replacing sequential carry-forward for primary parameters.

## What Was Built

Restructured `scripts/tune_association.py` to use a 2D joint grid sweep for the two primary parameters that interact through the soft scoring kernel, then carry the winning pair into sequential sweeps for remaining parameters.

### Key Changes

**`SWEEP_RANGES` (widened for overnight run):**
- `ray_distance_threshold`: [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15] (7 values)
- `score_min`: [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30] (8 values)
- `eviction_reproj_threshold`: [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10] (7 values)

**`SECONDARY_RANGES` (widened):**
- `leiden_resolution`: [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
- `early_k`: [5, 10, 15, 20, 25, 30]

**Structure replaced:**
- `PRIMARY_STAGES: list[str]` removed
- `JOINT_PARAMS: tuple[str, str] = ("ray_distance_threshold", "score_min")` added
- `SEQUENTIAL_PRIMARY: list[str] = ["eviction_reproj_threshold"]` added

**New functions:**
- `_run_joint_grid_sweep(param_a, values_a, param_b, values_b, ...)` — runs all 56 combinations, prints per-combo progress, sorts by score, returns best overrides dict and full results list
- `_print_joint_grid_matrix(param_a, values_a, param_b, values_b, results)` — prints 2D yield% matrix with rows=ray_distance_threshold, cols=score_min, best cell marked with asterisk

**Updated `main()` flow:**
1. Stage 1: Joint grid sweep (56 combos), print 2D matrix, lock best pair into carry_forward
2. Stage 2: Sequential sweep of eviction_reproj_threshold with carry_forward
3. Stages 3+: Secondary parameters (leiden_resolution, early_k) if target not reached
4. Top-N collection includes all joint grid combos as candidates (not just the winner)

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

Files exist:
- `scripts/tune_association.py` — FOUND

Functions present:
- `_run_joint_grid_sweep` — FOUND (line 386)
- `_print_joint_grid_matrix` — FOUND (line 496)
- `JOINT_PARAMS` — FOUND (line 46)
- `SEQUENTIAL_PRIMARY` — FOUND (line 49)
- `PRIMARY_STAGES` — REMOVED (confirmed absent)

Commits:
- `4905671` — feat(quick-16): restructure tune_association.py with joint grid sweep

Verification:
- Syntax check: PASSED
- `hatch run lint`: PASSED (all checks passed)
- `hatch run check`: lint PASSED, typecheck errors pre-existing (none in tune_association.py)

## Self-Check: PASSED
