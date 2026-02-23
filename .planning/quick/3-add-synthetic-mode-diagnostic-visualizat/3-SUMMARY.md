---
phase: quick-3
plan: 01
subsystem: visualization
tags: [synthetic, diagnostics, visualization, matplotlib, opencv]
dependency_graph:
  requires:
    - src/aquapose/visualization/diagnostics.py
    - src/aquapose/visualization/plot3d.py (_robust_bounds)
    - src/aquapose/reconstruction/triangulation.py (Midline3D, MidlineSet)
    - src/aquapose/synthetic/fish.py (FishConfig)
  provides:
    - vis_synthetic_3d_comparison
    - vis_synthetic_camera_overlays
    - vis_synthetic_error_distribution
    - write_synthetic_report
  affects:
    - scripts/diagnose_pipeline.py (_run_synthetic)
tech_stack:
  added: []
  patterns:
    - Lazy scipy/torch imports inside functions (matching existing pattern)
    - TYPE_CHECKING guard for FishConfig import
    - Individual try/except per visualization call (matching _run_real pattern)
key_files:
  created: []
  modified:
    - src/aquapose/visualization/diagnostics.py
    - src/aquapose/visualization/__init__.py
    - scripts/diagnose_pipeline.py
decisions:
  - "Lazy scipy.interpolate and torch imports inside each new function — matches existing vis_per_camera_spline_overlays pattern, avoids top-level import cost"
  - "FishConfig imported under TYPE_CHECKING only — used only in write_synthetic_report signature annotation"
  - "write_synthetic_report placed after vis_funcs_syn loop (not inside it) — report needs all vis files to exist for the Diagnostic Files section"
metrics:
  duration: ~10 min
  completed: 2026-02-23
  tasks_completed: 2
  files_modified: 3
---

# Quick Task 3: Add Synthetic Mode Diagnostic Visualizations — Summary

**One-liner:** 4 synthetic diagnostic functions (3D GT comparison, per-camera overlays, error distribution, markdown report) added to diagnostics.py and wired into _run_synthetic().

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Implement 4 synthetic diagnostic functions | 19060e4 | diagnostics.py, __init__.py |
| 2 | Wire synthetic visualizations into _run_synthetic() | b3a4dd7 | diagnose_pipeline.py |

## What Was Built

### Task 1: 4 New Functions in diagnostics.py

**`vis_synthetic_3d_comparison`** — 3D matplotlib figure overlaying GT (dashed) and reconstructed (solid) B-splines for all fish found in the best frame. Uses `_robust_bounds` from plot3d.py for stable axis scaling. Annotates per-fish mean control-point error in mm.

**`vis_synthetic_camera_overlays`** — For each camera in `models`, creates a 720x1280 gray canvas, projects GT midlines as dashed green segments (alternate segments to simulate dash) and reconstructed midlines as solid colored polylines. Computes per-fish pixel reprojection residual and annotates it. Saves one PNG per camera to `synthetic_camera_overlays/`.

**`vis_synthetic_error_distribution`** — 3-panel figure: (a) histogram of all per-control-point errors in mm with mean/median lines, (b) box plot grouped by fish ID, (c) scatter of error vs control-point index (head=0, tail=6) colored by fish. Handles empty data gracefully with a warning.

**`write_synthetic_report`** — Structured markdown with: config summary table, fish_configs sub-table, per-fish GT comparison (mean/max/std error mm, arc length error mm, mean residual px), per-camera mean residual table (if per_camera_residuals available), error percentile statistics, stage timing table, diagnostic file listing from `diag_dir`.

All 4 functions exported via `__init__.py` and `__all__` in alphabetical order.

### Task 2: Integration in diagnose_pipeline.py

Updated `_run_synthetic()` to:
- Import 4 new functions alongside existing `vis_arclength_histogram`, `vis_residual_heatmap`
- Add 3 new `(name, lambda)` pairs to `vis_funcs_syn` list for the 3 visualization functions
- Add `write_synthetic_report(...)` call in its own try/except block after the vis_funcs_syn loop and GT comparison print

## Verification

- `hatch run check` passes (ruff lint: 0 errors, basedpyright: 7 pre-existing errors in unrelated files, none in modified files)
- Dry run `python scripts/diagnose_pipeline.py --synthetic --n-fish 2 --stop-frame 3` executes without exceptions
- `output/test_synth_diag/diagnostics/` contains:
  - `synthetic_3d_comparison.png` (created)
  - `synthetic_camera_overlays/` with 12 per-camera PNGs (created)
  - `synthetic_report.md` (created)
  - `synthetic_error_distribution.png` skipped gracefully (triangulation produced 0 midlines on fabricated rig — warning logged, no crash)

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `src/aquapose/visualization/diagnostics.py` exists and contains all 4 functions
- `src/aquapose/visualization/__init__.py` exports all 4 new functions
- `scripts/diagnose_pipeline.py` imports and calls all 4 new functions
- Commits 19060e4 and b3a4dd7 exist in git log
