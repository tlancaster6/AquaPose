# Quick Task 1: Fix Triangulation Bugs

## Description
Fix triangulation bugs identified in `.planning/inbox/triangulation_debugging.md`: NaN contamination, coupled thresholds, and greedy orientation alignment.

## Tasks

### Task 1: Fix NaN contamination in residual computation
- **files**: `src/aquapose/reconstruction/triangulation.py`
- **action**: Add NaN check for `obs_pts[j]` in residual loop
- **verify**: Run `diagnose_pipeline.py` — residuals should be real numbers (~13px)
- **done**: Commit `cd9b78b`

### Task 2: Decouple snap_threshold and inlier_threshold
- **files**: `src/aquapose/reconstruction/triangulation.py`
- **action**: Add separate `snap_threshold=20.0` parameter to `triangulate_midlines()`
- **verify**: Run `diagnose_pipeline.py` — mean residual should drop
- **done**: Commit `c1cd279`

### Task 3: Replace greedy orientation with brute-force
- **files**: `src/aquapose/reconstruction/triangulation.py`
- **action**: Replace `_align_midline_orientations()` with brute-force enumeration
- **verify**: Run `diagnose_pipeline.py` — mean residual should improve
- **done**: REVERTED — both scoring approaches regressed to ~143px
