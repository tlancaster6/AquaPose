---
phase: 96-z-denoising-and-documentation
plan: "01"
subsystem: reconstruction
tags: [z-denoising, cli, docstrings, raw-keypoint, temporal-smoothing]
dependency_graph:
  requires: []
  provides: [z-denoising-raw-keypoint-fix, keypoint-native-docstrings]
  affects: [cli, reconstruction-types, temporal-smoothing]
tech_stack:
  added: []
  patterns: [nan-safe-shift, tdd]
key_files:
  created:
    - tests/unit/core/test_temporal_smoothing.py
  modified:
    - src/aquapose/cli.py
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/core/types/reconstruction.py
    - src/aquapose/core/types/midline.py
    - src/aquapose/core/reconstruction/temporal_smoothing.py
decisions:
  - "NaN-safe dual shift: both control_points and points shifted by dz; NaN + dz == NaN keeps unused dataset unchanged"
  - "Backward compat: points dataset read returns None for legacy HDF5 files; shift is skipped when None"
metrics:
  duration: "~4 min"
  completed: "2026-03-13"
  tasks_completed: 2
  files_modified: 6
---

# Phase 96 Plan 01: Z-Denoising Fix and Documentation Summary

**One-liner:** Fixed z-denoising CLI to shift raw `points` keypoints (not just `control_points`) and updated all reconstruction docstrings to describe keypoint-native, variable-point-count output.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Fix z-denoising CLI for raw-keypoint mode | 5c7c6e1 | cli.py, test_temporal_smoothing.py |
| 2 | Update docstrings for keypoint-native output | 75f88c7 | stage.py, reconstruction.py, midline.py, temporal_smoothing.py |

## What Was Built

### Task 1: Z-Denoising CLI Fix

In raw-keypoint mode (`spline_enabled=False`), `control_points` is all-NaN and the actual fish 3D positions live in the `points` dataset. The CLI previously only shifted `control_points`, silently doing nothing useful.

The fix:
- Reads `points` from HDF5 (returns `None` for legacy files without the dataset — backward compatible)
- Creates `shifted_pts = points.copy()` alongside `shifted_cp`
- In the per-fish loop: `shifted_pts[fi, si, :, 2] += dz` alongside the existing control_points shift
- Writes `shifted_pts` back to the `points` HDF5 dataset when present

NaN safety: the unused dataset (NaN-filled) stays NaN because `NaN + float = NaN`. No special casing needed.

Also added 5 unit tests for `smooth_centroid_z` verifying:
- 6-point contiguous array preserves shape `(T,)`
- Step function is smoothed (values change)
- NaN positions interpolated through during smoothing, then restored
- All-NaN input returns all-NaN
- Frame-index gap creates independent segments

### Task 2: Docstring Updates

Updated all stale docstrings across the reconstruction subsystem:

- **stage.py module docstring**: Now describes keypoint-native as primary mode (N anatomical keypoints, default 6, identity mapping when n_sample_points=6). Spline mode described as optional. Gap interpolation note updated to cover both modes.
- **ReconstructionStage class docstring**: Removed "3D B-spline midlines" framing; now "3D keypoints (optionally B-spline fitted)". Removed "dense Midline2D" language; explains when interpolation occurs vs identity mapping.
- **Midline3D**: `points` shape `(N, 3)` with variable N explained (typically 6 for raw-keypoint, higher for interpolated). `half_widths` and `z_offsets` shapes updated from `(n_sample_points,)` to `(N,)` with cross-reference.
- **Midline2D**: Added class-level note that N is variable; keypoint-native N=6 with named keypoints listed. `half_widths` notes N matches points count.
- **temporal_smoothing.py module docstring**: Clarified scope — computes smoothed centroid z only; shifting of 3D points is the caller's responsibility.

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `hatch run test`: 1208 passed, 3 skipped
- `hatch run check`: 0 errors, 0 warnings, 0 notes
- stage.py module docstring contains "keypoint-native": confirmed
- Midline3D docstring mentions variable point counts: confirmed
- Midline2D docstring mentions variable point counts: confirmed
- CLI `points` shift present: confirmed via `grep "shifted_pts"` in cli.py

## Self-Check: PASSED
