---
phase: 06-2d-medial-axis-and-arc-length-sampling
plan: 01
subsystem: reconstruction
tags: [midline, skeleton, arc-length, segmentation, orientation]
dependency_graph:
  requires:
    - src/aquapose/segmentation/crop.py (CropRegion coordinate transform)
    - src/aquapose/tracking/tracker.py (FishTrack velocity for orientation)
    - src/aquapose/calibration/projection.py (RefractiveProjectionModel.project for head projection)
  provides:
    - src/aquapose/reconstruction/midline.py (Midline2D, MidlineExtractor, all helpers)
    - src/aquapose/reconstruction/__init__.py (public package exports)
  affects:
    - Phase 7 (triangulation consumes Midline2D structs from this module)
tech_stack:
  added:
    - scikit-image>=0.21 (skeletonize, regionprops)
    - scipy.ndimage (distance_transform_edt, binary_closing, binary_opening)
    - scipy.interpolate (interp1d for arc-length resampling)
  patterns:
    - Two-pass BFS for longest-path skeleton pruning
    - Adaptive morphological smoothing via regionprops minor axis length
    - Arc-length normalized interpolation for fixed-point resampling
    - Back-correction buffering with cap for orientation inheritance
key_files:
  created:
    - src/aquapose/reconstruction/__init__.py
    - src/aquapose/reconstruction/midline.py
    - tests/unit/test_midline.py
  modified:
    - pyproject.toml (added scikit-image>=0.21 dependency)
decisions:
  - "axis_minor_length used (not deprecated minor_axis_length) for regionprops minor axis"
  - "skeletonize return wrapped in np.asarray(dtype=bool) for basedpyright compatibility"
  - "_orient_midline uses torch internally (lazy import) to call RefractiveProjectionModel.project"
metrics:
  duration: 11 min
  completed: 2026-02-21
  tasks_completed: 2
  files_created: 3
  files_modified: 1
---

# Phase 06 Plan 01: 2D Medial Axis Extraction and Arc-Length Sampling Summary

Implemented the full 2D medial axis extraction and arc-length sampling pipeline as `src/aquapose/reconstruction/midline.py`, producing 15-point Midline2D structs in full-frame coordinates with half-widths and consistent head-to-tail orientation via 3D velocity cues and back-correction buffering.

## What Was Built

### Core Data Structure

`Midline2D` dataclass: `points` (15,2) float32 full-frame pixels, `half_widths` (15,) float32, `fish_id`, `camera_id`, `frame_index`, `is_head_to_tail`.

### Pipeline Helpers

- `_check_skip_mask`: Rejects masks with area < min_area or nonzero pixels touching any crop edge
- `_adaptive_smooth`: Morphological closing then opening with elliptical kernel; radius derived from skimage `axis_minor_length`
- `_skeleton_and_widths`: Boolean skeleton via `skimage.morphology.skeletonize` + Euclidean distance transform
- `_longest_path_bfs`: Two-pass BFS on 8-connected skeleton to find and reconstruct the longest head-to-tail path
- `_resample_arc_length`: Cumulative arc-length normalization + `scipy.interpolate.interp1d` for exactly N evenly-spaced points
- `_crop_to_frame`: Scale + translate from resized mask space to full-frame coordinates, including half-width scaling
- `_orient_midline`: Projects predicted 3D head position into camera, compares to both endpoints; returns False when speed < 0.5 BL/s

### MidlineExtractor Class

Stateful class managing:
- Per-fish orientation dictionary for inheritance
- Back-correction buffer (max 30 frames or fps frames, whichever smaller)
- Full pipeline invocation per track per camera per frame
- Graceful skip handling at all stages

### Unit Tests (15 test cases)

All synthetic, no GPU, no real images:
1. `test_check_skip_mask_valid`
2. `test_check_skip_mask_too_small`
3. `test_check_skip_mask_boundary_clipped`
4. `test_adaptive_smooth_preserves_shape`
5. `test_skeleton_produces_thin_path`
6. `test_longest_path_bfs_returns_ordered_path`
7. `test_longest_path_bfs_empty_skeleton`
8. `test_resample_arc_length_count`
9. `test_resample_arc_length_endpoints`
10. `test_crop_to_frame_transform`
11. `test_crop_to_frame_with_resize`
12. `test_extract_midlines_full_pipeline`
13. `test_extract_midlines_skips_small_mask`
14. `test_orientation_inheritance`
15. `test_back_correction_cap`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed deprecated skimage API usage**
- **Found during:** Task 2 (FutureWarning in test output)
- **Issue:** `RegionProperties.minor_axis_length` deprecated since skimage 0.26; removed in 2.0
- **Fix:** Replaced with `axis_minor_length`
- **Files modified:** `src/aquapose/reconstruction/midline.py`
- **Commit:** 0bb5b4d

**2. [Rule 2 - Type safety] Added explicit ndarray cast for skeletonize return type**
- **Found during:** Task 1 typecheck
- **Issue:** `skimage.morphology.skeletonize` returns a complex type union that basedpyright couldn't narrow to `ndarray`
- **Fix:** `np.asarray(skeletonize(bool_mask), dtype=bool)` with explicit type annotation
- **Files modified:** `src/aquapose/reconstruction/midline.py`
- **Commit:** 50faf7a

## Verification Results

- Import check: PASS — `from aquapose.reconstruction import Midline2D, MidlineExtractor`
- Tests: 297 passed (15 new midline tests + 282 existing), 0 failures
- Lint: PASS — no errors in new files
- Typecheck: PASS — no errors in reconstruction/ (4 pre-existing errors in detector.py, out of scope)

## Self-Check: PASSED

All created files exist on disk. Both task commits verified in git log.
