---
phase: 63-pseudo-label-generation-source-a
plan: 01
subsystem: training
tags: [pseudo-labels, geometry, bspline, reprojection, confidence-scoring]

requires:
  - phase: 61-z-denoising
    provides: Midline3D type with plane metadata and per_camera_residuals
  - phase: 62-prep-infrastructure
    provides: keypoint_t_values fail-fast in MidlineConfig

provides:
  - training.geometry module with pca_obb, extrapolate_edge_keypoints, format_obb_annotation, format_pose_annotation
  - training.pseudo_labels module with reproject_spline_keypoints, compute_confidence_score, generate_fish_labels
  - Unit tests for both modules

affects: [63-02, 64-pseudo-label-generation-source-b, 65-dataset-assembly]

tech-stack:
  added: []
  patterns: [geometry-function-promotion-from-scripts, composite-confidence-scoring]

key-files:
  created:
    - src/aquapose/training/geometry.py
    - src/aquapose/training/pseudo_labels.py
    - tests/unit/training/test_geometry.py
    - tests/unit/training/test_pseudo_labels.py
  modified:
    - src/aquapose/training/__init__.py
    - scripts/build_yolo_training_data.py

key-decisions:
  - "Promoted geometry functions from scripts/ to training.geometry for code reuse"
  - "Confidence score weighted composite: 50% residual + 30% camera count + 20% variance"
  - "B-spline evaluation via scipy.interpolate.BSpline for spline reprojection"

patterns-established:
  - "Geometry function promotion: shared functions from scripts/ go to training.geometry"
  - "Composite confidence scoring: normalized component scores with configurable weights"

requirements-completed: [LABEL-01, LABEL-02, LABEL-03]

duration: 8min
completed: 2026-03-05
---

# Plan 63-01 Summary

**Promoted geometry functions from scripts/ and built core pseudo-label module with B-spline reprojection, composite confidence scoring, and per-camera label generation**

## Performance

- **Duration:** 8 min
- **Tasks:** 1
- **Files modified:** 6

## Accomplishments
- Promoted pca_obb, extrapolate_edge_keypoints, format_obb_annotation, format_pose_annotation from scripts/ to importable training.geometry module
- Built reproject_spline_keypoints: evaluates B-spline at arc fractions and projects via RefractiveProjectionModel
- Built compute_confidence_score: 0-1 composite from residual, camera count, and per-camera variance
- Built generate_fish_labels: per-camera label generation with residual filtering
- 27 unit tests covering all functions

## Task Commits

1. **Task 1: Promote geometry functions and build core pseudo-label module** - `437c871` (feat)

## Files Created/Modified
- `src/aquapose/training/geometry.py` - Promoted geometry functions (pca_obb, extrapolate_edge_keypoints, format_obb/pose_annotation)
- `src/aquapose/training/pseudo_labels.py` - Core pseudo-label generation (reprojection, confidence, label generation)
- `src/aquapose/training/__init__.py` - Updated exports for new modules
- `scripts/build_yolo_training_data.py` - Imports geometry functions from training.geometry instead of local definitions
- `tests/unit/training/test_geometry.py` - 13 tests for geometry functions
- `tests/unit/training/test_pseudo_labels.py` - 14 tests for pseudo-label functions

## Decisions Made
- Used scipy.interpolate.BSpline for spline evaluation (consistent with reconstruction code)
- Confidence formula: 0.5*residual + 0.3*camera + 0.2*variance (residual is primary quality signal)
- Used .cpu().detach().numpy() for CUDA safety per CLAUDE.md guidelines

## Deviations from Plan
None - plan executed as specified.

## Issues Encountered
None.

## Next Phase Readiness
- Core module ready for CLI wiring in Plan 63-02
- All functions are importable and tested

---
*Phase: 63-pseudo-label-generation-source-a*
*Completed: 2026-03-05*
