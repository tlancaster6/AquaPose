---
phase: 67-elastic-midline-deformation-augmentation-for-pose-training-data
plan: 01
subsystem: training
tags: [numpy, scipy, tps, augmentation, keypoints]

requires: []
provides:
  - elastic_deform.py module with C-curve, S-curve, TPS warp, label generation
  - generate_variants high-level API producing 4 deformed variants per image
affects: [67-02, training-pipeline]

tech-stack:
  added: [scipy.interpolate.RBFInterpolator for TPS]
  patterns: [centroid-preserving keypoint deformation, RBF backward-mapping image warp]

key-files:
  created:
    - src/aquapose/training/elastic_deform.py
    - tests/unit/training/test_elastic_deform.py
  modified:
    - src/aquapose/training/__init__.py

key-decisions:
  - "scipy RBFInterpolator for TPS warp (cv2.createThinPlateSplineShapeTransformer not available in opencv-python)"
  - "Midpoint of angle_range for deformation magnitude (deterministic, not random)"
  - "Test assertions adjusted for centroid re-centering geometry"

patterns-established:
  - "Keypoint deformation: arc-length parameterize, displace perpendicular, re-center centroid"
  - "TPS warp: build backward map via RBF, then cv2.remap with BORDER_REPLICATE"

requirements-completed: [AUG-01, AUG-02, AUG-03]

duration: 8min
completed: 2026-03-05
---

# Plan 67-01: Elastic Midline Deformation Core Summary

**C-curve and S-curve keypoint deformation with scipy TPS image warping and pca_obb label regeneration**

## Performance

- **Duration:** ~8 min
- **Tasks:** TDD cycle (RED -> GREEN)
- **Files modified:** 3

## Accomplishments
- C-curve deformation via uniform circular arc displacement with centroid preservation
- S-curve deformation via sinusoidal lateral displacement with centroid preservation
- TPS image warp using scipy RBFInterpolator backward mapping + cv2.remap
- Label generation reusing pca_obb, format_obb_annotation, format_pose_annotation
- generate_variants API producing 4 symmetric variants (c_pos, c_neg, s_pos, s_neg)
- 16 unit tests covering geometry, identity, mirror symmetry, shape, and label normalization

## Task Commits

1. **RED: Failing tests** - `f0c1b06` (test)
2. **GREEN: Implementation** - `bd00d82` (feat)

## Files Created/Modified
- `src/aquapose/training/elastic_deform.py` - Core deformation module
- `tests/unit/training/test_elastic_deform.py` - 16 unit tests
- `src/aquapose/training/__init__.py` - Added 5 new exports

## Decisions Made
- Used scipy RBFInterpolator instead of OpenCV TPS (cv2.createThinPlateSplineShapeTransformer unavailable)
- Used midpoint of angle_range for deterministic deformation magnitude
- Adjusted test assertions to account for centroid re-centering shifting displacement patterns

## Deviations from Plan
None significant -- adapted TPS implementation to available libraries as allowed by plan context.

## Issues Encountered
- cv2.createThinPlateSplineShapeTransformer not available in opencv-python (requires contrib). Switched to scipy RBF.

## Next Phase Readiness
- elastic_deform.py ready for CLI integration in Plan 67-02
- All public functions exported in training/__init__.py

---
*Plan: 67-01*
*Completed: 2026-03-05*
