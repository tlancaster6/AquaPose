---
phase: 39-migrate-legacy-domain-libraries-into-core-submodules
plan: "03"
subsystem: testing
tags: [imports, migration, test-refactoring, core-types, reconstruction, segmentation, tracking]

requires:
  - phase: 39-01
    provides: core/types/ package + relocated implementation files + YOLODetector merged into detection backend
  - phase: 39-02
    provides: all src/ consumer imports rewired to new core paths; shim files deleted

provides:
  - All test file imports updated to new core paths (zero legacy package references in tests/)
  - Full test suite passing (656 tests, 3 skipped, 31 deselected)
  - Complete import migration for test code

affects:
  - 39-04

tech-stack:
  added: []
  patterns:
    - "Test files import types from core/types/ (Detection, Midline2D, Midline3D, MidlineSet, CropRegion, AffineCrop)"
    - "Test files import implementations from core/<stage>/ (core/midline/crop, core/midline/midline, core/reconstruction/triangulation, core/reconstruction/curve_optimizer, core/detection/backends/yolo, core/tracking/ocsort_wrapper)"

key-files:
  created: []
  modified:
    - tests/unit/segmentation/test_affine_crop.py
    - tests/unit/segmentation/test_detector.py
    - tests/unit/tracking/test_ocsort_wrapper.py
    - tests/unit/test_midline.py
    - tests/unit/test_triangulation.py
    - tests/unit/test_curve_optimizer.py
    - tests/unit/core/reconstruction/test_confidence_weighting.py
    - tests/unit/core/reconstruction/test_reconstruction_stage.py
    - tests/unit/core/midline/test_segmentation_backend.py
    - tests/unit/core/midline/test_pose_estimation_backend.py
    - tests/unit/core/midline/test_midline_stage.py
    - tests/unit/core/midline/test_direct_pose_backend.py
    - tests/unit/synthetic/test_detection_gen.py
    - tests/unit/synthetic/test_synthetic.py
    - tests/unit/core/test_synthetic.py
    - tests/unit/io/test_midline_writer.py

key-decisions:
  - "Test imports follow same rewrite map as src/ (Plan 02): types to core/types/, implementations to core/<stage>/)"
  - "SPLINE_K, SPLINE_KNOTS, SPLINE_N_CTRL, N_SAMPLE_POINTS constants available from both core/reconstruction/triangulation and core/reconstruction/curve_optimizer"
  - "Remaining legacy package references in legacy src files (reconstruction/, segmentation/, tracking/) are self-references within the canonical implementations — not considered violations"

patterns-established:
  - "Split imports pattern: from aquapose.segmentation.detector import (Detection, YOLODetector) becomes two imports: types from core/types/, implementations from core/detection/backends/yolo"
  - "Zero legacy package references in tests/ enforced — verified with grep post-migration"

requirements-completed:
  - REORG-01

duration: 15min
completed: "2026-03-02"
---

# Phase 39 Plan 03: Migrate Test Imports to Core Paths Summary

**All 16 test files rewired from legacy aquapose.reconstruction/segmentation/tracking packages to new core/types/ and core/<stage>/ paths, with 656 unit tests passing**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-02T18:20:00Z
- **Completed:** 2026-03-02T18:35:00Z
- **Tasks:** 2
- **Files modified:** 16

## Accomplishments

- Updated all 16 test files to import from new core module paths
- Zero legacy package references remain in tests/ (verified with grep)
- 656 unit tests pass, 3 skipped, 31 deselected (slow/e2e)
- Tests lint clean (ruff passes on tests/)

## Task Commits

Each task was committed atomically:

1. **Task 1: Update all test file imports to new core paths** - `ba52398` (feat)
2. **Task 2: Run full test suite and fix any remaining import issues** - `ba52398` (no new files — verification only, all green on first run)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `tests/unit/segmentation/test_affine_crop.py` - Split `segmentation.crop` → `core/types/crop` + `core/midline/crop`
- `tests/unit/segmentation/test_detector.py` - Split `segmentation.detector` → `core/types/detection` + `core/detection/backends/yolo`
- `tests/unit/tracking/test_ocsort_wrapper.py` - `tracking.ocsort_wrapper` → `core/tracking/ocsort_wrapper` (3 lazy import sites)
- `tests/unit/test_midline.py` - `reconstruction.midline` → `core/midline/midline` + `core/types/midline`; `segmentation.crop` → `core/types/crop`
- `tests/unit/test_triangulation.py` - `reconstruction.midline` → `core/types/midline`; `reconstruction.triangulation` → `core/reconstruction/triangulation` + `core/types/reconstruction`
- `tests/unit/test_curve_optimizer.py` - `reconstruction.curve_optimizer` → `core/reconstruction/curve_optimizer`; `reconstruction.midline` → `core/types/midline`; constants from `reconstruction.triangulation` → `core/reconstruction/curve_optimizer`
- `tests/unit/core/reconstruction/test_confidence_weighting.py` - `reconstruction.curve_optimizer` → `core/reconstruction/curve_optimizer`; `reconstruction.midline` → `core/types/midline`; `reconstruction.triangulation` → `core/reconstruction/triangulation` + `core/types/reconstruction`
- `tests/unit/core/reconstruction/test_reconstruction_stage.py` - `reconstruction.midline` → `core/types/midline`; `reconstruction.triangulation` → `core/types/reconstruction`
- `tests/unit/core/midline/test_segmentation_backend.py` - `segmentation.crop` → `core/types/crop`; `segmentation.detector` → `core/types/detection`
- `tests/unit/core/midline/test_pose_estimation_backend.py` - `segmentation.crop` → `core/types/crop`; `segmentation.detector` → `core/types/detection`
- `tests/unit/core/midline/test_midline_stage.py` - `segmentation.detector` → `core/types/detection`
- `tests/unit/core/midline/test_direct_pose_backend.py` - `segmentation.detector` → `core/types/detection`
- `tests/unit/synthetic/test_detection_gen.py` - `segmentation.detector` → `core/types/detection`
- `tests/unit/synthetic/test_synthetic.py` - `reconstruction.triangulation` → `core/reconstruction/triangulation` + `core/types/reconstruction`
- `tests/unit/core/test_synthetic.py` - lazy `reconstruction.midline import Midline2D` → `core/types/midline`
- `tests/unit/io/test_midline_writer.py` - `reconstruction.triangulation` → `core/reconstruction/triangulation` + `core/types/reconstruction`

## Decisions Made

- SPLINE constants (SPLINE_K, SPLINE_KNOTS, SPLINE_N_CTRL, N_SAMPLE_POINTS) are available in both `core/reconstruction/triangulation` and `core/reconstruction/curve_optimizer` — test_curve_optimizer.py imports them from curve_optimizer to avoid cross-module dependency
- Legacy src/ files still contain self-referential imports within `reconstruction/`, `segmentation/`, `tracking/` — these are canonical implementations, not consumers, so not violations of the migration

## Deviations from Plan

None - plan executed exactly as written. All 16 test files updated on first pass, test suite passed green immediately.

## Issues Encountered

Ruff auto-fixed import ordering on `test_curve_optimizer.py` and `test_midline_writer.py` during pre-commit hook — standard import block sorting. Resolved automatically by re-staging.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plans 01+02+03 complete: entire repository (src/ + tests/) now uses only core/types/ and core/<stage>/ paths
- Legacy packages (reconstruction/, segmentation/, tracking/) remain as canonical implementations but have zero external consumers
- Plan 04 can proceed to delete legacy packages, completing the migration

---
*Phase: 39-migrate-legacy-domain-libraries-into-core-submodules*
*Completed: 2026-03-02*
