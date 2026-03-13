---
phase: 90-group-validation-with-changepoint-detection
plan: 02
subsystem: association
tags: [config-migration, pipeline-wiring, cleanup]

requires:
  - phase: 90-group-validation-with-changepoint-detection
    provides: validation.py with validate_groups() and ValidationConfigLike
provides:
  - AssociationConfig with validation_enabled, min_cameras_validate, min_segment_length
  - AssociationStage wired to validate_groups
  - Clean removal of refinement.py with no straggling references
affects: [singleton-recovery, parameter-tuning]

tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - src/aquapose/engine/config.py
    - src/aquapose/core/association/stage.py
    - src/aquapose/core/association/__init__.py
    - src/aquapose/core/association/types.py

key-decisions:
  - "min_cameras_validate defaults to 2 (was 3 for refinement) since multi-keypoint residuals give meaningful signal with 2 cameras"
  - "min_segment_length defaults to 10 (~0.3s at 30fps)"

patterns-established: []

requirements-completed: [CLEAN-02]

duration: 5min
completed: 2026-03-11
---

# Plan 90-02 Summary

**Pipeline wired to multi-keypoint validation; refinement.py deleted with zero straggling references**

## Performance

- **Duration:** 5 min
- **Tasks:** 1
- **Files modified:** 4 modified, 2 deleted

## Accomplishments
- AssociationStage.run() calls validate_groups() instead of refine_clusters()
- AssociationConfig migrated: validation_enabled, min_cameras_validate, min_segment_length
- __init__.py exports updated: ValidationConfigLike + validate_groups
- refinement.py and test_refinement.py deleted; grep audit confirmed zero straggling references
- All 1168 tests pass, lint and typecheck clean

## Task Commits

1. **Task 1: Migrate config, wire stage, update exports, delete refinement** - `cea5207` (feat)

## Files Created/Modified
- `src/aquapose/engine/config.py` - AssociationConfig with validation fields replacing refinement fields
- `src/aquapose/core/association/stage.py` - Step 4 uses validate_groups via lazy import
- `src/aquapose/core/association/__init__.py` - Exports ValidationConfigLike + validate_groups
- `src/aquapose/core/association/types.py` - Docstring updated from refinement to validation terminology
- `src/aquapose/core/association/refinement.py` - DELETED
- `tests/unit/core/association/test_refinement.py` - DELETED

## Decisions Made
- min_cameras_validate defaults to 2 (multi-keypoint residuals meaningful with 2 cameras)
- min_segment_length defaults to 10 (~0.3s at 30fps per user decision)

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Pipeline fully wired to new validation module
- Ready for Phase 91 (singleton recovery) and Phase 92 (parameter tuning)

---
*Phase: 90-group-validation-with-changepoint-detection*
*Completed: 2026-03-11*
