---
phase: 69-cli-workflow-cleanup
plan: 03
subsystem: cli
tags: [click, training, cleanup, dead-code-removal]

requires:
  - phase: 69-02
    provides: CLI config-to-project migration
  - phase: 68
    provides: Store-based data import and data assemble commands
provides:
  - Clean train group with obb command (no redundant augment-elastic or yolo-obb)
  - Clean pseudo-label group with only generate and inspect (no redundant assemble)
  - Removed dataset_assembly.py dead module
affects: []

tech-stack:
  added: []
  patterns:
    - "Dead code removal: delete CLI commands superseded by store-based equivalents"

key-files:
  created: []
  modified:
    - src/aquapose/training/cli.py
    - src/aquapose/training/pseudo_label_cli.py
    - src/aquapose/training/__init__.py
    - src/aquapose/training/elastic_deform.py
    - tests/unit/training/test_elastic_deform_cli.py
    - tests/unit/training/test_training_cli.py
    - tests/unit/training/test_pseudo_label_cli.py

key-decisions:
  - "Kept elastic_deform.py as library code (generate_variants, parse_pose_label, etc.) used by data import --augment"
  - "Deleted write_yolo_dataset and generate_preview_grid from elastic_deform.py (only callers were removed CLI)"

patterns-established: []

requirements-completed: [CLI-07, CLI-08, CLI-09]

duration: 5min
completed: 2026-03-06
---

# Phase 69 Plan 03: Deprecated Command and Dead Code Removal Summary

**Removed augment-elastic CLI, pseudo-label assemble CLI, and dataset_assembly.py; renamed yolo-obb to obb**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-06T20:00:55Z
- **Completed:** 2026-03-06T20:05:55Z
- **Tasks:** 2
- **Files modified:** 9 (7 modified, 2 deleted)

## Accomplishments
- Removed augment-elastic command from train group (superseded by data import --augment)
- Removed pseudo-label assemble command (superseded by data assemble)
- Deleted dataset_assembly.py and its tests (fully superseded by store-based assembly)
- Renamed yolo-obb to obb for cleaner CLI naming
- Removed dead code from elastic_deform.py (write_yolo_dataset, generate_preview_grid, _draw_keypoints_on_image)
- Cleaned __init__.py exports to match remaining code

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove augment-elastic, rename yolo-obb to obb, clean train group** - `0395c37` (feat)
2. **Task 2: Remove pseudo-label assemble, delete dataset_assembly.py, final cleanup** - `6f9e5e5` (feat)

## Files Created/Modified
- `src/aquapose/training/cli.py` - Removed augment-elastic command, renamed yolo-obb to obb
- `src/aquapose/training/pseudo_label_cli.py` - Removed assemble command (~210 lines)
- `src/aquapose/training/__init__.py` - Removed dead imports and __all__ entries
- `src/aquapose/training/elastic_deform.py` - Removed write_yolo_dataset, generate_preview_grid, _draw_keypoints_on_image
- `src/aquapose/training/dataset_assembly.py` - DELETED
- `tests/unit/training/test_dataset_assembly.py` - DELETED
- `tests/unit/training/test_elastic_deform_cli.py` - Kept parse_pose_label tests, removed write_yolo_dataset tests
- `tests/unit/training/test_training_cli.py` - Updated yolo-obb references to obb
- `tests/unit/training/test_pseudo_label_cli.py` - Removed TestAssembleCommand class

## Decisions Made
- Kept elastic_deform.py as library code since generate_variants and related functions are used by data import --augment
- Deleted write_yolo_dataset and generate_preview_grid since their only caller was the removed augment-elastic CLI
- Removed shutil import from elastic_deform.py (only used by deleted write_yolo_dataset)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 69 is now complete (all 3 plans done)
- CLI is clean with no redundant commands or dead modules

---
*Phase: 69-cli-workflow-cleanup*
*Completed: 2026-03-06*

## Self-Check: PASSED
