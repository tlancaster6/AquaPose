---
phase: 68-improved-training-data-storage-and-tracking
plan: 02
subsystem: cli
tags: [click, cli, training-data, import, convert, coco, yolo]

# Dependency graph
requires:
  - phase: 68-01
    provides: SampleStore class for SQLite-backed training data management
provides:
  - "aquapose data import" CLI command for YOLO directory ingestion with dedup and augmentation
  - "aquapose data convert" CLI command for COCO-to-YOLO conversion (OBB and Pose)
  - coco_convert.py module with conversion functions moved from scripts/
affects: [68-03, 68-04]

# Tech tracking
tech-stack:
  added: []
  patterns: [click-cli-data-commands, coco-to-yolo-conversion-module]

key-files:
  created:
    - src/aquapose/training/data_cli.py
    - src/aquapose/training/coco_convert.py
    - tests/unit/training/test_data_cli.py
  modified:
    - src/aquapose/cli.py
    - src/aquapose/training/__init__.py
    - tests/unit/test_build_yolo_training_data.py
  deleted:
    - scripts/build_yolo_training_data.py

key-decisions:
  - "Move conversion functions from scripts/ to src/aquapose/training/coco_convert.py (importable module)"
  - "Seg conversion functions (generate_seg_dataset, format_seg_annotation) not migrated per plan -- seg not in current workflow"
  - "Import uses content hash to check pre-existing children count before upsert for cascade-delete tracking"

patterns-established:
  - "data_group Click group pattern for training data management CLI"
  - "Config-based store path resolution: {project_dir}/training_data/{store}/store.db"

requirements-completed: [STORE-03, STORE-04]

# Metrics
duration: 8min
completed: 2026-03-06
---

# Phase 68 Plan 02: Data CLI Summary

**Click CLI commands for training data import (YOLO dirs with dedup/augmentation) and COCO-to-YOLO conversion replacing scripts/build_yolo_training_data.py**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-06T17:34:15Z
- **Completed:** 2026-03-06T17:42:44Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 8 (3 created, 3 modified, 1 deleted, 1 test updated)

## Accomplishments
- `aquapose data import` command ingests YOLO directories into SampleStore with content-hash dedup, optional elastic augmentation (pose only), batch-id, and metadata-json support
- `aquapose data convert` command replaces build_yolo_training_data.py with full parameter set (crop-width, crop-height, lateral-ratio, min-visible, edge-threshold-factor, val-split, seed)
- COCO-to-YOLO conversion functions extracted to importable coco_convert.py module
- 10 new CLI integration tests all passing, plus 22 existing conversion tests migrated from script import to module import

## Task Commits

Each task was committed atomically:

1. **RED: Failing tests for data CLI** - `d38bd65` (test)
2. **GREEN: Implement data CLI, coco_convert, wire CLI, delete script** - `aa2df9e` (feat)

## Files Created/Modified
- `src/aquapose/training/data_cli.py` - Click CLI commands: import and convert subcommands under data group
- `src/aquapose/training/coco_convert.py` - Pure conversion functions moved from scripts/ (load_coco, parse_keypoints, generate_obb_dataset, generate_pose_dataset, etc.)
- `tests/unit/training/test_data_cli.py` - 10 CLI integration tests
- `src/aquapose/cli.py` - Wired data_group into main CLI
- `src/aquapose/training/__init__.py` - Added coco_convert and data_cli exports
- `tests/unit/test_build_yolo_training_data.py` - Updated imports from script to module, removed seg tests
- `scripts/build_yolo_training_data.py` - Deleted (replaced by CLI + coco_convert module)

## Decisions Made
- Conversion functions moved to `coco_convert.py` as a proper importable module rather than keeping them in a standalone script
- Seg conversion functions excluded per plan (not part of current workflow)
- Existing test file for build_yolo_training_data updated to import from new module locations; seg-related test classes removed since those functions were intentionally not migrated

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated test_build_yolo_training_data.py imports after script deletion**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Deleting scripts/build_yolo_training_data.py broke the existing test file that loaded it via importlib
- **Fix:** Updated imports to use aquapose.training.coco_convert and aquapose.training.geometry; removed seg test classes whose functions were not migrated
- **Files modified:** tests/unit/test_build_yolo_training_data.py
- **Verification:** All 1055 tests pass
- **Committed in:** aa2df9e (task commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary fix for script deletion. No scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Data import and convert commands functional, ready for Plan 03 (dataset assembly CLI) and Plan 04 (model registry)
- SampleStore + CLI provide complete data ingestion pipeline

## Self-Check: PASSED

All 8 files verified. Both commit hashes confirmed.

---
*Phase: 68-improved-training-data-storage-and-tracking*
*Completed: 2026-03-06*
