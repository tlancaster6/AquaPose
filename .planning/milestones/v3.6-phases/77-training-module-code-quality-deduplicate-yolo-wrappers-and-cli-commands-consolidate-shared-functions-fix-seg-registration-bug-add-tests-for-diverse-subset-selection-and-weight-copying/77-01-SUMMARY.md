---
phase: 77-training-module-code-quality
plan: 01
subsystem: training
tags: [yolo, refactoring, cli, deduplication]

# Dependency graph
requires: []
provides:
  - "Consolidated train_yolo() entry point in yolo_training.py"
  - "Shared _run_training() CLI orchestrator with seg registration bug fix"
  - "Canonical compute_arc_length() in geometry.py (returns float, never None)"
  - "Canonical _LutConfigFromDict in common.py"
  - "Deduplicated affine_warp_crop and transform_keypoints (only in geometry.py)"
affects: [training-pipeline, pseudo-labels, cli]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single train_yolo() dispatches by model_type with _MODEL_DEFAULTS"
    - "_run_training() orchestrator handles full lifecycle for all CLI train commands"
    - "Shared geometry functions live in geometry.py, shared config in common.py"

key-files:
  created:
    - src/aquapose/training/yolo_training.py
  modified:
    - src/aquapose/training/cli.py
    - src/aquapose/training/geometry.py
    - src/aquapose/training/coco_convert.py
    - src/aquapose/training/pseudo_labels.py
    - src/aquapose/training/common.py
    - src/aquapose/training/prep.py
    - src/aquapose/training/pseudo_label_cli.py
    - src/aquapose/training/data_cli.py
    - src/aquapose/training/__init__.py

key-decisions:
  - "compute_arc_length returns 0.0 instead of None for consistency (float return type)"
  - "parse_pose_label called with crop_w=crop_h=1 for curvature since curvature is scale-invariant"

patterns-established:
  - "All YOLO training goes through train_yolo() in yolo_training.py"
  - "CLI train commands delegate to _run_training() for shared lifecycle"
  - "Geometry functions are canonical in geometry.py, imported everywhere else"

requirements-completed: [CQ-01, CQ-02, CQ-03, CQ-04, CQ-05, CQ-06]

# Metrics
duration: 9min
completed: 2026-03-09
---

# Phase 77 Plan 01: Consolidate YOLO Wrappers and Deduplicate Training Module Summary

**Unified train_yolo() replacing 3 wrapper files, shared _run_training() CLI orchestrator fixing seg registration bug, canonical geometry/config functions eliminating all duplicates**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-09T20:09:04Z
- **Completed:** 2026-03-09T20:18:49Z
- **Tasks:** 2
- **Files modified:** 15 (12 source + 3 test)

## Accomplishments
- Created yolo_training.py with unified train_yolo() and 3 convenience aliases, deleting 3 wrapper files (-199 LOC net)
- Extracted _run_training() CLI orchestrator, fixing seg command missing register_trained_model (bug fix)
- Deduplicated compute_arc_length, affine_warp_crop, transform_keypoints, _LutConfigFromDict -- each now has exactly one definition
- Replaced inline pose label parsing in data_cli.py with parse_pose_label()
- All 1113 tests pass, lint clean, no new typecheck errors

## Task Commits

Each task was committed atomically:

1. **Task 1: Create yolo_training.py and deduplicate shared functions** - `f270556` (refactor)
2. **Task 2: Consolidate CLI training commands with shared _run_training()** - `fcfc668` (fix)

## Files Created/Modified
- `src/aquapose/training/yolo_training.py` - Consolidated YOLO training: train_yolo() + 3 aliases
- `src/aquapose/training/cli.py` - Shared _run_training() orchestrator, slim CLI commands
- `src/aquapose/training/geometry.py` - Added canonical compute_arc_length()
- `src/aquapose/training/coco_convert.py` - Removed 3 duplicate functions, imports from geometry
- `src/aquapose/training/pseudo_labels.py` - Removed _compute_arc_length, uses geometry.compute_arc_length
- `src/aquapose/training/common.py` - Added _LutConfigFromDict canonical definition
- `src/aquapose/training/prep.py` - Imports _LutConfigFromDict from common
- `src/aquapose/training/pseudo_label_cli.py` - Imports _LutConfigFromDict from common
- `src/aquapose/training/data_cli.py` - Uses parse_pose_label() instead of inline parsing
- `src/aquapose/training/__init__.py` - Updated exports for new module locations
- `src/aquapose/training/yolo_obb.py` - DELETED
- `src/aquapose/training/yolo_seg.py` - DELETED
- `src/aquapose/training/yolo_pose.py` - DELETED
- `tests/unit/training/test_yolo_seg.py` - Updated import path
- `tests/unit/training/test_yolo_pose.py` - Updated import path
- `tests/unit/test_build_yolo_training_data.py` - Updated import, assertions for 0.0 return

## Decisions Made
- compute_arc_length returns 0.0 (not None) for fewer than min_visible keypoints -- simplifies all call sites
- parse_pose_label called with crop_w=crop_h=1 for curvature computation since curvature is scale-invariant

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test assertions for compute_arc_length return type change**
- **Found during:** Task 1 (verification)
- **Issue:** Tests expected None return, function now returns 0.0
- **Fix:** Changed assertions from `is None` to `== 0.0` and renamed test methods
- **Files modified:** tests/unit/test_build_yolo_training_data.py
- **Committed in:** f270556

**2. [Rule 3 - Blocking] Updated test imports for deleted modules**
- **Found during:** Task 1 (verification)
- **Issue:** test_yolo_seg.py and test_yolo_pose.py imported from deleted yolo_seg/yolo_pose modules
- **Fix:** Changed imports to use yolo_training module
- **Files modified:** tests/unit/training/test_yolo_seg.py, tests/unit/training/test_yolo_pose.py
- **Committed in:** f270556

**3. [Rule 1 - Bug] Adapted parse_pose_label call signature for data_cli.py**
- **Found during:** Task 1 (implementation)
- **Issue:** Plan suggested passing label_text string, but parse_pose_label takes (Path, crop_w, crop_h)
- **Fix:** Pass label_path directly with crop_w=crop_h=1 (curvature is scale-invariant)
- **Files modified:** src/aquapose/training/data_cli.py
- **Committed in:** f270556

---

**Total deviations:** 3 auto-fixed (1 bug, 1 blocking, 1 bug)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Training module is clean and deduplicated, ready for plan 77-02 (tests)
- All public APIs preserved with same signatures via convenience aliases

---
*Phase: 77-training-module-code-quality*
*Completed: 2026-03-09*
