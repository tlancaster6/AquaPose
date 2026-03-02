---
phase: 35-codebase-cleanup
plan: 01
subsystem: segmentation, training
tags: [cleanup, dead-code, unet, sam2, mog2, yolo]

# Dependency graph
requires: []
provides:
  - "segmentation package exports only Detection, YOLODetector, make_detector, and crop utilities"
  - "training package exports only CropDataset, common utils, prep_group, train_yolo_obb"
  - "make_detector('mog2') raises ValueError"
  - "DetectionConfig validates detector_kind at construction"
  - "training CLI has only yolo-obb subcommand"
affects: [36-training-wrappers, 37-pipeline-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "make_detector factory only accepts 'yolo' and 'yolo_obb' — raises ValueError for anything else"
    - "DetectionConfig validates detector_kind in __post_init__ for early error feedback"

key-files:
  created: []
  modified:
    - src/aquapose/segmentation/__init__.py
    - src/aquapose/segmentation/detector.py
    - src/aquapose/segmentation/crop.py
    - src/aquapose/training/__init__.py
    - src/aquapose/training/datasets.py
    - src/aquapose/training/cli.py
    - src/aquapose/engine/config.py
    - tests/unit/segmentation/test_detector.py
    - tests/unit/segmentation/test_dataset.py
    - tests/unit/training/test_training_cli.py
    - tests/unit/engine/test_config.py
    - tests/unit/engine/test_cli.py
    - tests/unit/core/midline/test_midline_stage.py

key-decisions:
  - "MOG2 and UNet/SAM2 files were already deleted in a prior docs commit (8f01608) — Plan 01 cleaned up the survivors (__init__.py, datasets.py, tests)"
  - "segment_then_extract.py and direct_pose.py retain stale UNetSegmentor/_PoseModel imports — Plan 02 will stub/replace those backends"
  - "test_midline_stage.py _build_stage helper mocks aquapose.segmentation.model via sys.modules so existing stage tests continue passing through Plan 02"
  - "DetectionConfig now validates detector_kind at construction, rejecting 'mog2' and other invalid values"

patterns-established:
  - "Lazy-import deleted modules must be handled by injecting mock into sys.modules in tests"

requirements-completed: [CLEAN-01, CLEAN-02, CLEAN-04, CLEAN-05]

# Metrics
duration: 16min
completed: 2026-03-01
---

# Phase 35 Plan 01: Codebase Cleanup — Delete Dead Code Summary

**Removed U-Net, SAM2, MOG2, and legacy training CLI code; cleaned all package exports and config validation to enforce Ultralytics-only backends**

## Performance

- **Duration:** ~16 min
- **Started:** 2026-03-01T20:12:22Z
- **Completed:** 2026-03-01T20:28:30Z
- **Tasks:** 3
- **Files modified:** 13

## Accomplishments
- `segmentation/__init__.py` now exports only crop utilities + Detection + YOLODetector + make_detector
- `training/__init__.py` now exports only CropDataset + common utils + prep_group + train_yolo_obb
- `make_detector('mog2')` raises `ValueError`; `DetectionConfig` validates at construction
- `aquapose train --help` shows only `yolo-obb` subcommand (unet and pose removed)
- Full test suite passes (608 tests) and lint passes cleanly

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete custom model files and SAM2 pseudo-labeler** - `2ebc1ba` (feat)
2. **Task 2: Remove MOG2 detector and old training CLI commands** - `0efa21a` (feat)
3. **Task 3: Run full test suite and fix remaining breakage** - (no additional changes needed; Tasks 1-2 covered all fixes)

## Files Created/Modified
- `src/aquapose/segmentation/__init__.py` - Trimmed to crop + Detection + YOLODetector + make_detector only
- `src/aquapose/segmentation/detector.py` - Removed MOG2Detector class; updated make_detector to reject mog2
- `src/aquapose/segmentation/crop.py` - Updated docstring (removed SAM/MaskRCNN references)
- `src/aquapose/training/__init__.py` - Trimmed to CropDataset + common + prep_group + train_yolo_obb
- `src/aquapose/training/datasets.py` - Removed BinaryMaskDataset class and UNET_INPUT_SIZE constant
- `src/aquapose/training/cli.py` - Removed unet and pose subcommands; kept yolo-obb only
- `src/aquapose/engine/config.py` - Added detector_kind validation in DetectionConfig.__post_init__
- `tests/unit/segmentation/test_detector.py` - Rewrote: removed MOG2 tests, added mog2-rejection test
- `tests/unit/segmentation/test_dataset.py` - Removed BinaryMaskDataset test classes
- `tests/unit/segmentation/test_training.py` - Deleted (tested deleted train_unet)
- `tests/unit/training/test_training_cli.py` - Rewrote: removed unet/pose tests, added negative test
- `tests/unit/engine/test_config.py` - Replaced mog2 test values with yolo/yolo_obb, added validation tests
- `tests/unit/engine/test_cli.py` - Replaced mog2 CLI override with yolo
- `tests/unit/core/midline/test_midline_stage.py` - Fixed _build_stage to mock deleted module via sys.modules; removed construction tests for deleted backends
- `tests/unit/core/midline/test_direct_pose_backend.py` - Deleted (all tests relied on deleted _PoseModel)

## Decisions Made
- The deleted files (model.py, pseudo_labeler.py, unet.py, pose.py, viz.py and their tests) were already absent from the codebase (deleted in commit `8f01608` as part of a docs commit). Plan 01 focused on cleaning the remaining survivors: __init__.py exports, datasets.py classes, training CLI, and test files.
- `segment_then_extract.py` and `direct_pose.py` still contain lazy imports to deleted modules — these are left in place intentionally per plan guidance; Plan 02 will stub/replace both backends.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed tests that patched the deleted aquapose.segmentation.model module**
- **Found during:** Task 1 (post-deletion verification)
- **Issue:** `test_direct_pose_backend.py` and `test_midline_stage.py` patched `aquapose.segmentation.model.UNetSegmentor` and `aquapose.training.pose._PoseModel` — both modules now deleted, causing `ModuleNotFoundError` at patch time
- **Fix:** Deleted `test_direct_pose_backend.py` entirely; removed backend-construction tests from `test_midline_stage.py`; fixed `_build_stage` helper to inject a mock module via `sys.modules` so stage protocol tests still pass
- **Files modified:** `tests/unit/core/midline/test_midline_stage.py`, deleted `tests/unit/core/midline/test_direct_pose_backend.py`
- **Committed in:** `2ebc1ba` (Task 1 commit)

**2. [Rule 1 - Bug] Removed BinaryMaskDataset tests from test_dataset.py**
- **Found during:** Task 1 verification
- **Issue:** `tests/unit/segmentation/test_dataset.py` imported and tested `BinaryMaskDataset` which was being deleted from `datasets.py`
- **Fix:** Removed all `BinaryMaskDataset` test classes; updated import to remove `BinaryMaskDataset`
- **Files modified:** `tests/unit/segmentation/test_dataset.py`
- **Committed in:** `2ebc1ba` (Task 1 commit)

**3. [Rule 1 - Bug] Removed test_training.py which imported deleted train_unet**
- **Found during:** Task 1 verification
- **Issue:** `tests/unit/segmentation/test_training.py` directly imported `from aquapose.training.unet import train_unet`
- **Fix:** Deleted the file entirely (all tests were for deleted functionality)
- **Committed in:** `2ebc1ba` (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (all Rule 1 — bugs caused by deleted modules)
**Impact on plan:** All fixes were necessary consequences of the planned deletions. No scope creep.

## Issues Encountered
- The source files targeted in Plan 01 were already deleted in a previous `docs(36)` commit (`8f01608`). The work of this plan was completing the cleanup: fixing `__init__.py` exports, removing `BinaryMaskDataset` from `datasets.py`, removing `MOG2Detector` from `detector.py`, stripping the CLI, and fixing tests.

## Next Phase Readiness
- Codebase is clean: no MOG2, no UNet, no SAM2, no custom pose model anywhere in `src/`
- `segment_then_extract.py` and `direct_pose.py` remain with stale imports — Plan 02 must stub/replace these before they can be used
- `DetectionConfig` validates detector kinds; config pipeline is enforced
- 608 tests pass; lint clean

---
*Phase: 35-codebase-cleanup*
*Completed: 2026-03-01*
