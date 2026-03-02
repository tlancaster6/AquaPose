---
phase: 39-migrate-legacy-domain-libraries-into-core-submodules
plan: "02"
subsystem: core
tags: [migration, imports, cleanup, refactor]
dependency_graph:
  requires: [39-01]
  provides: [clean-import-graph, no-legacy-packages]
  affects: [core/detection, core/midline, core/reconstruction, core/tracking, core/synthetic, io, synthetic, visualization]
tech_stack:
  added: []
  patterns: [core/types/ as canonical type location, core/<stage>/ as canonical implementation location]
key_files:
  created: []
  modified:
    - src/aquapose/core/detection/__init__.py
    - src/aquapose/core/detection/backends/yolo_obb.py
    - src/aquapose/core/detection/stage.py
    - src/aquapose/core/midline/__init__.py
    - src/aquapose/core/midline/backends/pose_estimation.py
    - src/aquapose/core/midline/backends/segmentation.py
    - src/aquapose/core/midline/types.py
    - src/aquapose/core/reconstruction/__init__.py
    - src/aquapose/core/reconstruction/backends/curve_optimizer.py
    - src/aquapose/core/reconstruction/backends/triangulation.py
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/core/synthetic.py
    - src/aquapose/core/tracking/stage.py
    - src/aquapose/io/midline_writer.py
    - src/aquapose/synthetic/detection.py
    - src/aquapose/synthetic/fish.py
    - src/aquapose/visualization/midline_viz.py
    - src/aquapose/visualization/overlay.py
    - src/aquapose/visualization/plot3d.py
    - src/aquapose/visualization/triangulation_viz.py
    - tests/unit/core/detection/test_detection_stage.py
  deleted:
    - src/aquapose/core/detection/types.py
    - src/aquapose/core/reconstruction/types.py
    - src/aquapose/reconstruction/ (entire directory, 4 files)
    - src/aquapose/segmentation/ (entire directory, 3 files)
    - src/aquapose/tracking/ (entire directory, 2 files)
decisions:
  - "core/midline/types.py converted from re-export shim to real type definition (AnnotatedDetection only)"
  - "Test import boundary list updated to remove deleted types.py shim (minor fix, not Plan 03 scope)"
metrics:
  duration: "~4 minutes"
  completed: "2026-03-02"
  tasks_completed: 2
  files_modified: 21
  files_deleted: 11
---

# Phase 39 Plan 02: Import Migration and Legacy Package Deletion Summary

One-liner: Rewired all 22 src/aquapose/ import sites from legacy reconstruction/segmentation/tracking packages to canonical core/types/ and core/<stage>/ paths, then deleted the 3 legacy directories and 2 re-export shim files.

## What Was Built

### Task 1: Rewire all src/ consumer imports to new core paths

Updated 20 source files to replace legacy import paths with canonical core paths:

**Detection imports** (`from aquapose.segmentation.detector import Detection`):
- Rewired to `from aquapose.core.types.detection import Detection` in 6 files

**Crop imports** (`from aquapose.segmentation.crop import ...`):
- `AffineCrop`, `CropRegion` rewired to `from aquapose.core.types.crop import ...`
- `extract_affine_crop`, `invert_affine_points` rewired to `from aquapose.core.midline.crop import ...`

**Midline2D imports** (`from aquapose.reconstruction.midline import Midline2D`):
- Rewired to `from aquapose.core.types.midline import Midline2D`

**Private helpers** (`from aquapose.reconstruction.midline import _adaptive_smooth, ...`):
- Rewired to `from aquapose.core.midline.midline import ...`

**Triangulation imports** (`from aquapose.reconstruction.triangulation import ...`):
- `Midline3D`, `MidlineSet` rewired to `from aquapose.core.types.reconstruction import ...`
- `triangulate_midlines` and constants rewired to `from aquapose.core.reconstruction.triangulation import ...`

**CurveOptimizer imports** (`from aquapose.reconstruction.curve_optimizer import ...`):
- Rewired to `from aquapose.core.reconstruction.curve_optimizer import ...`

**Tracker import** (`from aquapose.tracking.ocsort_wrapper import OcSortTracker`):
- Rewired lazy import to `from aquapose.core.tracking.ocsort_wrapper import OcSortTracker`

**Shim files deleted:**
- `core/detection/types.py` — was a re-export shim for Detection
- `core/reconstruction/types.py` — was a re-export shim for Midline2D, Midline3D, MidlineSet

**core/midline/types.py** converted from a shim (re-exporting from 3 legacy paths) to a real type definition containing only `AnnotatedDetection`, with imports from canonical `core/types/` locations.

### Task 2: Delete legacy directories and verify import integrity

Deleted the 3 top-level legacy directories:
- `src/aquapose/reconstruction/` — contained `__init__.py`, `curve_optimizer.py`, `midline.py`, `triangulation.py`
- `src/aquapose/segmentation/` — contained `__init__.py`, `crop.py`, `detector.py`
- `src/aquapose/tracking/` — contained `__init__.py`, `ocsort_wrapper.py`

Verified `import aquapose` succeeds and all 656 unit tests pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test list referenced deleted shim module**
- **Found during:** Task 2 (after deleting legacy directories and shim files)
- **Issue:** `tests/unit/core/detection/test_detection_stage.py` had `"aquapose.core.detection.types"` in `_CORE_DETECTION_MODULES` list for import boundary test. That module was deleted as part of Task 1's shim file cleanup.
- **Fix:** Removed `"aquapose.core.detection.types"` from the list. The deleted shim no longer needs to be covered by the import boundary test.
- **Files modified:** `tests/unit/core/detection/test_detection_stage.py`
- **Commit:** 9236537

## Verification Results

1. `grep -rn "from aquapose.reconstruction.\|from aquapose.segmentation.\|from aquapose.tracking." src/aquapose/ --include="*.py"` — **0 matches**
2. `hatch run python -c "import aquapose; print('Import OK')"` — **Import OK**
3. `ls reconstruction/ segmentation/ tracking/` — **all "No such file or directory"**
4. `core/detection/types.py` and `core/reconstruction/types.py` — **both deleted**
5. `core/midline/types.py` — **contains only AnnotatedDetection** with imports from `core/types/`
6. `hatch run test` — **656 passed, 3 skipped, 0 failed**

## Self-Check: PASSED

All task commits exist:
- `ed0fcd9` — Task 1: rewire imports, delete shim files
- `9236537` — Task 2: delete legacy directories, fix test

All key files verified deleted:
- `src/aquapose/reconstruction/` — DELETED
- `src/aquapose/segmentation/` — DELETED
- `src/aquapose/tracking/` — DELETED
- `src/aquapose/core/detection/types.py` — DELETED
- `src/aquapose/core/reconstruction/types.py` — DELETED
