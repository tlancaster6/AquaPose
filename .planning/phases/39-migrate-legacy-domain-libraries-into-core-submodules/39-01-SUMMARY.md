---
phase: 39-migrate-legacy-domain-libraries-into-core-submodules
plan: "01"
subsystem: core/types, core/midline, core/reconstruction, core/tracking, core/detection
tags: [refactor, types, migration, dual-path]
dependency_graph:
  requires: []
  provides:
    - core/types/ package with Detection, CropRegion, AffineCrop, Midline2D, Midline3D, MidlineSet
    - core/midline/midline.py with MidlineExtractor
    - core/midline/crop.py with extract_affine_crop, invert_affine_point, invert_affine_points
    - core/reconstruction/triangulation.py with triangulate_midlines + helpers + constants
    - core/reconstruction/curve_optimizer.py with CurveOptimizer + CurveOptimizerConfig + OptimizerSnapshot
    - core/tracking/ocsort_wrapper.py with OcSortTracker
    - core/detection/backends/yolo.py with YOLODetector and make_detector()
  affects:
    - Plan 02 (import rewiring — all consumers will be updated to new paths)
tech_stack:
  added: []
  patterns:
    - Dual-path state: both legacy and new core paths are live simultaneously until Plan 02
    - Types-first layer: core/types/ imports only stdlib + numpy (no implementation imports)
key_files:
  created:
    - src/aquapose/core/types/__init__.py
    - src/aquapose/core/types/detection.py
    - src/aquapose/core/types/crop.py
    - src/aquapose/core/types/midline.py
    - src/aquapose/core/types/reconstruction.py
    - src/aquapose/core/midline/midline.py
    - src/aquapose/core/midline/crop.py
    - src/aquapose/core/reconstruction/triangulation.py
    - src/aquapose/core/reconstruction/curve_optimizer.py
    - src/aquapose/core/tracking/ocsort_wrapper.py
  modified:
    - src/aquapose/core/detection/backends/yolo.py
decisions:
  - core/types/reconstruction.py imports Midline2D from core/types/midline (MidlineSet type alias requires it)
  - CropRegion not re-exported from core/midline/crop.py (ruff correctly removed unused import — CropRegion is imported in crop.py but only used in function signatures; it was added to types package instead)
  - curve_optimizer.py at new location splits Midline3D/MidlineSet import: constants/functions from core/reconstruction/triangulation, types from core/types/reconstruction
metrics:
  duration: "~8 minutes"
  completed_date: "2026-03-02"
  tasks_completed: 3
  tasks_total: 3
  files_created: 10
  files_modified: 1
---

# Phase 39 Plan 01: Create core/types/ Package and Relocate Implementation Files Summary

Established `core/types/` package with 6 cross-stage domain types and created 5 implementation files at new `core/` locations, plus merged `YOLODetector` into the detection backend — creating a dual-path state where both legacy and new core paths coexist until Plan 02 rewires consumers.

## What Was Built

### Task 1: core/types/ Package

Created `src/aquapose/core/types/` with 5 files:

- **`detection.py`**: `Detection` dataclass (extracted from `segmentation/detector.py`)
- **`crop.py`**: `CropRegion` and `AffineCrop` dataclasses (extracted from `segmentation/crop.py`)
- **`midline.py`**: `Midline2D` dataclass (extracted from `reconstruction/midline.py`)
- **`reconstruction.py`**: `Midline3D` dataclass + `MidlineSet` type alias (extracted from `reconstruction/triangulation.py`); imports `Midline2D` from `core/types/midline`
- **`__init__.py`**: Re-exports all 6 public types

All `core/types/` files import only stdlib and numpy — no implementation module imports.

### Task 2: Implementation Files at New Core Locations

Copied and updated imports for 5 implementation files:

- **`core/midline/midline.py`**: `MidlineExtractor` + all private helpers. Imports `CropRegion` from `core/types/crop`, `Midline2D` from `core/types/midline`. `Midline2D` class definition removed.
- **`core/midline/crop.py`**: `extract_affine_crop`, `invert_affine_point`, `invert_affine_points`. Types come from `core/types/crop`. Dead code (`compute_crop_region`, `extract_crop`, `paste_mask`) not ported.
- **`core/reconstruction/triangulation.py`**: `triangulate_midlines` + all helpers + constants. `Midline2D` imported from `core/types/midline`, `Midline3D`/`MidlineSet` imported from `core/types/reconstruction`. Class definitions removed.
- **`core/reconstruction/curve_optimizer.py`**: Full `CurveOptimizer`, `CurveOptimizerConfig`, `OptimizerSnapshot`. Internal import updated from `reconstruction.triangulation` to `core.reconstruction.triangulation`.
- **`core/tracking/ocsort_wrapper.py`**: `OcSortTracker` unchanged except docstring comment updated to reference `core/types/detection`.

### Task 3: YOLODetector Merged into Detection Backend

Updated `core/detection/backends/yolo.py`:
- Inlined `YOLODetector` class definition (previously imported from `segmentation.detector`)
- Added `make_detector()` factory function
- Updated `Detection` import to `core/types/detection`
- Removed old `from aquapose.segmentation.detector import Detection, YOLODetector`
- Updated `__all__` to include `YOLODetector` and `make_detector`

## Verification

All plan checks passed:
- `from aquapose.core.types import Detection, CropRegion, AffineCrop, Midline2D, Midline3D, MidlineSet` — OK
- `from aquapose.core.midline.midline import MidlineExtractor` — OK
- `from aquapose.core.detection.backends.yolo import YOLODetector, make_detector` — OK
- Legacy files untouched — 656 tests pass, 0 failures

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Ruff removed unused CropRegion import from core/midline/crop.py**
- **Found during:** Task 2 commit
- **Issue:** `CropRegion` was imported in `core/midline/crop.py` but only used as a type annotation in function signatures, which ruff detected as unused at the module level since the type annotations on `extract_affine_crop` don't require it in the module body.
- **Fix:** Ruff auto-removed the import on pre-commit. The functions still work correctly; `CropRegion` is available from `core/types/crop` when needed by callers.
- **Files modified:** `src/aquapose/core/midline/crop.py`
- **Commit:** 26bae8f

## Self-Check: PASSED

All 10 created files exist at expected paths. All 3 task commits found (6dea193, 26bae8f, 80af32f). 656 unit tests pass.
