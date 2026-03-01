---
phase: 32-yolo-obb-detection-backend
plan: "01"
subsystem: detection
tags: [yolo-obb, ultralytics, affine-crop, opencv, detection-backend]

requires:
  - phase: 30-config-and-contracts
    provides: Detection dataclass with angle/obb_points fields, DetectionConfig
  - phase: 30-config-and-contracts
    provides: get_backend() factory pattern and YOLOBackend to follow

provides:
  - YOLOOBBBackend class with eager loading and OBB -> Detection conversion
  - extract_affine_crop() for rotation-aligned crops from OBB detections
  - invert_affine_point() and invert_affine_points() for crop->frame back-projection
  - AffineCrop dataclass with invertible transform matrix M
  - crop_size field on DetectionConfig (default [256, 128])

affects:
  - 32-02 (pipeline integration of YOLO-OBB backend)
  - 33-keypoint-midline-backend (affine crop utilities are the primary consumer)

tech-stack:
  added: []
  patterns:
    - "OBB angle conversion happens once at the boundary (YOLOOBBBackend.detect) — negate ultralytics CW to standard math CCW"
    - "Affine crop uses cv2.warpAffine with BORDER_CONSTANT=0 for zero-fill letterboxing"
    - "DetectionConfig uses list[int] not tuple for crop_size to ensure YAML safe_load roundtrip"

key-files:
  created:
    - src/aquapose/core/detection/backends/yolo_obb.py
    - tests/unit/segmentation/test_affine_crop.py
  modified:
    - src/aquapose/core/detection/backends/__init__.py
    - src/aquapose/engine/config.py
    - src/aquapose/segmentation/crop.py
    - tests/unit/core/detection/test_detection_stage.py

key-decisions:
  - "crop_size stored as list[int] not tuple[int,int] — Python tuples serialize as !!python/tuple in PyYAML which safe_load cannot parse"
  - "OBB angle conversion (negate) happens once in YOLOOBBBackend.detect — no conversion anywhere else"
  - "extract_affine_crop accepts obb_w/obb_h for interface consistency but crop canvas is always crop_size"

patterns-established:
  - "Angle convention: YOLOOBBBackend.detect() negates ultralytics CW rad -> standard math CCW rad stored in Detection.angle"
  - "Affine crop pipeline: extract_affine_crop() -> model inference -> invert_affine_points() back to frame"

requirements-completed: [DET-01, DET-02, DET-03]

duration: 18min
completed: 2026-02-28
---

# Phase 32 Plan 01: YOLO-OBB Backend and Affine Crop Utilities Summary

**YOLOOBBBackend with single-point CW->CCW angle conversion plus invertible affine crop utilities (extract/invert) with < 1px round-trip accuracy validated across 6 rotation angles**

## Performance

- **Duration:** 18 min
- **Started:** 2026-02-28T23:08:06Z
- **Completed:** 2026-02-28T23:26:39Z
- **Tasks:** 2
- **Files modified:** 6 (2 new, 4 modified)

## Accomplishments
- YOLOOBBBackend eagerly loads YOLO-OBB weights, reads `result.obb` (not `result.boxes`), and converts ultralytics CW angle to standard math CCW — this is the ONE conversion point in the codebase
- `extract_affine_crop()` builds a cv2.warpAffine rotation-centred crop with zero-fill letterboxing for out-of-bounds regions
- `invert_affine_point()` and `invert_affine_points()` back-project crop-space predictions to frame space with < 1px error (validated for angles 0, pi/4, pi/2, -pi/3, pi/6, -pi)
- `crop_size` added to `DetectionConfig` as `list[int]` (not `tuple`) to preserve YAML `safe_load` roundtrip compatibility
- 17 new affine crop tests and 4 new backend registry tests all pass

## Task Commits

1. **Task 1: Create YOLOOBBBackend and extend backend registry** - `104d92b` (feat)
2. **Task 2: Implement affine crop utilities with invertible transform** - `6752ca3` / `887d602` (feat)

## Files Created/Modified
- `src/aquapose/core/detection/backends/yolo_obb.py` - YOLOOBBBackend class with OBB detection and angle conversion
- `src/aquapose/core/detection/backends/__init__.py` - Added yolo_obb case to get_backend() factory
- `src/aquapose/engine/config.py` - Added crop_size: list[int] = [256, 128] to DetectionConfig
- `src/aquapose/segmentation/crop.py` - Added AffineCrop dataclass, extract_affine_crop(), invert_affine_point(), invert_affine_points()
- `tests/unit/core/detection/test_detection_stage.py` - Added 4 yolo_obb backend tests
- `tests/unit/segmentation/test_affine_crop.py` - 17 affine crop tests including round-trip accuracy

## Decisions Made
- `crop_size` stored as `list[int]` not `tuple[int, int]`: `dataclasses.asdict()` preserves Python tuple identity, and PyYAML serializes tuples as `!!python/tuple` which `yaml.safe_load()` cannot parse. Using a list avoids the need for custom YAML representers.
- `extract_affine_crop()` accepts `obb_w`/`obb_h` for caller convenience but the crop canvas size is controlled solely by `crop_size`. This matches the plan's note that padding is handled downstream.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed tuple-in-YAML serialization breaking existing test**
- **Found during:** Task 1 (DetectionConfig.crop_size field)
- **Issue:** Using `tuple[int, int]` as the field type caused `yaml.dump(dataclasses.asdict(...))` to emit `!!python/tuple` tags that `yaml.safe_load()` rejects — broke existing `test_serialize_config_roundtrip`
- **Fix:** Changed field type to `list[int]` with `field(default_factory=lambda: [256, 128])`
- **Files modified:** `src/aquapose/engine/config.py`
- **Verification:** `test_serialize_config_roundtrip` passes; plan note explicitly anticipated list/tuple interchangeability
- **Committed in:** `104d92b` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug)
**Impact on plan:** Necessary correctness fix; plan's own note anticipated the list/tuple flexibility. No scope creep.

## Issues Encountered
- Ruff lint required two passes: unused unpacked variables (`cx`, `cy` -> `_cx`, `_cy`) in `yolo_obb.py` and unused test variables. Fixed before final commits.
- Pre-commit stash mechanism caused `crop.py` and `test_affine_crop.py` to appear in an earlier `docs(33)` commit (pre-existing commit in repo captured staged files during stash/restore cycle). Files are fully committed at HEAD.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- Plan 32-01 complete: YOLOOBBBackend and affine crop utilities ready
- Plan 32-02 can now integrate `get_backend("yolo_obb", ...)` into DetectionStage
- Phase 33 keypoint midline backend can import `extract_affine_crop` and `invert_affine_points` from `segmentation.crop`

---
*Phase: 32-yolo-obb-detection-backend*
*Completed: 2026-02-28*
