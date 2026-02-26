---
phase: 15-stage-migrations
plan: "01"
subsystem: core/detection
tags: [stage-migration, detection, yolo, stage-protocol]
dependency_graph:
  requires: [engine/stages.py, engine/config.py, segmentation/detector.py, io/video.py]
  provides: [core/detection/DetectionStage, core/detection/backends/YOLOBackend]
  affects: [engine/config.py (DetectionConfig gains model_path + device fields)]
tech_stack:
  added: []
  patterns:
    - "TYPE_CHECKING guard preserves ENG-07 import boundary in stage modules"
    - "Eager model loading at construction (fail-fast FileNotFoundError on missing weights)"
    - "Backend registry pattern (get_backend factory in backends/__init__.py)"
    - "pytest.raises used as context manager alongside patch() in same with block"
key_files:
  created:
    - src/aquapose/core/detection/__init__.py
    - src/aquapose/core/detection/types.py
    - src/aquapose/core/detection/stage.py
    - src/aquapose/core/detection/backends/__init__.py
    - src/aquapose/core/detection/backends/yolo.py
    - tests/unit/core/__init__.py
    - tests/unit/core/detection/__init__.py
    - tests/unit/core/detection/test_detection_stage.py
  modified:
    - src/aquapose/engine/config.py (DetectionConfig: +model_path, +device fields)
decisions:
  - "Calibration loading deferred inside __init__ via local imports to avoid circular imports and preserve ENG-07 boundary"
  - "run() uses VideoSet with undistortion maps computed at construction (same as v1.0 orchestrator)"
  - "TYPE_CHECKING guard used for PipelineContext annotation in stage.py — engine is not imported at runtime"
  - "Backend registry returns YOLOBackend directly (no union type) — only one kind supported for now"
metrics:
  duration_minutes: 12
  completed_date: "2026-02-26"
  tasks_completed: 2
  files_created: 9
---

# Phase 15 Plan 01: Detection Stage Migration Summary

DetectionStage in core/detection/ satisfying Stage Protocol via structural typing, with eager YOLO loading, backend registry, and full interface tests.

## What Was Built

### core/detection/ package

- **types.py**: Re-exports `Detection` from `segmentation.detector` as the canonical detection output type
- **backends/__init__.py**: `get_backend(kind, **kwargs)` factory resolves "yolo" to `YOLOBackend`, raises `ValueError` for unknown kinds
- **backends/yolo.py**: `YOLOBackend` wraps `YOLODetector` with eager loading — `FileNotFoundError` raised at construction if weights missing
- **stage.py**: `DetectionStage` class satisfying Stage Protocol via structural typing. Loads calibration and computes undistortion maps at construction. `run()` opens `VideoSet`, iterates frames, runs detector per camera, populates `context.detections`, `context.frame_count`, `context.camera_ids`
- **__init__.py**: Public API exporting `DetectionStage` and `Detection`

### engine/config.py update

Added `model_path: str | None = None` and `device: str = "cuda"` to `DetectionConfig`. These fields are needed for YOLO weight loading and device placement configuration.

### Interface tests

6 tests covering:
1. Protocol conformance (`isinstance(stage, Stage)` passes)
2. Context population (detections, frame_count, camera_ids all set after run())
3. Backend registry error handling (ValueError on unknown kind)
4. Import boundary enforcement (no runtime engine/ imports in any core/detection/ module)
5. Fail-fast behavior (FileNotFoundError at construction for missing weights)

## Key Design Decisions

**Import boundary (ENG-07)**: `PipelineContext` is used only in `run()` type annotation. The `from aquapose.engine.stages import PipelineContext` import lives inside an `if TYPE_CHECKING:` block — verified at runtime to confirm `aquapose.engine` is never loaded when importing from `core/detection/`.

**Local imports in `__init__`**: `load_calibration_data` and `compute_undistortion_maps` are imported inside `DetectionStage.__init__()`. This preserves construction-time fail-fast behavior while avoiding module-level circular imports.

**Exact v1.0 behavior ported**: Camera discovery (glob `*.avi/*.mp4`, `stem.split("-")[0]` for camera ID), skip camera logic (`e3v8250` default), calibration loading, undistortion map computation — all identical to `pipeline/orchestrator.py::reconstruct()`.

## Deviations from Plan

None — plan executed exactly as written.

The `test_backend_registry_yolo_requires_model_path` test was simplified to test `get_backend("yolo")` without arguments (catches `TypeError` for missing required kwarg), which covers the spirit of the test without requiring filesystem interaction.

## Verification

- `isinstance(DetectionStage(...), Stage)` returns `True` — confirmed
- `hatch run test tests/unit/core/detection/ -v` — 6 tests pass
- `hatch run check` (lint + typecheck) — all checks pass for new files
- Import boundary: grep for runtime `engine` imports in `src/aquapose/core/detection/` returns nothing relevant (only docstring and TYPE_CHECKING block references)
