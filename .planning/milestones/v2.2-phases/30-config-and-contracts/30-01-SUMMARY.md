---
phase: 30-config-and-contracts
plan: "01"
subsystem: core-contracts
tags: [dataclasses, config, validation, obb, keypoint-midline]
dependency_graph:
  requires: []
  provides:
    - Detection.angle and Detection.obb_points optional fields for YOLO-OBB backend
    - Midline2D.point_confidence optional field for keypoint midline backend
    - _filter_fields() strict-reject applied to all 8 config types
    - _RENAME_HINTS registry for actionable config error messages
  affects:
    - src/aquapose/segmentation/detector.py
    - src/aquapose/reconstruction/midline.py
    - src/aquapose/engine/config.py
    - src/aquapose/core/synthetic.py
    - src/aquapose/core/midline/backends/segment_then_extract.py
tech_stack:
  added: []
  patterns:
    - Strict-reject config validation with rename hint registry
    - Optional dataclass fields with None default for v2.2 extensibility
key_files:
  created: []
  modified:
    - src/aquapose/segmentation/detector.py
    - src/aquapose/reconstruction/midline.py
    - src/aquapose/engine/config.py
    - src/aquapose/core/synthetic.py
    - src/aquapose/core/midline/backends/segment_then_extract.py
    - tests/unit/engine/test_config.py
    - tests/unit/segmentation/test_detector.py
    - tests/unit/test_midline.py
decisions:
  - "Detection.angle uses standard math convention radians [-pi, pi], not ultralytics native; conversion deferred to YOLO-OBB backend (Plan 32)"
  - "segment_then_extract always fills point_confidence=1.0 — skeletonization has no per-point uncertainty model"
  - "_RENAME_HINTS does not yet include detection.device or detection.stop_frame renames — those fields still exist on DetectionConfig and will move in Plan 02"
  - "PipelineConfig top-level fields filtered after popping run_id/output_dir to avoid double-filtering positional args"
metrics:
  duration_minutes: 9
  tasks_completed: 3
  files_modified: 8
  completed_date: "2026-02-28"
requirements: [CFG-05, CFG-10, CFG-11]
---

# Phase 30 Plan 01: Extend Dataclasses and Universalize Config Validation Summary

**One-liner:** Extended Detection with OBB fields and Midline2D with point_confidence, then applied strict-reject _filter_fields() to all 8 config types with a rename-hint registry.

## What Was Built

This plan established the v2.2 data contract foundation: two dataclass extensions and universal config validation.

**Detection dataclass** now carries two optional OBB fields:
- `angle: float | None = None` — OBB rotation in radians, standard math convention
- `obb_points: np.ndarray | None = None` — shape (4, 2), clockwise from top-left

**Midline2D dataclass** now carries one optional confidence field:
- `point_confidence: np.ndarray | None = None` — shape (N,), float32, values in [0, 1]

All existing Midline2D construction sites in the active codebase (segment_then_extract backend and synthetic module) now fill `point_confidence=np.ones(N, dtype=np.float32)`, which is the locked policy for skeletonization-derived midlines.

**Config validation** is now strict-reject across all stages. The module-level `_filter_fields()` function and `_RENAME_HINTS` dict replace the old silently-dropping inner closure that only covered 3 of 8 config types. Any unknown YAML/CLI field now raises `ValueError` with the field name. Known renames produce a "did you mean?" hint.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Extend Detection and Midline2D dataclasses | 2f80584 | detector.py, midline.py, synthetic.py, segment_then_extract.py |
| 2 | Universalize _filter_fields() with strict reject | 83d7423 | config.py |
| 3 | Add tests for strict reject, rename hints, dataclass extensions | 5ea9c1c | test_config.py, test_detector.py, test_midline.py |

## Decisions Made

- Detection.angle uses standard math convention radians, not ultralytics native convention. The YOLO-OBB backend (Plan 32) will handle angle convention conversion at the boundary.
- segment_then_extract always fills `point_confidence=np.ones(N)` — this is a locked contract. Skeletonization does not produce per-point uncertainty, so uniform 1.0 is the correct representation.
- `_RENAME_HINTS` does not yet include `detection.device` or `detection.stop_frame` rename hints. Those fields still exist on `DetectionConfig` and will be moved to `PipelineConfig` in Plan 02. The hints will be added at that time.
- `PipelineConfig` top-level filtering correctly filters `top_kwargs` after `run_id` and `output_dir` have already been popped, so those positional args are not double-filtered.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Linting errors in new test code**
- **Found during:** Task 3 commit (pre-commit hook)
- **Issue:** `pytest.raises(ValueError)` without `match` parameter (PT011), and unused `crop` variable in test_midline2d_point_confidence_defaults_none
- **Fix:** Added `match="unknown field 'totally_fake'"` to the broad raises, removed unused `crop` variable assignment
- **Files modified:** tests/unit/engine/test_config.py, tests/unit/test_midline.py
- **Commit:** included in 5ea9c1c (same commit after fixes)

## Test Coverage

8 new tests added:
- `test_unknown_field_raises_value_error` — unknown detection field raises ValueError with field name
- `test_rename_hint_in_error_message` — expect_fish_count produces "did you mean" in error
- `test_unknown_top_level_field_raises` — unknown top-level field raises ValueError
- `test_valid_config_still_loads` — valid fields load without error
- `test_detection_angle_defaults_none` — angle/obb_points default to None
- `test_detection_with_obb_fields` — Detection constructed with angle=0.5, obb shape (4,2)
- `test_midline2d_point_confidence_defaults_none` — point_confidence defaults to None
- `test_midline2d_with_point_confidence` — populated array stores correctly

Full suite: 568 passed, 0 failures.

## Self-Check: PASSED
