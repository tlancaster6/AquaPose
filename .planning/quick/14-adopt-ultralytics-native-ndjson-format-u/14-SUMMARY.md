---
phase: quick-14
plan: 01
subsystem: training
tags: [ndjson, ultralytics, training, refactor, build-script]
dependency_graph:
  requires: []
  provides: [ultralytics-native-ndjson-format, simplified-training-wrappers]
  affects: [scripts/build_yolo_training_data.py, src/aquapose/training/]
tech_stack:
  added: []
  patterns: [ultralytics-native-ndjson, direct-model-train]
key_files:
  created: []
  modified:
    - scripts/build_yolo_training_data.py
    - src/aquapose/training/common.py
    - src/aquapose/training/yolo_obb.py
    - src/aquapose/training/yolo_pose.py
    - src/aquapose/training/yolo_seg.py
    - src/aquapose/training/cli.py
    - src/aquapose/training/__init__.py
    - tests/unit/training/test_yolo_pose.py
    - tests/unit/training/test_yolo_seg.py
decisions:
  - "build_yolo_training_data.py now emits single dataset.ndjson per task type (not per split) with Ultralytics-native header and flat annotation arrays"
  - "Training wrappers pass dataset.ndjson path directly to model.train(data=...) — no txt conversion, no data.yaml rewriting"
  - "format_obb_annotation, format_pose_annotation, format_seg_annotation now return flat list[float] rows instead of nested dicts"
  - "OBB: flat [cls, x1, y1, x2, y2, x3, y3, x4, y4]; Pose: flat [cls, cx, cy, w, h, x1, y1, v1, ...]; Seg: flat [cls, x1, y1, x2, y2, ...]"
metrics:
  duration_minutes: 25
  tasks_completed: 2
  files_modified: 9
  completed_date: "2026-03-01"
---

# Quick Task 14: Adopt Ultralytics-Native NDJSON Format Summary

One-liner: Eliminated ~200 lines of NDJSON-to-txt conversion plumbing by making the build script emit Ultralytics-native NDJSON directly and having training wrappers pass the file path straight to model.train().

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Convert build_yolo_training_data.py to emit Ultralytics-native NDJSON | 3e8bda5 |
| 2 | Simplify training wrappers to pass NDJSON directly to model.train() | c8cd9eb |

## What Was Built

### Task 1: Ultralytics-Native NDJSON Build Script

Updated `scripts/build_yolo_training_data.py`:

- **Dataset header line** (first line of each `.ndjson` file): `{"type": "dataset", "task": "obb|segment|pose", "class_names": {"0": "fish"}}` — pose also includes `"kpt_shape": [6, 3]`
- **Image lines** now use `"type": "image"`, `"file"` (not `"image"`), and `"split"` fields
- **Flat annotation arrays** under task-specific keys:
  - OBB: `{"obb": [[cls, x1, y1, x2, y2, x3, y3, x4, y4], ...]}`
  - Pose: `{"pose": [[cls, cx, cy, w, h, x1, y1, v1, ...], ...]}`
  - Seg: `{"segments": [[cls, x1, y1, x2, y2, ...], ...]}`
- **Single combined file** `dataset.ndjson` per task type (train+val in one file, each line has `"split"` field)
- **Removed** `data.yaml` generation from all three `generate_*_dataset` functions

Format functions (`format_obb_annotation`, `format_pose_annotation`, `format_seg_annotation`) now return `list[float]` flat rows instead of nested dicts.

### Task 2: Simplified Training Wrappers

**Deleted from `common.py`:**
- `convert_ndjson_to_txt` — converted NDJSON records to YOLO .txt label files
- `rewrite_data_yaml` — rewrote data.yaml with absolute paths for Ultralytics
- `train_yolo_ndjson` — orchestrated the full NDJSON-to-txt-to-train pipeline

**Deleted from `yolo_obb.py`, `yolo_pose.py`, `yolo_seg.py`:**
- `_format_*_label_line` — formatted annotation dicts as YOLO label lines
- `_convert_*_ndjson_to_txt` — thin wrappers around convert_ndjson_to_txt
- `_rewrite_data_yaml_*` — thin wrappers around rewrite_data_yaml
- `_POSE_YAML_KEYS` constant from yolo_pose.py

**Simplified `train_yolo_{obb,pose,seg}`:**
Each function now:
1. Finds `dataset.ndjson` in `data_dir`
2. Initializes YOLO model (`YOLO(weights)` or `YOLO(model.pt)`)
3. Calls `model.train(data=str(ndjson_path), ...)`
4. Copies best/last weights to `output_dir`

**Updated:**
- `cli.py`: `--data-dir` help text updated to "Directory containing dataset.ndjson file"
- `__init__.py`: Removed `train_yolo_ndjson` from imports and `__all__`
- Tests: Replaced tests for deleted functions with importability + `FileNotFoundError` tests for missing `dataset.ndjson`

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written, with one minor Rule 1 fix:

**1. [Rule 1 - Bug] Removed unused `shutil` import from common.py**
- **Found during:** Task 2 lint check
- **Issue:** After removing the three dead functions, `shutil` was no longer used in `common.py`
- **Fix:** Removed the `import shutil` line
- **Files modified:** `src/aquapose/training/common.py`

**2. [Rule 1 - Bug] Fixed RUF043 lint warning in test files**
- **Found during:** Task 2 lint check
- **Issue:** `match="dataset.ndjson"` in `pytest.raises` uses literal `.` which is a regex metacharacter
- **Fix:** Changed to raw strings: `match=r"dataset\.ndjson"`
- **Files modified:** `tests/unit/training/test_yolo_pose.py`, `tests/unit/training/test_yolo_seg.py`

### Out-of-Scope Pre-existing Issues

Lint errors in `tests/unit/core/midline/test_pose_estimation_backend.py` and `test_segmentation_backend.py` (unused vars, import issues, style warnings) — pre-existing, not caused by this task. Logged to deferred items.

## Self-Check

### Files verified to exist
- `scripts/build_yolo_training_data.py` — FOUND
- `src/aquapose/training/common.py` — FOUND (no convert_ndjson_to_txt, rewrite_data_yaml, train_yolo_ndjson)
- `src/aquapose/training/yolo_obb.py` — FOUND (simplified, uses model.train(data=ndjson_path))
- `src/aquapose/training/yolo_pose.py` — FOUND (simplified, uses model.train(data=ndjson_path))
- `src/aquapose/training/yolo_seg.py` — FOUND (simplified, uses model.train(data=ndjson_path))

### Zero references to deleted functions in src/
```
grep -r "convert_ndjson_to_txt|rewrite_data_yaml|format_label_line" src/  # 0 hits
```

### Commits verified
- 3e8bda5: feat(quick-14): convert build script to emit Ultralytics-native NDJSON
- c8cd9eb: refactor(quick-14): simplify training wrappers to pass NDJSON directly to model.train()

### Tests: 29 passed, 0 failed

## Self-Check: PASSED
