---
phase: quick-13
plan: 01
subsystem: training
tags: [yolo-obb, ndjson, training, cli, refactor]
dependency_graph:
  requires: []
  provides: [OBB NDJSON training pipeline, unified CLI --model flag]
  affects: [scripts/build_yolo_training_data.py, src/aquapose/training/yolo_obb.py, src/aquapose/training/cli.py]
tech_stack:
  added: []
  patterns: [NDJSON-to-YOLO-txt conversion via train_yolo_ndjson, thin format_label_line adapter]
key_files:
  created: []
  modified:
    - scripts/build_yolo_training_data.py
    - src/aquapose/training/yolo_obb.py
    - src/aquapose/training/cli.py
    - tests/unit/training/test_training_cli.py
decisions:
  - OBB NDJSON key renamed from "obbs" to "annotations" to match convert_ndjson_to_txt expectation
  - model_size: str = "s" replaced with model: str = "yolov8s-obb" maintaining same default architecture
  - CLI yolo-obb now uses --model and --weights flags consistent with seg/pose commands
metrics:
  duration: ~10 minutes
  completed: 2026-03-01
  tasks_completed: 2
  files_modified: 4
---

# Quick Task 13: Convert yolo_obb.py to Use NDJSON Format — Summary

**One-liner:** Unified OBB training wrapper to use train_yolo_ndjson from common.py, matching seg/pose pattern, and renamed NDJSON "obbs" key to "annotations".

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Fix build_yolo_training_data.py OBB NDJSON to use "annotations" key | 0c6af2d |
| 2 | Rewrite yolo_obb.py to use train_yolo_ndjson, update CLI and test | 7e98ac3 |

## What Was Built

### Task 1: OBB NDJSON key fix (0c6af2d)

In `scripts/build_yolo_training_data.py`, `generate_obb_dataset()` was writing NDJSON records with key `"obbs"` for the annotations list. The `convert_ndjson_to_txt` helper in `common.py` reads `record["annotations"]`. The key was renamed to `"annotations"` so the OBB NDJSON output is compatible with the shared conversion pipeline.

### Task 2: yolo_obb.py rewrite (7e98ac3)

`yolo_obb.py` was completely rewritten to follow the identical pattern as `yolo_seg.py`:

- Removed: `shutil`, `torch`, `MetricsLogger` imports and manual training loop
- Added: `_format_obb_label_line(ann)` — formats one OBB annotation as `class_id x1 y1 x2 y2 x3 y3 x4 y4`
- Added: `_convert_obb_ndjson_to_txt` and `_rewrite_data_yaml_obb` thin wrapper helpers
- `train_yolo_obb` now delegates entirely to `train_yolo_ndjson` with `format_label_line=_format_obb_label_line`
- Parameter `model_size: str = "s"` replaced with `model: str = "yolov8s-obb"` and `weights: Path | None = None` added

CLI (`cli.py`):
- `--model-size` option removed; replaced with `--model` (str, default `"yolov8s-obb"`)
- `--weights` option added (matching seg/pose commands)
- Function signature and `train_yolo_obb` call updated accordingly

Test (`test_training_cli.py`):
- `test_train_yolo_obb_help_shows_expected_flags` updated to expect `--model` and `--weights` instead of `--model-size`

## Verification

All 34 training unit tests pass:

```
tests/unit/training/test_common.py           18 passed
tests/unit/training/test_training_cli.py      7 passed
tests/unit/training/test_yolo_pose.py         4 passed
tests/unit/training/test_yolo_seg.py          5 passed
```

Lint: clean (ruff check passes). Pre-existing typecheck errors in unrelated modules (association/stage.py, engine/pipeline.py, etc.) are out of scope.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

### Created files
- `.planning/quick/13-convert-yolo-obb-py-to-use-ndjson-format/13-SUMMARY.md` — this file

### Modified files
- `scripts/build_yolo_training_data.py` — "obbs" -> "annotations" key
- `src/aquapose/training/yolo_obb.py` — rewritten to use train_yolo_ndjson
- `src/aquapose/training/cli.py` — --model-size replaced with --model + --weights
- `tests/unit/training/test_training_cli.py` — updated expected flags

### Commits
- 0c6af2d: fix(quick-13): change OBB NDJSON key from "obbs" to "annotations"
- 7e98ac3: refactor(quick-13): rewrite yolo_obb.py to use NDJSON pipeline via train_yolo_ndjson
