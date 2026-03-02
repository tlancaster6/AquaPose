---
phase: 38-stabilization-and-tech-debt-cleanup
plan: "02"
subsystem: training
tags: [training, data-pipeline, yolo, format-migration, tech-debt]
dependency_graph:
  requires: []
  provides: [standard-yolo-txt-yaml-dataset-format]
  affects: [scripts/build_yolo_training_data.py, src/aquapose/training/]
tech_stack:
  added: [pyyaml (yaml.dump)]
  patterns: [YOLO txt label files, dataset.yaml metadata, labels/ directory mirroring images/]
key_files:
  created: []
  modified:
    - scripts/build_yolo_training_data.py
    - src/aquapose/training/yolo_obb.py
    - src/aquapose/training/yolo_pose.py
    - src/aquapose/training/yolo_seg.py
    - src/aquapose/training/cli.py
    - tests/unit/training/test_yolo_pose.py
    - tests/unit/training/test_yolo_seg.py
    - tests/unit/test_build_yolo_training_data.py
decisions:
  - "Identity flip_idx (list(range(n_keypoints))) used for pose dataset.yaml — no bilateral symmetry in fish keypoints"
  - "kpt_shape derived dynamically from annotation data with N_KEYPOINTS (6) as fallback"
  - "labels/ directory structure mirrors images/ directory (YOLO standard convention)"
  - "Task 2 changes (training wrappers) were already committed by the 38-01 prior agent as commit 88ee4b8"
metrics:
  duration_minutes: 9
  tasks_completed: 2
  files_modified: 8
  completed_date: "2026-03-02"
---

# Phase 38 Plan 02: NDJSON-to-txt+yaml Migration Summary

Migrated training data generation from NDJSON format to standard YOLO txt labels + dataset.yaml, and updated all three training wrappers to consume the new format.

## What Was Built

**Task 1 — Build script migration (commit 69bfe02):**

The three `generate_*` functions in `scripts/build_yolo_training_data.py` now produce standard Ultralytics YOLO directory structure:

```
{mode}/
  images/train/     (source images)
  images/val/
  labels/train/     (per-image .txt label files — NEW)
  labels/val/
  dataset.yaml      (metadata — NEW, replaces dataset.ndjson)
```

Each `.txt` label file contains one line per annotation with space-separated normalized float values:
- OBB: `cls x1 y1 x2 y2 x3 y3 x4 y4`
- Pose: `cls cx cy w h x1 y1 v1 x2 y2 v2 ...`
- Seg: `cls x1 y1 x2 y2 ...` (polygon vertices)

`dataset.yaml` fields:
- OBB/Seg: `path`, `train`, `val`, `nc`, `names`
- Pose: adds `kpt_shape: [6, 3]` and `flip_idx: [0, 1, 2, 3, 4, 5]`

All NDJSON writing code (header lines, JSON records, `.ndjson` file writes) removed. Added `import yaml` to the script.

The `format_obb_annotation`, `format_pose_annotation`, and `format_seg_annotation` functions were already returning flat `list[float]` — only the file-writing infrastructure changed.

**Task 2 — Training wrappers and CLI (commit 88ee4b8, from prior agent):**

The three training wrappers (`yolo_obb.py`, `yolo_pose.py`, `yolo_seg.py`) now:
- Look for `dataset.yaml` instead of `dataset.ndjson`
- Pass `yaml_path` to `model.train(data=...)`
- Raise `FileNotFoundError` mentioning `dataset.yaml`

CLI `--data-dir` help text updated from "dataset.ndjson" to "dataset.yaml" in all three subcommands.

Tests updated: `pytest.raises(FileNotFoundError, match=r"dataset\.yaml")` in both `test_yolo_pose.py` and `test_yolo_seg.py`.

**Test suite (`tests/unit/test_build_yolo_training_data.py`) fully rewritten** to test the new txt+yaml format:
- `TestFormatObbAnnotation`: verifies flat list `[cls, x1, y1, ..., x4, y4]` (9 values)
- `TestFormatPoseAnnotation`: verifies flat list `[cls, cx, cy, w, h, x1, y1, v1, ...]` (23 values for N=6)
- `TestFormatSegAnnotation`: verifies flat list `[cls, x1, y1, ...]`
- `TestIntegrationPipeline`: verifies `labels/` directory, `.txt` files, `dataset.yaml` content
- `TestSegConverter`: verifies seg-specific behavior (multi-ring, polygon transform, intruder labeling)

## Deviations from Plan

### Auto-fixed Issues

None — plan executed as written.

### Discovery: Task 2 Already Completed by Prior Agent

When attempting to commit Task 2, git showed the working tree was already clean for those files. Investigation revealed that commit `88ee4b8` (labeled `docs(38-01): complete config field consolidation plan`) included the training wrapper and test changes bundled with the plan 38-01 SUMMARY commit. The changes were identical to what was planned, so no rework was needed.

### Pre-existing Test Failures (Out of Scope)

Two test failures in `tests/unit/engine/test_pipeline.py` pre-exist before this plan and are unrelated to training data format:
- `test_pipeline_writes_config_artifact`
- `test_config_artifact_written_before_stages`

These are deferred to a future plan.

### Pre-existing Hook Infrastructure Bug (Out of Scope)

The `.hooks/check-import-boundary.sh` calls `python` (not `python3`), which fails in this environment with exit code 127. Manual verification with `python3 tools/import_boundary_checker.py` confirmed no actual import boundary violations. Deferred to a future fix.

## Verification Results

1. `hatch run test` passes — 654 passed, 3 skipped, 2 pre-existing failures (unrelated)
2. `grep -rni 'ndjson' src/` — zero source code hits
3. `grep -rni 'ndjson' scripts/` — zero hits
4. `grep -rn 'dataset\.yaml' scripts/build_yolo_training_data.py` — 9 hits confirming new format
5. `grep -rn 'dataset\.yaml' src/aquapose/training/` — all three wrappers confirmed

## Note for Users with Local tmp/convert_all_annotations.py

If you have a local copy of `tmp/convert_all_annotations.py` (not tracked in the repo), it may reference the old NDJSON format. Update it to use the new txt+yaml output structure if needed.

## Self-Check: PASSED

Files exist:
- `scripts/build_yolo_training_data.py` — FOUND
- `src/aquapose/training/yolo_obb.py` — FOUND
- `src/aquapose/training/yolo_pose.py` — FOUND
- `src/aquapose/training/yolo_seg.py` — FOUND
- `tests/unit/test_build_yolo_training_data.py` — FOUND

Commits exist:
- `69bfe02` (Task 1: build script) — FOUND
- `88ee4b8` (Task 2: training wrappers, already committed by prior agent) — FOUND
