---
phase: 18-fix-pseudo-label-pose-output
plan: 01
subsystem: training
tags: [pseudo-labels, pose, crop, cli]
dependency_graph:
  requires: []
  provides: [affine_warp_crop, transform_keypoints, crop-based-pose-output]
  affects: [pseudo_label_cli, geometry]
tech_stack:
  added: []
  patterns: [multi-fish-per-crop, affine-warp-cropping]
key_files:
  created: []
  modified:
    - src/aquapose/training/geometry.py
    - src/aquapose/training/__init__.py
    - src/aquapose/training/pseudo_label_cli.py
    - tests/unit/training/test_geometry.py
    - tests/unit/training/test_pseudo_label_cli.py
decisions:
  - Extracted affine_warp_crop and transform_keypoints from scripts into training.geometry for reuse
  - Pose output uses per-fish crop filenames with fish-index suffix pattern
metrics:
  duration: ~5 min
  completed: "2026-03-05T20:04:37Z"
  tasks_completed: 2
  tasks_total: 2
---

# Quick Task 18: Fix Pseudo-Label Pose Output to Use OBB Crops Summary

Pose pseudo-labels now write OBB-cropped stretch-fitted images with crop-space normalized keypoints, matching the training pipeline format from scripts/build_yolo_training_data.py.

## What Changed

### Task 1: Extract affine_warp_crop and transform_keypoints (7d7f256)

Added two geometry functions to `src/aquapose/training/geometry.py`:
- `affine_warp_crop`: Maps 3 OBB corners (TL, TR, BL) to crop rectangle via cv2 affine transform
- `transform_keypoints`: Projects keypoints through affine matrix, marks out-of-bounds as invisible

Both exported from `training.__init__` with 4 unit tests covering output shapes, identity-like affine, in-bounds handling, and OOB marking.

### Task 2: Refactor pseudo_label_cli.py (dae27f6)

- Added `--crop-width` (default 128) and `--crop-height` (default 64) CLI options
- OBB output unchanged: full-frame images + full-frame OBB labels
- Pose output refactored: per-fish OBB crops (crop_w x crop_h) with crop-space keypoints
- Multi-fish-per-crop logic: each crop includes annotations for ALL fish visible in that crop
- Pose filenames: `{frame:06d}_{cam}_{fish:03d}.jpg` (with fish-index suffix)
- Added `_write_pose_crops` helper function encapsulating the crop + annotate logic
- Label accumulation dicts now store `fish_data` (keypoints_2d, visibility) per fish
- Tests updated for crop-based filenames, crop-normalized coordinates, and new CLI options

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- `hatch run test tests/unit/training/` - 920 passed
- `hatch run lint` - all checks passed
- `hatch run typecheck` - no new errors (42 pre-existing)
