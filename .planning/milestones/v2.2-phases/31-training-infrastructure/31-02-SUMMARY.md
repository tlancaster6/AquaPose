---
phase: 31-training-infrastructure
plan: "02"
subsystem: training
tags: [training, cli, yolo-obb, pose, keypoint, transfer-learning, migration, cleanup]
dependency_graph:
  requires:
    - aquapose.training package (from 31-01)
    - aquapose.segmentation.model._UNet encoder
  provides:
    - training/yolo_obb.py with train_yolo_obb() wrapping ultralytics
    - training/pose.py with _PoseModel, KeypointDataset, train_pose()
    - aquapose train yolo-obb CLI subcommand
    - aquapose train pose CLI subcommand with --backbone-weights / --unfreeze
    - stratified_split extended to accept any _HasImages dataset (Protocol)
  affects:
    - src/aquapose/segmentation/__init__.py (training/dataset exports removed)
    - tests/unit/segmentation/test_dataset.py (imports updated to training.datasets)
    - tests/unit/segmentation/test_training.py (migrated to train_unet() API)
tech_stack:
  added:
    - training/yolo_obb.py (new)
    - training/pose.py (new — _PoseModel, KeypointDataset, train_pose)
  patterns:
    - _HasImages Protocol for duck-typed stratified_split
    - Frozen-backbone transfer learning with conditional differential LR
    - YOLO-OBB wraps ultralytics with consistent output naming (best_model.pt / last_model.pt)
    - Regression head: AdaptiveAvgPool2d(1) -> flatten -> Linear(96,256) -> ReLU -> Linear(256, n_kp*2) -> Sigmoid
key_files:
  created:
    - src/aquapose/training/yolo_obb.py
    - src/aquapose/training/pose.py
    - tests/unit/training/test_pose.py
  modified:
    - src/aquapose/training/cli.py (yolo-obb and pose subcommands added)
    - src/aquapose/training/__init__.py (train_yolo_obb, train_pose, KeypointDataset exported)
    - src/aquapose/training/datasets.py (_HasImages Protocol, stratified_split broadened)
    - src/aquapose/segmentation/__init__.py (removed training/dataset re-exports)
    - tests/unit/training/test_training_cli.py (7 new tests for 3 subcommands + consistent flags)
    - tests/unit/segmentation/test_dataset.py (import from training.datasets)
    - tests/unit/segmentation/test_training.py (migrated to train_unet API)
  deleted:
    - src/aquapose/segmentation/training.py
    - src/aquapose/segmentation/dataset.py
decisions:
  - _PoseModel borrows only enc0-enc4 from _UNet; decoder is discarded and replaced by regression head
  - stratified_split broadened to _HasImages Protocol so KeypointDataset can be stratified without importing it into datasets.py
  - yolo_obb.py guards results.save_dir for None to handle ultralytics API edge cases
  - test_training.py fully migrated to train_unet() with data_dir convention (old evaluate() not ported — no equivalent in training/)
  - test_dataset.py updated to import from training.datasets (identical API)
metrics:
  duration: "15 min"
  completed_date: "2026-02-28"
  tasks_completed: 2
  files_created: 3
  files_modified: 7
  files_deleted: 2
  tests_added: 21
---

# Phase 31 Plan 02: YOLO-OBB and Pose Training Subcommands Summary

**One-liner:** `aquapose train yolo-obb` and `aquapose train pose` CLI subcommands backed by ultralytics YOLO wrapper and U-Net encoder + regression head for keypoint regression, with frozen-backbone transfer learning and full cleanup of superseded segmentation/training.py and segmentation/dataset.py.

## What Was Built

### Task 1: YOLO-OBB and Pose Training Subcommands

**`training/yolo_obb.py`** — Ultralytics wrapper:
- `train_yolo_obb(data_dir, output_dir, ...)` — expects `data.yaml` in data_dir, delegates to `YOLO("yolov8{model_size}-obb.pt").train(...)`, copies `best.pt` → `best_model.pt` and `last.pt` → `last_model.pt` to `output_dir` for consistent naming
- Auto-detects device (cuda:0 / cpu), guards against None `results.save_dir`
- Uses MetricsLogger to write a summary line after training

**`training/pose.py`** — Keypoint regression:
- `_PoseModel(n_keypoints, pretrained)` — U-Net enc0-enc4 encoder + regression head (AdaptiveAvgPool2d(1) → flatten → Linear(96,256) → ReLU → Linear(256,n_kp*2) → Sigmoid)
- `KeypointDataset` — COCO-format keypoint JSON → (image [3,128,128], keypoints [n_kp*2]) normalised to [0,1]
- `train_pose(data_dir, output_dir, ...)` — full training loop with MSE loss, EarlyStopping (mode="min"), MetricsLogger, save_best_and_last
- Transfer learning: with `backbone_weights` → loads enc* keys from U-Net checkpoint; without `unfreeze` → freezes encoder; with `unfreeze` → differential LR (encoder lr*0.1, head lr*1.0)
- Helpers: `_load_backbone_weights`, `_freeze_encoder`, `_mean_keypoint_error`

**`training/cli.py`** — Two new subcommands:
- `yolo-obb`: `--data-dir`, `--output-dir`, `--epochs`, `--batch-size`, `--device`, `--val-split`, `--imgsz`, `--model-size`
- `pose`: `--data-dir`, `--output-dir`, `--epochs`, `--batch-size`, `--lr`, `--val-split`, `--patience`, `--device`, `--num-workers`, `--backbone-weights`, `--unfreeze`, `--n-keypoints`

**`training/datasets.py`** — Extended:
- Added `_HasImages` Protocol so `stratified_split` accepts any dataset with `_images: list[dict]`
- `stratified_split` type signature broadened from `CropDataset | BinaryMaskDataset` to `_HasImages`

**Tests (21 new):**
- `test_training_cli.py`: 5 new tests — yolo-obb and pose in --help, pose flags, shared consistent flags across all 3 subcommands
- `test_pose.py` (13 tests): model output shape/range, encoder param lists, backbone weight loading, freeze/unfreeze behaviour, mean_keypoint_error utility

### Task 2: Migration Cleanup

**Deleted:**
- `src/aquapose/segmentation/training.py` — superseded by `training/unet.py`
- `src/aquapose/segmentation/dataset.py` — superseded by `training/datasets.py`

**Updated:**
- `segmentation/__init__.py`: removed `from .dataset import ...` and `from .training import ...` re-exports; kept crop/detector/model/pseudo_labeler exports
- `tests/unit/segmentation/test_dataset.py`: import path updated to `training.datasets` (same API)
- `tests/unit/segmentation/test_training.py`: fully migrated to `train_unet(data_dir=...)` convention; `evaluate()` not ported (no equivalent in training/)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed type error in stratified_split for KeypointDataset**
- **Found during:** Task 1 type-check pass
- **Issue:** `stratified_split` only accepted `CropDataset | BinaryMaskDataset`; `KeypointDataset` is a third dataset type needing the same split logic
- **Fix:** Introduced `_HasImages` Protocol in `datasets.py`; broadened `stratified_split` parameter type
- **Files modified:** `src/aquapose/training/datasets.py`

**2. [Rule 1 - Bug] Fixed potential None access on ultralytics results.save_dir**
- **Found during:** Task 1 type-check pass
- **Issue:** `results.save_dir` typed as `str | None` in ultralytics stubs; bare `Path(str(results.save_dir))` would raise
- **Fix:** Added guard: `save_dir = results.save_dir if results is not None and results.save_dir is not None else fallback`
- **Files modified:** `src/aquapose/training/yolo_obb.py`

**3. [Rule 3 - Blocking] Updated test_training.py API migration during cleanup**
- **Found during:** Task 2 deletion step
- **Issue:** `tests/unit/segmentation/test_training.py` used old `train(coco_json, image_root, ...)` and `evaluate()` APIs from the deleted file
- **Fix:** Rewrote to use `train_unet(data_dir=...)` convention; `evaluate()` not ported since no equivalent exists in `training/unet.py` (different scope)
- **Files modified:** `tests/unit/segmentation/test_training.py`

**4. [Rule 3 - Blocking] Ruff auto-fixes on commit**
- **Found during:** Pre-commit hooks on both task commits
- **Issues:** Import sort (I001), line-length wrapping (ruff format)
- **Fix:** Re-staged after ruff modifications

## Self-Check

Files created:
- `src/aquapose/training/yolo_obb.py`: exists
- `src/aquapose/training/pose.py`: exists
- `tests/unit/training/test_pose.py`: exists

Files deleted:
- `src/aquapose/segmentation/training.py`: deleted (confirmed absent)
- `src/aquapose/segmentation/dataset.py`: deleted (confirmed absent)

Commits:
- 41a7ad3: feat(31-02): add YOLO-OBB and pose training subcommands
- c9ea371: feat(31-02): migration cleanup — delete superseded segmentation files and update imports

Tests: 627 passed, 0 failed
Lint: clean (ruff)
Import boundary: 3 warnings (pre-existing in engine/pipeline.py), no violations
Type-check: 0 errors in training/ directory

## Self-Check: PASSED
