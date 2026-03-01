---
phase: 31-training-infrastructure
plan: "01"
subsystem: training
tags: [training, cli, datasets, common-utilities, unet, import-boundary]
dependency_graph:
  requires: []
  provides:
    - aquapose.training package with EarlyStopping, MetricsLogger, save_best_and_last, make_loader
    - aquapose.training.datasets with BinaryMaskDataset, CropDataset, stratified_split
    - aquapose.training.unet with train_unet() function
    - aquapose train unet CLI subcommand
    - import boundary enforcement for training/ -> engine/ prohibition
  affects:
    - src/aquapose/cli.py (train_group wired in)
    - tools/import_boundary_checker.py (training/ added to legacy dirs)
tech_stack:
  added:
    - training/ package (new)
  patterns:
    - EarlyStopping with patience and min/max mode
    - MetricsLogger writing to console + CSV
    - Differential LR AdamW (encoder 1/10th of decoder)
    - BCE + Dice combined loss for binary segmentation
    - CosineAnnealingLR scheduler
key_files:
  created:
    - src/aquapose/training/__init__.py
    - src/aquapose/training/common.py
    - src/aquapose/training/datasets.py
    - src/aquapose/training/unet.py
    - src/aquapose/training/cli.py
    - tests/unit/training/__init__.py
    - tests/unit/training/test_training_cli.py
    - tests/unit/training/test_common.py
  modified:
    - src/aquapose/cli.py (added train_group)
    - tools/import_boundary_checker.py (added "training" to _LEGACY_COMPUTATION_DIRS)
decisions:
  - training/ reuses BinaryMaskDataset and stratified_split as fresh rewrites matching segmentation/dataset.py behavior
  - train_unet() imports UNetSegmentor from segmentation.model (allowed: training/ -> segmentation/ is permitted)
  - train_unet() uses data_dir convention (annotations.json + images in same dir) vs old coco_json + image_root pair
  - unet subcommand lazy-imports train_unet inside command body to avoid circular at import time
metrics:
  duration: "11 min"
  completed_date: "2026-02-28"
  tasks_completed: 2
  files_created: 8
  files_modified: 2
  tests_added: 21
---

# Phase 31 Plan 01: Training Infrastructure Foundation Summary

**One-liner:** `aquapose training/` package with EarlyStopping/MetricsLogger utilities, BinaryMaskDataset/stratified_split dataset classes, and `aquapose train unet` CLI subcommand backed by a differential-LR BCE+Dice training loop.

## What Was Built

### training/ Package (new)

**`training/common.py`** — Shared utilities for all training subcommands:
- `EarlyStopping(patience, mode)` — tracks best metric, signals stop when patience exhausted; supports both "min" and "max" modes
- `MetricsLogger(output_dir, fields)` — logs epoch summaries to console and `metrics.csv` with elapsed time
- `save_best_and_last(model, output_dir, metric, best_metric, metric_name)` — saves `last_model.pth` always, `best_model.pth` when metric improves; direction determined by "loss" in metric_name
- `make_loader(dataset, batch_size, shuffle, device, num_workers)` — creates DataLoader with pin_memory and persistent_workers set correctly per device

**`training/datasets.py`** — Dataset classes rewritten from `segmentation/dataset.py`:
- `BinaryMaskDataset` — COCO JSON + images → (image [3,128,128], mask [1,128,128]) float32
- `CropDataset` — COCO JSON + images → (image, target_dict) for Mask R-CNN style training
- `stratified_split` — per-camera stratified split returning (train_indices, val_indices)
- `apply_augmentation` — flips, rotation, brightness/contrast/HSV jitter
- `UNET_INPUT_SIZE = 128` constant

**`training/unet.py`** — U-Net training loop:
- `train_unet(data_dir, output_dir, ...)` — full training loop with differential LR, gradient clipping, early stopping, MetricsLogger, save_best_and_last
- Supports pre-split `train.json`/`val.json` or single `annotations.json` with stratified split
- Imports `UNetSegmentor` from `segmentation.model` for model construction
- Local `_dice_loss`, `_bce_dice_loss`, `_compute_val_iou` helpers

**`training/cli.py`** — CLI:
- `train_group` Click group registered as `aquapose train`
- `unet` subcommand with all required flags: `--data-dir`, `--output-dir`, `--epochs`, `--batch-size`, `--lr`, `--val-split`, `--patience`, `--device`, `--num-workers`

### CLI Integration
- `src/aquapose/cli.py`: `cli.add_command(train_group)` added so `aquapose train` is accessible from the main CLI

### Import Boundary Enforcement
- `tools/import_boundary_checker.py`: `"training"` added to `_LEGACY_COMPUTATION_DIRS` — pre-commit hook now rejects any `training/` → `aquapose.engine` or `aquapose.cli` import

### Tests (21 total)

**`tests/unit/training/test_training_cli.py`** (3 tests):
- CLI help shows "unet" subcommand
- `aquapose train unet --help` lists all 9 expected flags
- AST-based import boundary check verifies training/ modules don't import engine/ or cli/

**`tests/unit/training/test_common.py`** (18 tests):
- EarlyStopping: max/min modes, patience counting, patience=0 disables stopping, best tracking, invalid mode
- MetricsLogger: CSV header creation, row appending, multi-row accumulation, console format
- save_best_and_last: always saves last, saves best on improvement, skips best when no improvement, loss minimization, output_dir creation, checkpoint loadability

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect test assertion logic**
- **Found during:** Task 2 test run
- **Issue:** `test_min_mode_patience_exceeded` asserted `not es.step(0.6)` but EarlyStopping correctly returns True when patience=1 and count reaches 1
- **Fix:** Changed assertion to `assert es.step(0.6)` with corrected docstring
- **Files modified:** `tests/unit/training/test_common.py`

**2. [Rule 3 - Blocking] Fixed ruff lint errors on commit**
- **Found during:** Task 1 commit (pre-commit hook)
- **Issues:** SIM108 (ternary instead of if-else), RUF005 (list unpacking), I001 (import sort), F401 (unused import)
- **Fix:** Applied ruff fixes manually and via `ruff check --fix`
- **Files modified:** `src/aquapose/training/common.py`, `tests/unit/training/test_common.py`

## Self-Check

Files created:
- `src/aquapose/training/__init__.py`: exists
- `src/aquapose/training/common.py`: exists
- `src/aquapose/training/datasets.py`: exists
- `src/aquapose/training/unet.py`: exists
- `src/aquapose/training/cli.py`: exists
- `tests/unit/training/__init__.py`: exists
- `tests/unit/training/test_training_cli.py`: exists
- `tests/unit/training/test_common.py`: exists

Commits:
- 5dda12d: feat(31-01): create training/ package scaffold
- aba2645: feat(31-01): implement U-Net training subcommand with tests

Tests: 21 passed, 0 failed
Lint: clean
Import boundary: OK — no violations found
