---
phase: 11-add-training-visualizations
plan: "01"
type: quick
subsystem: training
tags: [visualization, training, unet, pose, diagnostics]
dependency_graph:
  requires: []
  provides: [training-visualization-grids]
  affects: [train_unet, train_pose]
tech_stack:
  added: [matplotlib-agg]
  patterns: [try-except-viz-calls, lazy-cv2-import, torch-nn-module-type]
key_files:
  created:
    - src/aquapose/training/viz.py
  modified:
    - src/aquapose/training/unet.py
    - src/aquapose/training/pose.py
decisions:
  - "Use torch.nn.Module as model parameter type in val grid functions — avoids object-typed typecheck errors while remaining compatible with _UNet and _PoseModel"
  - "matplotlib.use('Agg') at module top for headless compatibility; cv2 lazily imported inside functions to avoid import overhead"
  - "plt.colormaps.get_cmap() used instead of deprecated plt.cm.get_cmap() (deprecated Matplotlib 3.7+)"
  - "Wrap all viz calls in try/except in train_unet/train_pose — visualization failures never crash training"
metrics:
  duration: "7 min"
  completed: "2026-03-01"
  tasks_completed: 2
  files_changed: 3
---

# Quick Task 11: Add Training Visualizations (Augmented Data + Val Predictions)

**One-liner:** Four matplotlib grid functions in viz.py — augmented data with GT overlays (masks/keypoints) and validation predictions with GT-vs-pred comparisons — wired into train_unet and train_pose with try/except guards.

## What Was Built

### viz.py (new module)

Four public functions in `src/aquapose/training/viz.py`:

- **`save_unet_augmented_grid(dataset, output_path, n_samples=16)`** — Samples the augmented BinaryMaskDataset, renders each image with GT mask overlaid as semi-transparent red fill + contour. Saved as `augmented_data_grid.png`.

- **`save_unet_val_grid(model, val_dataset, device, output_path, n_samples=16)`** — Runs the trained _UNet on validation samples, draws GT mask contours in green and predicted mask contours in red on each image cell. Saved as `val_predictions_grid.png`.

- **`save_pose_augmented_grid(dataset, output_path, n_samples=16, input_size=(128,64))`** — Samples the KeypointDataset, de-normalizes keypoints, renders visible as colored circles (tab10 colormap) and invisible as gray X marks. Saved as `augmented_data_grid.png`.

- **`save_pose_val_grid(model, val_dataset, device, output_path, n_samples=16, input_size=(128,64))`** — Runs _PoseModel on val samples, overlays GT keypoints in green and predicted in red with yellow connecting lines showing error magnitude. Saved as `val_predictions_grid.png`.

All functions use `torch.manual_seed(0)` + `random.sample` for reproducible index selection and `plt.close(fig)` to prevent memory leaks.

### train_unet wiring

```python
# Before model construction (after loaders):
try:
    save_unet_augmented_grid(train_dataset, output_dir / "augmented_data_grid.png")
except Exception as exc:
    print(f"[viz] Warning: augmented data grid failed: {exc}")

# After training loop (before return):
try:
    best_state = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    save_unet_val_grid(model, val_dataset, device, output_dir / "val_predictions_grid.png")
except Exception as exc:
    print(f"[viz] Warning: val predictions grid failed: {exc}")
```

### train_pose wiring

Same pattern: augmented grid before model construction, val grid after training loop with best weights loaded.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1: Create viz.py | c9504e3 | feat(11-01): add training visualization grid utilities |
| Task 2: Wire into trainers | f062bdc | feat(11-02): wire viz calls into train_unet and train_pose |

## Deviations from Plan

**1. [Rule 1 - Bug] Removed deprecated plt.cm.get_cmap() call**
- **Found during:** Task 1 verification (test run showed MatplotlibDeprecationWarning)
- **Issue:** `plt.cm.get_cmap()` was deprecated in Matplotlib 3.7 and scheduled for removal in 3.11
- **Fix:** Replaced with `plt.colormaps.get_cmap("tab10")`
- **Files modified:** src/aquapose/training/viz.py

**2. [Rule 2 - Type safety] Changed model parameter type from object to torch.nn.Module**
- **Found during:** Task 1 verification (basedpyright flagged .eval() and __call__ on object)
- **Issue:** `model: object` caused typecheck errors for .eval() and model(inp) calls
- **Fix:** Changed to `model: torch.nn.Module` in save_unet_val_grid and save_pose_val_grid
- **Files modified:** src/aquapose/training/viz.py

## Self-Check

**Files exist:**
- FOUND: src/aquapose/training/viz.py
- FOUND: src/aquapose/training/unet.py (modified)
- FOUND: src/aquapose/training/pose.py (modified)

**Commits exist:**
- FOUND: c9504e3 — feat(11-01): add training visualization grid utilities
- FOUND: f062bdc — feat(11-02): wire viz calls into train_unet and train_pose

**Import check:** `from aquapose.training.viz import save_unet_augmented_grid, save_unet_val_grid, save_pose_augmented_grid, save_pose_val_grid` — OK

**Test suite:** 690 passed, 0 failures

## Self-Check: PASSED
