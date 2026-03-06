---
phase: 11-add-training-visualizations
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/training/viz.py
  - src/aquapose/training/unet.py
  - src/aquapose/training/pose.py
  - src/aquapose/training/__init__.py
autonomous: true
requirements: [VIZ-01]

must_haves:
  truths:
    - "train_unet saves an augmented data grid image to output_dir before training starts"
    - "train_unet saves a validation predictions grid image to output_dir after training completes"
    - "train_pose saves an augmented data grid image to output_dir before training starts"
    - "train_pose saves a validation predictions grid image to output_dir after training completes"
    - "Augmented grid shows images with GT overlays so user can verify augmentation correctness"
    - "Validation grid shows GT vs predicted overlays so user can judge model performance"
  artifacts:
    - path: "src/aquapose/training/viz.py"
      provides: "Shared visualization grid utilities for both training pipelines"
      exports: ["save_unet_augmented_grid", "save_unet_val_grid", "save_pose_augmented_grid", "save_pose_val_grid"]
  key_links:
    - from: "src/aquapose/training/unet.py"
      to: "src/aquapose/training/viz.py"
      via: "import and call at start/end of train_unet"
      pattern: "save_unet_augmented_grid|save_unet_val_grid"
    - from: "src/aquapose/training/pose.py"
      to: "src/aquapose/training/viz.py"
      via: "import and call at start/end of train_pose"
      pattern: "save_pose_augmented_grid|save_pose_val_grid"
---

<objective>
Add training visualizations to both `train_unet` and `train_pose` so they always save diagnostic grids to `output_dir`:

(a) **Augmented data grid** — saved before training starts. Shows a grid of ~16 samples from the augmented training dataset with GT overlays (masks for U-Net, keypoints for pose) so the user can confirm augmentations are treating labels correctly.

(b) **Validation prediction grid** — saved after training completes (using best model weights). Shows a grid of ~16 validation samples with GT vs predicted overlays so the user can judge model quality.

Purpose: Give the user immediate visual feedback on data quality and model performance without needing separate scripts.
Output: New `viz.py` module, modified `train_unet` and `train_pose` with visualization calls.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/training/unet.py (U-Net training loop — insert viz calls)
@src/aquapose/training/pose.py (Pose training loop — insert viz calls)
@src/aquapose/training/datasets.py (BinaryMaskDataset, KeypointDataset definitions)
@src/aquapose/segmentation/model.py (_UNet model, UNET_INPUT_SIZE)
@src/aquapose/training/common.py (shared training utilities)

<interfaces>
From src/aquapose/training/datasets.py:
```python
class BinaryMaskDataset(_CocoDataset):
    # Returns (image_tensor [3,H,W] float32 [0,1], mask_tensor [1,H,W] float32 {0,1})
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
```

From src/aquapose/training/pose.py:
```python
class KeypointDataset(Dataset):
    # Returns (image_tensor [3,H,W] float32 [0,1], kp_flat [n_kp*2] float32 [0,1], visibility [n_kp] bool)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class _PoseModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns (B, n_keypoints*2) in [0,1]. Even=x, odd=y.
        ...

_INPUT_SIZE: tuple[int, int] = (128, 64)  # (width, height)
```

From src/aquapose/training/unet.py:
```python
def train_unet(data_dir, output_dir, ...) -> Path:
    # Uses BinaryMaskDataset, _UNet model
    # val_iou computed, best model saved to output_dir/best_model.pth
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create viz.py with grid visualization functions</name>
  <files>src/aquapose/training/viz.py, src/aquapose/training/__init__.py</files>
  <action>
Create `src/aquapose/training/viz.py` with four public functions. All use matplotlib to compose grids and save as PNG files. Use `matplotlib.use("Agg")` at module top to avoid display backend issues.

**`save_unet_augmented_grid(dataset, output_path, n_samples=16)`**
- Takes a BinaryMaskDataset (augmented) and samples `n_samples` random indices
- For each sample: get `(image, mask)` tuple from dataset
- Compose a grid (4x4 for 16 samples): show the image with GT mask overlaid as a semi-transparent colored overlay (e.g. red at alpha=0.4)
- Save to `output_path` (e.g. `output_dir / "augmented_data_grid.png"`)

**`save_unet_val_grid(model, val_dataset, device, output_path, n_samples=16)`**
- Takes the trained _UNet model (already loaded with best weights), a val dataset, and device
- Samples `n_samples` random indices from val_dataset
- For each sample: run inference, get predicted mask (threshold 0.5)
- Compose a grid with 2 rows per sample (or side-by-side): top shows image + GT mask overlay (green), bottom shows image + predicted mask overlay (red). Use a 4-column layout: each column is one sample, 2 rows (GT row, pred row) -> 8 rows total for 16 samples, or simpler: 4x4 grid where each cell shows image with GT contour (green) and predicted contour (red) overlaid
- Simpler approach preferred: single grid, each cell has image with GT mask contour in green and predicted mask contour in red, so differences are immediately visible
- Save to `output_path` (e.g. `output_dir / "val_predictions_grid.png"`)

**`save_pose_augmented_grid(dataset, output_path, n_samples=16, input_size=(128, 64))`**
- Takes a KeypointDataset (augmented) and samples `n_samples` random indices
- For each sample: get `(image, kp_flat, visibility)`. Denormalize keypoints: `x_px = kp[2*k] * width`, `y_px = kp[2*k+1] * height`
- Compose grid: show image with visible keypoints as colored circles (e.g. matplotlib scatter), invisible ones as gray X marks
- Save to `output_path`

**`save_pose_val_grid(model, val_dataset, device, output_path, n_samples=16, input_size=(128, 64))`**
- Takes trained _PoseModel, val dataset, device
- For each sample: run inference, get predicted keypoints
- Each cell: image with GT keypoints as green circles and predicted keypoints as red circles, connected by thin lines showing error magnitude. Only show visible GT keypoints.
- Save to `output_path`

Implementation notes:
- Use `torch.manual_seed(0)` + `random.sample` for reproducible sample selection
- Handle datasets wrapped in `Subset` or `ConcatDataset` by accessing via integer index (they support `__getitem__`)
- Use `plt.subplots(nrows, ncols, figsize=...)` with `fig.savefig(output_path, dpi=150, bbox_inches="tight")`
- Call `plt.close(fig)` after saving to prevent memory leaks
- All tensor-to-numpy conversions via `.cpu().numpy()` (per project convention)
- Images from datasets are already float32 [0,1] with shape (3,H,W) — transpose to (H,W,3) for matplotlib
- For mask contours, use `cv2.findContours` on the binary mask and `cv2.drawContours` on the image copy

Update `src/aquapose/training/__init__.py`: do NOT add viz functions to __init__.py or __all__ — these are internal helpers called only from train_unet/train_pose, not public API.
  </action>
  <verify>
    `python -c "from aquapose.training.viz import save_unet_augmented_grid, save_unet_val_grid, save_pose_augmented_grid, save_pose_val_grid; print('imports ok')"` succeeds.
    `hatch run check` passes (lint + typecheck).
  </verify>
  <done>viz.py exists with all four grid functions, imports cleanly, passes lint and typecheck.</done>
</task>

<task type="auto">
  <name>Task 2: Wire visualization calls into train_unet and train_pose</name>
  <files>src/aquapose/training/unet.py, src/aquapose/training/pose.py</files>
  <action>
**In `train_unet` (src/aquapose/training/unet.py):**

1. Add import at top: `from .viz import save_unet_augmented_grid, save_unet_val_grid`

2. After `train_loader` and `val_loader` are created (around line 176), before model construction, add:
```python
# Save augmented training data grid for visual inspection
save_unet_augmented_grid(train_dataset, output_dir / "augmented_data_grid.png")
```

3. After the training loop ends (after the early stopping break / loop completion, around line 265), but before the final `return best_model_path`, add:
```python
# Save validation predictions grid using best model
best_state = torch.load(best_model_path, map_location=device, weights_only=True)
model.load_state_dict(best_state)
save_unet_val_grid(model, val_dataset, device, output_dir / "val_predictions_grid.png")
```

**In `train_pose` (src/aquapose/training/pose.py):**

1. Add import at top: `from .viz import save_pose_augmented_grid, save_pose_val_grid`

2. After `val_loader` is created (around line 479), before model construction, add:
```python
# Save augmented training data grid for visual inspection
save_pose_augmented_grid(train_dataset, output_dir / "augmented_data_grid.png", input_size=input_size)
```

3. After the training loop ends (after early stopping break / loop completion, around line 575), before final `return best_model_path`, add:
```python
# Save validation predictions grid using best model
best_state = torch.load(best_model_path, map_location=device, weights_only=True)
model.load_state_dict(best_state)
save_pose_val_grid(model, val_dataset, device, output_dir / "val_predictions_grid.png", input_size=input_size)
```

Important: Wrap each viz call in try/except (catch Exception, print warning, continue) so visualization failures never crash training. Training results are more important than diagnostic images.
  </action>
  <verify>
    `hatch run check` passes (lint + typecheck).
    `hatch run test` passes (existing tests still work — viz calls won't fire in unit tests since they don't run full training).
  </verify>
  <done>Both train_unet and train_pose call the viz functions: augmented grid before training, val predictions grid after training. Failures are caught and logged as warnings without interrupting training.</done>
</task>

</tasks>

<verification>
- `hatch run check` passes (lint + typecheck)
- `hatch run test` passes (no regressions)
- Manual verification: run a short training (e.g. 2 epochs) and confirm PNG files appear in output_dir
</verification>

<success_criteria>
- `train_unet` produces `augmented_data_grid.png` and `val_predictions_grid.png` in output_dir
- `train_pose` produces `augmented_data_grid.png` and `val_predictions_grid.png` in output_dir
- Augmented grids show images with GT overlays (masks or keypoints)
- Validation grids show GT vs predicted overlays for quality assessment
- No CLI toggle needed — always runs
- Visualization failures do not crash training
</success_criteria>

<output>
After completion, create `.planning/quick/11-add-training-visualizations-augmented-da/11-SUMMARY.md`
</output>
