# Stack Research

**Domain:** Multi-view 3D fish pose estimation via direct triangulation with refractive ray casting
**Researched:** 2026-02-19 | **Updated:** 2026-02-28 (v2.2 Backends additions)
**Confidence:** HIGH (primary pipeline has no exotic build dependencies; all choices are stable PyPI packages)

> **Pivot note (2026-02-21):** AquaPose pivoted from analysis-by-synthesis (differentiable mesh rendering + Adam optimization) to a direct triangulation pipeline. The primary pipeline uses skeletonization, spline fitting, multi-view triangulation, and Levenberg-Marquardt refinement — no differentiable rendering. The old analysis-by-synthesis pipeline is shelved but retained.

> **v2.2 update (2026-02-28):** Added YOLO-OBB detection backend, keypoint regression head for midline extraction, training CLI infrastructure, and config system additions. All new additions use libraries already present in the dependency set — no new runtime dependencies are required.

---

## Critical Version Warning

**Post-pivot (2026-02-21):** The primary pipeline has **no PyTorch3D dependency**. PyTorch is used only for U-Net segmentation inference and YOLO detection — both work on any recent PyTorch version. **PyTorch can be upgraded freely.** The version-pinning constraint (PyTorch 2.4.1) applies only if you need to run the shelved analysis-by-synthesis pipeline.

---

## v2.2 Backends: New Stack Requirements

This section covers what is needed specifically for the v2.2 milestone. All items use libraries already in the dependency set unless explicitly noted.

### YOLO-OBB Detection Backend

**No new dependencies required.** OBB support was added to ultralytics in v8.1.0 (released early 2024) and the project already requires `ultralytics>=8.0`. The project should pin to `ultralytics>=8.1` to guarantee OBB availability.

| Capability | How It Works | API Surface |
|------------|--------------|-------------|
| OBB inference | `YOLO("yolov8n-obb.pt")` model variant | `result.obb.xywhr` (tensor: cx, cy, w, h, angle_rad), `result.obb.xyxyxyxy` (4-corner polygon), `result.obb.conf`, `result.obb.cls` |
| OBB training | Same `model.train(data=..., epochs=...)` API as standard YOLO | Label format: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (4 corner points, normalized 0-1) |
| Affine crop extraction | OpenCV `cv2.getRotationMatrix2D` + `cv2.warpAffine` | Already available via `opencv-python>=4.8` in dependencies |
| Angle representation | `xywhr` angle is in **radians**, range `[-pi/4, 3*pi/4)` | Convert to degrees with `angle_rad * 180 / pi` for display |

**OBB label format for custom annotation:**
```
# fish_obb.txt (one line per fish per frame)
0 0.512 0.341 0.548 0.312 0.564 0.401 0.528 0.430
```
Each row: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (clockwise from top-left, normalized).

**Dataset YAML for OBB training:**
```yaml
path: /path/to/fish_obb_dataset
train: images/train
val: images/val
nc: 1
names: [fish]
```

**Why OBB over standard bbox for fish:** Fish midlines are elongated and oriented — OBB provides the heading angle directly from the detector, enabling affine-aligned crops that orient the fish consistently for the keypoint regression head downstream.

**Integration point:** OBB angle feeds directly into `CropRegion` or a new `OBBCropRegion` dataclass. The rotation angle is used in `cv2.warpAffine` to produce upright fish crops before passing to the keypoint regression U-Net.

**Version pinning:** Update `pyproject.toml` from `ultralytics>=8.0` to `ultralytics>=8.1`.

---

### Keypoint Regression Head (U-Net + Regression)

**No new dependencies required.** The keypoint head is a pure PyTorch module built on top of the existing U-Net encoder.

#### Architecture Decision: Direct Coordinate Regression vs Heatmap

For this project, **direct coordinate regression with a lightweight MLP head** is recommended over heatmap regression. Rationale:

- Fish midline has N ordered points along a 1D curve — spatial ordering is known. Heatmaps solve an unordered detection problem (find where each keypoint is). Ordered regression is simpler.
- Input crops are 128x128 (already fixed size) — heatmap resolution at this size is already coarse (4x4 at bottleneck). A small regression head on bottleneck features is sufficient.
- N_SAMPLE_POINTS is configurable (typically 10-20 points) — directly regress `(x, y, confidence)` per point = `3 * N` output values.
- Heatmap approach adds post-processing (argmax, subpixel refinement, Gaussian target generation) for no benefit on ordered curve data.

| Architecture | When to Use | Why Not Here |
|-------------|-------------|--------------|
| Heatmap regression | Unordered keypoints, large spatial resolution, detection task | Adds Gaussian target generation, argmax decoding — unnecessary for ordered midline points |
| Direct coordinate regression (MLP head) | Ordered curve, small crop, known point count | Simple, efficient, directly produces `(x, y, conf)` per point |
| SimCC (classification-based) | Low resolution inputs needing subpixel accuracy | Adds complexity not justified for 128x128 crops |

**Recommended keypoint head implementation** (builds on existing `_UNet`):

```python
# Added as a new nn.Module — no new dependencies
class _KeypointHead(nn.Module):
    """Regresses N ordered midline keypoints from U-Net bottleneck features.

    Output: (B, N, 3) where last dim is (x_norm, y_norm, confidence).
    x_norm, y_norm are in [0, 1] (crop-relative, sigmoid activated).
    confidence is in [0, 1] (sigmoid activated, indicates point visibility).
    """
    def __init__(self, in_channels: int, n_points: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(4)  # 96ch -> (B, 96, 4, 4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_points * 3),  # (x, y, conf) per point
        )
        self.n_points = n_points

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        x = self.pool(bottleneck)
        x = self.head(x)
        x = x.reshape(-1, self.n_points, 3)
        return torch.cat([torch.sigmoid(x[..., :2]), torch.sigmoid(x[..., 2:3])], dim=-1)
```

**Loss function for keypoint training:**

Use Wing Loss (smooth near zero, log far from zero) or MSE. Wing Loss from literature is better for keypoint regression because it provides finer gradients near the true location. Wing Loss is a ~10-line custom implementation — no additional library needed.

Alternatively, standard `F.mse_loss` weighted by confidence ground truth is sufficient for a first pass.

**Pseudo-labels for keypoint training:** Existing skeletonization pipeline (`skimage.morphology.skeletonize` + arc-length resampling) already generates ordered N-point midlines from binary masks. Use these as keypoint supervision targets — no new annotation tool needed.

**Partial midline handling:** Points with visibility confidence below threshold are masked out of the loss. The existing `BinaryMaskDataset` pattern can be extended with a `KeypointDataset` that yields `(image, keypoints, visibility_mask)` tuples using the same COCO-style JSON format with keypoint annotations.

---

### Training CLI Infrastructure

**No new dependencies required.** Click is already in the dependency set (`click>=8.1`). The pattern follows the existing `aquapose run` CLI.

#### Recommended Pattern: Subcommand Group Under Existing CLI

Add a `train` command group to the existing `cli` group in `src/aquapose/cli.py`:

```python
@cli.group()
def train() -> None:
    """Training commands for detection and segmentation models."""

@train.command("segmentation")
@click.option("--coco-json", required=True, type=click.Path(exists=True))
@click.option("--image-root", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--epochs", default=100)
@click.option("--batch-size", default=8)
@click.option("--lr", default=1e-4)
@click.option("--device", default=None)
def train_segmentation(...) -> None:
    """Train U-Net binary segmentation model."""
    ...

@train.command("keypoints")
# similar options + --n-points
def train_keypoints(...) -> None:
    """Train U-Net + keypoint regression head."""
    ...

@train.command("detection")
@click.option("--data-yaml", required=True, type=click.Path(exists=True))
@click.option("--model", default="yolov8n.pt")
@click.option("--epochs", default=100)
@click.option("--imgsz", default=640)
@click.option("--obb", is_flag=True, default=False, help="Use OBB model variant.")
def train_detection(...) -> None:
    """Train YOLO detection model (standard or OBB)."""
    ...
```

This produces the CLI surface:
- `aquapose train segmentation --coco-json ... --image-root ... --output-dir ...`
- `aquapose train keypoints --coco-json ... --n-points 15 --output-dir ...`
- `aquapose train detection --data-yaml ... --obb --epochs 100`

**Why subcommands under existing `cli` group vs a separate entrypoint:** The existing `aquapose` entrypoint is registered in `pyproject.toml` under `[project.scripts]`. Adding training as a subcommand reuses the entrypoint and keeps all AquaPose functionality under one namespace. No `pyproject.toml` changes needed.

**Progress reporting:** The existing training loop in `src/aquapose/segmentation/training.py` uses `print()`. For the CLI training commands, wrap the epoch loop with `tqdm` for progress display. `tqdm` is a transitive dependency of `ultralytics` and will already be available in the environment. Do not add it as an explicit dependency — just import it.

```python
# tqdm is available via ultralytics transitive dependency
from tqdm import tqdm

for epoch in tqdm(range(epochs), desc="Training"):
    ...
```

If `tqdm` import fails (edge case), fall back to `print()`. Do not add `tqdm` to `pyproject.toml` dependencies — it is a transitive dependency and making it explicit creates a pinning conflict risk.

---

### Config System: N_SAMPLE_POINTS and Device Propagation

**No new dependencies required.** The existing frozen dataclass config system handles this cleanly.

#### N_SAMPLE_POINTS Propagation

The current pipeline has `N_SAMPLE_POINTS` scattered as module-level constants (e.g., in reconstruction and midline stages). The fix is to:

1. Add `n_sample_points: int = 20` to the appropriate nested config dataclass (likely `MidlineConfig` or a new `ReconstructionConfig`).
2. Propagate it from `PipelineConfig` down through `PipelineContext` to both the Midline stage (arc-length resampling) and the keypoint regression head (model instantiation).
3. The keypoint head's `n_points` parameter must match `n_sample_points` at model load time.

No library changes needed — this is a config plumbing refactor within the existing frozen dataclass system.

#### Device Propagation

Current pattern: stages auto-detect device (`"cuda" if torch.cuda.is_available() else "cpu"`). Problem: inconsistent when multiple stages need the same device or when the user wants CPU-only for testing.

Add `device: str = "auto"` to `PipelineConfig`. At pipeline initialization, resolve `"auto"` to `"cuda"` or `"cpu"` once and store the resolved value in `PipelineContext`. All stages read `context.device` instead of calling `torch.cuda.is_available()` themselves.

No library changes needed.

---

## Recommended Stack (Full Picture)

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11 | Runtime | Project targets >=3.11; stable, widely supported. |
| PyTorch | latest stable | U-Net inference, keypoint head, YOLO detection, tensor ops | No longer pinned to 2.4.1. Primary pipeline has no PyTorch3D coupling. |
| torchvision | match PyTorch | MobileNetV3-Small backbone for U-Net encoder | Follows PyTorch version; needed for image transforms and pretrained weights. |
| CUDA | 12.x | GPU acceleration | U-Net, keypoint head, and YOLO inference only. No PyTorch3D CUDA ops in primary pipeline. |
| scikit-image | >=0.22 | Skeletonization, morphology, distance transform | Core of reconstruction: `skeletonize` for midlines, `distance_transform_edt` for width profiles. Also generates keypoint pseudo-labels. |
| scipy | >=1.13 | Spline fitting, LM refinement, Hungarian assignment | `splprep`/`splev` for midline splines, `least_squares` for refractive triangulation refinement. |
| numpy | >=1.24 | Array operations, calibration interface | Foundation for all non-tensor computation. AquaCal is numpy-based. |

### Perception Pipeline Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| opencv-python | >=4.8 | MOG2 background subtraction, video I/O, affine crop extraction | Detection (Stage 1), video frame reading, OBB-aligned crop extraction via `cv2.warpAffine`. |
| ultralytics | **>=8.1** | Primary fish detector (standard bbox or OBB) | Standard detection: `yolov8n.pt`. OBB detection: `yolov8n-obb.pt`. Version bump from `>=8.0` to `>=8.1` required for OBB support. |
| U-Net + Keypoint Head (custom) | — | Binary mask segmentation + ordered midline keypoint regression | Shared MobileNetV3-Small encoder. Two heads: segmentation (existing) and keypoint regression (new in v2.2). Weights stored separately per head. |
| sam2 (segment-anything-2) | latest from source | Zero-shot pseudo-label generation via box prompts | Offline annotation tool only — not deployed at inference time. Also generates keypoint supervision via skeletonization of SAM2 masks. |

### Reconstruction Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-image | >=0.22 | Skeletonize masks, morphological cleanup, distance-based width profiles | Every frame: extract midline skeleton from binary mask, compute width via `distance_transform_edt`, prune skeleton branches. Also generates keypoint training targets via arc-length resampling. |
| scipy | >=1.13 | Spline fitting, LM refinement, Hungarian tracking | Every frame: fit smooth spline to skeleton points, refine 3D triangulation via Levenberg-Marquardt, associate fish across frames. |

### Tracking and Association Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| boxmot | >=11.0 | OC-SORT per-camera 2D tracking | 2D Tracking stage. Replaces Hungarian-only tracker. Robust to occlusion via virtual trajectories. |
| leidenalg | >=0.10 | Leiden graph clustering for cross-camera association | Association stage. Graph-based with must-not-link constraints. |
| igraph | >=0.11 | Graph construction for Leiden clustering | Dependency of leidenalg. |

### Storage and Visualization Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| h5py | >=3.9 | HDF5 trajectory data storage | Primary output format. Efficient random access to per-fish, per-frame data. |
| plotly | >=5.18 | 3D midline animation | 3D visualization observer. |
| click | >=8.1 | CLI entrypoint and training subcommands | All CLI surface: `aquapose run`, `aquapose train segmentation`, `aquapose train keypoints`, `aquapose train detection`. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| hatch | Project management, virtual envs, test/lint/typecheck scripts | Already configured in `pyproject.toml`. Use `hatch run test` / `hatch run lint` throughout. |
| ruff | Linting and formatting | Already configured. Line length 88, targets py311. |
| basedpyright | Type checking | Already configured in basic mode. |
| pytest | Testing | Already configured with `slow` and `e2e` markers. |

---

## v2.2 Dependency Changes

### pyproject.toml Updates Required

```toml
# CHANGE: bump ultralytics lower bound from >=8.0 to >=8.1
# OBB support was added in ultralytics 8.1.0 (January 2024)
"ultralytics>=8.1",  # was >=8.0
```

No other dependency additions required. All new capabilities (OBB inference, keypoint head, training CLI, config params) use:
- `torch` and `torchvision` (already required)
- `ultralytics` (already required, just needs version bump)
- `opencv-python` (already required)
- `click` (already required)
- `scikit-image` (already required, for keypoint pseudo-label generation)
- `tqdm` (transitive via ultralytics — do not add explicitly)

### What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `tqdm` as explicit dependency | Already available as ultralytics transitive dep; explicit dep risks version pinning conflicts | Import directly; fall back to print if unavailable |
| `albumentations` for training augmentation | Heavy dep with many transitive requirements; existing augmentation uses `torchvision.transforms.v2` and OpenCV | `torchvision.transforms.v2` (already available) for geometric augmentation |
| `timm` for alternative encoder backbones | Adds 100+ MB of model weights and registry boilerplate; MobileNetV3-Small via torchvision is sufficient | `torchvision.models.mobilenet_v3_small` (already used) |
| `mmdet` / `mmpose` | Heavyweight frameworks with complex installation (MMCV build issues on Windows); kill the install simplicity that is a project strength | Direct ultralytics API for OBB; custom head for keypoints |
| `segmentation_models_pytorch` | Replaces custom U-Net; adds dependency without architectural benefit for this specific use case | Custom `_UNet` in `src/aquapose/segmentation/model.py` (already exists) |
| `lightning` / `pytorch-lightning` | Adds trainer abstraction overhead for simple training loops; the existing `training.py` pattern is sufficient | Bare PyTorch training loop (existing pattern) |
| Pydantic for config | Already decided against in v2.0; frozen dataclasses are simpler and stdlib-only | Frozen dataclasses with `dataclasses.asdict()` (existing) |

---

## Shelved Pipeline Dependencies

The following libraries are required **only** for the shelved analysis-by-synthesis pipeline. Not needed for v2.2.

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| PyTorch | **2.4.1** (pinned) | Autograd for differentiable rendering | Must be pinned to 2.4.1 for PyTorch3D 0.7.9 compatibility. |
| torchvision | 0.19.1 | Paired with PyTorch 2.4.1 | Must match PyTorch version exactly. |
| CUDA | 12.1 | GPU acceleration for PyTorch3D CUDA ops | Supported by PyTorch 2.4.1. |
| PyTorch3D | 0.7.9 (source) | Differentiable silhouette rendering, mesh rasterization | Install from source against PyTorch 2.4.1. |
| kornia | >=0.7 | Differentiable IoU loss, image morphological ops | Pure PyTorch, no CUDA extension compile needed. |

---

## Installation

### Primary Pipeline (unchanged from v2.1)

```bash
# Using hatch (preferred):
pip install hatch
hatch env create

# Or: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Then: pip install -e .

# SAM2 for pseudo-label generation (offline annotation only)
git clone https://github.com/facebookresearch/sam2.git
pip install -e sam2/
```

### v2.2 Additions (no new install steps)

OBB model variant downloads automatically on first use:
```python
from ultralytics import YOLO
model = YOLO("yolov8n-obb.pt")  # downloads ~6MB on first call
```

No additional installation required for keypoint head, training CLI, or config changes.

---

## Alternatives Considered (v2.2 scope)

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| YOLO-OBB via ultralytics (same library) | DOTA-pretrained YOLO-OBB fine-tuned from scratch | Already have detection training data; ultralytics OBB fine-tuning is direct extension of existing YOLO workflow |
| Direct coordinate regression (custom MLP head) | Heatmap regression (e.g., ViTPose, HRNet style) | Ordered curve points don't benefit from heatmap representation; adds post-processing complexity for no accuracy gain on 128x128 crops |
| SimCC (coordinate classification) | — | Subpixel accuracy improvement marginal at 128x128; adds complexity |
| Click subcommand group (`aquapose train ...`) | Separate `aquapose-train` entrypoint in pyproject.toml | Single entrypoint is cleaner; no pyproject.toml changes; consistent with existing CLI pattern |
| Bare PyTorch training loop | PyTorch Lightning | Lightning adds abstraction overhead not justified for simple training loops without distributed training |

---

## Stack Patterns by Variant

**If OBB angle is needed for affine crop alignment:**
- Access via `result.obb.xywhr[:, 4]` (angle in radians)
- Convert to degrees: `angle_deg = float(result.obb.xywhr[0, 4]) * 180 / math.pi`
- Apply with `cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)` then `cv2.warpAffine`

**If keypoint head quality is insufficient after initial training:**
- Increase `n_sample_points` supervision diversity by using multiple augmented skeleton passes
- Consider a larger bottleneck: add a 1x1 conv to project 96ch bottleneck to 256ch before the regression head
- Do NOT switch to heatmaps — the crop size (128x128) is too small for heatmap decoding quality to improve on direct regression

**If training CLI is too slow for interactive use:**
- `--epochs 5 --batch-size 32` for quick smoke-test runs
- Use `hatch run python -m aquapose.training.segmentation` as an escape hatch (no Click overhead)

**If N_SAMPLE_POINTS changes at inference time (config change):**
- Keypoint head weights are tied to the `n_points` the model was trained with — changing `n_sample_points` requires retraining the keypoint head
- Skeletonization-based midline can always be resampled to any N at inference time without retraining
- Document this constraint clearly in the config YAML comment

**For Windows development (all v2.2 additions):**
- OBB inference: no issues — ultralytics installs cleanly on Windows
- Keypoint head: no issues — pure PyTorch
- Training CLI: no issues — Click works identically on Windows
- `cv2.warpAffine` for affine crops: no issues

---

## Version Compatibility

### Primary Pipeline (v2.2)

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| PyTorch (latest) | torchvision (matching) | Standard pairing; no special constraints |
| PyTorch (latest) | ultralytics >=8.1 | ultralytics tracks PyTorch releases; OBB requires >=8.1 |
| ultralytics >=8.1 | YOLO-OBB models | OBB introduced in 8.1.0; xywhr angle in radians |
| scikit-image >=0.22 | numpy >=1.24 | scikit-image 0.22+ supports numpy 2.x |
| scipy >=1.11 | numpy >=1.24 | scipy 1.11+ supports numpy 2.x |
| Python 3.11+ | All listed libraries | Safe baseline; 3.12 also works for primary pipeline |

---

## Sources

- Ultralytics OBB docs — tasks, format, inference API: https://docs.ultralytics.com/tasks/obb/
- Ultralytics OBB datasets overview — label format (4-corner): https://docs.ultralytics.com/datasets/obb/
- Ultralytics v8.1.0 release discussion — OBB introduction: https://github.com/orgs/ultralytics/discussions/7472
- PyTorch torchvision transforms v2 — affine/rotated box support: https://docs.pytorch.org/vision/stable/transforms.html
- scikit-image morphology docs — skeletonize, distance_transform_edt: https://scikit-image.org/docs/stable/api/skimage.morphology.html
- scipy interpolate docs — splprep, splev for spline fitting: https://docs.scipy.org/doc/scipy/reference/interpolate.html
- Click docs — command groups and subcommands: https://click.palletsprojects.com/en/stable/commands/
- SimCC paper (ECCV 2022) — coordinate classification vs heatmap: https://github.com/leeyegy/SimCC
- OpenCV releases — v4.13.0: https://opencv.org/releases/
- PyTorch official get-started — current stable: https://pytorch.org/get-started/locally/

---

*Stack research for: 3D fish pose estimation via direct triangulation with refractive ray casting*
*Researched: 2026-02-19 | Updated: 2026-02-28 (v2.2 Backends additions)*
