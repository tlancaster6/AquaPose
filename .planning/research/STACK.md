# Stack Research

**Domain:** Multi-view 3D fish pose estimation via direct triangulation with refractive ray casting
**Researched:** 2026-02-19 | **Updated:** 2026-03-01 (v3.0 Ultralytics Unification additions)
**Confidence:** HIGH (primary pipeline has no exotic build dependencies; all choices are stable PyPI packages)

> **Pivot note (2026-02-21):** AquaPose pivoted from analysis-by-synthesis (differentiable mesh rendering + Adam optimization) to a direct triangulation pipeline. The primary pipeline uses skeletonization, spline fitting, multi-view triangulation, and Levenberg-Marquardt refinement — no differentiable rendering. The old analysis-by-synthesis pipeline is shelved but retained.

> **v2.2 update (2026-02-28):** Added YOLO-OBB detection backend, keypoint regression head for midline extraction, training CLI infrastructure, and config system additions. All new additions use libraries already present in the dependency set — no new runtime dependencies required.

> **v3.0 update (2026-03-01):** Replacing custom U-Net segmentation and custom keypoint regression with Ultralytics-native YOLO-seg and YOLO-pose models. The custom `_UNet`, `_KeypointHead`, and `BinaryMaskDataset` are deleted. Ultralytics already in the stack — this is a training workflow and annotation format change, not a dependency change.

---

## Critical Version Warning

**Post-pivot (2026-02-21):** The primary pipeline has **no PyTorch3D dependency**. PyTorch is used only for YOLO-seg/pose inference and YOLO detection — both work on any recent PyTorch version. **PyTorch can be upgraded freely.** The version-pinning constraint (PyTorch 2.4.1) applies only if you need to run the shelved analysis-by-synthesis pipeline.

---

## v3.0 Ultralytics Unification: Stack Requirements

This milestone replaces both custom model types with Ultralytics-native equivalents. The change is primarily about **training workflows and annotation formats**, not new library dependencies.

### What Changes

| Component | Old (v2.2) | New (v3.0) | Reason |
|-----------|-----------|-----------|--------|
| Segmentation model | Custom `_UNet` (MobileNetV3 encoder) | `YOLO("yolo11n-seg.pt")` | IoU 0.623 insufficient; Ultralytics has pretrained COCO backbone and battle-tested training |
| Keypoint model | Custom `_KeypointHead` (MLP on bottleneck) | `YOLO("yolo11n-pose.pt")` | Poor performance even with augmentation; Ultralytics pose integrates detection + keypoints natively |
| Training infrastructure | `src/aquapose/segmentation/training.py` + bare PyTorch loop | `model.train(data="fish.yaml", epochs=N)` unified API | One training API for detect, seg, and pose — no custom training loop to maintain |
| Dataset format (seg) | COCO JSON with binary masks | YOLO seg polygon format (`.txt` per image) | Ultralytics native format |
| Dataset format (pose) | COCO JSON with keypoints | YOLO pose format (`.txt` per image with kpt_shape in YAML) | Ultralytics native format |
| Deleted code | `_UNet`, `_KeypointHead`, `BinaryMaskDataset`, `training.py` | None (stripped) | Clean removal; SAM2 pipeline kept for generating labels in new format |

### What Does NOT Change

- `ultralytics` is already a dependency (`ultralytics>=8.1` from v2.2)
- YOLO detection (`yolov8n.pt` or `yolov8n-obb.pt`) is unchanged
- SAM2 pseudo-label pipeline is kept — it generates the source masks that are converted to YOLO seg format
- PyTorch, torchvision, OpenCV, scikit-image — unchanged
- No new `pyproject.toml` dependencies required

---

### YOLO-Seg: Instance Segmentation Backend

**Model selection: YOLO11n-seg over YOLOv8n-seg**

YOLO11 (released September 2024) is the current recommended generation in the ultralytics package. It outperforms YOLOv8 at equal model size (higher mask mAP, faster inference), uses identical training API, same annotation format, and same `.pt` weight format. YOLOv8 variants remain fully supported but YOLO11 is the preferred starting point for new training runs.

For this project, start with `yolo11n-seg.pt` (nano, 2.7M params, ~53ms CPU). Upgrade to `yolo11s-seg.pt` if mask quality is insufficient — the nano backbone has proven sufficient for comparable fish-scale detection tasks.

**Annotation format — YOLO seg polygon:**

One `.txt` file per image, named to match the image file. One row per fish instance:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

- `class_id`: Integer class index (0 for fish — single class)
- `x1 y1 ... xn yn`: Polygon boundary coordinates, normalized to [0, 1] relative to image width/height
- Minimum 3 coordinate pairs (6 values) per instance
- Variable polygon length per row — fish body outlines typically use 20-60 points
- One empty `.txt` file for images with no fish (not missing — an empty file)

Example (two fish in one image):
```
0 0.312 0.421 0.324 0.408 0.341 0.399 0.389 0.401 0.412 0.418 0.398 0.432 0.312 0.421
0 0.621 0.234 0.638 0.219 0.655 0.228 0.671 0.247 0.659 0.261 0.638 0.258 0.621 0.234
```

**Known issue — SAM2 multi-region masks:** SAM2 sometimes produces disconnected mask regions for a single fish (e.g., fin separated from body). YOLO seg training expects one polygon per instance row. Multi-region SAM2 masks must be reduced to the single largest connected component before polygon extraction. This is a data preparation step, not a training configuration issue.

**Dataset YAML for seg training:**

```yaml
path: /path/to/fish_seg_dataset
train: images/train
val: images/val
nc: 1
names: [fish]
```

No additional seg-specific YAML fields required. The polygon format is self-describing.

**Training configuration:**

```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")  # pretrained COCO backbone, fine-tune
model.train(
    data="fish_seg.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=1e-3,
    weight_decay=0.0005,
    mosaic=1.0,          # keep enabled — multi-fish scenes benefit
    degrees=15.0,        # fish rotate freely; enable rotation augmentation
    flipud=0.5,          # fish can be upside-down (top-down camera)
    fliplr=0.5,
    scale=0.3,
    overlap_mask=True,   # default; handles overlapping fish masks
    mask_ratio=4,        # default mask downsample ratio
    device="0",          # GPU
)
```

**Key training settings for this domain:**

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| `imgsz` | 640 | Full camera frame (1600x1200) scaled to 640; fish are ~50px at this scale — small object regime |
| `mosaic` | 1.0 | Leave enabled — creates multi-fish training scenes; helps with multi-instance cases |
| `degrees` | 15.0 | Top-down camera, fish orient in all directions; rotation augmentation improves generalization |
| `flipud` | 0.5 | Top-down camera makes vertical flip a valid augmentation |
| `overlap_mask` | True | Fish overlap in crowded schools; keep enabled to handle mask merging |
| `close_mosaic` | 10 | Disable mosaic in last 10 epochs for stable fine-tuning convergence |

**Inference API:**

```python
from ultralytics import YOLO

model = YOLO("path/to/fish_seg_best.pt")
results = model(frame, verbose=False)

for r in results:
    if r.masks is not None:
        masks = r.masks.data        # (N, H, W) bool tensor — per-fish binary masks
        boxes = r.boxes.xyxy        # (N, 4) float tensor — bounding boxes
        confs = r.boxes.conf        # (N,) float tensor — detection confidence
```

This replaces the existing `UNetSegmentor` inference path. The output (`masks.data`) is a binary mask tensor matching the current `BinaryMaskDataset` contract — downstream skeletonization is unchanged.

---

### YOLO-Pose: Keypoint Midline Backend

**Model selection: YOLO11n-pose**

YOLO11n-pose (nano, 2.9M params) is the starting point. The pose task integrates bounding box detection and keypoint regression in one forward pass — it replaces both the separate detection step and the custom keypoint head. For fish with 6 anatomical keypoints (head, pectoral-left, pectoral-right, dorsal, caudal-base, tail-tip — or a project-specific set), the nano model is appropriate. Scale up to YOLO11s-pose if localization accuracy is insufficient.

**Custom keypoint count — kpt_shape configuration:**

YOLO pose models are not hard-coded to 17 human keypoints. Custom anatomy is specified via `kpt_shape` in the dataset YAML. The tiger-pose dataset (12 keypoints) and dog-pose dataset (24 keypoints) demonstrate this pattern — it is fully supported and documented.

For 6 midline keypoints (the project's planned anatomy for curve fitting):
```yaml
kpt_shape: [6, 3]  # 6 keypoints, each with (x, y, visibility)
```

For 2D-only (no visibility flag):
```yaml
kpt_shape: [6, 2]  # 6 keypoints, each with (x, y) only
```

**Recommendation: use `[6, 3]` with visibility.** The visibility channel (0=unlabeled, 1=labeled-occluded, 2=labeled-visible) allows marking partial fish frames (fish partially out of frame) without discarding the annotation. This matches the prior project decision to handle partial midlines via NaN+conf=0.

**flip_idx for fish:** Fish have bilateral symmetry but the keypoint ordering must be decided at annotation time. For a straight-line midline (head to tail) with no left/right pairs, flip_idx can be identity (no swapping on horizontal flip). For a keypoint scheme with paired left/right fins, flip_idx must swap the pair indices.

**Dataset YAML for pose training:**

```yaml
path: /path/to/fish_pose_dataset
train: images/train
val: images/val
nc: 1
names: [fish]
kpt_shape: [6, 3]
flip_idx: [0, 1, 2, 3, 4, 5]  # identity — adjust if anatomy has L/R pairs
```

**Annotation format — YOLO pose label:**

One `.txt` file per image. One row per fish:

```
<class_id> <cx> <cy> <w> <h> <kp1x> <kp1y> <kp1v> <kp2x> <kp2y> <kp2v> ... <kpNx> <kpNy> <kpNv>
```

- `class_id`: 0 (fish)
- `cx cy w h`: Bounding box center and dimensions, normalized [0, 1]
- `kpNx kpNy`: Keypoint position, normalized [0, 1]
- `kpNv`: Visibility flag — 0=unlabeled, 1=occluded, 2=visible

Example (one fish, 6 keypoints at head, shoulder, mid-body, pelvis, caudal-base, tail):
```
0 0.512 0.341 0.180 0.060 0.421 0.335 2 0.452 0.338 2 0.490 0.341 2 0.528 0.343 2 0.562 0.344 2 0.591 0.346 2
```

**Generating keypoint pseudo-labels from existing SAM2 masks:**

The existing skeletonization pipeline already produces ordered N-point midlines from binary masks. For YOLO pose labels, reduce the midline from N sampled points to the 6 anatomical keypoint positions (or subsample to 6 from the arc-length-parameterized curve). This is a data preparation script, not a model change.

**Training configuration:**

```python
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
model.train(
    data="fish_pose.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=1e-3,
    degrees=15.0,
    flipud=0.5,
    fliplr=0.5,
    scale=0.3,
    pose=12.0,    # pose loss weight (default; increase to 24.0 if keypoints dominate over box)
    kobj=1.0,     # keypoint objectness loss weight (default)
    device="0",
)
```

**Inference API:**

```python
from ultralytics import YOLO

model = YOLO("path/to/fish_pose_best.pt")
results = model(frame, verbose=False)

for r in results:
    if r.keypoints is not None:
        kpts = r.keypoints.xy      # (N, K, 2) — pixel coords, N fish, K keypoints
        conf = r.keypoints.conf    # (N, K) — per-keypoint confidence
        # kpts[i, :, :] is the ordered midline for fish i
        # use conf[i, :] > threshold to filter invalid keypoints
```

The output replaces the existing keypoint head output contract. Downstream midline spline fitting receives the (K, 2) ordered keypoint array per fish — same interface as the arc-length-resampled skeleton points.

---

### Annotation Tooling: SAM2 → YOLO Format Conversion

No new annotation tool is required. The existing SAM2 pseudo-label pipeline generates binary masks. Two conversion paths are needed:

**For seg labels (masks → polygons):**

```python
import cv2
import numpy as np

def mask_to_yolo_seg(mask: np.ndarray, class_id: int = 0) -> str:
    """Convert binary mask to YOLO seg polygon annotation row.

    Uses largest connected component to handle multi-region SAM2 output.
    Contour is normalized to [0, 1] by image dimensions.
    """
    h, w = mask.shape
    # Take largest connected component (handles SAM2 multi-region issue)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels < 2:
        return ""  # empty mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = (labels == largest).astype(np.uint8) * 255

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim < 2 or len(contour) < 3:
        return ""

    # Normalize to [0, 1]
    xy = contour.astype(float)
    xy[:, 0] /= w
    xy[:, 1] /= h
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in xy)
    return f"{class_id} {coords}"
```

**For pose labels (skeleton points → keypoints):**

The existing `skimage.morphology.skeletonize` + arc-length resampling pipeline produces N ordered points. Select 6 (or K) evenly spaced points from the arc-length-parameterized curve as the keypoint targets. Write each fish annotation as one row in the YOLO pose label format.

Ultralytics also provides a built-in `auto_annotate()` function that can generate YOLO seg annotations from bounding boxes using SAM2 internally — this overlaps with the existing SAM2 pipeline. The existing SAM2 pipeline with box prompts has already been validated on this dataset and should be the conversion source.

---

### Model Variant Selection Guide

**YOLO-seg:**

| Variant | Params | Purpose | When to Use |
|---------|--------|---------|-------------|
| `yolo11n-seg.pt` | 2.7M | Starting point, fastest | Default — start here |
| `yolo11s-seg.pt` | 10.4M | Improved mask quality | If nano mask IoU < 0.80 on val set |
| `yolo11m-seg.pt` | 23.6M | High accuracy | If small is still insufficient; diminishing returns for fish |

**YOLO-pose:**

| Variant | Params | Purpose | When to Use |
|---------|--------|---------|-------------|
| `yolo11n-pose.pt` | 2.9M | Starting point | Default — start here |
| `yolo11s-pose.pt` | 10.4M | Improved keypoint accuracy | If nano PCK < 0.85 on val set |
| `yolo11m-pose.pt` | 21.5M | High accuracy | Only if small is insufficient |

**Why nano first:** The task is single-class (fish), top-down camera with consistent scale, small dataset (~hundreds to low thousands of images). Nano models are typically sufficient for narrow-domain tasks. Larger models are slower to train, harder to overfit-diagnose, and rarely necessary for single-class custom domains.

---

## Dependency Changes for v3.0

### pyproject.toml: No New Dependencies Required

The ultralytics package (already `>=8.1` from v2.2) ships YOLOv8 and YOLO11 seg/pose models. The current version is `8.4.19` (released 2026-02-28). The version constraint `ultralytics>=8.1` already covers all seg and pose capabilities needed.

```toml
# No change needed — existing constraint covers v3.0
"ultralytics>=8.1",
```

YOLO11 seg and pose pretrained weights download automatically on first use:
```python
YOLO("yolo11n-seg.pt")   # downloads ~6MB on first call
YOLO("yolo11n-pose.pt")  # downloads ~6MB on first call
```

### Code to Delete (v3.0 stripping)

| File / Symbol | Replaced By |
|---------------|-------------|
| `src/aquapose/segmentation/model.py` → `_UNet`, `UNetSegmentor` | `YOLO("yolo11n-seg.pt")` inference wrapper |
| `src/aquapose/segmentation/training.py` → `train_unet()`, `train_pose()` | `model.train(data=yaml, ...)` calls |
| `src/aquapose/segmentation/dataset.py` → `BinaryMaskDataset` | YOLO dataset YAML + annotation `.txt` files |
| `src/aquapose/core/` → `_KeypointHead`, `_PoseModel` (if extracted to core) | `YOLO("yolo11n-pose.pt")` inference wrapper |
| `tests/unit/` → model and training unit tests for above | Tests for new Ultralytics wrappers |

The `src/aquapose/segmentation/crop.py` shared crop utilities are retained — they are used by the OBB pipeline and may be used for inference preprocessing.

---

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `mmdet` / `mmpose` | Heavyweight frameworks, complex MMCV build on Windows, no benefit over ultralytics for this task | `ultralytics` (already in stack) |
| `detectron2` | Meta's framework, poor Windows support, complex setup | `ultralytics` |
| `timm` for custom backbone | Not needed — YOLO11 provides its own pretrained backbone; adding timm for a different encoder defeats the purpose of switching to Ultralytics | `yolo11n-seg.pt` pretrained |
| `segmentation_models_pytorch` | Custom U-Net is being deleted — no reason to add another custom seg framework | `yolo11n-seg.pt` |
| `albumentations` | Training augmentation is handled by ultralytics internally; the `degrees`, `flipud`, `mosaic` etc. parameters cover all needed augmentations | Built-in ultralytics augmentation |
| `supervision` (Roboflow) | Visualization helper for YOLO results; not needed — existing visualization observers handle output rendering | Existing `src/aquapose/visualization/` |
| Label Studio / CVAT | External annotation tools; unnecessary — SAM2 pseudo-labels generate annotations programmatically | SAM2 pipeline → conversion scripts |
| `roboflow` package | Dataset management SaaS — not appropriate for private research data | Local YOLO dataset directories |
| `pytorch-lightning` | Training loop abstraction — unnecessary given ultralytics wraps training internally | `model.train(...)` ultralytics API |

---

## Recommended Stack (Full Picture, v3.0)

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11 | Runtime | Project targets >=3.11; stable, widely supported. |
| PyTorch | latest stable | YOLO inference, tensor ops | No longer pinned to 2.4.1. Primary pipeline has no PyTorch3D coupling. |
| torchvision | match PyTorch | Image transforms (inference preprocessing) | Follows PyTorch version; transform utilities used in data prep. |
| CUDA | 12.x | GPU acceleration | YOLO seg/pose inference and training. No PyTorch3D CUDA ops in primary pipeline. |
| scikit-image | >=0.22 | Skeletonization (midline extraction), distance transform (width profiles) | Still used in reconstruction stage and for generating keypoint pseudo-labels from SAM2 masks. |
| scipy | >=1.13 | Spline fitting, LM refinement, Hungarian assignment | `splprep`/`splev` for midline splines, LM for refractive triangulation refinement. |
| numpy | >=1.24 | Array operations, calibration interface | Foundation for all non-tensor computation. AquaCal is numpy-based. |

### Perception Pipeline Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| opencv-python | >=4.8 | MOG2 background subtraction, video I/O, affine crop extraction, contour extraction for seg labels | Detection fallback, video I/O, OBB crop extraction, mask-to-polygon conversion for annotation prep. |
| ultralytics | **>=8.1** (current: 8.4.19) | YOLO detection (standard + OBB), YOLO-seg inference, YOLO-pose inference, unified training API | `yolov8n.pt` / `yolov8n-obb.pt` for detection; `yolo11n-seg.pt` for segmentation; `yolo11n-pose.pt` for midline keypoints. One library covers all three model types. |
| sam2 (segment-anything-2) | latest from source | Zero-shot pseudo-label mask generation via box prompts | Offline annotation tool only — not deployed at inference time. Generates source masks converted to YOLO seg/pose format. |

### Reconstruction Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-image | >=0.22 | Skeletonize masks, morphological cleanup, distance-based width profiles | Every frame: extract midline skeleton from binary mask, compute width via `distance_transform_edt`, prune skeleton branches. Also generates keypoint training targets. |
| scipy | >=1.13 | Spline fitting, LM refinement | Every frame: fit smooth spline to skeleton/keypoint points, refine 3D triangulation. |

### Tracking and Association Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| boxmot | >=11.0 | OC-SORT per-camera 2D tracking | 2D Tracking stage. Robust to occlusion via virtual trajectories. |
| leidenalg | >=0.10 | Leiden graph clustering for cross-camera association | Association stage. Graph-based with must-not-link constraints. |
| igraph | >=0.11 | Graph construction for Leiden clustering | Dependency of leidenalg. |

### Storage and Visualization Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| h5py | >=3.9 | HDF5 trajectory data storage | Primary output format. |
| plotly | >=5.18 | 3D midline animation | 3D visualization observer. |
| click | >=8.1 | CLI entrypoint and training subcommands | `aquapose run`, `aquapose train detection`, `aquapose train segmentation`, `aquapose train pose`. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| hatch | Project management, virtual envs, test/lint/typecheck scripts | Already configured in `pyproject.toml`. Use `hatch run test` throughout. |
| ruff | Linting and formatting | Line length 88, targets py311. |
| basedpyright | Type checking | Configured in basic mode. |
| pytest | Testing | Configured with `slow` and `e2e` markers. |

---

## v3.0 Dependency Changes

### pyproject.toml Updates Required

None. The existing constraint `ultralytics>=8.1` covers all YOLO11 seg and pose functionality. No new packages needed.

### What Changes in pyproject.toml

Nothing. All v3.0 capabilities are already within the `ultralytics>=8.1` dependency:
- YOLO11 seg and pose support: added in ultralytics ~8.2+ (well within constraint)
- YOLO11 model naming convention (`yolo11n-seg.pt`): handled by ultralytics model hub
- Training API (`model.train(...)`): unchanged API across YOLOv8 and YOLO11

---

## Annotation Format Reference Card

### YOLO Seg Label Format

```
# labels/<image_name>.txt (one file per image)
# One row per fish instance
# Empty file (not missing) for images with no fish

<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

# class_id: integer (0 = fish)
# x1 y1 ... xn yn: polygon boundary, normalized [0, 1], min 3 pairs
```

### YOLO Pose Label Format

```
# labels/<image_name>.txt (one file per image)
# One row per fish instance

<class_id> <cx> <cy> <w> <h> <kp1x> <kp1y> <kp1v> ... <kpKx> <kpKy> <kpKv>

# class_id: integer (0 = fish)
# cx cy w h: bounding box center + dims, normalized [0, 1]
# kpNx kpNy: keypoint position, normalized [0, 1]
# kpNv: visibility — 0=unlabeled, 1=labeled-occluded, 2=labeled-visible
```

### YOLO Seg Dataset YAML

```yaml
path: /path/to/fish_seg_dataset
train: images/train
val: images/val
nc: 1
names: [fish]
```

### YOLO Pose Dataset YAML

```yaml
path: /path/to/fish_pose_dataset
train: images/train
val: images/val
nc: 1
names: [fish]
kpt_shape: [6, 3]           # [num_keypoints, dims] — 3 = (x, y, visibility)
flip_idx: [0, 1, 2, 3, 4, 5]  # identity for straight midline; adjust if L/R paired anatomy
```

### Dataset Directory Structure

```
fish_seg_dataset/
  images/
    train/
      frame_000001_cam1.jpg
      ...
    val/
      frame_000100_cam1.jpg
      ...
  labels/
    train/
      frame_000001_cam1.txt   # one polygon per fish
      ...
    val/
      frame_000100_cam1.txt
      ...

fish_pose_dataset/         # same directory structure, different label format
  images/...
  labels/...               # one row per fish with bbox + keypoints
```

---

## Alternatives Considered (v3.0 scope)

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| YOLO11n-seg as starting model | YOLOv8n-seg | YOLO11 is the current generation in the same ultralytics package — same API, same format, better accuracy. No reason to use the older generation for a new training run. |
| YOLO11n-pose for keypoints | Custom `_KeypointHead` on U-Net bottleneck | Custom head underperformed even with augmentation; Ultralytics provides pretrained backbone + battle-tested pose head out of the box. |
| YOLO11n-pose for keypoints | HRNet / ViTPose (mmpose) | mmpose has complex MMCV build requirements; fails silently on Windows; heavy dependencies incompatible with project's install simplicity goal. |
| Contour extraction for seg labels | Roboflow auto-annotation API | Private research data cannot be sent to a cloud API. Local SAM2 → OpenCV contour conversion is already in-house. |
| 6 anatomical keypoints | N arc-length-sampled points (e.g., 20) | YOLO pose is not designed for dense ordered curves — it predicts unordered discrete keypoints. Use 6 anatomical landmarks (head, mid-body, tail) for YOLO-pose, then fit a spline through them in post-processing. Dense midline sampling stays in the reconstruction stage. |
| kpt_shape: [6, 3] (with visibility) | kpt_shape: [6, 2] (no visibility) | Visibility flag allows annotating partially-visible fish without discarding the sample. Aligns with prior project decision: NaN+conf=0 for occluded keypoints. |

---

## Stack Patterns by Variant (v3.0)

**If seg mask quality is insufficient (IoU < 0.80):**
- Upgrade from `yolo11n-seg.pt` to `yolo11s-seg.pt` in the training run
- Do NOT switch to custom U-Net — the whole point of this milestone is eliminating custom model maintenance
- Verify SAM2 pseudo-label quality first — training on noisy labels is the more likely root cause

**If pose keypoint accuracy is insufficient (PCK < 0.85):**
- Upgrade from `yolo11n-pose.pt` to `yolo11s-pose.pt`
- Increase `pose` loss weight from 12.0 to 24.0 to emphasize keypoint accuracy over box accuracy
- Verify that the 6 keypoints are anatomically consistent across annotators/frames

**If female fish (low-contrast) are hard to segment:**
- Increase contrast augmentation: `hsv_v=0.6` (default 0.4) in training config
- Do not add albumentations — ultralytics `hsv_v` parameter covers this

**If fish partially out of frame:**
- Mark out-of-frame keypoints with visibility=0 (unlabeled), not 1 (occluded)
- The model learns to not predict for unlabeled keypoints
- Include these frames in training data — partial fish are common in this rig

**For Windows development (all v3.0 additions):**
- YOLO11-seg training: no issues — ultralytics installs cleanly on Windows
- YOLO11-pose training: no issues — same training API
- Contour extraction for labels: no issues — OpenCV on Windows
- Dataset directory structure: use forward slashes or `pathlib.Path` in Python code regardless of platform

---

## Version Compatibility (v3.0)

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| ultralytics 8.4.x (latest) | Python 3.11, PyTorch latest | YOLO11 seg and pose fully supported |
| ultralytics >=8.1 | YOLO-OBB (from v2.2) | OBB introduced in 8.1.0 — still satisfied |
| PyTorch (latest) | ultralytics >=8.1 | ultralytics tracks PyTorch releases; no pinning needed |
| scikit-image >=0.22 | numpy >=1.24 | scikit-image 0.22+ supports numpy 2.x |
| Python 3.11 | All listed libraries | Safe baseline; 3.12 also works for primary pipeline |

---

## Sources

- Ultralytics GitHub releases — latest version 8.4.19 (2026-02-28): https://github.com/ultralytics/ultralytics/releases
- Ultralytics YOLO11 docs — model variants, tasks, naming: https://docs.ultralytics.com/models/yolo11/
- Ultralytics YOLOv8 model variants — seg and pose suffix names: https://docs.ultralytics.com/models/yolov8/
- Ultralytics seg task docs — annotation format, training config: https://docs.ultralytics.com/tasks/segment/
- Ultralytics pose task docs — annotation format, kpt_shape, training config: https://docs.ultralytics.com/tasks/pose/
- Ultralytics seg dataset format — polygon format specification: https://docs.ultralytics.com/datasets/segment/
- Ultralytics pose dataset format — kpt_shape, visibility flags, flip_idx: https://docs.ultralytics.com/datasets/pose/
- Tiger-pose dataset example — 12-keypoint non-human kpt_shape: [12, 2] configuration: https://docs.ultralytics.com/datasets/pose/tiger-pose/
- YOLOv8 vs YOLO11 comparison — seg and pose task support, performance: https://docs.ultralytics.com/compare/yolov8-vs-yolo11/
- Ultralytics training config reference — all training parameters: https://docs.ultralytics.com/usage/cfg/
- SAM2 masks to YOLO format — multi-region challenge discussion: https://github.com/ultralytics/ultralytics/issues/15380

---

*Stack research for: 3D fish pose estimation via direct triangulation with refractive ray casting*
*Researched: 2026-02-19 | Updated: 2026-03-01 (v3.0 Ultralytics Unification additions)*


---

## v3.2 Evaluation Ecosystem: Stack Additions

> **v3.2 update (2026-03-03):** Adding unified `aquapose eval` and `aquapose tune` CLI subcommands with per-stage proxy metrics, single-stage sweeps, cascade tuning (association → reconstruction), and partial pipeline execution. This section covers only the NEW stack requirements — everything in the sections above remains unchanged.

### What Already Exists (Do Not Re-Research)

All of the following are already available and validated:

- `click>=8.1` — CLI entrypoint, already used for `aquapose run` and `aquapose train` subcommand groups
- `json` (stdlib) — already used in `evaluation/output.py` for regression JSON
- `numpy>=1.24` — already used throughout evaluation/metrics
- `itertools` (stdlib) — Cartesian product for grid search; already used in codebase
- `dataclasses` (stdlib) — frozen config hierarchy already established
- `pathlib` (stdlib) — path handling throughout

**Transitive dependencies already installed in the hatch environment** (confirmed present via `hatch run python -c "import X"`):

- `tqdm 4.67.3` — installed as transitive dependency (likely via ultralytics)
- `joblib 1.5.3` — installed as transitive dependency (likely via scikit-image/scipy)
- `sklearn 1.8.0` (scikit-learn) — installed as transitive dependency

None of these need to be added to `pyproject.toml` — they are already available. If any of them need to become explicit dependencies (because the upstream transitive chain changes), add them at that point.

---

### New Capabilities Needed and Recommended Approach

#### 1. Grid Search / Parameter Sweep

**Recommendation: `itertools.product` with a small wrapper — no new library.**

The existing `tune_association.py` already implements grid search without any library. The sweep logic is 10–20 lines of pure Python:

```python
import itertools
from typing import Any

def iter_grid(grid: dict[str, list[Any]]):
    """Yield all parameter combinations from a grid dict."""
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))
```

`sklearn.model_selection.ParameterGrid` is equivalent and already available as a transitive dependency, but importing sklearn for a 10-line stdlib replacement adds a conceptual coupling to ML tooling that doesn't belong here. Use `itertools.product` directly.

**Why not Optuna/Ray Tune/W&B Sweeps:** The seed document explicitly rules out Bayesian optimization — grids are small (5–10 values per parameter, 1–2 dimensions at a time), interpretable, and reproducible. Optuna adds 3MB+ of dependencies and async complexity for no benefit at this scale.

#### 2. Intermediate Caching (Sweep Upstream Cache)

**Recommendation: `pickle` (stdlib) for per-stage cache files. No new library.**

The seed document's decision is explicit: pickle for within-session intermediate caches, structured output (JSON) for final results. This is correct for the following reasons:

- The pipeline objects being cached (`Detection`, `Tracklet2D`, `TrackletGroup`, `MidlineSet`, etc.) are custom frozen dataclasses with numpy arrays and torch tensors. They serialize cleanly with pickle.
- Pickle is exact (no round-trip fidelity loss). The pipeline code won't change mid-session.
- The cache is explicitly discardable after tuning completes — it is not a persistent artifact.
- `joblib.dump`/`joblib.load` would be marginally faster for large numpy arrays (memory mapping), but the cache items here are Python object graphs, not raw arrays. `joblib` provides no meaningful advantage over `pickle` for this shape of data.
- Cache files are written to a temporary work directory per tuning session and discarded after.

**Cache file convention:**

```python
import pickle
from pathlib import Path

# Write per-stage cache
cache_path = work_dir / f"stage_{stage_name}_cache.pkl"
with cache_path.open("wb") as f:
    pickle.dump(stage_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)

# Read per-stage cache
with cache_path.open("rb") as f:
    stage_outputs = pickle.load(f)
```

Use `pickle.HIGHEST_PROTOCOL` (currently protocol 5 in Python 3.11) for maximum efficiency.

**Why not joblib.Memory (automatic function-level caching):** `joblib.Memory` caches by function signature + argument hash. The pipeline stage functions don't have stable hashable signatures (they accept `PipelineContext` which contains frame data). Manual explicit pickle cache files are simpler and give full control over what gets cached and when it gets invalidated.

**Why not msgpack/orjson/arrow:** These are for structured/columnar data. The stage outputs are heterogeneous Python objects — not JSON-serializable without custom encoders. Pickle handles them correctly out of the box.

#### 3. Progress Reporting During Sweeps

**Recommendation: `tqdm` — already available as transitive dependency.**

`tqdm` is already installed (4.67.3 via ultralytics). It wraps any iterable and provides a real-time progress bar with ETA — exactly right for the sweep outer loop:

```python
from tqdm import tqdm

for params in tqdm(list(iter_grid(grid)), desc="Sweeping association"):
    result = run_stage(params)
    results.append(result)
```

`tqdm` requires zero additional dependencies and has negligible overhead. It is the standard choice for CLI progress in scientific Python code.

**Why not `rich.progress`:** `rich` is not currently installed in the hatch environment. Adding it for progress bars when `tqdm` is already available would add a dependency (~3MB) for no incremental benefit. If `rich` is added later for table formatting (see below), the progress bar could be migrated, but tqdm is correct for now.

#### 4. Sweep Results Table Formatting

**Recommendation: plain Python string formatting (stdlib) — extend existing `output.py` pattern.**

The existing `evaluation/output.py` already produces ASCII tables using f-strings. The same approach should be extended for sweep result tables:

```python
def format_sweep_table(rows: list[dict], metric_name: str) -> str:
    """Format sweep results as a sorted ASCII comparison table."""
    ...
```

This is consistent with the existing codebase pattern and keeps zero new dependencies.

**Why not `rich` tables:** `rich` is not currently installed. The existing output format is working and consistent. Adding `rich` for cosmetic improvement would require adding a new dependency and changing the output character set — both are unnecessary for a research tool where the developer is the primary consumer.

**Why not `tabulate`:** Another library that adds a dependency for something stdlib handles adequately. The existing `format_summary_table` in `output.py` is proof that custom formatting is maintainable here.

#### 5. CLI Subcommand Groups

**Recommendation: `click` groups — the exact pattern already used for `train`.**

The `aquapose train` group in `training/cli.py` is the blueprint. Two new groups follow the same pattern:

```python
# src/aquapose/evaluation/cli.py
import click

@click.group("eval")
def eval_group() -> None:
    """Evaluate a diagnostic run."""

@eval_group.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--stage", type=click.Choice([...]), default=None)
@click.option("--report", type=click.Choice(["text", "json"]), default="text")
def run_eval(...): ...
```

```python
# src/aquapose/evaluation/tune_cli.py
import click

@click.group("tune")
def tune_group() -> None:
    """Sweep parameters for a pipeline stage."""

@tune_group.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--stage", type=click.Choice([...]), required=True)
@click.option("--cascade", is_flag=True, default=False)
@click.option("--n-frames", type=int, default=30)
@click.option("--n-frames-validate", type=int, default=100)
@click.option("--param", "param_overrides", multiple=True, type=str)
@click.option("--range", "range_overrides", multiple=True, type=str)
@click.option("--top-n", type=int, default=3)
def run_tune(...): ...
```

Register both groups in `cli.py`:

```python
from aquapose.evaluation.cli import eval_group
from aquapose.evaluation.tune_cli import tune_group

cli.add_command(eval_group)
cli.add_command(tune_group)
```

**`--param`/`--range` parsing for custom sweep ranges:** Use `multiple=True` string options. The orchestrator parses `--param outlier_threshold --range 5:50:5` as a pair, extracting `(name, start, stop, step)` from the range string with a small helper:

```python
def parse_range(range_str: str) -> list[float]:
    """Parse 'min:max:step' into a list of float values."""
    start, stop, step = (float(x) for x in range_str.split(":"))
    values = []
    v = start
    while v <= stop + 1e-9:
        values.append(round(v, 10))
        v += step
    return values
```

No `click.ParamType` subclass needed — string parsing in the command body is simpler and more testable.

#### 6. Final Output Format (Structured Summary)

**Recommendation: JSON via stdlib `json` — extend the existing `write_regression_json` pattern.**

The existing `write_regression_json` in `output.py` already establishes the pattern for machine-readable output. The sweep summary output should follow the same format — a JSON file written alongside the run directory:

```json
{
  "stage": "association",
  "timestamp": "2026-03-03T14:22:00",
  "n_frames_sweep": 30,
  "n_frames_validate": 100,
  "baseline": {"fish_yield": 0.42, "singleton_rate": 0.58},
  "sweep_results": [
    {"params": {...}, "primary_metric": 0.67, "tiebreaker": 0.31}
  ],
  "winner": {"params": {...}, "primary_metric": 0.71, "tiebreaker": 0.28},
  "validation": {"fish_yield": 0.70, "mean_reprojection_px": 3.21},
  "config_diff": {"association.ray_distance_threshold": [0.03, 0.04]}
}
```

No new libraries needed. `json.dump` with `indent=2` is already used in the codebase.

---

### pyproject.toml Changes for v3.2

**No new runtime dependencies required.**

All libraries needed for the evaluation/tuning system are either:
1. Already explicit dependencies (`click>=8.1`, `numpy>=1.24`, `scipy>=1.11`)
2. Already available as transitive dependencies (`tqdm`, `joblib`, `sklearn`)
3. Python stdlib (`itertools`, `pickle`, `json`, `pathlib`, `dataclasses`)

The only code additions are new modules under `src/aquapose/evaluation/`:

```
src/aquapose/evaluation/
├── __init__.py              (existing — add new public symbols)
├── harness.py               (existing — refactor for multi-stage support)
├── metrics.py               (existing — extend with per-stage metric functions)
├── output.py                (existing — extend with sweep table formatters)
├── orchestrator.py          (NEW — EvalRunner + TuningOrchestrator)
├── stage_metrics/           (NEW — per-stage metric modules)
│   ├── __init__.py
│   ├── detection.py
│   ├── tracking.py
│   ├── association.py
│   ├── midline.py
│   └── reconstruction.py
├── cli.py                   (NEW — aquapose eval subcommand group)
└── tune_cli.py              (NEW — aquapose tune subcommand group)
```

---

### What NOT to Add for v3.2

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `optuna` | Bayesian optimization overkill for 5–10-point 1D/2D grids; adds async machinery and a large dependency | `itertools.product` + custom grid dict |
| `ray[tune]` or W&B sweeps | Distributed sweep infrastructure for ML research at scale; massively over-engineered for a single-machine research tool with <50 combos per sweep | `tqdm` + sequential loop |
| `joblib.Memory` | Function-level automatic caching by hash — doesn't match the explicit stage-output cache model where invalidation is by design (new tuning session = fresh cache) | Explicit `pickle.dump`/`pickle.load` per stage |
| `rich` | Not currently installed; ASCII tables in `output.py` are sufficient and consistent with existing style | Extend `format_summary_table` pattern in `output.py` |
| `tabulate` | Yet another string-formatting library adding a dependency for something `output.py` already does with f-strings | f-string formatting in `output.py` |
| `typer` | Click alternative; would require rewriting existing CLI | `click` (already used throughout) |
| `pydantic` | Already explicitly ruled out for config — using frozen dataclasses | Frozen `dataclasses` |
| Hydra / `omegaconf` | Config override frameworks; the existing `load_config + dataclasses.replace` pattern is sufficient for sweep parameter injection | `dataclasses.replace` in orchestrator |
| `mlflow` / `wandb` | Experiment tracking dashboards — overkill for a single-researcher tool; output is JSON + stdout | JSON summary files + console output |

---

### Integration Points with Existing Architecture

| New Component | Integrates With | Integration Notes |
|---------------|----------------|-------------------|
| `EvalRunner` | `DiagnosticObserver`, per-stage diagnostic files | Reads per-stage files written by observer; does not call `PosePipeline` directly |
| `TuningOrchestrator` | `PosePipeline`, `PipelineContext`, `DiagnosticObserver` | Creates `PipelineContext` with pre-loaded upstream cache, calls `PosePipeline.run(initial_context=...)` |
| `stage_metrics/association.py` | `TrackletGroup` (from `core/types/`), `PipelineContext` | Reads `context.tracklet_groups` to compute fish yield and singleton rate |
| `stage_metrics/reconstruction.py` | Existing `compute_tier1`, `compute_tier2` in `metrics.py` | Wraps existing functions; reconstruction metrics already implemented |
| `eval_group`, `tune_group` | `cli.py` main group | Added via `cli.add_command(eval_group)` and `cli.add_command(tune_group)` |
| Per-stage pickle caches | `PipelineContext` field types | Cache files store the typed fields from `PipelineContext` directly; deserialize back into the same types |
| `DEFAULT_GRIDS` per stage | `stage_metrics/<stage>.py` | Lives in the metric module for that stage; orchestrator imports it as the default sweep range |

---

### Version Compatibility for v3.2

| Component | Version | Notes |
|-----------|---------|-------|
| `click` | >=8.1 (current: ~8.1.8) | `@click.group` with `add_command` already validated in codebase |
| `tqdm` | 4.67.3 (transitive) | No API changes needed; `tqdm(iterable, desc=...)` is stable |
| `pickle` | stdlib (protocol 5, Python 3.11) | `HIGHEST_PROTOCOL=5` for Python 3.11; deterministic for round-trip of frozen dataclasses |
| `itertools.product` | stdlib | No version concerns |
| `json` | stdlib | `_NumpySafeEncoder` pattern in `output.py` already handles numpy scalar types |

---

### Sources for v3.2 Stack Research

- Python stdlib `itertools.product` docs: https://docs.python.org/3/library/itertools.html
- Python stdlib `pickle` protocol reference: https://docs.python.org/3/library/pickle.html#data-stream-format
- `tqdm` PyPI (4.67.3 confirmed via `hatch run python -c "import tqdm; print(tqdm.__version__)"`): https://pypi.org/project/tqdm/
- `joblib` Memory class docs (1.5.3 transitive dep, considered and rejected for this use): https://joblib.readthedocs.io/en/stable/memory.html
- `click` Commands and Groups documentation (8.3.x current): https://click.palletsprojects.com/en/stable/commands/
- `sklearn.model_selection.ParameterGrid` (1.8.0 transitive dep, considered and rejected for stdlib itertools): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html
- Existing `src/aquapose/cli.py` — `train_group` pattern for new subcommand groups
- Existing `src/aquapose/evaluation/output.py` — `format_summary_table` pattern for new table formatters
- Confirmed via `hatch run python -c "import X; print(X.__version__)"`: tqdm=4.67.3, joblib=1.5.3, sklearn=1.8.0

---

*Stack research for v3.2 Evaluation Ecosystem additions*
*Researched: 2026-03-03*
