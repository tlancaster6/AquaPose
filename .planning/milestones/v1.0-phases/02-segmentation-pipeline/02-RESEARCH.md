# Phase 2: Segmentation Pipeline - Research

**Researched:** 2026-02-19
**Domain:** Computer vision segmentation — MOG2 background subtraction, SAM2 pseudo-labeling, Label Studio annotation, Mask R-CNN training
**Confidence:** HIGH (core stack), MEDIUM (Label Studio RLE workflow), MEDIUM (Detectron2 on Windows)

## Summary

Phase 2 builds a three-stage segmentation pipeline: (1) MOG2 background subtraction for fish detection and bounding box extraction, (2) SAM2 pseudo-label generation from those boxes fed into Label Studio for human correction, and (3) Mask R-CNN training on corrected 256x256 crops to produce the final inference model. All three stages interact with established libraries that are well-documented and production-tested.

The central implementation decision is the choice of Mask R-CNN backend. The locked decision specifies Detectron2, but Detectron2 has significant Windows compatibility problems and has received no official releases since v0.6 in October 2021. The `torchvision` built-in `maskrcnn_resnet50_fpn` is a drop-in alternative that is fully cross-platform, actively maintained, and already a project dependency. This is flagged as a critical open question for planning (see below). SAM2's `SAM2ImagePredictor` with box prompts is straightforward and well-supported. The Label Studio RLE import/export workflow for brush masks requires the `label-studio-converter` helper library and a non-obvious conversion step.

**Primary recommendation:** Use `torchvision.models.detection.maskrcnn_resnet50_fpn` instead of Detectron2 for Mask R-CNN training, since Detectron2 is not officially supported on Windows (the dev environment), is unmaintained, and torchvision ships with the same ResNet-50 FPN backbone with equivalent training outcomes.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Detection strategy:** Global MOG2 parameters across all 13 cameras — no per-camera tuning. Balanced recall/precision — avoid over-aggressive detection. No secondary fallback for stationary fish — accept the gap; trained Mask R-CNN handles stationary cases later. MOG2 outputs both bounding boxes AND rough foreground masks — both feed into SAM as prompts.
- **Annotation workflow:** All SAM pseudo-labels get human review and correction in Label Studio — no confidence-based skip. Temporal sampling for frame selection (every Nth frame). Include negative (empty/no-fish) frames in annotation set. Random 80/20 train/validation split across all annotated frames.
- **Model training:** Mask R-CNN operates on fixed 256x256 crops around MOG2 detections — one fish per crop. ImageNet-pretrained ResNet-50 backbone. Single "fish" class (no male/female distinction) — track female IoU separately during evaluation, only intervene if target not met. Standard augmentation: flips, rotations, brightness/contrast jitter.
- **Pipeline interface:** Return all detections above 0.1 confidence threshold. Fish count per frame is N-max (9), often fewer visible per camera — segmentation pipeline is per-camera, per-frame only. RLE-encoded masks per detection, with bounding box and confidence score. Batch frame API — accept batches of frames for GPU throughput. Expect pre-extracted frames (numpy arrays/tensors) as input.

### Claude's Discretion

- MOG2 hyperparameters (history length, variance threshold, learning rate)
- SAM model variant and prompt engineering details
- Label Studio project configuration specifics
- Mask R-CNN training hyperparameters (learning rate schedule, epochs, etc.)
- Exact temporal sampling rate (every Nth frame — N TBD based on dataset size)
- Augmentation intensity parameters

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `opencv-python` | >=4.8 (already in deps) | MOG2 background subtraction, morphological ops, contour extraction | The canonical background subtraction implementation; MOG2 is built into cv2 |
| `sam2` | 2.1 (latest checkpoint Sept 2024) | SAM2 pseudo-label generation from bounding box prompts | Meta's SAM2 is the current state-of-the-art promptable segmentor; SAM2.1 checkpoints outperform SAM on static images |
| `torchvision` | >=0.15 (follows torch) | Mask R-CNN training and inference | Ships `maskrcnn_resnet50_fpn` with COCO-pretrained weights; already a transitive dep via torch; cross-platform |
| `label-studio` | >=1.x (pip install) | Human annotation interface | Open-source, pip-installable, supports brush/RLE mask annotations, Docker-runnable |
| `label-studio-converter` | latest | RLE mask format conversion for Label Studio import | Official converter library; provides `mask2rle()`, `encode_rle()`, `image2annotation()` in `brush.py` |
| `pycocotools` | latest | COCO annotation format, RLE encode/decode for mask serialization | Standard COCO API; `mask.encode()` / `mask.decode()` with Fortran-order arrays |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torchvision.transforms.v2` | >=0.15 | Augmentation pipeline for training | Flips, rotations, brightness/contrast jitter — use v2 API (supports masks natively) |
| `scipy.ndimage` | >=1.11 (already in deps) | Morphological cleanup of MOG2 masks if needed | Alternative to OpenCV morphology when working on numpy arrays |
| `h5py` | >=3.9 (already in deps) | Persistent storage of annotation metadata or mask caches | Use for anything that needs to persist alongside the HDF5 pipeline output |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `torchvision` Mask R-CNN | Detectron2 | Detectron2 has richer model zoo but is unmaintained (v0.6, Oct 2021), not officially supported on Windows, and requires source compilation. torchvision Mask R-CNN is equivalent for single-class 256x256 crop use case and is cross-platform. |
| `label-studio` (local) | Roboflow, CVAT, or cloud Label Studio | All alternatives involve cloud data upload or licensing. Label Studio pip install is fully local and free. |
| `sam2` (Meta) | `segment-geospatial` SAM wrapper | Direct `sam2` package is lower-level but avoids unnecessary geo-spatial abstraction layer |
| `pycocotools` RLE | Custom RLE | Never hand-roll RLE; pycocotools handles edge cases in column-major (Fortran) ordering that trip up custom implementations |

**Installation:**
```bash
# SAM2 — not on PyPI stable, install from source
git clone https://github.com/facebookresearch/sam2.git && pip install -e sam2/

# Download SAM2.1 checkpoint (choose one — small recommended for pseudo-labeling speed)
# Checkpoints: sam2.1_hiera_tiny (38.9M), sam2.1_hiera_small (46M), sam2.1_hiera_base_plus (80.8M), sam2.1_hiera_large (224.4M)
cd sam2/checkpoints && ./download_ckpts.sh  # or download individually

# Label Studio
pip install label-studio label-studio-converter

# pycocotools (for RLE encoding in pipeline output)
pip install pycocotools
```

Add to `pyproject.toml` dependencies: `pycocotools>=2.0`, `label-studio-converter` (dev-only). SAM2 install instruction goes in a comment like pytorch3d.

---

## Architecture Patterns

### Recommended Module Structure

```
src/aquapose/segmentation/
├── __init__.py          # Public API: MOG2Detector, SAMPseudoLabeler, MaskRCNNSegmentor, SegmentationResult
├── detector.py          # MOG2Detector — background subtraction, bounding box extraction
├── pseudo_labeler.py    # SAMPseudoLabeler — SAM2 inference from MOG2 prompts
├── model.py             # MaskRCNNSegmentor — Mask R-CNN inference on 256x256 crops
└── training/
    ├── __init__.py
    ├── dataset.py       # CropDataset — COCO-format loader for 256x256 annotated crops
    └── train.py         # train() entry point — configurable via dataclass config
```

### Pattern 1: MOG2 Detector

**What:** Stateful background model with global parameters. `apply()` called per-frame returns foreground mask. Contours extracted, filtered by area, padded into bounding boxes.

**When to use:** First stage — always runs before SAM or Mask R-CNN.

```python
# Source: OpenCV docs https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
import cv2
import numpy as np

subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,        # Claude's discretion — start here, tune down if stationary fish absorbed
    varThreshold=16,    # Claude's discretion — default; lower = more sensitive
    detectShadows=True, # Suppress shadow false positives; shadows marked as 127
)

def detect(frame: np.ndarray, padding: int = 20) -> list[tuple[int, int, int, int]]:
    """Returns list of (x1, y1, x2, y2) padded bounding boxes."""
    fg_mask = subtractor.apply(frame)
    # Threshold to binary — shadow pixels (127) become 0
    _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Contour → bounding boxes with area filter
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = frame.shape[:2]
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)
        boxes.append((x1, y1, x2, y2))
    return boxes
```

**MOG2 hyperparameter guidance (Claude's discretion):**
- `history=500`: safe default for ~30fps video. 500 frames = ~17 seconds of background model. Reduce to 200-300 if fish frequently hold position for >10 seconds (faster relearning of truly stationary fish into background is acceptable loss given Mask R-CNN fallback).
- `varThreshold=16`: default. Lower (8-12) increases recall at cost of more noise. Given the ≥95% recall requirement and the explicit acceptance of false positives, start at 12.
- `detectShadows=True`: mandatory — aquarium lighting creates strong shadows on the tank floor that would produce false foreground regions.
- `MIN_AREA`: start at 200px² for 1600x1200 frames; fish at max distance should still exceed this.

### Pattern 2: SAM2 Image Predictor with Box Prompts

**What:** Single-image predictor. Set image once, call predict with box array. Returns binary masks.

```python
# Source: https://github.com/facebookresearch/sam2/blob/main/README.md
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def generate_pseudo_labels(
    image: np.ndarray,  # HxWx3, uint8
    boxes: list[tuple[int, int, int, int]],  # (x1, y1, x2, y2)
) -> list[np.ndarray]:
    """Returns list of binary masks (HxW, bool) for each box."""
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        input_boxes = np.array(boxes, dtype=np.float32)  # shape (N, 4)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,  # one mask per box
        )
    # masks shape: (N, 1, H, W) → squeeze to (N, H, W)
    return [masks[i, 0].astype(bool) for i in range(len(boxes))]
```

**SAM2 variant recommendation (Claude's discretion):** Use `sam2.1_hiera_small` (46M params, 84.8 FPS on A100). It provides near-large accuracy at faster throughput. The tiny variant (91.2 FPS) is worth testing if pseudo-label quality is acceptable — pseudo-labels go to human review anyway, so slightly coarser masks are acceptable.

**Box + foreground mask prompting:** MOG2 also produces a rough foreground mask (the `fg_mask` binary after threshold). SAM2 accepts both box AND point prompts simultaneously. The foreground mask centroid can be used as a positive point prompt alongside the box, improving mask quality in cluttered scenes. However, for simplicity, start with box-only and add centroid point if quality is insufficient after review.

### Pattern 3: Label Studio Import Workflow

**What:** SAM2 masks (binary numpy arrays) must be converted to Label Studio's internal RLE format for import as pre-annotations. Annotators then correct masks in the brush tool and export.

**Key insight:** Label Studio uses its OWN RLE variant (not pycocotools RLE) for BrushLabels. The `label-studio-converter` library bridges this gap.

```python
# Source: https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/brush.py
from label_studio_converter.brush import mask2rle

# Convert binary mask (HxW, bool/uint8) to Label Studio RLE format
mask_uint8 = mask.astype(np.uint8) * 255  # LS expects 0/255 not 0/1
rle = mask2rle(mask_uint8)  # returns list of ints (Label Studio RLE)

# Build pre-annotation task JSON
task = {
    "data": {"image": "/path/to/frame.jpg"},
    "predictions": [{
        "result": [{
            "type": "brushlabels",
            "value": {
                "format": "rle",
                "rle": rle,
                "brushlabels": ["fish"],
            },
            "original_width": W,
            "original_height": H,
            "from_name": "label",
            "to_name": "image",
        }]
    }]
}
```

**Export after annotation:** Label Studio exports BrushLabels annotations as JSON with RLE. Use `label-studio-converter` to decode back to binary masks for training:
```python
from label_studio_converter.brush import decode_rle
binary_mask = decode_rle(rle_list, width=W, height=H)
```

**Label Studio project template:** Use `BrushLabels` tag (not `PolygonLabels`) — polygon annotation is slower and less accurate for fish silhouettes. Configure one label: `fish`.

### Pattern 4: Mask R-CNN Training (torchvision)

**What:** Fine-tune `maskrcnn_resnet50_fpn_v2` on 256x256 crop dataset. One class ("fish") + background.

```python
# Source: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_model(num_classes: int = 2) -> torch.nn.Module:
    """Build Mask R-CNN for fine-tuning. num_classes = 1 fish + 1 background."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers=3,
    )
    # Replace box head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Replace mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model
```

**Dataset format for torchvision Mask R-CNN:** Each sample is `(image_tensor, target_dict)` where `target_dict` contains:
- `boxes`: `FloatTensor[N, 4]` in `[x1, y1, x2, y2]` format
- `labels`: `Int64Tensor[N]`
- `masks`: `UInt8Tensor[N, H, W]` — binary masks (0/1)

This is simpler than COCO registration required by Detectron2.

**Training hyperparameters (Claude's discretion):**
- Optimizer: SGD with momentum=0.9, weight_decay=0.0005
- LR: start at 0.005 (single GPU, batch size 4-8), use step LR scheduler, decay by 0.1 at 80% of epochs
- Epochs: 30-50 for a small dataset (~500-2000 annotated frames); monitor val mask IoU to stop early
- Batch: 4-8 images per batch (256x256 crops fit in VRAM)
- Use `torch.utils.data.DataLoader` with `collate_fn=lambda x: tuple(zip(*x))` (standard torchvision detection pattern)

### Pattern 5: Pipeline Output Interface (RLE masks)

**What:** Segmentation pipeline returns detections as structured dataclass with RLE-encoded masks.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    """Single fish detection result."""
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in original frame coords
    mask_rle: dict  # pycocotools RLE dict: {"size": [H, W], "counts": bytes}
    score: float

@dataclass
class FrameSegmentation:
    """Segmentation result for one camera frame."""
    detections: list[Detection]  # empty list = no fish detected above threshold
    frame_hw: tuple[int, int]   # (H, W) of original frame
```

RLE encoding with pycocotools:
```python
import pycocotools.mask as mask_util

# Encode: mask must be Fortran-contiguous uint8
mask_f = np.asfortranarray(binary_mask.astype(np.uint8))
rle = mask_util.encode(mask_f)  # {"size": [H, W], "counts": bytes}

# Decode:
binary_mask = mask_util.decode(rle).astype(bool)
```

### Anti-Patterns to Avoid

- **Running SAM2 on full 1600x1200 frames:** SAM2 can handle it but pseudo-labeling throughput will be slow. Since MOG2 gives bounding boxes, run SAM2 on the full frame but mask results back to detection regions. Do NOT crop before SAM2 — SAM2 uses the full image context for better mask boundaries.
- **Using PolygonLabels in Label Studio for fish masks:** Fish silhouettes require dozens of polygon points. BrushLabels with the paint brush tool is significantly faster for annotators and produces equivalent or better masks.
- **Registering dataset in COCO format for torchvision training:** torchvision detection training expects `__getitem__` returning `(image, target_dict)`, not COCO JSON registration (that's a Detectron2 pattern). Do NOT use `register_coco_instances`.
- **Applying MOG2 to pre-extracted still frames in random order:** MOG2 is a temporal model — it must process frames in sequential order to build the background model. If using pre-extracted frames as numpy arrays (per the interface decision), they must be fed in temporal order per camera.
- **Storing masks as PNG files for training:** Store annotations in COCO JSON with RLE — avoids filesystem overhead and integrates cleanly with the pipeline output format.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Background subtraction | Custom Gaussian mixture model | `cv2.createBackgroundSubtractorMOG2` | MOG2 handles adaptive Gaussian mixture, shadow detection, and per-pixel learning rates — 200+ edge cases |
| RLE mask encoding | Custom run-length encoding | `pycocotools.mask.encode` | Column-major (Fortran) ordering is a common gotcha; pycocotools handles it correctly |
| Label Studio RLE format | Custom JSON serializer | `label_studio_converter.brush.mask2rle` | LS uses its own RLE variant distinct from pycocotools; the converter handles the difference |
| Image segmentation foundation model | Fine-tuned detector from scratch | SAM2 `SAM2ImagePredictor` | Training a segmentation foundation model requires millions of images; SAM2 provides zero-shot quality masks from box prompts |
| Mask R-CNN head replacement | Custom detection head | `FastRCNNPredictor` + `MaskRCNNPredictor` from `torchvision.models.detection` | These handle the correct weight initialization for fine-tuning |
| Data augmentation with mask support | Custom augmentation that transforms masks consistently with images | `torchvision.transforms.v2` | v2 API natively supports simultaneous image+mask+box transforms maintaining geometric consistency |

**Key insight:** Every stage in this pipeline has a well-established library solution. The only custom code should be the glue: extracting boxes from MOG2 output, converting masks between formats, and wiring the three stages together.

---

## Common Pitfalls

### Pitfall 1: MOG2 State Leak Between Cameras

**What goes wrong:** The same `BackgroundSubtractorMOG2` instance is used across different cameras, corrupting the background model.

**Why it happens:** 13 cameras each need their own background model — a single global subtractor would mix scenes.

**How to avoid:** Instantiate one `BackgroundSubtractorMOG2` per camera. Store as a dict keyed by camera ID in the `MOG2Detector`.

**Warning signs:** Detection bounding boxes don't match the camera being processed.

### Pitfall 2: Label Studio RLE vs. pycocotools RLE Format Mismatch

**What goes wrong:** Masks encoded with `pycocotools.mask.encode` cannot be directly imported into Label Studio as BrushLabels pre-annotations. The two formats are different.

**Why it happens:** Label Studio uses its own RLE variant (row-major integers); pycocotools uses COCO RLE (column-major bytes).

**How to avoid:** Use `label_studio_converter.brush.mask2rle()` for import into LS. Use `pycocotools.mask.encode()` for pipeline output (external interface).

**Warning signs:** Masks appear empty or completely inverted in Label Studio brush tool after import.

### Pitfall 3: MOG2 Foreground Mask Contains Shadow Pixels (value=127)

**What goes wrong:** Shadow regions (marked as 127 by MOG2 with `detectShadows=True`) are treated as foreground, inflating bounding boxes.

**Why it happens:** `cv2.threshold` with `THRESH_BINARY` at threshold 127 will include shadows if threshold < 127.

**How to avoid:** Threshold at 200 (or use `THRESH_BINARY` with threshold=200) to exclude shadow pixels (127) and keep only foreground (255).

**Warning signs:** Bounding boxes extend far beyond the visible fish, especially toward light sources.

### Pitfall 4: SAM2 `multimask_output` with Multiple Boxes

**What goes wrong:** Calling `predictor.predict(box=boxes_array, multimask_output=True)` with multiple boxes returns ambiguous mask stacks.

**Why it happens:** When multiple boxes are provided, `multimask_output` behavior changes — SAM2 returns one mask per box when `multimask_output=False`.

**How to avoid:** Always set `multimask_output=False` when passing multiple bounding boxes. When passing a single box and wanting multiple mask proposals, set `multimask_output=True`.

**Warning signs:** Mask output shape is unexpected (extra dimension); masks don't align with individual boxes.

### Pitfall 5: Fortran-Order Requirement for pycocotools RLE

**What goes wrong:** `pycocotools.mask.encode(mask)` returns garbage RLE if `mask` is C-contiguous (row-major).

**Why it happens:** COCO RLE encoding traverses the mask column-by-column (Fortran order). If the array is C-contiguous, `encode` will either error or silently produce incorrect output.

**How to avoid:** Always call `np.asfortranarray(mask.astype(np.uint8))` before encoding.

**Warning signs:** Decoded mask looks like a transposition of the original mask.

### Pitfall 6: Temporal Frame Ordering for MOG2

**What goes wrong:** Background model does not converge because frames are fed in non-sequential order.

**Why it happens:** MOG2 is a temporal model that expects frames in video order to learn the background distribution. Random access to pre-extracted frames breaks this.

**How to avoid:** The `MOG2Detector` must receive frames in sequential temporal order per camera. The caller is responsible for ensuring this when providing pre-extracted numpy arrays. Add an assertion or docstring warning in the public API.

**Warning signs:** Very high or very low foreground detection rates from the first frame of every sequence.

### Pitfall 7: Detectron2 on Windows

**What goes wrong:** Detectron2 fails to compile or import on Windows due to missing MSVC tools, CUDA version mismatches, or missing Linux-specific system calls.

**Why it happens:** Detectron2 explicitly does not support Windows in official documentation. Last release was v0.6, October 2021 — CUDA 11.x era.

**How to avoid:** Use `torchvision.models.detection.maskrcnn_resnet50_fpn_v2` instead. It is functionally equivalent for this use case, is actively maintained, and is already a project dependency.

**Warning signs:** `import detectron2` fails with `ModuleNotFoundError` or C extension build errors.

---

## Code Examples

Verified patterns from official sources:

### MOG2 Shadow Pixel Handling

```python
# Source: OpenCV docs https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
fg_mask = subtractor.apply(frame)
# Shadow pixels are set to 127; foreground is 255; background is 0
# To suppress shadows: threshold at 200
_, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
```

### SAM2 Box Prompts (multiple boxes in one call)

```python
# Source: https://github.com/facebookresearch/sam2 README
# input_boxes: np.ndarray of shape (N, 4), dtype=float32, format [x1, y1, x2, y2]
input_boxes = np.array([[x1, y1, x2, y2], ...], dtype=np.float32)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=input_boxes,
        multimask_output=False,
    )
# masks.shape: (N, 1, H, W)
```

### torchvision Mask R-CNN Head Replacement

```python
# Source: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes=2)
```

### pycocotools RLE Encode/Decode

```python
# Source: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
import pycocotools.mask as mask_util
import numpy as np

# Encode
mask_f = np.asfortranarray(binary_mask.astype(np.uint8))  # MUST be Fortran order
rle = mask_util.encode(mask_f)  # {"size": [H, W], "counts": bytes}

# Decode
decoded = mask_util.decode(rle)  # uint8 array, 0/1
```

### Label Studio BrushLabels Pre-annotation JSON

```python
# Source: https://labelstud.io/guide/predictions
# Source: https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/brush.py
from label_studio_converter.brush import mask2rle

rle = mask2rle(mask_uint8)  # mask_uint8: HxW, values 0 or 255
task = {
    "data": {"image": "s3://...or/local/path/frame.jpg"},
    "predictions": [{
        "result": [{
            "type": "brushlabels",
            "value": {
                "format": "rle",
                "rle": rle,
                "brushlabels": ["fish"],
            },
            "original_width": W,
            "original_height": H,
            "from_name": "label",
            "to_name": "image",
        }]
    }]
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SAM (ViT-H, 636M params) | SAM2.1 Hiera variants (38-224M params) | Sept 2024 | SAM2 is 6x faster at equivalent quality; SAM2.1 improves further; use SAM2.1, not original SAM |
| Detectron2 for Mask R-CNN | torchvision `maskrcnn_resnet50_fpn_v2` | 2022+ | torchvision now ships first-class detection models; Detectron2 unmaintained since 2021 |
| `torchvision.transforms` (v1) | `torchvision.transforms.v2` | torchvision >=0.15 | v2 supports joint image+mask+box transforms; v1 only transforms images |
| `pretrained=True` kwarg | `weights=ModelWeights.DEFAULT` | torchvision >=0.13 | `pretrained` deprecated; use `weights=` kwarg |

**Deprecated/outdated:**
- `torchvision.models.detection.maskrcnn_resnet50_fpn` with `pretrained=True`: deprecated — use `weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT`
- Original SAM (segment-anything, not sam2): superseded by SAM2; don't use for new work
- Detectron2: last release 2021, Windows unsupported — do not use in this project

---

## Open Questions

1. **Detectron2 vs. torchvision Mask R-CNN — locked decision review**
   - What we know: The CONTEXT.md specifies "Mask R-CNN (Detectron2)" but Detectron2 is unmaintained and officially unsupported on Windows (the dev platform). torchvision provides an equivalent implementation.
   - What's unclear: Whether the user has a specific reason to prefer Detectron2 (e.g., existing familiarity, planned use of Detectron2-specific features like panoptic or PointRend).
   - Recommendation: **Plan for torchvision Mask R-CNN** (same ResNet-50 FPN backbone, same training outcomes, cross-platform). Flag this in planning notes. The CONTEXT.md says "Mask R-CNN (Detectron2)" — the planner should surface this as a deviation with justification and ask user to confirm.

2. **SAM2 on Windows with CUDA + bfloat16**
   - What we know: SAM2 uses `torch.autocast("cuda", dtype=torch.bfloat16)`. bfloat16 requires Ampere+ GPU (RTX 3000+) or falls back to float16.
   - What's unclear: The user's GPU generation — if it's pre-Ampere, bfloat16 may not be supported.
   - Recommendation: Use `dtype=torch.float16` as fallback. Add a runtime check in the SAMPseudoLabeler.

3. **Temporal sampling rate N (every Nth frame)**
   - What we know: N is explicitly deferred to dataset-size assessment. The dataset is 13 cameras of aquarium footage.
   - What's unclear: Total video duration and whether annotating every 5th vs 10th vs 30th frame significantly affects IoU targets.
   - Recommendation: Plan for a configurable N with a reasonable default (N=10 = ~3fps at 30fps). Make this a parameter to the pseudo-labeling script, not a hardcoded constant.

4. **Label Studio deployment — local pip vs. Docker**
   - What we know: Both work; Docker adds volume-mount complexity but isolates the database.
   - What's unclear: Whether the annotator needs network access (remote annotation) or is local.
   - Recommendation: Plan for `pip install label-studio` (local), document the Docker alternative in a comment. Local SQLite database is sufficient for this project scale.

---

## Sources

### Primary (HIGH confidence)
- SAM2 official GitHub README (https://github.com/facebookresearch/sam2/blob/main/README.md) — model variants, checkpoint sizes, FPS benchmarks, SAM2ImagePredictor API
- torchvision official docs (https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html) — model API, weights, fine-tuning parameters
- OpenCV official docs (https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html) — MOG2 parameters, shadow pixel value (127), threshold pattern
- Label Studio official docs (https://labelstud.io/guide/export, https://labelstud.io/guide/predictions) — export formats, pre-annotation import format
- label-studio-converter brush.py (https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/brush.py) — mask2rle, decode_rle functions
- pycocotools cocoapi mask.py (https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py) — RLE encode/decode, Fortran-order requirement

### Secondary (MEDIUM confidence)
- Detectron2 install docs (https://detectron2.readthedocs.io/en/latest/tutorials/install.html) — version v0.6, Linux/macOS requirement, Windows unofficial
- GitHub Issue: Detectron2 Windows support (https://github.com/facebookresearch/detectron2/issues/4015) — confirms no official Windows support
- Label Studio community forum — BrushLabels RLE import (https://community.labelstud.io/t/how-to-use-api-post-an-image-with-segmentation-mask-in-rle-format-to-label-studio-project/248) — RLE format requirement confirmed

### Tertiary (LOW confidence)
- WebSearch result: MOG2 minimum area thresholds — not from official source; treat as starting point only
- WebSearch result: SAM2 bfloat16 fallback behavior on pre-Ampere GPUs — not verified against official SAM2 docs

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all core libraries verified against official docs or official GitHub
- Architecture: HIGH — patterns derived directly from official API documentation
- Pitfalls: MEDIUM — MOG2 shadow pitfall and pycocotools Fortran order verified; Label Studio RLE mismatch verified via community + converter source; Detectron2 Windows issue verified via official docs and GitHub issues
- Training hyperparameters: LOW — Claude's discretion items; reasonable starting points but must be tuned empirically

**Research date:** 2026-02-19
**Valid until:** 2026-05-19 (90 days — stable libraries; SAM2 checkpoints unlikely to change significantly)
