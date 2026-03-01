# Feature Research

**Domain:** 3D fish pose estimation — replacing custom U-Net segmentation and keypoint regression with Ultralytics-native YOLOv8-seg and YOLOv8-pose (v3.0 Ultralytics Unification)
**Researched:** 2026-03-01
**Confidence:** HIGH (Ultralytics documentation verified; fish-domain specifics from primary literature)

---

## Milestone Scope

This is a subsequent-milestone research document for **v3.0 Ultralytics Unification**. The following are already shipped and NOT re-researched here:

- YOLO v8n standard bbox detection (`YOLODetector`)
- YOLO-OBB detection (`YOLOOBBDetector`) — v2.2
- U-Net segmentation + skeletonize midline — **being replaced**
- Custom keypoint regression head (`UNetKeypointHead`) — **being replaced**
- `Midline2D` dataclass with optional confidences
- Confidence-weighted DLT triangulation
- `aquapose train` CLI group with unet/yolo-obb/keypoint subcommands
- Multi-view triangulation + B-spline reconstruction
- 5-stage pipeline: Detection → 2D Tracking → Association → Midline → Reconstruction

**What this milestone adds:** YOLOv8-seg as instance segmentation backend, YOLOv8-pose as keypoint-based midline backend, training data preparation for both tasks, pipeline integration as swappable backends.

---

## Context: Who Are the "Users"?

AquaPose users for this milestone are the researchers themselves. "Table stakes" means: missing this makes the Ultralytics backend unusable or produces outputs that break the existing pipeline. "Differentiators" are features that make YOLOv8-seg/pose qualitatively better than the custom U-Net models they replace.

---

## Background: How YOLOv8-seg and YOLOv8-pose Work

### YOLOv8-seg

YOLOv8-seg extends the YOLO detection head with a segmentation branch. Each detected object receives:
- A bounding box (same as standard YOLO detect)
- A per-instance binary mask at the input image resolution

The mask is generated via a prototype-based approach (not pixel-level decoder like U-Net): 32 prototype masks are predicted globally, then per-instance coefficients linearly combine them and crop to the bounding box. The result is a full-image-size mask tensor but only the bbox region has meaningful data.

**Inference outputs (Python API):**
```python
results = model("image.jpg")
for result in results:
    result.masks.xy      # list of (K,2) polygon arrays, one per instance
    result.masks.xyn     # same, normalized [0,1]
    result.masks.data    # (N_instances, H, W) bool tensor, full image size
    result.boxes         # bounding boxes, same as detect
```

**Model sizes (YOLOv8-seg, pretrained COCO):**
- yolov8n-seg: 3.4M params — nano, fastest
- yolov8s-seg: 11.8M params — small
- yolov8m-seg: 27.3M params — medium
- yolov8l-seg: 46.0M params — large
- yolov8x-seg: 71.8M params — xlarge, most accurate

**Training data format (YOLO segmentation):**
One `.txt` file per image. Each row is one instance:
```
<class_idx> <x1> <y1> <x2> <y2> ... <xN> <yN>
```
Where `x1 y1 ... xN yN` are normalized `[0,1]` polygon vertices tracing the object contour. Minimum 3 vertices. Polygon is stored as a closed contour (no need to repeat first point).

### YOLOv8-pose

YOLOv8-pose extends the YOLO detection head with a keypoint regression branch. Each detected object receives:
- A bounding box
- N keypoints, each with (x, y) or (x, y, visibility)

The number and meaning of keypoints is entirely user-defined via `kpt_shape` in the dataset YAML. The default pretrained model uses 17 COCO human body keypoints, but custom counts are fully supported (fish midline points, anatomical landmarks, etc.).

**Inference outputs (Python API):**
```python
results = model("image.jpg")
for result in results:
    result.keypoints.xy    # (N_instances, N_kpts, 2) — pixel coords
    result.keypoints.xyn   # (N_instances, N_kpts, 2) — normalized coords
    result.keypoints.data  # (N_instances, N_kpts, 3) — x, y, confidence
    result.boxes           # bounding boxes
```

**Training data format (YOLO pose):**
One `.txt` file per image. Each row is one instance:
```
<class_idx> <cx> <cy> <w> <h> <px1> <py1> [<vis1>] <px2> <py2> [<vis2>] ... <pxN> <pyN> [<visN>]
```
Where `cx cy w h` is the normalized bounding box center/size, and `px py` are normalized keypoint coordinates. Visibility is optional: `0` = not labeled/occluded, `1` = labeled but possibly occluded, `2` = labeled and visible.

**Dataset YAML requires `kpt_shape`:**
```yaml
kpt_shape: [6, 3]   # 6 keypoints, each with x/y/visibility
```
For the fish domain, 6 anatomical keypoints along the spine (head → tail) is the established choice from v2.2 planning.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that anyone using the Ultralytics backends will assume exist. Missing these makes the milestone feel incomplete or broken.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| YOLOv8-seg inference on full frames returns per-fish instance masks | Core output of instance segmentation; masks replace U-Net output | LOW | `result.masks.data[i]` is (H,W) bool tensor; must filter to fish class; one mask per detected fish instance |
| YOLOv8-seg mask extraction in crop coordinates | MidlineStage receives crops from DetectionStage; seg model must return masks aligned to crop, not full frame | MEDIUM | Two approaches: (a) run seg on crop directly — mask is already crop-aligned; (b) run seg on full frame, then slice `masks.data[i]` to bbox region; approach (a) is simpler but changes inference granularity |
| YOLOv8-pose inference returns keypoints per fish | Core output of pose model; keypoints replace custom regression head output | LOW | `result.keypoints.data` is (N_instances, N_kpts, 3); each row is (x, y, confidence) in full-frame pixel coords |
| `kpt_shape: [6, 3]` configuration in dataset YAML | Custom keypoint count must be declared; model head dimension is set at training time | LOW | Cannot retrain with different kpt_shape without rebuilding model; choose keypoint count once and commit |
| YOLOv8-seg training data: polygon contours from SAM2 masks | Training pipeline must convert existing SAM2 binary masks to YOLO polygon format | MEDIUM | `ultralytics.data.converter.convert_segment_masks_to_yolo_seg()` exists for directory-level conversion; alternative: `cv2.findContours()` → normalize → write txt file per instance; existing SAM2 pseudo-labels are the raw material |
| YOLOv8-pose training data: keypoints derived from skeletonizer or medial axis | Pose model needs (x, y, visibility) keypoints; cannot reuse seg labels directly; must generate programmatically from masks | HIGH | Pipeline: SAM2 mask → skeletonize → BFS pruning → arc-length sample 6 points → normalize → write YOLO pose txt; this is the primary new annotation tooling; visibility can be set to 2 (visible) for all programmatically extracted points |
| Both models integrate as swappable backends in MidlineStage | Existing pipeline calls `midline_backend.extract(crop, detection)` → `Midline2D`; Ultralytics backends must satisfy the same interface | MEDIUM | Add `YOLOSegMidlineBackend` and `YOLOPoseMidlineBackend` classes implementing existing backend protocol; `make_midline_backend()` factory gains `"yolo_seg"` and `"yolo_pose"` branches |
| Mask → midline conversion for YOLOv8-seg backend | Seg model gives a mask; downstream still needs ordered midline points for triangulation; must run medial axis / skeletonization on the mask | MEDIUM | Same existing skeletonize → BFS → arc-length pipeline applied to the Ultralytics-provided mask; this is the same logic as the current U-Net path, reused unchanged |
| Keypoints → `Midline2D` conversion for YOLOv8-pose backend | Pose model gives (N_kpts, 3) array; must map to `Midline2D(points, confidences)` | LOW | Reshape `result.keypoints.data[0][:, :2]` → points, `result.keypoints.data[0][:, 2]` → confidences; straightforward |
| Training data format validation / dry-run check | Users must be able to verify label format is correct before committing to a full training run | LOW | Ultralytics provides `model.train(data="dataset.yaml")` which validates format on startup; add a `--dry-run` flag or lean on the built-in validation error messages |
| Custom class index = 0 (single-class fish dataset) | Both seg and pose datasets have exactly one class: fish; class index 0 throughout all label files | LOW | Dataset YAML: `nc: 1`, `names: ['fish']`; all label files start with `0 ...` |

### Differentiators (Competitive Advantage)

Features that make the Ultralytics backends qualitatively better than the custom models they replace.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Pretrained COCO backbone weights for seg and pose models | ImageNet/COCO pretrained weights transfer well to fish domain even with limited labeled data; custom U-Net trained from SAM2 pseudo-labels only; Ultralytics models start from a much stronger feature extractor | LOW | Use `yolov8n-seg.pt` or `yolov8n-pose.pt` as starting weights; pass `pretrained=True` (default); dramatically reduces labeled data needed to reach good generalization |
| Instance-aware masks from YOLOv8-seg (vs U-Net binary mask per crop) | U-Net operates on pre-cropped patches and produces one mask per call; YOLOv8-seg produces N instance masks in one forward pass on the full frame, separating overlapping fish | HIGH | When fish partially overlap, U-Net gets confused by the combined crop; YOLOv8-seg's instance-aware output can distinguish individual fish; critical for the 9-fish cichlid school scenario |
| Direct keypoint output from YOLOv8-pose (vs skeletonize + BFS from mask) | Skeletonization on a noisy mask produces disconnected or multi-branch skeletons requiring BFS pruning; YOLOv8-pose directly regresses ordered keypoint positions with confidence, bypassing the fragile skeletonization step | MEDIUM | Quality ceiling: YOLOv8-pose accuracy is bounded by pseudo-label quality; if pseudo-labels are generated from skeletonizer, that noise is baked in; however, training forces consistency and smoothing across the dataset |
| Per-keypoint confidence from YOLOv8-pose (native output) | `result.keypoints.data[:, :, 2]` gives per-point confidence natively; flows into existing confidence-weighted DLT triangulation without custom head architecture | LOW | Already integrated: `Midline2D.confidences` field and weighted DLT exist from v2.2; YOLOv8-pose provides the confidence naturally, avoiding the need for a custom sigmoid output head |
| Unified Ultralytics training API for all model types | `model.train(data="fish.yaml", epochs=100, ...)` works identically for detect, seg, and pose; no custom training loop code required | LOW | Eliminates custom `train_unet()` function, custom loss, custom checkpointing; Ultralytics handles mixed precision, cosine LR, augmentation, validation loop; simplifies `aquapose train` subcommands dramatically |
| Native augmentation in Ultralytics training (Mosaic, MixUp, Albumentations) | Ultralytics applies aggressive augmentation by default; custom U-Net training used basic augmentation only | LOW | Mosaic augmentation combines 4 images; for fish domain, this can be disabled or reduced (`mosaic=0.0`) if it creates unrealistic multi-fish images; default augmentation otherwise suitable |
| Export to ONNX / TensorRT for deployment | Ultralytics supports `model.export(format="onnx")` and `format="engine"` for TensorRT; custom U-Net required separate export tooling | LOW | Out of scope for this milestone but architecture supports it; note for future |
| Single-model multi-fish inference (YOLOv8-seg and -pose run on full frame) | Instead of running U-Net once per detection crop (N=9 times per frame per camera), seg and pose models run once per full frame and return all instances | MEDIUM | Changes inference granularity from crop-level to frame-level; reduces total inference calls from O(N_fish) to O(1) per frame per camera; must update MidlineStage to consume frame-level results and match instances to tracked fish |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Run YOLOv8-seg on pre-cropped patches (same as U-Net) | Seems like the simplest drop-in swap — just replace U-Net with YOLO-seg on the same crop | YOLO-seg is designed for full-frame inference; running on crops means re-running the full model N times per frame, losing the instance-separation advantage; crops at the edge of the frame also lose context the model needs for good boundary detection | Run YOLOv8-seg on the full frame once, extract the mask for each fish by matching bounding boxes to tracked fish identities |
| Manual keypoint annotation for pose training | Highest quality labels; standard in pose estimation literature | 9 fish × 30fps × hours of footage = millions of frames; manual keypoint annotation is infeasible at this scale | Programmatic pseudo-label generation: SAM2 mask → medial axis → arc-length sample 6 keypoints → YOLO pose txt; imperfect but scalable |
| Using COCO pretrained pose model weights directly without fine-tuning | Pretrained model might "just work" on fish | COCO pose model has `kpt_shape: [17, 3]` for human body; fish requires `kpt_shape: [6, 3]`; kpt_shape is baked into the model head at training time — cannot use COCO pose weights with a different keypoint count | Fine-tune from COCO detect backbone weights (not pose weights); or use `yolov8n.pt` (detect) as base and train the pose head from scratch with the correct kpt_shape |
| Joint seg+pose model in a single forward pass | Architecturally attractive; some community implementations exist (DmitryCS/yolov8_segment_pose on GitHub) | Not officially supported in Ultralytics; requires custom model surgery; maintenance burden; Ultralytics does not expose combined seg+pose heads in the standard API | Run seg and pose as separate models; if inference speed is a concern, run pose only (keypoints give enough for midline) and skip seg |
| Replacing skeletonize midline with pose-only and removing the seg path entirely | Simplifies codebase; pose is the "better" approach | Seg masks are still needed for other consumers (width profile extraction, visualization overlays, SAM2 bootstrapping); removing seg removes capabilities that are useful even after pose model is deployed | Keep both backends; select via `midline_backend: yolo_seg` or `midline_backend: yolo_pose` in config; default to yolo_pose, but preserve yolo_seg path |
| Retraining on all 13 cameras simultaneously as one dataset | More data = better model; seems obvious | Camera geometry varies significantly; top-center camera (e3v8250) is wide-angle overhead producing very different fish appearances and was already identified as a quality problem and skipped in v1.0; mixing it in degrades per-camera model quality | Per-camera datasets or pooled from only the 12 ring cameras; skip e3v8250 as established |
| YOLO11 instead of YOLOv8 for seg/pose | YOLO11 is 2024 current and achieves ~2-3% higher mAP with fewer parameters | The milestone is "Ultralytics Unification" — the key is consistent, working integration, not maximum mAP; YOLOv8 has more community documentation for fish/animal pose; YOLO11 API is identical (same `model.train()` call) so upgrading later is trivial | Use YOLOv8n-seg and YOLOv8n-pose for this milestone to minimize variables; document YOLO11 as a trivial future upgrade |

---

## Feature Dependencies

```
YOLOv8-seg Backend
    └──requires──> SAM2 masks → YOLO polygon label conversion tooling
    └──requires──> Frame-level inference path in MidlineStage (not crop-level)
    └──requires──> Instance matching: seg output detections → tracked fish IDs
    └──outputs──>  binary mask per fish instance
                       └──feeds──> existing skeletonize → BFS → Midline2D path (REUSED)
                       └──feeds──> width profile extraction (REUSED)

YOLOv8-pose Backend
    └──requires──> Keypoint pseudo-label generation tooling:
                       SAM2 mask → medial axis → arc-length sample 6 pts → YOLO pose txt
    └──requires──> kpt_shape: [6, 3] in dataset YAML (committed choice, cannot change later)
    └──requires──> Frame-level inference path in MidlineStage
    └──requires──> Instance matching: pose output detections → tracked fish IDs
    └──outputs──>  (N_kpts, 3) array: x, y, confidence per fish
                       └──feeds──> Midline2D(points=kpts[:,:2], confidences=kpts[:,2])
                       └──feeds──> existing confidence-weighted DLT (REUSED)

Frame-Level Instance Matching
    └──requires──> AssociationStage has already resolved fish identity across cameras
    └──requires──> DetectionStage bounding boxes available for IoU matching
    └──independent of──> which midline backend is selected

Keypoint Pseudo-Label Generation
    └──requires──> SAM2 pseudo-labels (already exist from v1.0)
    └──reuses──>   existing skeletonize + BFS + arc-length sampling code from MidlineStage
    └──outputs──>  YOLO pose .txt files for training

Seg Polygon Label Generation
    └──requires──> SAM2 binary masks (already exist from v1.0)
    └──uses──>     cv2.findContours() or ultralytics.data.converter.convert_segment_masks_to_yolo_seg()
    └──outputs──>  YOLO seg .txt files for training

Training (seg and pose)
    └──requires──> dataset YAML with correct kpt_shape (pose) or nc/names (seg)
    └──requires──> label files in YOLO format
    └──uses──>     ultralytics model.train() — standard API, no custom training loop
    └──outputs──>  best.pt weights
```

### Dependency Notes

- **Instance matching is the hardest new integration point:** When running seg/pose on a full frame, the returned instances are ordered by detection confidence, not by fish identity. Must match `result.boxes` to tracked fish bounding boxes (IoU or centroid distance) to assign the right mask/keypoints to the right `Midline2D`. This is a new step not present in the U-Net crop-based pipeline.
- **Pose pseudo-label quality determines pose model quality ceiling:** If the skeletonizer produces noisy or incomplete medial axes on SAM2 masks, those errors are baked into the pose training labels. The pose model cannot be better than its labels. Consider filtering: only use frames where the skeletonizer produces clean single-branch medial axes.
- **kpt_shape must be decided before any training begins:** 6 keypoints along the midline is the established choice. This is irreversible without retraining.
- **Seg and pose backends do not compete — both are useful:** Seg masks provide body silhouette (width profile, visualization); pose keypoints provide ordered midline (triangulation input). The long-term architecture may run both and use pose for midline, seg for width. Keep both.
- **Existing skeletonize path is REUSED in seg backend:** YOLOv8-seg replaces U-Net inference only; the mask→midline conversion (skeletonize + BFS + arc-length) is unchanged. This limits the improvement: seg helps when the mask quality is better, but noisy masks still produce noisy midlines.

---

## MVP Definition (for v3.0 Ultralytics Unification)

### Launch With

- [ ] `convert_sam_masks_to_yolo_seg()` — batch converts existing SAM2 binary masks to YOLO polygon label files
- [ ] `generate_pose_labels_from_masks()` — batch generates YOLO pose labels from SAM2 masks via existing skeletonize + arc-length pipeline; outputs (6, 3) with visibility=2
- [ ] Dataset YAML files for fish-seg and fish-pose with correct `nc`, `names`, `kpt_shape`
- [ ] `aquapose train seg` and `aquapose train pose` CLI subcommands using `model.train()`
- [ ] `YOLOSegMidlineBackend` — runs yolov8n-seg on full frame, matches instances to tracked boxes, extracts mask, runs existing skeletonize path → `Midline2D`
- [ ] `YOLOPoseMidlineBackend` — runs yolov8n-pose on full frame, matches instances to tracked boxes, maps keypoints → `Midline2D(points, confidences)`
- [ ] `make_midline_backend()` factory updated: `"yolo_seg"` and `"yolo_pose"` branches, each configurable with a `model_path`
- [ ] Instance matching utility: `match_detections_to_tracks(result_boxes, tracked_bboxes)` → index mapping using IoU; shared between seg and pose backends
- [ ] Integration test: run pose backend on a synthetic frame, verify `Midline2D` output has correct shape and flows through triangulation unchanged

### Add After Validation

- [ ] Width profile extraction from YOLOv8-seg masks — seg masks are higher quality than U-Net masks; improves B-spline width profile; defer until seg model is trained and quality is assessed
- [ ] Filter pose pseudo-labels by skeleton quality (single-branch, minimum arc length) — improves pose model training data; defer until label generation pipeline is working
- [ ] `aquapose train pose --transfer-from-detect` — initialize pose backbone from YOLO detect weights; improves convergence; add once basic training pipeline works

### Future Consideration (v3.1+)

- [ ] Upgrade to YOLO11-seg and YOLO11-pose — trivial API-compatible swap; defer until v3.0 is validated
- [ ] Run seg+pose simultaneously for combined width+midline output in a single inference pass — requires instance matching once, not twice
- [ ] Pose label refinement via human annotation on a small subset — addresses quality ceiling of pseudo-labels
- [ ] Confidence calibration on pose keypoints (temperature scaling) — useful if confidence values are systematically over- or under-confident

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Seg polygon label generation from SAM2 masks | HIGH | LOW | P1 |
| Pose keypoint label generation from SAM2 masks | HIGH | MEDIUM | P1 |
| Dataset YAMLs (seg and pose) | HIGH | LOW | P1 |
| `aquapose train seg` subcommand | HIGH | LOW | P1 |
| `aquapose train pose` subcommand | HIGH | LOW | P1 |
| `YOLOPoseMidlineBackend` (keypoints → Midline2D) | HIGH | MEDIUM | P1 |
| Instance matching utility (result_boxes → track IDs) | HIGH | MEDIUM | P1 |
| `YOLOSegMidlineBackend` (mask → skeletonize → Midline2D) | MEDIUM | MEDIUM | P1 |
| `make_midline_backend()` factory update | MEDIUM | LOW | P1 |
| Integration test: pose backend through triangulation | HIGH | LOW | P1 |
| Width profile from seg masks | MEDIUM | MEDIUM | P2 |
| Pose label quality filtering | MEDIUM | LOW | P2 |
| YOLO11 upgrade | LOW | LOW | P3 |
| Joint seg+pose inference | LOW | HIGH | P3 |

**Priority key:**
- P1: Must ship in v3.0
- P2: Add when core is validated
- P3: Future milestone

---

## Technical Implementation Notes

### Seg Label Generation from Binary Masks

```python
import cv2
import numpy as np

def mask_to_yolo_polygon(binary_mask: np.ndarray, class_idx: int = 0) -> str:
    """Convert a (H, W) binary mask to a YOLO segmentation label row."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return ""
    # Take largest contour (outermost fish boundary)
    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim < 2:
        return ""
    h, w = binary_mask.shape
    normalized = contour.astype(float) / np.array([w, h])
    flat = normalized.flatten()  # x1, y1, x2, y2, ...
    coords_str = " ".join(f"{v:.6f}" for v in flat)
    return f"{class_idx} {coords_str}"
```

Alternatively, use Ultralytics built-in utility (handles directory-level conversion):
```python
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
convert_segment_masks_to_yolo_seg(
    masks_dir="path/to/binary_masks/",
    output_dir="path/to/labels/",
    classes=1  # single fish class
)
```

### Pose Label Generation from Masks

```python
def mask_to_pose_label(
    binary_mask: np.ndarray,
    n_keypoints: int = 6,
    class_idx: int = 0
) -> str:
    """
    Convert a (H, W) binary mask to a YOLO pose label row.
    Uses existing medial axis extraction to derive ordered spine keypoints.
    """
    from aquapose.reconstruction.midline import extract_medial_axis  # existing code
    from aquapose.reconstruction.midline import arc_length_sample     # existing code

    h, w = binary_mask.shape
    # Existing skeletonize + BFS pipeline
    medial_pts = extract_medial_axis(binary_mask)    # (M, 2) ordered points
    if medial_pts is None or len(medial_pts) < n_keypoints:
        return ""
    # Arc-length resample to exactly n_keypoints
    kpts = arc_length_sample(medial_pts, n_keypoints)  # (N, 2) pixel coords

    # Bounding box from mask (required for YOLO pose format)
    ys, xs = np.where(binary_mask)
    cx = (xs.min() + xs.max()) / 2 / w
    cy = (ys.min() + ys.max()) / 2 / h
    bw = (xs.max() - xs.min()) / w
    bh = (ys.max() - ys.min()) / h

    # Normalize keypoints, set visibility=2 (labeled and visible)
    kpts_norm = kpts / np.array([w, h])
    kpts_str = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in kpts_norm)
    return f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kpts_str}"
```

### Instance Matching: Seg/Pose Results to Tracked Fish

When YOLOv8-seg or -pose runs on the full frame, it returns detections in confidence order (not fish identity order). Must match to `tracked_bboxes` from the 2D tracking stage:

```python
def match_detections_to_tracks(
    result_boxes: np.ndarray,   # (N_det, 4) x1y1x2y2 from result.boxes.xyxy
    tracked_boxes: np.ndarray,  # (N_tracks, 4) x1y1x2y2 from tracking stage
    iou_threshold: float = 0.3,
) -> dict[int, int]:
    """Returns {track_idx: detection_idx} mapping."""
    # Compute pairwise IoU, assign greedily by max IoU
    ...
```

This utility is shared between `YOLOSegMidlineBackend` and `YOLOPoseMidlineBackend`.

### Dataset YAML Example (pose)

```yaml
path: /path/to/fish_pose_dataset
train: images/train
val: images/val

nc: 1
names: ['fish']

kpt_shape: [6, 3]   # 6 keypoints, each (x, y, visibility)
```

### Training (standard Ultralytics API, no custom loop)

```python
from ultralytics import YOLO

# Segmentation
model = YOLO("yolov8n-seg.pt")
model.train(
    data="fish_seg.yaml",
    epochs=100,
    imgsz=640,
    device="cuda",
    project="runs/seg",
    name="fish_v1",
)

# Pose
model = YOLO("yolov8n-pose.pt")  # NOTE: kpt_shape mismatch — use yolov8n.pt instead
# When kpt_shape differs from pretrained, start from detect weights:
model = YOLO("yolov8n.pt")
model.train(
    data="fish_pose.yaml",  # kpt_shape: [6, 3] declared here
    task="pose",
    epochs=100,
    imgsz=640,
    device="cuda",
    project="runs/pose",
    name="fish_v1",
)
```

**Important:** `yolov8n-pose.pt` has its pose head configured for 17 COCO keypoints. Training with `kpt_shape: [6, 3]` (6 keypoints) requires either (a) using a detection backbone as base (`yolov8n.pt`) or (b) accepting that the pretrained pose head weights will be discarded anyway since the head dimensions don't match. Use option (a): start from `yolov8n.pt` with `task="pose"`.

### YOLOv8-pose vs YOLOv8-seg for Midline Extraction: Decision

| Criterion | YOLOv8-seg backend | YOLOv8-pose backend |
|-----------|-------------------|---------------------|
| Label generation | Automatic from SAM2 contours | Requires skeletonizer pseudo-labels (noisier) |
| Midline quality | Depends on skeletonize quality (same as current U-Net path) | Directly regresses ordered keypoints; smoother |
| Per-point confidence | Derived from mask quality (no native confidence) | Native per-keypoint confidence; feeds weighted DLT |
| Training data scale | Large (all SAM2 masks usable) | Medium (must filter to clean skeletonizer outputs) |
| Complexity to integrate | Medium (instance matching + existing skeletonize reused) | Medium (instance matching + keypoint → Midline2D) |
| **Recommended primary** | Secondary (useful for width profile) | **Primary** (better midline; native confidence) |

**Recommendation:** YOLOv8-pose is the primary midline backend. YOLOv8-seg is the secondary backend (width profile extraction + fallback mask). Both should be implemented.

---

## Competitor Feature Analysis

| Feature | DeepLabCut + Anipose | SLEAP + 3D | AquaPose v2.2 | AquaPose v3.0 Target |
|---------|---------------------|-----------|---------------|----------------------|
| Segmentation | None (no masks) | None (no masks) | U-Net (IoU 0.623) | YOLOv8-seg (instance-aware) |
| Midline extraction | Fixed anatomical keypoints (heatmap) | Fixed anatomical keypoints (heatmap) | Skeletonize + BFS | YOLOv8-pose keypoints (direct regression) |
| Per-point confidence | Yes (heatmap peak) | Yes | No (v2.1), Yes (v2.2 custom head) | Yes (native YOLOv8-pose output) |
| Training data prep | Label Studio / DLC GUI | SLEAP GUI labeler | Manual scripts | `aquapose train seg/pose` with pseudo-labels |
| Multi-fish instance separation | Manual ID assignment | Manual ID assignment | U-Net per crop (no separation) | YOLOv8-seg instance-aware masks |
| Refractive optics | No | No | Yes | Unchanged |
| Training framework | TensorFlow (DeepLabCut) | TF/PyTorch | Custom PyTorch | Ultralytics (unified) |

---

## Sources

- [Ultralytics YOLOv8 Instance Segmentation Task Docs](https://docs.ultralytics.com/tasks/segment/) — output format, model sizes, training (HIGH confidence — official docs)
- [Ultralytics YOLOv8 Pose Estimation Task Docs](https://docs.ultralytics.com/tasks/pose/) — output format, kpt_shape, keypoint API (HIGH confidence — official docs)
- [Ultralytics Pose Estimation Datasets Overview](https://docs.ultralytics.com/datasets/pose/) — kpt_shape YAML field, annotation format with visibility (HIGH confidence — official docs)
- [Ultralytics Simple Utilities: convert_segment_masks_to_yolo_seg](https://docs.ultralytics.com/usage/simple-utilities/) — binary mask → YOLO seg format conversion (HIGH confidence — official docs)
- [Ultralytics YOLOv8 vs YOLO11 Comparison](https://docs.ultralytics.com/compare/yolov8-vs-yolo11/) — YOLO11 is 22% faster, ~2% higher mAP, API-identical to YOLOv8 (HIGH confidence — official docs)
- [Animal Pose Estimation: Fine-tuning YOLOv8 Pose Models — LearnOpenCV](https://learnopencv.com/animal-pose-estimation/) — custom kpt_shape for non-human subjects, animal pose training workflow (MEDIUM confidence — verified tutorial)
- [Train Custom YOLOv8 Pose — Roboflow Blog](https://blog.roboflow.com/train-a-custom-yolov8-pose-estimation-model/) — custom keypoint annotation and training workflow (MEDIUM confidence)
- [Improved YOLOv8-Pose for Albacore Tuna — MDPI Marine Science 2024](https://www.mdpi.com/2077-1312/12/5/784) — fish-domain validation; head/jaw/tail keypoints extracted via YOLOv8-pose (MEDIUM confidence — peer-reviewed, paywalled, fetched summary only)
- [SAM segmentation masks to YOLO format — Ultralytics Discussion #6421](https://github.com/orgs/ultralytics/discussions/6421) — community pattern for SAM mask → YOLO seg conversion (MEDIUM confidence)
- [YOLOv8 combined seg+pose community implementation](https://github.com/DmitryCS/yolov8_segment_pose) — joint seg+pose model exists but is not officially supported; cited as anti-feature rationale (LOW confidence — community project)
- Existing codebase: `src/aquapose/reconstruction/midline.py`, `segmentation/training.py`, `core/pipeline_context.py`, `.planning/PROJECT.md` — authoritative for existing interfaces (HIGH confidence)

---

*Feature research for: AquaPose v3.0 Ultralytics Unification — YOLOv8-seg and YOLOv8-pose as production backends*
*Researched: 2026-03-01*
