# Architecture Research

**Domain:** Ultralytics YOLOv8-seg and YOLOv8-pose integration into existing AquaPose 5-stage pipeline
**Researched:** 2026-03-01
**Confidence:** HIGH (existing codebase read directly; Ultralytics API verified against official docs)

---

## Existing Architecture: What v3.0 Inherits (v2.1 Baseline)

### System Overview

```
PosePipeline (engine/pipeline.py)
    |
    |-- Stage 1: DetectionStage        (core/detection/)
    |       backends/yolo.py           <- wraps YOLODetector (segmentation/detector.py)
    |       backends/yolo_obb.py       <- wraps YOLO-OBB + affine crop
    |
    |-- Stage 2: TrackingStage         (core/tracking/)
    |       ocsort_wrapper.py          <- per-camera OC-SORT
    |
    |-- Stage 3: AssociationStage      (core/association/)
    |       scoring, clustering, refinement
    |
    |-- Stage 4: MidlineStage          (core/midline/)
    |       backends/segment_then_extract.py  <- UNetSegmentor + skeletonization
    |       backends/direct_pose.py           <- _PoseModel (UNet enc + regression head)
    |
    |-- Stage 5: ReconstructionStage   (core/reconstruction/)
            backends/triangulation.py
            backends/curve_optimizer.py

Observers (engine/*_observer.py) -- no stage coupling
PipelineContext (core/context.py)  -- accumulator passed through all stages
```

### Import Boundary (enforced by AST pre-commit hook)

```
core/     -> calibration/, reconstruction/, segmentation/ (NO engine/ imports)
engine/   -> core/, calibration/, reconstruction/, segmentation/, tracking/
cli.py    -> engine/ only
training/ -> core/, segmentation/, reconstruction/ (NOT engine/)
```

This boundary is the most critical architectural constraint. Every new file must be classified by layer before writing the first import statement.

### What Flows Into and Out of MidlineStage (Stage 4)

MidlineStage reads from PipelineContext:
- `context.detections` — per-frame per-camera `Detection` lists from Stage 1
- `context.tracklet_groups` — cross-camera fish identity clusters from Stage 3
- `context.camera_ids` — list of active camera IDs

MidlineStage writes to PipelineContext:
- `context.annotated_detections` — per-frame per-camera `AnnotatedDetection` lists

`AnnotatedDetection` contract (unchanged in v3.0):
```python
@dataclass
class AnnotatedDetection:
    detection: Detection           # source detection (bbox, confidence)
    mask: np.ndarray | None        # binary crop-space mask uint8 (H_crop, W_crop)
    crop_region: CropRegion | None # crop placement in full frame
    midline: Midline2D | None      # N-point midline in full-frame px coords
    camera_id: str
    frame_index: int
```

`Midline2D` contract (unchanged in v3.0):
```python
@dataclass
class Midline2D:
    points: np.ndarray             # (N, 2) float32 -- may contain NaN for gaps
    half_widths: np.ndarray        # (N,) float32
    fish_id: int
    camera_id: str
    frame_index: int
    is_head_to_tail: bool
    point_confidence: np.ndarray   # (N,) float32 in [0, 1]
```

**`Midline2D` and `AnnotatedDetection` are the unchanged data contracts downstream of Stage 4. ReconstructionStage (Stage 5) is completely unaffected by this migration.**

---

## What Changes vs What Stays in v3.0

**The 5-stage pipeline structure does not change.**
**The PipelineContext accumulator does not change.**
**The AnnotatedDetection and Midline2D data contracts do not change.**
**The backend pattern (get_backend factory, kind string) does not change.**
**The crop utilities (segmentation/crop.py) do not change.**
**All observers are completely unaffected.**
**Stages 1, 2, 3, 5 are completely unaffected.**

All changes are localized to:
- `core/midline/backends/` — replace two custom backends with two Ultralytics backends
- `segmentation/model.py` — delete _UNet, UNetSegmentor
- `training/` — delete unet.py, pose.py, datasets.py; add yolo_seg.py, yolo_pose.py, prep_seg.py, prep_pose.py
- `engine/config.py` (MidlineConfig) — add 3 new fields

---

## Component Change Map

### DELETED

| Component | Path | Why |
|-----------|------|-----|
| `_UNet`, `UNetSegmentor`, `MaskRCNNSegmentor` | `segmentation/model.py` | Replaced by Ultralytics seg model |
| `_PoseModel`, `KeypointDataset`, `train_pose()` | `training/pose.py` | Replaced by Ultralytics pose model |
| `BinaryMaskDataset`, `train_unet()` | `training/unet.py` | Replaced by Ultralytics seg training |
| `stratified_split`, `_load_image` | `training/datasets.py` | Custom dataset classes replaced |
| `SegmentThenExtractBackend` | `core/midline/backends/segment_then_extract.py` | Replaced by YOLOSegBackend |
| `DirectPoseBackend` | `core/midline/backends/direct_pose.py` | Replaced by YOLOPoseBackend |
| `unet`, `pose` CLI subcommands | `training/cli.py` | Replaced by `yolo-seg`, `yolo-pose` |

### ADDED

| Component | Path | Purpose |
|-----------|------|---------|
| `YOLOSegInferencer` | `segmentation/yolo_seg.py` | Wraps Ultralytics YOLO seg inference API |
| `YOLOSegBackend` | `core/midline/backends/yolo_seg.py` | New midline backend: YOLO seg -> skeletonization |
| `YOLOPoseBackend` | `core/midline/backends/yolo_pose.py` | New midline backend: YOLO pose -> spline fit |
| `train_yolo_seg()` | `training/yolo_seg.py` | Ultralytics seg training wrapper |
| `train_yolo_pose()` | `training/yolo_pose.py` | Ultralytics pose training wrapper |
| `prepare_seg_dataset()` | `training/prep_seg.py` | Convert binary masks -> YOLO polygon labels |
| `prepare_pose_dataset()` | `training/prep_pose.py` | Convert COCO JSON -> YOLO pose txt labels |
| `yolo-seg`, `yolo-pose` CLI subcommands | `training/cli.py` | Replacement training entrypoints |

### MODIFIED

| Component | Path | Change |
|-----------|------|--------|
| `MidlineConfig` | `engine/config.py` | Add `yolo_seg_model_path`, `yolo_pose_model_path`, `yolo_imgsz` fields |
| `get_backend()` registry | `core/midline/backends/__init__.py` | Add `"yolo_seg"` and `"yolo_pose"` entries |
| `build_stages()` | `engine/pipeline.py` | Wire new MidlineConfig fields to new backend kwargs |
| `load_config()` | `engine/config.py` | Add new path fields to project_dir resolution |

---

## Ultralytics API: What New Backends Must Know

### Segmentation Model Inference (HIGH confidence — official docs)

```python
from ultralytics import YOLO

model = YOLO("fish_seg.pt")   # loaded ONCE at __init__, not per frame

# Pass numpy BGR crop directly -- Ultralytics accepts np.ndarray
results = model(crop_bgr, verbose=False)   # returns list[Results]

result = results[0]
if result.masks is None or len(result.masks) == 0:
    return None   # no fish found in crop

# masks.data: torch.Tensor shape (num_detections, H_model, W_model)
# H_model/W_model are model resolution (e.g. 640), NOT crop dimensions
mask_tensor = result.masks.data[0]          # take highest-conf detection
mask_numpy = mask_tensor.cpu().numpy()      # always .cpu() before numpy

# Resize back to crop dimensions
import cv2
mask_uint8 = (mask_numpy > 0.5).astype(np.uint8) * 255
mask_crop = cv2.resize(mask_uint8, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
```

**Batch inference:** Pass a list of numpy arrays: `model(list_of_crops, verbose=False)`. Returns `list[Results]`, one per crop. This replaces the manual tensor stacking in the old `UNetSegmentor`.

**Training label format:** Binary masks (SAM2 pseudo-labels) must be converted to YOLO polygon format. Use `cv2.findContours` on each binary mask to extract contour, normalize to [0,1]. Label txt per line: `<class_idx> <x1> <y1> <x2> <y2> ... <xN> <yN>` (min 3 points). Ultralytics also provides `convert_segment_masks_to_yolo_seg()` utility.

### Pose Model Inference (HIGH confidence — official docs)

```python
from ultralytics import YOLO

model = YOLO("fish_pose.pt")   # loaded ONCE at __init__

results = model(crop_bgr, verbose=False)

result = results[0]
if result.keypoints is None or len(result.keypoints) == 0:
    return None

# keypoints.data: torch.Tensor shape (num_detections, num_keypoints, 3)
# dim 2 = [x_pixel, y_pixel, confidence] when kpt_shape=[N, 3]
kp_data = result.keypoints.data[0].cpu().numpy()   # shape (6, 3)
xy_crop = kp_data[:, :2]   # (6, 2) pixel coords in CROP SPACE (not normalized)
conf = kp_data[:, 2]       # (6,) confidence [0, 1]
```

**Critical coordinate system difference:** Ultralytics keypoints are in pixel coordinates (not normalized [0,1] as in the old `_PoseModel`). The downstream `invert_affine_points(kp_crop_px, affine.M)` call already expects pixel coords, so this is actually simpler — no denormalization step needed.

**Training label format:** YOLO pose txt per line: `<class> <cx> <cy> <bw> <bh> <kx1> <ky1> <kv1> ... <kx6> <ky6> <kv6>` (all normalized 0-1 except visibility kv which is 0 or 1). data.yaml must include `kpt_shape: [6, 3]`.

---

## New Component Boundaries

### segmentation/yolo_seg.py

```
YOLOSegInferencer
    __init__(model_path, confidence_threshold, device)
    segment_batch(crops: list[np.ndarray]) -> list[np.ndarray | None]
        Returns: per-crop binary mask uint8 (H_crop, W_crop), or None if no detection
```

Mirrors existing pattern: `YOLODetector` lives in `segmentation/detector.py`, wrapped by `core/detection/backends/yolo.py`. `YOLOSegInferencer` lives in `segmentation/`, wrapped by the midline backend. Raw Ultralytics API calls stay in `segmentation/`; pipeline logic stays in `core/`.

### core/midline/backends/yolo_seg.py

```
YOLOSegBackend
    __init__(model_path, confidence_threshold, n_points, min_area, device)
    process_frame(frame_idx, frame_dets, frames, camera_ids)
        -> dict[str, list[AnnotatedDetection]]
```

Internal data flow:
1. For each camera: `compute_crop_region` -> `extract_crop` (unchanged, segmentation/crop.py)
2. `YOLOSegInferencer.segment_batch(crops)` -> binary masks
3. For each mask: existing skeletonization pipeline unchanged (`_skeleton_and_widths`, `_longest_path_bfs`, `_resample_arc_length`, `_crop_to_frame` from `reconstruction/midline.py`)
4. Build `AnnotatedDetection` with mask + `Midline2D`

**The skeletonization logic does not change. Only the mask source changes.**

### core/midline/backends/yolo_pose.py

```
YOLOPoseBackend
    __init__(model_path, n_points, n_keypoints, keypoint_t_values,
             confidence_floor, min_observed_keypoints, device)
    process_frame(frame_idx, frame_dets, frames, camera_ids)
        -> dict[str, list[AnnotatedDetection]]
```

Internal data flow:
1. `extract_affine_crop` (unchanged, segmentation/crop.py)
2. YOLO pose model inference -> `(6, 3)` tensor [x_px, y_px, conf] in crop coords
3. Confidence floor filter, `min_observed_keypoints` check (same logic as `DirectPoseBackend`)
4. Sort keypoints by x-coord (same monotone ordering logic as `DirectPoseBackend`)
5. `invert_affine_points(kp_crop_px, affine.M)` -> frame pixel coords (unchanged)
6. `CubicSpline` fit + resample to `n_points` (same logic as `DirectPoseBackend`)
7. NaN-pad outside observed arc-span, interpolate confidence (same logic)
8. Build `Midline2D` with identical contract

**The spline fitting, NaN-padding, and confidence interpolation logic does not change. Only the keypoint source changes.**

### training/yolo_seg.py

```
train_yolo_seg(data_yaml, output_dir, model_size, epochs, imgsz, device, ...)
    -> Path  # path to best.pt
```

Thin wrapper: `ultralytics.YOLO(f"yolov8{size}-seg.pt").train(data=data_yaml, ...)`. Mirrors existing `training/yolo_obb.py` exactly.

### training/yolo_pose.py

```
train_yolo_pose(data_yaml, output_dir, model_size, epochs, imgsz, device, ...)
    -> Path  # path to best.pt
```

Thin wrapper: `ultralytics.YOLO(f"yolov8{size}-pose.pt").train(data=data_yaml, ...)`.

### training/prep_seg.py (NEW)

```
prepare_seg_dataset(mask_dir, image_dir, output_dir, val_fraction) -> Path  # data.yaml
```

Converts SAM2 binary masks to YOLO seg format:
- `cv2.findContours` on each binary mask -> polygon contour
- Normalize contour to [0,1] relative to image dimensions
- Write `labels/train/` and `labels/val/` txt files
- Write `data.yaml` with `nc: 1, names: ['fish']`

### training/prep_pose.py (NEW)

```
prepare_pose_dataset(coco_json, image_dir, output_dir, val_fraction) -> Path  # data.yaml
```

Converts existing COCO-format `annotations.json` (6-keypoint format already used by `KeypointDataset`) to YOLO pose txt labels:
- Per instance: `0 <cx> <cy> <bw> <bh> <kx1> <ky1> <kv1> ... <kx6> <ky6> <kv6>` (normalized)
- Write `data.yaml` with `kpt_shape: [6, 3]`

---

## Config Changes (engine/config.py)

`MidlineConfig` gains three new fields (existing fields preserved until deletion phase):

```python
@dataclass(frozen=True)
class MidlineConfig:
    # ALL EXISTING FIELDS PRESERVED (unchanged until deletion phase)
    confidence_threshold: float = 0.5
    weights_path: str | None = None          # legacy U-Net weights
    backend: str = "segment_then_extract"
    n_points: int = 15
    min_area: int = 300
    detection_tolerance: float = 50.0
    speed_threshold: float = 2.0
    orientation_weight_geometric: float = 1.0
    orientation_weight_velocity: float = 0.5
    orientation_weight_temporal: float = 0.3
    keypoint_weights_path: str | None = None  # legacy _PoseModel weights
    n_keypoints: int = 6
    keypoint_t_values: list[float] | None = None
    keypoint_confidence_floor: float = 0.1
    min_observed_keypoints: int = 3

    # NEW fields for Ultralytics backends
    yolo_seg_model_path: str | None = None   # YOLOv8-seg .pt path
    yolo_pose_model_path: str | None = None  # YOLOv8-pose .pt path
    yolo_imgsz: int = 640                    # inference image size (try 256-320 for crops)
```

`get_backend()` in `core/midline/backends/__init__.py` adds two entries:

```python
if kind == "yolo_seg":
    from aquapose.core.midline.backends.yolo_seg import YOLOSegBackend
    return YOLOSegBackend(**kwargs)

if kind == "yolo_pose":
    from aquapose.core.midline.backends.yolo_pose import YOLOPoseBackend
    return YOLOPoseBackend(**kwargs)
```

`build_stages()` in `engine/pipeline.py` maps `yolo_seg_model_path` / `yolo_pose_model_path` to backend kwargs when backend kind is `"yolo_seg"` or `"yolo_pose"`.

---

## Data Flow: Before and After

### Before (v2.1): segment_then_extract backend

```
Detection bbox
    -> compute_crop_region (segmentation/crop.py)
    -> extract_crop
    -> UNetSegmentor.segment([crop])          <- REPLACED
    -> binary mask (H_crop, W_crop)
    -> skeletonization pipeline (reconstruction/midline.py)
    -> Midline2D
    -> AnnotatedDetection
```

### After (v3.0): yolo_seg backend

```
Detection bbox
    -> compute_crop_region (segmentation/crop.py)  [UNCHANGED]
    -> extract_crop                                 [UNCHANGED]
    -> YOLOSegInferencer.segment_batch([crop])      <- NEW
    -> binary mask (H_crop, W_crop)                 [SAME SHAPE]
    -> skeletonization pipeline (reconstruction/midline.py)  [UNCHANGED]
    -> Midline2D                                    [UNCHANGED]
    -> AnnotatedDetection                           [UNCHANGED]
```

### Before (v2.1): direct_pose backend

```
Detection bbox + OBB angle
    -> extract_affine_crop (segmentation/crop.py)
    -> _PoseModel(img_tensor) -> (N_kp*2,) normalized [0,1]   <- REPLACED
    -> confidence heuristic from boundary deviation
    -> sort by x, invert_affine_points, CubicSpline, resample
    -> Midline2D
    -> AnnotatedDetection
```

### After (v3.0): yolo_pose backend

```
Detection bbox + OBB angle
    -> extract_affine_crop (segmentation/crop.py)  [UNCHANGED]
    -> YOLO pose model -> (N_kp, 3) [x_px, y_px, conf]  <- NEW
    -> conf from result.keypoints.data[:, 2]             <- DIFFERENT SOURCE
    -> sort by x, invert_affine_points, CubicSpline, resample  [UNCHANGED]
    -> Midline2D                                         [UNCHANGED]
    -> AnnotatedDetection                                [UNCHANGED]
```

---

## Training Workflow: Before and After

### Before (v2.1)

```
SAM2 masks (binary PNGs)
    -> BinaryMaskDataset (training/datasets.py)
    -> custom training loop in train_unet()
    -> best_model.pth (U-Net state dict, ~2.5M params)
    -> MidlineConfig.weights_path -> segment_then_extract backend

COCO JSON (annotations.json)
    -> KeypointDataset (training/pose.py)
    -> custom training loop with masked MSE loss in train_pose()
    -> best_model.pth (_PoseModel state dict)
    -> MidlineConfig.keypoint_weights_path -> direct_pose backend
```

### After (v3.0)

```
SAM2 masks (binary PNGs)
    -> prepare_seg_dataset()    <- NEW: cv2.findContours -> YOLO polygon labels
    -> data.yaml (nc:1, names:['fish'])
    -> train_yolo_seg()         <- NEW: ultralytics YOLO seg .train()
    -> runs/segment/train/weights/best.pt
    -> MidlineConfig.yolo_seg_model_path -> yolo_seg backend

COCO JSON (annotations.json)
    -> prepare_pose_dataset()   <- NEW: normalize + convert -> YOLO pose txt labels
    -> data.yaml (kpt_shape: [6, 3])
    -> train_yolo_pose()        <- NEW: ultralytics YOLO pose .train()
    -> runs/pose/train/weights/best.pt
    -> MidlineConfig.yolo_pose_model_path -> yolo_pose backend
```

The existing `aquapose train yolo-obb` pattern (training/yolo_obb.py) is the direct precedent for `train_yolo_seg` and `train_yolo_pose` — follow it exactly.

---

## Suggested Build Order (Dependency-Driven)

### Phase 1: Data Preparation Tooling

**Rationale:** Training cannot start without data in YOLO format. These scripts have zero dependency on pipeline code. Start training immediately after completing this phase so models train while pipeline integration proceeds.

1. `training/prep_seg.py` — binary mask -> YOLO seg polygon labels
2. `training/prep_pose.py` — COCO JSON -> YOLO pose txt labels
3. `aquapose prep-seg` and `aquapose prep-pose` CLI subcommands
4. Validate outputs (spot-check labels visually, confirm data.yaml loads)
5. **Kick off model training** (long-running background process)

### Phase 2: Training Wrappers

**Rationale:** Thin wrappers, low risk. Can be built while models are training. Establishes the full training workflow before pipeline integration is needed.

6. `training/yolo_seg.py` — `train_yolo_seg()` wrapping Ultralytics
7. `training/yolo_pose.py` — `train_yolo_pose()` wrapping Ultralytics
8. `aquapose train yolo-seg` and `aquapose train yolo-pose` CLI subcommands

### Phase 3: Inference Backends

**Rationale:** Can be developed and unit-tested against pre-trained COCO weights (`yolov8n-seg.pt`, `yolov8n-pose.pt`) before fish-specific models are ready. Does not block on Phase 1/2 completion.

9. `segmentation/yolo_seg.py` — `YOLOSegInferencer` (wraps Ultralytics seg API)
10. Unit tests for `YOLOSegInferencer` with dummy crops
11. `core/midline/backends/yolo_seg.py` — `YOLOSegBackend`
12. Unit tests: `YOLOSegBackend.process_frame` with mock inferencer
13. `core/midline/backends/yolo_pose.py` — `YOLOPoseBackend`
14. Unit tests: `YOLOPoseBackend.process_frame` with mock YOLO model
15. Update `core/midline/backends/__init__.py` `get_backend()` registry

### Phase 4: Config Wiring and Integration Test

**Rationale:** Backends exist; now wire them into the config system. Integration test validates end-to-end path before deletion.

16. Add `yolo_seg_model_path`, `yolo_pose_model_path`, `yolo_imgsz` to `MidlineConfig`
17. Update `engine/pipeline.py` `build_stages()` for new backend kinds
18. Update `load_config()` path resolution for new fields (same pattern as `weights_path`)
19. Integration test: run pipeline with `backend: "yolo_seg"` using pre-trained COCO weights

### Phase 5: Deletion Pass

**Rationale:** Only delete after new backends pass integration tests. Deleting before validation breaks the pipeline with no fallback.

20. Delete `_UNet`, `UNetSegmentor` from `segmentation/model.py`
21. Delete `training/unet.py`, `training/pose.py`, `training/datasets.py`
22. Remove `unet` and `pose` CLI subcommands from `training/cli.py`
23. Delete `core/midline/backends/segment_then_extract.py`
24. Delete `core/midline/backends/direct_pose.py`
25. Remove legacy fields from `MidlineConfig` (`weights_path`, `keypoint_weights_path`, `n_keypoints`, `keypoint_t_values`, `keypoint_confidence_floor`, `min_observed_keypoints`)
26. Update all tests referencing deleted code
27. Full test suite pass

---

## Architectural Patterns to Follow

### Pattern 1: Thin Backend Wrapper with Eager Loading

**What:** Backend class loads model at `__init__`. Raises `FileNotFoundError` immediately if weights missing. Never defers loading to `process_frame`.

**Precedent in codebase:** `SegmentThenExtractBackend.__init__`, `DirectPoseBackend.__init__`, `YOLOBackend.__init__`.

**Apply to:** `YOLOSegInferencer.__init__`, `YOLOSegBackend.__init__`, `YOLOPoseBackend.__init__`.

### Pattern 2: Factory Registry with Kind String

**What:** `get_backend(kind, **kwargs)` dispatches to backend class by string key. New backends registered with new string identifiers.

**Precedent in codebase:** `core/midline/backends/__init__.py::get_backend()`, `core/detection/backends/__init__.py`.

**Apply to:** Add `"yolo_seg"` and `"yolo_pose"` to midline backend registry.

### Pattern 3: process_frame Protocol (Structural Typing)

**What:** All midline backends implement `process_frame(frame_idx, frame_dets, frames, camera_ids) -> dict[str, list[AnnotatedDetection]]` via structural typing (not ABC inheritance).

**Precedent in codebase:** Both existing backends implement this without inheriting from a base class.

**Apply to:** Both new backends must implement identical signature.

### Pattern 4: Training Wrapper Delegates to Ultralytics

**What:** Training functions are thin wrappers calling `ultralytics.YOLO(...).train(...)` and returning the path to `best.pt`. No custom training loop.

**Precedent in codebase:** `training/yolo_obb.py::train_yolo_obb()` — copy this structure exactly.

**Apply to:** `train_yolo_seg()` and `train_yolo_pose()`.

### Pattern 5: Lazy Imports for Heavy Dependencies

**What:** `ultralytics`, `torch` imported inside `__init__()` or factory functions, not at module top-level. Prevents import failures when optional deps are absent.

**Precedent in codebase:** Detection backends, `DirectPoseBackend._process_single_detection()`.

**Apply to:** `from ultralytics import YOLO` inside `__init__()` of new inferencer and backend classes.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Importing engine/ from core/

**What people do:** Import `MidlineConfig` or other engine types inside `core/midline/backends/`.

**Why it's wrong:** The AST-based import boundary checker enforces `core/` never imports `engine/`. Violation breaks the pre-commit hook.

**Do this instead:** Extract config values in `MidlineStage.__init__` and pass as primitives to backends (model_path string, conf threshold float, etc.). Backends receive only stdlib types and domain types from `core/`, `segmentation/`, `reconstruction/`.

### Anti-Pattern 2: Returning CUDA Tensors Without .cpu()

**What people do:** Forget `.cpu().numpy()` on tensors returned from Ultralytics inference.

**Why it's wrong:** `result.masks.data` and `result.keypoints.data` may be CUDA tensors. NumPy rejects them. Runtime error.

**Do this instead:** Always `.cpu().numpy()` immediately after extracting from Ultralytics result objects. This is an established project convention documented in CLAUDE.md.

### Anti-Pattern 3: Loading the YOLO Model per Frame

**What people do:** Instantiate `YOLO(model_path)` inside `process_frame()` or `_process_single_detection()`.

**Why it's wrong:** Model loading takes ~100ms+. Per-frame loading destroys throughput on a 30fps pipeline.

**Do this instead:** Load once in `__init__`, store as `self._model`. Same pattern as `YOLOBackend`, `UNetSegmentor`.

### Anti-Pattern 4: Confusing Pixel vs Normalized Coordinates

**What people do:** Forget that Ultralytics keypoints are pixel coords while `_PoseModel` returned normalized [0,1] coords.

**Why it's wrong:** Silently produces garbled geometry (no crash, just wrong midlines).

**Do this instead:** `result.keypoints.data[0, :, :2].cpu().numpy()` gives crop pixel coordinates directly. Pass directly to `invert_affine_points(kp_crop_px, affine.M)` — no denormalization needed.

### Anti-Pattern 5: Deleting Old Code Before New Backends Validate

**What people do:** Delete `UNetSegmentor`, `_PoseModel`, and existing backends as the first migration step.

**Why it's wrong:** Leaves the pipeline broken with no fallback if validation takes longer than expected.

**Do this instead:** Build and test new backends (Phases 1-4) while old backends remain. Both coexist in `MidlineConfig` via the `backend` field. Delete only after integration test passes (Phase 5).

### Anti-Pattern 6: Putting New Inference Classes in engine/

**What people do:** Place `YOLOSegInferencer` in `engine/` because the pipeline orchestrator lives there.

**Why it's wrong:** Import boundary violation — `core/midline/backends/` would need to import from `engine/`. The AST checker rejects this.

**Do this instead:** `YOLOSegInferencer` lives in `segmentation/` (same layer as `YOLODetector`). `YOLOSegBackend` lives in `core/midline/backends/` and imports from `segmentation/`.

---

## Integration Points Summary

| Boundary | Communication | Change Required |
|----------|---------------|-----------------|
| Stage 1 -> Stage 4 | `Detection` objects via `context.detections` | None |
| Stage 3 -> Stage 4 | `TrackletGroup` via `context.tracklet_groups` | None |
| Stage 4 -> Stage 5 | `AnnotatedDetection` via `context.annotated_detections` | None |
| MidlineStage -> backends | `get_backend(kind, **kwargs)` | Add 2 new entries to registry |
| MidlineConfig -> MidlineStage | Frozen dataclass fields | Add 3 new fields; old fields kept until Phase 5 |
| crop.py utilities | `compute_crop_region`, `extract_affine_crop`, `invert_affine_points` | None — both new backends reuse unchanged |
| reconstruction/midline.py | `_skeleton_and_widths`, `_longest_path_bfs`, `_resample_arc_length`, `_crop_to_frame` | None — `yolo_seg` backend reuses unchanged |
| segmentation/detector.py | `Detection` dataclass | None — no new fields needed |
| Observers | Event subscriptions | None |

---

## Scaling Considerations

Processing is batch mode on a single GPU workstation targeting 5-30 minute clips. Concern is throughput per frame, not distributed systems.

| Concern | Current (v2.1) | v3.0 |
|---------|----------------|------|
| Seg inference | UNet: 128x64 crops, fast (~1ms/batch) | YOLO-seg: 640px default, more overhead but better masks |
| Pose inference | _PoseModel: lightweight custom, fast | YOLO-pose: comparable overhead to YOLO-seg |
| Batch strategy | Manual tensor stacking in UNetSegmentor | Ultralytics native list-of-arrays batch |
| VRAM pressure | Two separate custom models | Two Ultralytics models (can share GPU memory pool) |

**Key optimization lever:** `MidlineConfig.yolo_imgsz` defaults to 640. For fish crops (~128x64 native size), `imgsz=256` or `imgsz=320` likely sufficient and halves inference time. Tune empirically after integration.

---

## Sources

- Ultralytics Predict Mode docs — masks.data shape, keypoints.data format, numpy array input:
  [docs.ultralytics.com/modes/predict/](https://docs.ultralytics.com/modes/predict/)
- Ultralytics Segmentation task docs — binary mask to polygon conversion, min 3 polygon points:
  [docs.ultralytics.com/tasks/segment/](https://docs.ultralytics.com/tasks/segment/)
- Ultralytics Pose task docs — kpt_shape, keypoint visibility flags, custom keypoint counts:
  [docs.ultralytics.com/tasks/pose/](https://docs.ultralytics.com/tasks/pose/)
- Ultralytics Pose dataset format — YOLO pose txt label format with visibility:
  [docs.ultralytics.com/datasets/pose/](https://docs.ultralytics.com/datasets/pose/)
- Ultralytics Segmentation dataset format — polygon label format, data.yaml structure:
  [docs.ultralytics.com/datasets/segment/](https://docs.ultralytics.com/datasets/segment/)
- Existing codebase read directly (HIGH confidence):
  - `src/aquapose/core/midline/stage.py`
  - `src/aquapose/core/midline/backends/segment_then_extract.py`
  - `src/aquapose/core/midline/backends/direct_pose.py`
  - `src/aquapose/core/midline/backends/__init__.py`
  - `src/aquapose/core/midline/types.py`
  - `src/aquapose/segmentation/model.py`
  - `src/aquapose/training/pose.py`
  - `src/aquapose/training/cli.py`
  - `src/aquapose/engine/config.py`
  - `src/aquapose/core/detection/backends/yolo.py`

---

*Architecture research for: AquaPose v3.0 Ultralytics Unification — YOLOv8-seg and YOLOv8-pose integration*
*Researched: 2026-03-01*
