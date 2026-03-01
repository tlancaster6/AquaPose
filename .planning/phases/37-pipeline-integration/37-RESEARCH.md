# Phase 37: Pipeline Integration - Research

**Researched:** 2026-03-01
**Domain:** Ultralytics YOLO inference API, skeletonization midline extraction, spline fitting from sparse keypoints, config system rename
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Backend naming:**
- Rename backends from `segment_then_extract` / `direct_pose` to `segmentation` / `pose_estimation`
- Rename files: `segmentation.py` → `SegmentationBackend`, `pose_estimation.py` → `PoseEstimationBackend`
- Config key: `midline.backend: segmentation` or `midline.backend: pose_estimation`
- Default backend: `segmentation`
- Names describe the algorithmic approach, not the model — allows future non-YOLO models without config changes

**Failure handling:**
- Failed extractions return `AnnotatedDetection(midline=None)` — flagged empty midline, no exceptions
- Segmentation backend: skip skeletonization below a configurable minimum area threshold (existing config field likely already present)
- Pose backend: require at least 3 visible keypoints to attempt spline fitting — fewer can't define a meaningful curve
- Keypoint visibility determined by a configurable `confidence_floor` in MidlineConfig (default ~0.3) — keypoints below this threshold are treated as not visible

**Crop preparation:**
- Let Ultralytics handle all preprocessing (resize, pad, normalize) — pass raw OBB crop directly to model
- Use rotation-aligned (affine warp) crops from OBB detections, not axis-aligned bounding rects
- Coordinate back-projection (crop-space → full-frame) lives inside each backend, not shared — each backend is self-contained
- Critical: verify coordinate round-trips at crop→model→back-project boundary (cross-cutting v3.0 concern)

### Claude's Discretion
- Exact Ultralytics API calls for inference (model.predict vs model() etc.)
- Skeletonization algorithm choice for segmentation backend
- Spline fitting implementation details for pose backend
- Whether to batch crops or process one at a time per frame

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PIPE-01 | YOLOSegBackend produces binary masks per detection for midline extraction via skeletonization | Segmentation backend: YOLO-seg inference → mask tensor → skeletonize → Midline2D via existing MidlineExtractor logic |
| PIPE-02 | YOLOPoseBackend produces keypoint coordinates with per-point confidence for direct midline construction | Pose backend: YOLO-pose inference → keypoints + confidence → spline resample → Midline2D with point_confidence |
| PIPE-03 | Config system supports backend selection (yolo_seg, yolo_pose) via midline.backend field | MidlineConfig.__post_init__ rename: `{"segment_then_extract","direct_pose"}` → `{"segmentation","pose_estimation"}` |
</phase_requirements>

---

## Summary

Phase 37 wires two trained Ultralytics YOLO models into the existing midline backend stubs as real backends. The two stub files (`segment_then_extract.py`, `direct_pose.py`) become real implementations renamed to `segmentation.py` and `pose_estimation.py`. The backend registry `__init__.py` is updated to register the new names. The config system (`MidlineConfig`, `get_backend`) gets the name changes. `MidlineStage` and `build_stages` propagate the right kwargs. No new abstractions are needed — the insertion points were designed specifically for this wiring.

The segmentation backend receives an OBB-aligned affine crop, runs `model.predict()` for a binary instance mask, then feeds that mask through the existing `_adaptive_smooth` / `skeletonize` / `_longest_path_bfs` / `_resample_arc_length` logic from `reconstruction/midline.py`. The only missing piece is the affine inversion step to back-project crop-space skeleton coordinates to full-frame space — `invert_affine_points()` already exists in `segmentation/crop.py` for exactly this purpose.

The pose estimation backend receives the same OBB-aligned crop, runs `model.predict()` for keypoint predictions, filters by `confidence_floor`, and fits a parametric spline through the surviving keypoints to resample to `n_points`. The 6 keypoints (nose, head, spine1, spine2, spine3, tail) have a natural ordering so spline fitting is straightforward — no need for BFS or skeleton analysis. Per-keypoint model confidence goes directly into `Midline2D.point_confidence`.

**Primary recommendation:** Implement both backends as wrappers around existing crop, affine-invert, and skeletonize infrastructure — do not re-implement any of those utilities. The coordinate-space correctness concern from v3.0 is the dominant risk; verify round-trips explicitly in tests.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ultralytics | >=8.0 (already in deps) | YOLO-seg and YOLO-pose inference | Already used in detection backends; `model.predict()` is the standard inference call |
| scikit-image | >=0.21 (already in deps) | `skeletonize()` for mask thinning | Already used in `reconstruction/midline.py`; proven correct |
| scipy | >=1.11 (already in deps) | `interp1d` for arc-length resampling, `distance_transform_edt` for half-widths | Already used in `reconstruction/midline.py` |
| numpy | project dep | Array math throughout | Standard |
| opencv-python | project dep (cv2) | Affine warp, invert, `warpAffine` | Already used in `segmentation/crop.py`; `invert_affine_points` is the exact back-projection needed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.interpolate.UnivariateSpline | >=1.11 | Spline fitting through sparse keypoints (pose backend) | Pose backend: fit curve through 3-6 visible keypoints then resample to n_points |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `skeletonize` (scikit-image) | `medial_axis` | `medial_axis` gives distance transform simultaneously but is slower; `skeletonize` + `distance_transform_edt` is the existing pattern |
| `scipy.interpolate.interp1d` for pose resampling | `numpy.interp` | `interp1d` handles the existing arc-length parameterization pattern already in the codebase |

**Installation:** No new dependencies needed — all libraries already in pyproject.toml.

---

## Architecture Patterns

### File Layout After Phase 37

```
src/aquapose/core/midline/
├── __init__.py                       # exports updated (add SegmentationBackend, PoseEstimationBackend)
├── backends/
│   ├── __init__.py                   # get_backend: register "segmentation", "pose_estimation"
│   ├── segmentation.py               # NEW — replaces segment_then_extract.py stub
│   └── pose_estimation.py            # NEW — replaces direct_pose.py stub
│   # segment_then_extract.py and direct_pose.py: DELETE after rename
├── stage.py                          # Updated: backend kwarg routing for new names
├── types.py                          # No change
└── orientation.py                    # No change
src/aquapose/engine/
├── config.py                         # Updated: MidlineConfig backend validation set
```

### Pattern 1: Segmentation Backend — YOLO-seg Inference

**What:** Run YOLO-seg on an OBB-aligned crop, extract the binary mask from the result, apply adaptive smoothing + skeletonize, then invert affine transform to produce Midline2D in full-frame space.

**When to use:** `midline.backend: segmentation`

**Key concern:** YOLO-seg returns masks that may be in letterboxed input space OR already mapped to original image coordinates depending on how you call predict. Passing the raw OBB crop directly (not the full frame) means the mask is already in crop space — no additional letterbox undo is needed. The mask shape from Ultralytics will match the crop dimensions after `results[0].masks.data` is accessed. Confirm with `.orig_shape` vs `.shape`.

```python
# Source: ultralytics YOLO API pattern (established in yolo_obb.py)
# Inference — pass crop directly; Ultralytics resizes internally
results = self._model.predict(crop.image, conf=self._conf, verbose=False)

# Extract mask for the highest-confidence detection
if results and results[0].masks is not None:
    # masks.data shape: (N, crop_h, crop_w) — already in crop space
    # because we passed a crop, not the full frame
    mask_tensor = results[0].masks.data[0]   # first detection
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    # mask_np is crop-space binary mask — feed directly to skeletonization

# Skeletonize (same logic as MidlineExtractor in reconstruction/midline.py)
smooth = _adaptive_smooth(mask_np)
skeleton_bool, dt = _skeleton_and_widths(smooth)
path_yx = _longest_path_bfs(skeleton_bool)
xy_crop, half_widths = _resample_arc_length(path_yx, dt, n_points=self.n_points)

# Back-project from crop space to full-frame space
xy_frame = invert_affine_points(xy_crop, crop.M)
# half_widths scale: crop pixels → frame pixels via affine scale factor
```

**IMPORTANT:** The YOLO-seg mask from `results[0].masks.data` is returned in the resized input space (Ultralytics letterboxes to imgsz). When passing a crop directly, the mask is already de-letterboxed to the crop's pixel space. Verify shape matches `crop.image.shape[:2]` — if not, resize with `cv2.resize` using INTER_NEAREST before skeletonizing.

### Pattern 2: Pose Estimation Backend — YOLO-pose Inference

**What:** Run YOLO-pose on an OBB-aligned crop, extract keypoint coordinates + confidences, filter by `confidence_floor`, fit a spline through visible keypoints, resample to `n_points`, and invert affine to full-frame space.

**When to use:** `midline.backend: pose_estimation`

```python
# Source: ultralytics YOLO API pattern
results = self._model.predict(crop.image, conf=self._conf, verbose=False)

if results and results[0].keypoints is not None:
    # keypoints.xy shape: (N_det, N_kpts, 2) — crop-space pixel coords
    # keypoints.conf shape: (N_det, N_kpts) — per-point confidence
    kpts_xy = results[0].keypoints.xy[0].cpu().numpy()    # (6, 2)
    kpts_conf = results[0].keypoints.conf[0].cpu().numpy() # (6,)

    # Filter by confidence_floor
    visible_mask = kpts_conf >= self.confidence_floor
    visible_kpts = kpts_xy[visible_mask]                  # (K, 2), K >= 3

    if visible_kpts.shape[0] >= self.min_observed_keypoints:
        # Fit spline through visible keypoints (ordered nose→tail by index)
        t_visible = self.keypoint_t_values[visible_mask]  # arc-fraction positions
        # Spline fit + resample to n_points
        xy_crop_resampled = _spline_resample(visible_kpts, t_visible, self.n_points)

        # Back-project to full-frame space
        xy_frame = invert_affine_points(xy_crop_resampled, crop.M)

        # Per-point confidence: interpolate from keypoint confidences
        conf_resampled = _interpolate_confidence(kpts_conf, visible_mask,
                                                  t_visible, self.n_points)
```

**Keypoint ordering:** The 6 keypoints are ordered anatomically: index 0 = nose, 1 = head, 2 = spine1, 3 = spine2, 4 = spine3, 5 = tail. Their t-values (arc-fraction positions in [0, 1]) are from `MidlineConfig.keypoint_t_values` — if None, default to `np.linspace(0, 1, 6)`.

### Pattern 3: Config System Rename

**What:** Update `MidlineConfig.backend` validation set and `get_backend()` registry.

```python
# In engine/config.py — MidlineConfig.__post_init__:
_valid_backends = {"segmentation", "pose_estimation"}  # was: {"segment_then_extract", "direct_pose"}

# In core/midline/backends/__init__.py — get_backend():
if kind == "segmentation":
    from aquapose.core.midline.backends.segmentation import SegmentationBackend
    return SegmentationBackend(**kwargs)

if kind == "pose_estimation":
    from aquapose.core.midline.backends.pose_estimation import PoseEstimationBackend
    return PoseEstimationBackend(**kwargs)
```

### Pattern 4: Crop Extraction for Midline Backends

**What:** Both backends receive OBB-aligned affine crops, not axis-aligned crops. The Detection object from Stage 1 carries `angle` and `obb_points` (set by `YOLOOBBBackend`). Use `extract_affine_crop()` in each backend's `process_frame()`.

```python
# In process_frame(), for each detection:
from aquapose.segmentation.crop import AffineCrop, extract_affine_crop

det = detection  # Detection with angle, obb_points, bbox
cx = det.bbox[0] + det.bbox[2] / 2
cy = det.bbox[1] + det.bbox[3] / 2
obb_w = det.bbox[2]  # approximate — better to compute from obb_points if available
obb_h = det.bbox[3]

crop = extract_affine_crop(
    frame=frame,
    center_xy=(cx, cy),
    angle_math_rad=det.angle or 0.0,
    obb_w=obb_w,
    obb_h=obb_h,
    crop_size=self.crop_size,   # e.g. (256, 128) from DetectionConfig.crop_size
    fit_obb=True,               # scale OBB to fit canvas — required for YOLO models
    mask_background=True,       # zero outside OBB (matches training data format)
)
# crop.M is the (2,3) affine matrix for back-projection
```

**Caution on `fit_obb` and `mask_background`:** The YOLO-seg and YOLO-pose models were trained on crops produced by `build_yolo_training_data.py`. Check whether that script uses `fit_obb=True` and `mask_background=True` — if training used those flags, inference must too. Training/inference crop mismatch is the leading silent failure mode.

### Pattern 5: MidlineStage Backend Kwarg Routing

The existing `MidlineStage.__init__` has a special branch for `direct_pose` that passes extra kwargs. This must be updated for the new names:

```python
# In core/midline/stage.py — MidlineStage.__init__:
if backend == "pose_estimation":
    mc = midline_config
    self._backend = get_backend(
        "pose_estimation",
        weights_path=mc.keypoint_weights_path if mc is not None else None,
        device=device,
        n_points=n_points,
        n_keypoints=mc.n_keypoints if mc is not None else 6,
        keypoint_t_values=mc.keypoint_t_values if mc is not None else None,
        confidence_floor=mc.keypoint_confidence_floor if mc is not None else 0.3,
        min_observed_keypoints=mc.min_observed_keypoints if mc is not None else 3,
        crop_size=crop_size,
    )
else:  # "segmentation"
    self._backend = get_backend(
        backend,
        weights_path=weights_path,
        confidence_threshold=confidence_threshold,
        n_points=n_points,
        min_area=min_area,
        device=device,
        crop_size=crop_size,
    )
```

Note: `MidlineConfig.weights_path` is noted as "deprecated" in the current config but will become the segmentation model path in this phase. It already propagates from config into `MidlineStage` via `config.midline.weights_path`. Rename the docstring but keep the field name.

Similarly, `MidlineConfig.keypoint_weights_path` becomes the pose model path. No rename needed on the field.

**ALSO:** Update `build_stages()` in `engine/pipeline.py` — the backend string check `if backend == "direct_pose"` must become `if backend == "pose_estimation"`. The kwargs routing is already in place.

### Anti-Patterns to Avoid

- **Passing full frame to YOLO instead of crop:** This would produce masks/keypoints in full-frame space and bypass the OBB alignment, breaking the coordinate contract.
- **Using INTER_LINEAR for mask resize:** Binary masks must use INTER_NEAREST when resizing; bilinear creates gray values that break the binary threshold.
- **Accessing `.masks.xy` instead of `.masks.data`:** `.xy` gives polygon contour coordinates; `.data` gives the rasterized binary mask tensor. Use `.data` for skeletonization.
- **Accessing `.keypoints.xyn` instead of `.keypoints.xy`:** `.xyn` is normalized [0,1] coordinates; `.xy` is pixel coordinates. Use `.xy` for affine inversion.
- **Not checking `results[0].masks is not None` before accessing:** If the model produces no detections, `.masks` is None and indexing raises AttributeError.
- **Skipping the `fit_obb=True` flag mismatch check:** Training crops that were fit-to-OBB and inference crops that are not (or vice versa) produce silent accuracy degradation.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Skeletonizing a binary mask | Custom thinning algorithm | `skimage.morphology.skeletonize` (already in `reconstruction/midline.py`) | Proven correct; handles branches, single-pixel paths, boundary conditions |
| Skeleton path extraction | Custom DFS | `_longest_path_bfs` (already in `reconstruction/midline.py`) | Two-pass BFS finds the longest path correctly; already tested |
| Arc-length resampling | Manual interpolation | `_resample_arc_length` (already in `reconstruction/midline.py`) | Already handles degenerate paths, uniform spacing, half-width extraction |
| Affine inversion | Custom matrix math | `invert_affine_points` from `segmentation/crop.py` | Uses `cv2.invertAffineTransform` + `cv2.transform`; numerically stable |
| YOLO inference preprocessing | Manual resize/normalize/pad | `model.predict(crop_image, ...)` | Ultralytics handles letterboxing, normalization, batch formation internally |
| Spline fitting | Manual polynomial regression | `scipy.interpolate.UnivariateSpline` or `interp1d` | Handles non-uniform input spacing, edge cases at endpoints |

**Key insight:** The entire midline extraction pipeline for the segmentation backend already exists in `reconstruction/midline.py`. The segmentation backend is a thin wrapper: YOLO → crop mask → call existing helpers → affine invert. Do not copy-paste the helper code — import it directly.

---

## Common Pitfalls

### Pitfall 1: YOLO Mask Is Not in Crop-Pixel Space

**What goes wrong:** `results[0].masks.data` returns a tensor whose spatial dimensions do not match the crop image. The mask may be at the model's inference resolution (e.g., 160×160 for mask heads) rather than the crop resolution.

**Why it happens:** YOLO-seg uses a separate mask head that operates at a lower resolution. Ultralytics upsamples masks to `orig_shape` when `retina_masks=False` (default). When you pass a crop, `orig_shape` is the crop shape, so the final mask should match. But this depends on the Ultralytics version behavior.

**How to avoid:** After getting `mask_np`, assert `mask_np.shape == crop.image.shape[:2]`. If not equal, resize with `cv2.resize(mask_np, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)` before skeletonizing.

**Warning signs:** Skeleton coordinates consistently off by a constant factor from expected positions.

### Pitfall 2: Training/Inference Crop Mismatch

**What goes wrong:** Model produces low-quality masks or keypoints that don't match the fish body, even though inference runs without error.

**Why it happens:** The training data was generated with specific `extract_affine_crop` parameters (`fit_obb`, `mask_background`, `crop_size`). Inference with different parameters changes the crop appearance enough to degrade model performance.

**How to avoid:** Check `scripts/build_yolo_training_data.py` for the exact `extract_affine_crop` kwargs used during dataset generation. Replicate those exactly in the backend's `process_frame()`. Use `crop_size` from `DetectionConfig.crop_size` (currently `[256, 128]` but may differ for seg/pose).

**Warning signs:** Mask predictions are blobs far from fish body; keypoint predictions cluster near image center.

### Pitfall 3: Coordinate Back-Projection Half-Width Scaling

**What goes wrong:** Midline points are at correct positions in full-frame space but `half_widths` are in crop-pixel units (small numbers like 5-8 pixels) instead of full-frame units (30-60 pixels).

**Why it happens:** `invert_affine_points` correctly inverts position coordinates, but the affine transform has a scale factor (when `fit_obb=True`) that must also be applied to the half-width values extracted from the distance transform.

**How to avoid:** The affine matrix `M` encodes the scale. Extract the scale factor as `scale = np.sqrt(M[0,0]**2 + M[0,1]**2)` and divide half-widths by it: `hw_frame = half_widths / scale`. Alternatively, compare skeleton half-widths to known fish body width as a sanity check.

**Warning signs:** Reconstruction stage produces 3D midlines with incorrect estimated body thickness; visualization overlays show very thin or very thick fish silhouettes.

### Pitfall 4: Config Validation Still Rejects New Backend Names

**What goes wrong:** Setting `midline.backend: segmentation` in YAML raises `ValueError: Unknown midline backend: 'segmentation'`.

**Why it happens:** `MidlineConfig.__post_init__` validates against `{"segment_then_extract", "direct_pose"}`. The config code must be updated in the same change as the backend registry.

**How to avoid:** Update `MidlineConfig.__post_init__` and `get_backend()` in the same plan wave. Also update `MidlineStage.__init__` branch from `if backend == "direct_pose"` to `if backend == "pose_estimation"`.

**Warning signs:** Tests that construct `MidlineConfig(backend="segmentation")` raise `ValueError`.

### Pitfall 5: Old Backend Name Strings Scattered in Tests

**What goes wrong:** After renaming, existing tests that construct `MidlineConfig(backend="segment_then_extract")` or call `get_backend("direct_pose")` start failing.

**Why it happens:** Test files reference old backend name strings. The rename must be propagated consistently.

**How to avoid:** After renaming, search all test files for `"segment_then_extract"`, `"direct_pose"`, `SegmentThenExtractBackend`, `DirectPoseBackend` and update to new names. Files to check: `tests/unit/core/midline/test_midline_stage.py`, `tests/unit/core/midline/test_direct_pose_backend.py`, `tests/unit/engine/test_config.py`, `tests/unit/engine/test_build_stages.py`.

**Warning signs:** `ModuleNotFoundError` for `backends.segment_then_extract` or `backends.direct_pose` after renaming.

### Pitfall 6: Missing Detection `angle` or `obb_points` for Non-OBB Detections

**What goes wrong:** `detect.angle` is `None` when YOLO (not YOLO-OBB) is used for detection. `extract_affine_crop` receives `angle_math_rad=None`, causing a crash.

**Why it happens:** The project uses YOLO-OBB as the detection backend, but the midline backends must be defensive about `angle` being `None`.

**How to avoid:** In `process_frame()`, check `det.angle is not None` before using it; fall back to `angle=0.0` if None. This preserves correctness for the common YOLO-OBB case and degrades gracefully for any edge case.

---

## Code Examples

### Ultralytics YOLO-seg Inference on a Crop

```python
# Source: established pattern in aquapose/core/detection/backends/yolo_obb.py
from ultralytics import YOLO

self._model = YOLO(str(model_path))

# Inference — pass crop image directly (BGR uint8 numpy array)
results = self._model.predict(crop.image, conf=self._conf, verbose=False)

# Extract binary mask from result
if results and results[0].masks is not None and len(results[0].masks.data) > 0:
    # Pick highest-confidence detection (index 0 — ultralytics sorts by conf)
    mask_tensor = results[0].masks.data[0]   # (H_mask, W_mask), float [0,1]
    mask_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255
    # Resize to crop dimensions if needed
    ch, cw = crop.image.shape[:2]
    if mask_np.shape != (ch, cw):
        mask_np = cv2.resize(mask_np, (cw, ch), interpolation=cv2.INTER_NEAREST)
```

### Ultralytics YOLO-pose Inference on a Crop

```python
# Source: established pattern in aquapose/core/detection/backends/yolo_obb.py
results = self._model.predict(crop.image, conf=self._conf, verbose=False)

if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
    # kpts_xy: (N_det, N_kpts, 2) — pick detection 0
    kpts_xy = results[0].keypoints.xy[0].cpu().numpy()    # (6, 2)
    kpts_conf = results[0].keypoints.conf[0].cpu().numpy() # (6,)
    # kpts_xy is in crop-pixel coordinates
```

### Affine Crop Extraction with OBB Detection

```python
# Source: aquapose/segmentation/crop.py extract_affine_crop
from aquapose.segmentation.crop import extract_affine_crop, invert_affine_points

det = detection  # Detection from Stage 1 (has .angle, .obb_points, .bbox)
cx = det.bbox[0] + det.bbox[2] / 2.0
cy = det.bbox[1] + det.bbox[3] / 2.0
angle = det.angle if det.angle is not None else 0.0

# Compute OBB w/h from obb_points if available
if det.obb_points is not None:
    pts = det.obb_points   # (4, 2)
    side0 = np.linalg.norm(pts[1] - pts[0])
    side1 = np.linalg.norm(pts[2] - pts[1])
    obb_w, obb_h = max(side0, side1), min(side0, side1)
else:
    obb_w, obb_h = det.bbox[2], det.bbox[3]

crop = extract_affine_crop(
    frame=frame,
    center_xy=(cx, cy),
    angle_math_rad=angle,
    obb_w=obb_w,
    obb_h=obb_h,
    crop_size=self.crop_size,
    fit_obb=True,
    mask_background=True,
)
```

### Back-Projection with Half-Width Scaling

```python
# Source: aquapose/segmentation/crop.py invert_affine_points
from aquapose.segmentation.crop import invert_affine_points

# xy_crop: (N, 2) in crop space — from skeleton arc-length resampling
xy_frame = invert_affine_points(xy_crop, crop.M)  # (N, 2) in full-frame space

# Scale factor from affine matrix: M encodes rotation + scale
# M[0,0] = scale*cos(angle), M[0,1] = -scale*sin(angle) => scale = norm of row
scale = float(np.sqrt(crop.M[0, 0] ** 2 + crop.M[0, 1] ** 2))
hw_frame = (half_widths_crop / max(scale, 1e-6)).astype(np.float32)
```

### Spline Resample from Keypoints (Pose Backend)

```python
# Source: pattern from reconstruction/midline.py _resample_arc_length
import numpy as np
from scipy.interpolate import interp1d

def _keypoints_to_midline(
    kpts_xy: np.ndarray,    # (K, 2) visible keypoints in crop space
    t_values: np.ndarray,   # (K,) arc-fraction positions in [0,1]
    confidences: np.ndarray, # (K,) per-keypoint confidence
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample K keypoints to n_points along arc parameterization."""
    t_out = np.linspace(0.0, 1.0, n_points)

    interp_x = interp1d(t_values, kpts_xy[:, 0], kind="linear",
                         fill_value="extrapolate")
    interp_y = interp1d(t_values, kpts_xy[:, 1], kind="linear",
                         fill_value="extrapolate")
    interp_c = interp1d(t_values, confidences, kind="linear",
                         fill_value=(confidences[0], confidences[-1]),
                         bounds_error=False)

    xy = np.stack([interp_x(t_out), interp_y(t_out)], axis=1).astype(np.float32)
    conf = interp_c(t_out).astype(np.float32)
    return xy, conf
```

### Segmentation Backend Midline Construction

```python
# Full path through existing helpers (import, don't duplicate)
from aquapose.reconstruction.midline import (
    _adaptive_smooth,
    _longest_path_bfs,
    _resample_arc_length,
    _skeleton_and_widths,
    Midline2D,
)

smooth = _adaptive_smooth(mask_np)
skeleton_bool, dt = _skeleton_and_widths(smooth)

n_skel = int(np.sum(skeleton_bool))
if n_skel < self.n_points:
    return None  # midline=None — not enough skeleton

path_yx = _longest_path_bfs(skeleton_bool)
if not path_yx:
    return None

xy_crop, hw_crop = _resample_arc_length(path_yx, dt, self.n_points)
xy_frame = invert_affine_points(xy_crop, crop.M)
scale = float(np.sqrt(crop.M[0, 0] ** 2 + crop.M[0, 1] ** 2))
hw_frame = (hw_crop / max(scale, 1e-6)).astype(np.float32)

return Midline2D(
    points=xy_frame,
    half_widths=hw_frame,
    fish_id=0,          # populated by caller
    camera_id=cam_id,
    frame_index=frame_idx,
    is_head_to_tail=False,
    point_confidence=None,  # segmentation backend: uniform confidence
)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `segment_then_extract` backend name | `segmentation` backend name | Phase 37 | Config files updated; old name no longer accepted |
| `direct_pose` backend name | `pose_estimation` backend name | Phase 37 | Config files updated; old name no longer accepted |
| No-op stubs returning `midline=None` | Real YOLO inference | Phase 37 | Pipeline produces actual Midline2D objects |
| U-Net segmentation (v2.2, removed in Phase 35) | YOLO-seg inference | v3.0 | Ultralytics handles all preprocessing |
| Custom keypoint regression (v2.2, removed in Phase 35) | YOLO-pose inference | v3.0 | Standard Ultralytics output format |

**Deprecated/outdated:**
- `segment_then_extract.py` stub: deleted and replaced by `segmentation.py` with real model
- `direct_pose.py` stub: deleted and replaced by `pose_estimation.py` with real model
- `MidlineConfig.weights_path` docstring says "deprecated" — it becomes the segmentation model path; update docstring but keep field name
- `MidlineConfig.keypoint_weights_path` docstring says "deprecated" — it becomes the pose model path; update docstring

---

## Open Questions

1. **Were training crops produced with `fit_obb=True, mask_background=True`?**
   - What we know: `extract_affine_crop` has both flags; training data was built by `scripts/build_yolo_training_data.py`
   - What's unclear: Exact kwargs that script uses when generating seg/pose crops — not read during this research
   - Recommendation: First task of Wave 1 should read `scripts/build_yolo_training_data.py` and confirm exact crop parameters before writing backend inference code

2. **YOLO-seg mask resolution: does `results[0].masks.data` have the crop shape or model output shape?**
   - What we know: Ultralytics default upsamples masks to `orig_shape` (the input image size)
   - What's unclear: Whether `retina_masks` flag is needed; version-specific behavior
   - Recommendation: Add an assertion `assert mask_np.shape == crop.image.shape[:2]` with fallback resize; test with a real model before relying on this

3. **Confidence floor default: CONTEXT.md says ~0.3 but `MidlineConfig.keypoint_confidence_floor` currently defaults to 0.1**
   - What we know: CONTEXT.md says "default ~0.3"; config currently has 0.1
   - What's unclear: Whether to update the MidlineConfig default value in this phase
   - Recommendation: Update `MidlineConfig.keypoint_confidence_floor` default from 0.1 to 0.3 in this phase as part of the config rename task

4. **OBB dimensions from `obb_points` vs `bbox`:**
   - What we know: `Detection.obb_points` is `(4, 2)` corners; `Detection.bbox` is AABB (axis-aligned bounding box) — larger than OBB for rotated fish
   - What's unclear: Whether using AABB w/h as `obb_w`/`obb_h` in `extract_affine_crop` degrades crop quality
   - Recommendation: Compute OBB w/h from `obb_points` side lengths when available; fall back to bbox only when `obb_points is None`

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — `src/aquapose/core/midline/backends/segment_then_extract.py`, `direct_pose.py`, `__init__.py`
- Direct codebase inspection — `src/aquapose/core/midline/stage.py` — full kwarg routing logic
- Direct codebase inspection — `src/aquapose/engine/config.py` — `MidlineConfig` fields and validation
- Direct codebase inspection — `src/aquapose/engine/pipeline.py` — `build_stages()` wiring
- Direct codebase inspection — `src/aquapose/segmentation/crop.py` — `extract_affine_crop`, `invert_affine_points`
- Direct codebase inspection — `src/aquapose/reconstruction/midline.py` — `_adaptive_smooth`, `_longest_path_bfs`, `_resample_arc_length`, `Midline2D`
- Direct codebase inspection — `src/aquapose/core/detection/backends/yolo_obb.py` — Ultralytics inference pattern

### Secondary (MEDIUM confidence)
- Ultralytics documentation (known from training code in `training/common.py`) — `model.predict()` API, `results[0].masks.data`, `results[0].keypoints.xy`, `results[0].keypoints.conf`
- `.planning/phases/37-pipeline-integration/37-CONTEXT.md` — locked decisions and discretion areas

### Tertiary (LOW confidence)
- YOLO-seg mask output resolution behavior: based on known Ultralytics behavior but not verified against installed version — treat as hypothesis until confirmed with assertion

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in the project; no new deps
- Architecture: HIGH — existing stubs define the exact interface; existing helpers cover all the hard parts
- Pitfalls: HIGH — coordinate space bugs are documented in the MEMORY.md as a known project-wide concern; crop mismatch is well-understood from v2.2 development

**Research date:** 2026-03-01
**Valid until:** 2026-04-01 (Ultralytics API is stable; coordinate logic is internal to the project)
