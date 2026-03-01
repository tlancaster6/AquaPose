# Phase 32: YOLO-OBB Detection Backend - Research

**Researched:** 2026-02-28
**Domain:** Oriented bounding box detection, affine crop extraction, OpenCV geometry, ultralytics OBB API
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Affine crop behavior**: Proportional padding around the OBB (percentage of OBB dimensions, not fixed pixels). Fixed rectangular canvas with aspect ratio preserved via letterbox padding. Crop dimensions are configurable via config (e.g. `crop_size: [256, 128]`), not hardcoded. Always use the affine crop utility even for non-OBB detections (angle=0 produces identity rotation) — one unified code path.
- **OBB overlay styling**: OBB polygon color matches fish ID color (same palette as tracklet trails). OBB polygon replaces axis-aligned bounding box — no reason to draw both. Labels show fish ID + confidence score (same as existing AABB overlays). No orientation axis line — the polygon shape itself conveys orientation.
- **Model loading & training**: Fine-tune a pre-trained YOLOv8-OBB model on fish data. Training set already exists at `C:\Users\tucke\aquapose\projects\YH\models\obb\training_set`. Model output convention: training output goes to `<project_dir>/models/<shorthand>/`, best weights copied to `<project_dir>/models/<shorthand>_best.pt`. For OBB: shorthand `obb`, so `projects/YH/models/obb/` and `projects/YH/models/obb_best.pt`.
- **Fallback & compatibility**: Same confidence threshold filtering as regular YOLO — no separate OBB threshold. Always use affine transform regardless of rotation angle (no threshold fallback to axis-aligned). Both midline backends use OBB rotation-aligned crops when OBB data is available. Unified crop code path: non-OBB detections flow through the same affine utility with angle=0.

### Claude's Discretion

- Exact proportional padding percentage (likely 10-20% range)
- Affine interpolation method (bilinear vs bicubic)
- OBB polygon line thickness and style details
- Internal YOLO-OBB inference details (batch size, NMS parameters)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DET-01 | Pipeline supports YOLO-OBB as a configurable detection model selectable via `detector_kind: yolo_obb` in config | `get_backend()` factory extension + new `YOLOOBBBackend` class in `core/detection/backends/yolo_obb.py`; `DetectionConfig.detector_kind` already accepts a string |
| DET-02 | OBB detections produce rotation-aligned affine crops suitable for downstream segmentation and keypoint models | `cv2.getRotationMatrix2D` + `cv2.warpAffine` pipeline; new `AffineTransform` dataclass captures the 2×3 matrix for back-projection |
| DET-03 | Affine crop utilities support back-projection from crop coordinates to full-frame pixel coordinates via inverse transform | `cv2.invertAffineTransform(M)` returns the exact inverse 2×3 matrix; combine with `cv2.transform` for point mapping |
| VIZ-01 | Diagnostic mode renders OBB polygon overlays on detection frames | `cv2.polylines` with 4 `obb_points`; driven from `TrackletTrailObserver` or a new detection-frame observer |
| VIZ-02 | Tracklet trail visualization includes bounding box overlays (both axis-aligned and OBB when available) | `TrackletTrailObserver._draw_trail()` already draws dots/trails; need to extend with bbox drawing per-frame using `detection.obb_points` or `detection.bbox` |
</phase_requirements>

## Summary

Phase 32 adds YOLO-OBB as a selectable detection backend (`detector_kind: yolo_obb`) that produces `Detection` objects with populated `angle` and `obb_points` fields, and introduces a unified affine crop utility that all downstream stages use (angle=0 for non-OBB detections is a no-op rotation). The ultralytics `result.obb` API provides `xywhr` and `xyxyxyxy` — the phase requires careful angle convention conversion at the boundary (ultralytics uses clockwise radians in `[-pi/4, 3pi/4)`; the project's `Detection.angle` field uses standard math radians in `[-pi, pi]`).

The affine crop design is the most complex piece: extract a rotation-aligned rectangular crop, attach the 2×3 affine transform to the detection (or a companion dataclass), and provide an `invert_affine_transform` utility so downstream keypoint coordinates can be back-projected to frame space. `cv2.invertAffineTransform` handles the inversion exactly — no need to hand-roll matrix math.

Visualization changes are additive: `TrackletTrailObserver._draw_trail()` and the overlay observer need OBB polygon rendering using `cv2.polylines` with the four `obb_points` corners. The FISH_COLORS_BGR palette in `tracklet_trail_observer.py` is the authoritative source for fish ID colors, and both observers should draw OBB polygons using the same palette lookup.

**Primary recommendation:** Implement the phase as three tasks: (1) `YOLOOBBBackend` + `get_backend` extension, (2) affine crop utilities with invertible transform, (3) visualization extensions for both observers.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ultralytics | >=8.0 (already in project) | YOLOv8-OBB inference (`result.obb.xywhr`, `.xyxyxyxy`, `.conf`) | Project already uses YOLO; OBB is the same package |
| cv2 (OpenCV) | >=4.5 (already in project) | `getRotationMatrix2D`, `warpAffine`, `invertAffineTransform`, `polylines` | All affine/warp operations are in OpenCV; already a project dependency |
| numpy | >=1.24 (already in project) | Array manipulation for OBB point transforms | Already a project dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch (lazy import) | project version | Device string passed to ultralytics `.predict()` | Only for device string; ultralytics handles GPU internally |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cv2.invertAffineTransform` | Manual 2×3 inverse math | cv2 is already imported everywhere; hand-rolling adds risk with no benefit |
| `cv2.warpAffine` (bilinear) | `cv2.warpAffine` with `INTER_CUBIC` | Bilinear is faster and accurate enough for 256×128 crops; bicubic adds ~25% compute cost |

## Architecture Patterns

### Recommended Project Structure

The phase touches exactly these locations:

```
src/aquapose/
├── core/detection/backends/
│   ├── __init__.py          # extend get_backend() to handle "yolo_obb"
│   ├── yolo.py              # existing YOLOBackend (unchanged)
│   └── yolo_obb.py          # NEW: YOLOOBBBackend wrapping ultralytics OBB
├── segmentation/
│   ├── detector.py          # Detection dataclass already has angle/obb_points fields (Phase 30)
│   └── crop.py              # extend with affine crop utilities: AffineCrop, extract_affine_crop, invert_affine_point
├── engine/
│   ├── overlay_observer.py  # extend _draw_bbox to render OBB polygon when obb_points present
│   └── tracklet_trail_observer.py  # extend trail drawing to include per-frame OBB polygon
tests/unit/
├── segmentation/
│   └── test_detector.py     # extend: YOLOOBBDetector mock tests
├── core/detection/
│   └── test_detection_stage.py  # extend: yolo_obb backend registry test
└── engine/
    └── test_overlay_observer.py  # extend: OBB polygon rendering test
```

A new test file `tests/unit/segmentation/test_affine_crop.py` covers the affine crop round-trip (DET-02, DET-03).

### Pattern 1: YOLOOBBBackend — Thin Wrapper with Angle Convention Conversion

**What:** `YOLOOBBBackend` wraps ultralytics `YOLO` (OBB variant). The `detect()` method runs inference, reads `result.obb.xywhr` (center_x, center_y, w, h, angle_clockwise_rad) and `result.obb.xyxyxyxy` (4 corner points), applies the confidence threshold, converts the ultralytics angle to standard math convention, and returns `Detection` objects with `angle` and `obb_points` populated.

**When to use:** Always when `detector_kind == "yolo_obb"`.

**Angle conversion formula** (CRITICAL — see STATE.md concern):
```python
# ultralytics: clockwise radians in [-pi/4, 3pi/4)
# Detection.angle: standard math radians in [-pi, pi]
# Standard math: counter-clockwise positive, CW = negative
# Conversion: negate (clockwise -> counter-clockwise)
detection_angle = -float(angle_clockwise_rad)
```

This is confirmed by the STATE.md note: "Detection.angle uses standard math radians [-pi, pi]; YOLO-OBB backend (Plan 32) handles angle convention conversion at the boundary."

**Example — ultralytics OBB result API:**
```python
# Source: https://docs.ultralytics.com/tasks/obb/
results = model.predict(frame, conf=self._conf, iou=self._iou, verbose=False)
for r in results:
    if r.obb is None:
        continue
    xywhr = r.obb.xywhr   # tensor (N, 5): cx, cy, w, h, angle_cw_rad
    xyxyxyxy = r.obb.xyxyxyxy  # tensor (N, 4, 2): 4 corner points in pixel coords
    confs = r.obb.conf    # tensor (N,)
    for i in range(len(confs)):
        conf = float(confs[i])
        cx, cy, w, h, angle_cw = xywhr[i].tolist()
        corners = xyxyxyxy[i].cpu().numpy()  # shape (4, 2)
        # Convert angle convention
        angle_math = -angle_cw  # standard math: CCW positive
        detection = Detection(
            bbox=_obb_to_aabb(corners),
            mask=None,
            area=int(w * h),
            confidence=conf,
            angle=angle_math,
            obb_points=corners,
        )
```

**obb_points corner order:** ultralytics `xyxyxyxy` returns points in pixel coordinates, clockwise from top-left of the *rotated* box. This matches the `Detection.obb_points` docstring: "clockwise from top-left of the oriented box."

**AABB fallback bbox from OBB corners:**
```python
def _obb_to_aabb(corners: np.ndarray) -> tuple[int, int, int, int]:
    """Compute axis-aligned bbox from 4 OBB corners."""
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
```

### Pattern 2: Affine Crop Utility — Extract, Letterbox, Invert

**What:** New functions in `segmentation/crop.py` that extract rotation-aligned crops from full frames. The crop is extracted by rotating the frame around the OBB center, then slicing a rectangular region. The transform matrix is saved so downstream code can back-project crop coordinates to frame coordinates.

**Design: `AffineCrop` dataclass:**
```python
@dataclass
class AffineCrop:
    """Rotation-aligned crop with its invertible affine transform.

    Attributes:
        image: Cropped image array of shape (crop_h, crop_w, C) or (crop_h, crop_w).
        M: 2×3 affine transform matrix (float64) that maps full-frame pixel
            coordinates to crop pixel coordinates.
        crop_size: (width, height) of the output canvas.
        frame_shape: (height, width) of the source frame.
    """
    image: np.ndarray
    M: np.ndarray          # shape (2, 3), float64
    crop_size: tuple[int, int]
    frame_shape: tuple[int, int]
```

**Core extraction logic:**
```python
def extract_affine_crop(
    frame: np.ndarray,
    center_xy: tuple[float, float],
    angle_math_rad: float,
    obb_w: float,
    obb_h: float,
    crop_size: tuple[int, int],
    padding_fraction: float = 0.15,
    interpolation: int = cv2.INTER_LINEAR,
) -> AffineCrop:
    """Extract a rotation-aligned crop from frame.

    Args:
        frame: Source BGR or grayscale image.
        center_xy: OBB center in full-frame pixel coords (x, y).
        angle_math_rad: OBB angle in standard math convention (CCW positive).
        obb_w: OBB width in pixels (long axis for fish).
        obb_h: OBB height in pixels (short axis for fish).
        crop_size: Output canvas (width, height) in pixels.
        padding_fraction: Proportional padding relative to OBB dimensions.
        interpolation: cv2 interpolation flag (default INTER_LINEAR).

    Returns:
        AffineCrop with image, invertible transform M, crop_size, frame_shape.
    """
    crop_w, crop_h = crop_size
    # Convert standard math angle back to cv2 degrees (cv2 uses CW degrees)
    angle_cv2_deg = math.degrees(-angle_math_rad)  # negate: math CCW -> cv2 CW

    # Step 1: Build rotation matrix around OBB center, then translate to crop center
    cx, cy = center_xy
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_cv2_deg, scale=1.0)

    # Step 2: Add translation to map OBB center to crop canvas center
    M_rot[0, 2] += crop_w / 2.0 - cx
    M_rot[1, 2] += crop_h / 2.0 - cy

    # Step 3: Apply warpAffine — the resulting image is the rotation-aligned crop
    crop_image = cv2.warpAffine(
        frame, M_rot, (crop_w, crop_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return AffineCrop(
        image=crop_image, M=M_rot,
        crop_size=crop_size, frame_shape=frame.shape[:2],
    )
```

**Note on letterbox padding:** The `crop_size` parameter plus proportional padding implicitly handles letterboxing — the affine canvas is `crop_size` and black pixels fill any area outside the rotated source region. No explicit letterbox step is needed because `warpAffine` with `BORDER_CONSTANT` zero-fills.

**Back-projection:**
```python
def invert_affine_point(
    crop_xy: tuple[float, float],
    M: np.ndarray,
) -> tuple[float, float]:
    """Back-project a crop-space point to full-frame coordinates.

    Args:
        crop_xy: (x, y) in crop pixel coordinates.
        M: 2×3 affine transform from extract_affine_crop().

    Returns:
        (x, y) in full-frame pixel coordinates.
    """
    M_inv = cv2.invertAffineTransform(M)  # returns 2×3 float64
    pt = np.array([[[crop_xy[0], crop_xy[1]]]], dtype=np.float64)
    result = cv2.transform(pt, M_inv)
    return float(result[0, 0, 0]), float(result[0, 0, 1])
```

**Round-trip accuracy:** `cv2.invertAffineTransform` is numerically exact for rotation+translation matrices (orthogonal rotation part). Round-trip error for a pure rotation+translation affine will be sub-pixel (< 0.001 px in practice). The 1-pixel threshold in the success criteria is easily achieved.

### Pattern 3: OBB Polygon Overlay in Visualization Observers

**What:** Two observers need extension for OBB rendering:

1. `TrackletTrailObserver._draw_trail()` — extend to draw the OBB polygon at the trail head frame. The OBB polygon is available when `detection.obb_points is not None`.
2. `Overlay2DObserver._draw_bbox()` — extend to draw OBB polygon using `obb_points` corners when available, falling back to AABB rectangle otherwise.

**OBB polygon rendering:**
```python
def _draw_obb_polygon(
    frame: np.ndarray,
    obb_points: np.ndarray,  # shape (4, 2), float
    color: tuple[int, int, int],
    fish_id: int | None = None,
    conf: float | None = None,
    thickness: int = 2,
) -> None:
    """Draw OBB polygon (4-corner) with optional label.

    Args:
        frame: BGR image to draw on (modified in-place).
        obb_points: Array of 4 corner points in pixel coords, shape (4, 2).
        color: BGR color tuple matching fish ID.
        fish_id: If provided, draw "ID: conf" label near top-left corner.
        conf: Confidence score for label.
        thickness: Line thickness.
    """
    pts = obb_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    if fish_id is not None:
        label = str(fish_id) if conf is None else f"{fish_id} {conf:.2f}"
        x, y = int(obb_points[:, 0].min()), int(obb_points[:, 1].min())
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)
```

**Key detail for TrackletTrailObserver:** The observer currently draws centroid dots and trails, but does not have access to per-frame `Detection` objects (only `Tracklet2D`). To draw OBB polygons, the observer needs access to detection data. There are two options:
- Option A: The `Tracklet2D` object carries OBB data (angle, obb_points) per frame — cleanest but requires Tracklet2D changes.
- Option B: The observer accesses `context.detections` and cross-references by camera_id + frame_idx + centroid proximity — no Tracklet2D changes.

**Recommendation:** Option B (access `context.detections` directly). The observer already has access to `context.tracks_2d` and `context.detections` is on PipelineContext. This avoids modifying Tracklet2D. The lookup is: for each active tracklet at frame_idx, find the Detection in `context.detections[frame_idx][cam_id]` closest to the tracklet centroid. The `detections` field on context is `list[dict[str, list[Detection]]]` — indexed by frame then camera.

**VIZ-01 scope:** VIZ-01 says "diagnostic mode renders OBB polygon overlays on detection frames." This is distinct from trail frames (VIZ-02). VIZ-01 is the `overlay_observer.py` detection bounding-box rendering path (`show_bbox=True`). The `_draw_bbox` method already exists — it needs an OBB branch.

**VIZ-02 scope:** "Tracklet trail visualization includes bounding box overlays (both axis-aligned and OBB when available)." This is `TrackletTrailObserver` — currently no bbox rendering; needs new drawing logic.

### Anti-Patterns to Avoid

- **Don't store the affine matrix on `Detection`**: `Detection` is used widely; adding a numpy array to it makes serialization and equality checks brittle. Keep `AffineCrop` as a separate return value from the crop extraction utility.
- **Don't convert angle twice**: The angle convention conversion happens once in `YOLOOBBBackend.detect()` at the ultralytics boundary. All downstream code uses standard math convention. Do not re-apply negation in the crop utility.
- **Don't add `cv2.INTER_CUBIC` by default**: Bilinear (`INTER_LINEAR`) is the correct default. Bicubic is for upsampling; at 256×128 output sizes the quality difference is negligible.
- **Don't skip the `r.obb is None` check**: For OBB models, `result.obb` is the attribute to use (not `result.boxes`). If no OBB detections exist in a frame, `result.obb` may be None or have zero rows — both must be handled.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Affine matrix inversion | Manual 2×3 matrix inverse math | `cv2.invertAffineTransform(M)` | Numerically stable, already in OpenCV, 1-liner |
| OBB → AABB conversion | Custom geometry code | `corners.min(axis=0)` / `.max(axis=0)` | Affine corners already in pixel space; bounding box is trivial numpy |
| Angle convention check | Runtime assertions in multiple places | Single conversion in `YOLOOBBBackend.detect()` | One boundary, one conversion, all downstream uses standard convention |
| Crop letterboxing | Separate letterbox step | `cv2.warpAffine` with `BORDER_CONSTANT=0` | warpAffine zero-fills anything outside the rotated source; letterbox is implicit |

## Common Pitfalls

### Pitfall 1: Angle Convention Mismatch
**What goes wrong:** ultralytics outputs angles clockwise in `[-pi/4, 3pi/4)`; OpenCV `getRotationMatrix2D` expects clockwise degrees; project `Detection.angle` uses standard math (CCW, `[-pi, pi]`). Mixing conventions produces crops rotated 90° off.
**Why it happens:** Three different systems with three different conventions.
**How to avoid:** Convert once at the ultralytics boundary in `YOLOOBBBackend.detect()`. Document the convention in every docstring. The crop utility receives `angle_math_rad` and internally converts to `cv2` degrees with `math.degrees(-angle_math_rad)`.
**Warning signs:** Orientation smoke test fails; crops show fish sideways or inverted.

### Pitfall 2: `result.obb` vs `result.boxes` for OBB Models
**What goes wrong:** Using `result.boxes` on an OBB model returns None or axis-aligned boxes without rotation.
**Why it happens:** OBB model results store oriented boxes in `result.obb`, not `result.boxes`.
**How to avoid:** Always use `result.obb` in `YOLOOBBBackend`. Guard with `if r.obb is None: continue`.
**Warning signs:** All detections have `angle=None` despite using `yolo_obb` detector.

### Pitfall 3: `obb_points` Coordinate Origin
**What goes wrong:** `result.obb.xyxyxyxy` returns coordinates in the frame's pixel space (not normalized 0-1). If the frame has been undistorted, the coordinates are in undistorted pixel space.
**Why it happens:** ultralytics returns pixel coordinates for the input image as-is.
**How to avoid:** No conversion needed — the project uses undistorted frames throughout. The OBB corners are already in undistorted pixel space, matching everything else.
**Warning signs:** OBB polygons drawn far off from actual fish positions.

### Pitfall 4: `xywhr` Width/Height vs Long/Short Axis
**What goes wrong:** ultralytics `xywhr` width and height may or may not correspond to the long and short axes. For fish (elongated), `w` in `xywhr` is the "width" of the box in the oriented frame, not necessarily the long axis.
**Why it happens:** ultralytics `cv2.minAreaRect` uses a convention where width ≤ height, or uses the raw box dimensions before/after rotation normalization. The angle range `[-pi/4, 3pi/4)` constrains which dimension is width vs height.
**How to avoid:** Use `max(obb_w, obb_h)` for the crop canvas's long dimension and `min(obb_w, obb_h)` for the short dimension when computing proportional padding. For the affine extraction, use the OBB `w` and `h` as-is from `xywhr` but pass `crop_size` explicitly from config (`DetectionConfig`).
**Warning signs:** Crops are transposed (fish appears vertical when it should be horizontal).

### Pitfall 5: `get_backend()` Return Type Annotation
**What goes wrong:** `get_backend()` currently has return type `YOLOBackend`. Adding `YOLOOBBBackend` requires updating the return type annotation to a union or Protocol.
**Why it happens:** Strict typing with basedpyright.
**How to avoid:** Define a `DetectorBackend` Protocol with `detect(frame: np.ndarray) -> list[Detection]` and type `get_backend` to return it. Or use `YOLOBackend | YOLOOBBBackend`. The Protocol approach is cleaner and already aligns with the Stage Protocol pattern in the project.

### Pitfall 6: OBB Polygon Drawing with `cv2.polylines`
**What goes wrong:** `cv2.polylines` expects `pts` shape `(N, 1, 2)` (not `(N, 2)`).
**Why it happens:** OpenCV's contour/polyline functions use the `(N, 1, 2)` convention.
**How to avoid:** Always `.reshape((-1, 1, 2))` before passing to `cv2.polylines`.
**Warning signs:** `cv2.error: (-5:Bad argument)` from `cv2.polylines`.

## Code Examples

Verified patterns from official sources and the existing codebase:

### ultralytics OBB Inference (verified against docs.ultralytics.com/tasks/obb)
```python
# Source: https://docs.ultralytics.com/tasks/obb/
# Model loaded with YOLO("yolov8n-obb.pt") or fine-tuned weights
results = model.predict(frame, conf=0.5, iou=0.45, verbose=False)
for r in results:
    if r.obb is None:
        continue
    for i in range(len(r.obb.conf)):
        conf = float(r.obb.conf[i])
        xywhr = r.obb.xywhr[i].cpu().numpy()   # [cx, cy, w, h, angle_cw_rad]
        corners = r.obb.xyxyxyxy[i].cpu().numpy()  # shape (4, 2), pixel coords
```

### Affine Crop Extraction (cv2, verified pattern)
```python
# Source: cv2.getRotationMatrix2D + cv2.warpAffine standard usage
import math
import cv2
import numpy as np

cx, cy = center_xy
angle_cv2_deg = math.degrees(-angle_math_rad)  # standard math -> cv2 CW degrees
M = cv2.getRotationMatrix2D((cx, cy), angle_cv2_deg, scale=1.0)
# Shift so OBB center maps to crop canvas center
M[0, 2] += crop_w / 2.0 - cx
M[1, 2] += crop_h / 2.0 - cy
crop = cv2.warpAffine(frame, M, (crop_w, crop_h),
                       flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
```

### Affine Back-Projection (cv2, verified)
```python
# Source: cv2.invertAffineTransform standard usage
M_inv = cv2.invertAffineTransform(M)  # exact inverse of 2×3 rotation+translation
pt_crop = np.array([[[x_crop, y_crop]]], dtype=np.float64)
pt_frame = cv2.transform(pt_crop, M_inv)  # shape (1, 1, 2)
x_frame, y_frame = float(pt_frame[0, 0, 0]), float(pt_frame[0, 0, 1])
```

### OBB Polygon Drawing (cv2, verified)
```python
# CRITICAL: polylines needs (N, 1, 2) not (N, 2)
pts = obb_points.astype(np.int32).reshape((-1, 1, 2))
cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
```

### DetectionConfig Extension for crop_size (config.py pattern)
```python
# DetectionConfig already has detector_kind and model_path.
# Add crop_size as a tuple field:
@dataclass(frozen=True)
class DetectionConfig:
    detector_kind: str = "yolo"
    model_path: str | None = None
    crop_size: tuple[int, int] = (256, 128)  # (width, height) for affine crops
    extra: dict[str, Any] = field(default_factory=dict)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Axis-aligned bbox crops | OBB rotation-aligned affine crops | Phase 32 | Fish bodies are axis-aligned within crop; downstream models train better |
| Single detection kind in `get_backend` | Multiple backend kinds via Protocol | Phase 32 | `yolo_obb` added alongside `yolo`; both share the Detection output contract |
| Draw AABB for all detections | Draw OBB polygon when available, AABB as fallback | Phase 32 | More informative diagnostic visualization |

## Open Questions

1. **Where should affine crop data flow to downstream stages?**
   - What we know: `Detection` already has `angle` and `obb_points`. Downstream midline stages use crops extracted from detections.
   - What's unclear: Whether `MidlineStage` currently calls the crop utilities directly or if it re-extracts from `detection.bbox`. If it re-extracts from `bbox`, it needs to use the new affine utility instead.
   - Recommendation: Check `core/midline/stage.py` to confirm crop extraction point. The planner should ensure the affine crop utility is the single extraction path (DET-02 contract) — but the actual midline integration is Phase 33 scope, not Phase 32. Phase 32 only needs to provide the utility; Phase 33 consumes it.

2. **`TrackletTrailObserver` detection access pattern**
   - What we know: The observer has `context.tracks_2d` and `context.tracklet_groups`. It needs to draw OBB polygons at tracklet head positions.
   - What's unclear: Whether `context.detections` is available at the time `TrackletTrailObserver` runs (post-pipeline).
   - Recommendation: `PipelineComplete` carries the full context; `context.detections` is populated by Stage 1 and remains on context. The observer can access it via `getattr(context, "detections", None)`.

3. **`crop_size` config placement**
   - What we know: The context says `crop_size: [256, 128]` in config. `DetectionConfig` is the natural home.
   - What's unclear: YAML serializes `tuple` as a list; the frozen dataclass must accept both `list` and `tuple`.
   - Recommendation: Store as `tuple[int, int]` in the dataclass; in `_filter_fields` / `load_config`, convert incoming list values to tuple. Alternatively, store as two separate fields `crop_width: int = 256` and `crop_height: int = 128` to avoid the list/tuple YAML issue.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` (hatch env) |
| Quick run command | `hatch run test` |
| Full suite command | `hatch run test-all` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DET-01 | `detector_kind: yolo_obb` produces detections with non-None `angle` and `obb_points` | unit | `hatch run test tests/unit/core/detection/test_detection_stage.py` | ✅ (extend existing) |
| DET-01 | `get_backend("yolo_obb", ...)` returns `YOLOOBBBackend`; `get_backend("unknown")` raises `ValueError` | unit | `hatch run test tests/unit/core/detection/test_detection_stage.py` | ✅ (extend existing) |
| DET-02 | `extract_affine_crop` with known angle produces fish-axis-aligned crop (orientation smoke test) | unit | `hatch run test tests/unit/segmentation/test_affine_crop.py` | ❌ Wave 0 |
| DET-03 | Round-trip error `frame → crop → back-projected_frame` < 1 px for random points | unit | `hatch run test tests/unit/segmentation/test_affine_crop.py` | ❌ Wave 0 |
| VIZ-01 | `_draw_obb_polygon` called when `obb_points` present; `cv2.polylines` invoked with correct shape | unit | `hatch run test tests/unit/engine/test_overlay_observer.py` | ✅ (extend existing) |
| VIZ-02 | `TrackletTrailObserver` renders OBB polygon at trail head when detections available | unit | `hatch run test tests/unit/engine/test_tracklet_trail_observer.py` | ✅ (extend existing) |

### Sampling Rate
- **Per task commit:** `hatch run test`
- **Per wave merge:** `hatch run test`
- **Phase gate:** `hatch run test` green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/segmentation/test_affine_crop.py` — covers DET-02, DET-03 (round-trip accuracy, orientation smoke test)

## Sources

### Primary (HIGH confidence)
- https://docs.ultralytics.com/tasks/obb/ — OBB result fields (`result.obb.xywhr`, `.xyxyxyxy`, `.conf`), Python inference API
- https://docs.ultralytics.com/reference/engine/results/ — Results class reference
- Existing codebase: `src/aquapose/segmentation/detector.py` — `Detection` dataclass with `angle`/`obb_points` fields already in place (Phase 30)
- Existing codebase: `src/aquapose/core/detection/backends/__init__.py` — `get_backend()` factory to extend
- Existing codebase: `src/aquapose/engine/tracklet_trail_observer.py` — `FISH_COLORS_BGR` palette and trail drawing patterns
- Existing codebase: `.planning/STATE.md` — "OBB angle convention risk" concern and "Detection.angle uses standard math radians" contract

### Secondary (MEDIUM confidence)
- https://github.com/ultralytics/ultralytics/issues/13003 — Angle convention confirmed as clockwise radians; STATE.md independently notes the conversion needed
- OpenCV docs (`cv2.getRotationMatrix2D`, `cv2.warpAffine`, `cv2.invertAffineTransform`) — standard OpenCV API, well-established

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — same libraries already used; OBB is same ultralytics package
- Architecture: HIGH — follows existing backend + observer patterns exactly; cv2 affine API is well-known
- Pitfalls: HIGH — angle convention pitfall is documented in STATE.md; cv2.polylines shape requirement is a well-known gotcha

**Research date:** 2026-02-28
**Valid until:** 2026-03-28 (ultralytics OBB API stable; OpenCV API very stable)
