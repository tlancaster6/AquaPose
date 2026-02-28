# Architecture Research

**Domain:** 3D fish pose estimation — swappable ML backends, training infrastructure, config cleanup
**Researched:** 2026-02-28
**Confidence:** HIGH (analysis of live codebase, not conjecture)

---

## Existing Architecture: What v2.2 Inherits

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LAYER 3: Observers (side effects)                │
│  Timing  Console  HDF5  Overlay2D  Animation3D  Diagnostic  [new: OBB] │
├─────────────────────────────────────────────────────────────────────────┤
│                        LAYER 2: PosePipeline (orchestrator)             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ build_stages(config) → [Stage1..Stage5] → PipelineContext        │   │
│  │ event dispatch → registered observers                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                        LAYER 1: Core Computation (5 stages)             │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌────────────┐  │
│  │Detection │ │2DTracking│ │Association│ │ Midline  │ │Reconstruct.│  │
│  │ Stage 1  │ │ Stage 2  │ │  Stage 3  │ │ Stage 4  │ │  Stage 5   │  │
│  │          │ │          │ │           │ │          │ │            │  │
│  │ backends/│ │(OC-SORT) │ │ (Leiden)  │ │ backends/│ │ backends/  │  │
│  │ yolo.py  │ │          │ │           │ │ seg_ext  │ │ triangul.  │  │
│  │ [+obb]   │ │          │ │           │ │ dir_pose │ │ curve_opt  │  │
│  └──────────┘ └──────────┘ └───────────┘ └──────────┘ └────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                        INFRASTRUCTURE                                    │
│  engine/config.py (frozen dataclasses)  engine/events.py               │
│  core/context.py (PipelineContext)      cli.py (Click thin wrapper)     │
│  [new: training/ package]              [new: aquapose train CLI group]  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Import Boundary (enforced by AST pre-commit hook)

```
core/     → calibration/, reconstruction/, segmentation/ (NO engine/ imports)
engine/   → core/, calibration/, reconstruction/, segmentation/, tracking/
cli.py    → engine/ only
training/ → core/, segmentation/, reconstruction/ (NOT engine/)
```

The import boundary is the most critical architectural constraint. The AST checker
enforces it at commit time. Every new module must be classified by layer before
writing the first import.

---

## Component Responsibilities: Current State

| Component | Layer | Responsibility | Output |
|-----------|-------|----------------|--------|
| `core/detection/stage.py` | Core | Orchestrates per-camera detection via backend | `context.detections` |
| `core/detection/backends/yolo.py` | Core | Wraps YOLODetector (XYXY bbox) | `list[Detection]` |
| `core/tracking/stage.py` | Core | Per-camera OC-SORT 2D tracking | `context.tracks_2d` |
| `core/association/stage.py` | Core | Leiden cross-camera clustering | `context.tracklet_groups` |
| `core/midline/stage.py` | Core | Backend dispatch + frame loop | `context.annotated_detections` |
| `core/midline/backends/segment_then_extract.py` | Core | U-Net then skeleton then BFS | `dict[str, list[AnnotatedDetection]]` |
| `core/midline/backends/direct_pose.py` | Core | Stub (NotImplementedError) | — |
| `core/reconstruction/stage.py` | Core | Backend dispatch + frame assembly | `context.midlines_3d` |
| `core/reconstruction/backends/triangulation.py` | Core | RANSAC triangulation wrapper | `dict[int, Midline3D]` |
| `core/reconstruction/backends/curve_optimizer.py` | Core | Chamfer 3D B-spline optimizer | `dict[int, Midline3D]` |
| `engine/pipeline.py` | Engine | Stage ordering, event dispatch | — |
| `engine/config.py` | Engine | Frozen dataclass config hierarchy | `PipelineConfig` |
| `segmentation/detector.py` | Segmentation | MOG2Detector, YOLODetector, make_detector() | `list[Detection]` |
| `segmentation/model.py` | Segmentation | UNetSegmentor (MobileNetV3-Small encoder) | `list[SegResult]` |
| `segmentation/crop.py` | Segmentation | CropRegion, compute_crop_region(), extract_crop() | `CropRegion`, `np.ndarray` |
| `reconstruction/midline.py` | Reconstruction | Midline2D dataclass + extraction helpers | `Midline2D` |
| `reconstruction/triangulation.py` | Reconstruction | triangulate_midlines(), Midline3D, MidlineSet | `Midline3D` |
| `cli.py` | CLI | `aquapose run` Click command | — |

---

## v2.2 Integration Map: What Changes, What's New

### 1. YOLO-OBB Detection Backend

**Where it lives:** `core/detection/backends/yolo_obb.py` (NEW)
**What calls it:** `core/detection/backends/__init__.py` — extend `get_backend()` to dispatch `kind="yolo_obb"`
**What it produces:** Same `list[Detection]` contract as `YOLOBackend`

The key new behavior is the OBB-aware crop:

- YOLO-OBB returns oriented bounding boxes with angle `theta` in addition to XYXY
- The `Detection` dataclass must carry OBB metadata for downstream affine crop
- Two design options:

  **Option A (recommended):** Extend `Detection` with optional OBB fields (`angle`, `obb_points`). Downstream backends check for OBB presence to choose affine vs axis-aligned crop. No breaking change — existing code sees `angle=None`.

  **Option B:** Subclass `Detection` as `OBBDetection`. Rejected: breaks all isinstance checks, doubles maintenance burden.

**Affine crop logic location:** `segmentation/crop.py` — add `compute_obb_crop_region()` and `extract_affine_crop()`. Lives in `segmentation/` (not `core/`) because crop geometry is a reusable utility that both the detection backend and the midline backend need.

**Config change:** `DetectionConfig` — existing `detector_kind` field dispatches to `"yolo_obb"`. OBB-specific params (e.g. `obb_conf_threshold`) go in `DetectionConfig.extra` dict until they stabilize, then get promoted to first-class fields.

**Overlay change:** `engine/overlay_observer.py` — add OBB bbox drawing alongside existing axis-aligned overlay.

**Integration point summary:**
```
YOLOOBBBackend.detect(frame) -> list[Detection]  # Detection.angle is set
    |
    v
SegmentThenExtractBackend.process_frame()
    -> compute_obb_crop_region() if detection.angle is not None
    -> extract_affine_crop() (OpenCV warpAffine)
    -> UNetSegmentor.segment(rotated_crop)
    -> untransform midline points back to frame coordinates
```

### 2. Keypoint-Based Midline Backend (direct_pose implementation)

**Where it lives:** `core/midline/backends/direct_pose.py` (MODIFY — replace NotImplementedError stub)
**Supporting model:** `segmentation/keypoint_model.py` (NEW) — U-Net + keypoint regression head

**Contract preserved:** `process_frame()` method signature is already defined by the stub. It must return `dict[str, list[AnnotatedDetection]]` — same as `SegmentThenExtractBackend`.

**Midline2D contract change — per-point confidence:**

Current `Midline2D`:
```python
@dataclass
class Midline2D:
    points: np.ndarray      # (N, 2) float32
    half_widths: np.ndarray # (N,) float32
    fish_id: int
    camera_id: str
    frame_index: int
    is_head_to_tail: bool = False
```

Required addition:
```python
    point_confidence: np.ndarray | None = None  # (N,) float32, None = uniform weight
```

`None` default preserves backward compatibility. The `segment_then_extract` backend produces `None` (no per-point confidence from skeleton). The `direct_pose` backend populates it with per-keypoint confidence from the regression head.

**Where per-point confidence is consumed:** Both reconstruction backends must be updated.

In `reconstruction/triangulation.py`:
- The existing `triangulate_midlines()` function takes `MidlineSet` (dict of `Midline2D`)
- When `point_confidence` is present, use it to weight RANSAC inlier votes: high-confidence points have stronger vote
- Implementation: pass confidence as per-point weights to the triangulation loop for each body position
- The `is_low_confidence` flag on `Midline3D` can incorporate average confidence

In `reconstruction/curve_optimizer.py`:
- Chamfer distance computation can weight contributions by point confidence
- Lower-confidence points contribute less to the fitting objective

**Partial midline handling:**
- Keypoint backend may produce sparser outputs when some keypoints are occluded
- `Midline2D.point_confidence` values near 0.0 signal "not observed"
- Both reconstruction backends must skip (or down-weight) near-zero-confidence points rather than treating them as valid observations

### 3. Training Infrastructure

**Where it lives:** `src/aquapose/training/` (NEW PACKAGE)
**Layer classification:** Same level as `segmentation/` — NOT inside `engine/`. Training code imports from `core/`, `segmentation/`, and `reconstruction/` but NEVER from `engine/`.

**Package structure:**
```
src/aquapose/training/
├── __init__.py             # exports: train_unet, train_keypoint, TrainingConfig
├── unet.py                 # moves/wraps segmentation/training.py logic
├── keypoint.py             # NEW: keypoint model training loop
├── dataset.py              # NEW: keypoint dataset (OBB crops + point annotations)
└── config.py               # NEW: TrainingConfig frozen dataclass
```

**Why `training/` not `segmentation/training.py`:**
The existing `segmentation/training.py` is U-Net specific. The new `training/` package consolidates all model training (U-Net segmentation, YOLO fine-tuning, keypoint regression) under a single CLI namespace. `segmentation/training.py` can be deprecated and re-exported from `training/unet.py` for backward compat, or left in place with `training/unet.py` as a thin wrapper.

**CLI entrypoints:**
```python
# cli.py extension — new Click group
@cli.group()
def train() -> None:
    """Train AquaPose models."""

@train.command("unet")
def train_unet(...) -> None: ...

@train.command("keypoint")
def train_keypoint(...) -> None: ...

@train.command("yolo")
def train_yolo(...) -> None: ...  # wraps scripts/train_yolo.py logic
```

**`pyproject.toml` entrypoint:**
No changes needed. `aquapose` is already registered as the CLI entry point. The new `train` subcommand attaches to the existing `cli` Click group via `@cli.group()`.

**OBB crop dataset for keypoint training:**
- `training/dataset.py` defines `KeypointDataset`
- Loads from COCO-format JSON with keypoint annotations (or OBB annotations)
- Affine-crops each fish using the OBB angle from YOLO-OBB or manual annotation
- Returns `(rotated_crop, keypoints_in_crop_space, confidence_flags)`

**Import path for training:**
```
training/unet.py     -> segmentation/model.py, segmentation/dataset.py
training/keypoint.py -> segmentation/keypoint_model.py, training/dataset.py
training/config.py   -> (stdlib only, no internal imports)
cli.py               -> training/ (CLI layer can import training layer)
```

### 4. Config System Cleanup

**Affected file:** `engine/config.py`

**Change 1: Configurable N_SAMPLE_POINTS**
- `MidlineConfig.n_points` already exists (default 15)
- Problem: `reconstruction/triangulation.py` has `N_SAMPLE_POINTS: int = 15` hardcoded as a module constant
- Fix: `TriangulationBackend.__init__()` must receive `n_points` from `ReconstructionConfig` (or derive it from context at runtime)
- The reconstruction backend needs to know N at construction time for knot vector precomputation
- Solution: Add `n_points: int = 15` to `ReconstructionConfig` and thread it through `build_stages()` into `TriangulationBackend`

**Change 2: Device propagation**
- Current state: `MidlineConfig` has no `device` field; `SegmentThenExtractBackend` hardcodes `device="cuda"` default
- `DetectionConfig.device` exists but is not propagated to Midline or Reconstruction stages
- Fix: Add `device: str = "cuda"` to `MidlineConfig` and `ReconstructionConfig`
- `load_config()` should propagate a top-level `device` override to all sub-configs (same pattern as `n_animals -> expected_fish_count`)

**Change 3: `init-config` subcommand**
- New CLI command: `aquapose init-config [--output path.yaml]`
- Serializes a default `PipelineConfig` to YAML for user customization
- Implementation: call `serialize_config(load_config())` and write to file
- No new dataclass fields needed — uses existing `serialize_config()`

**Change 4: Promote OBB params from `extra`**
- Ensure YOLO-OBB params (`conf_threshold`, `iou_threshold`) are first-class `DetectionConfig` fields rather than buried in `extra`
- `extra` dict remains for truly backend-specific experimental params

---

## Data Flow: v2.2 End-to-End

### Standard Path (YOLO-OBB + keypoint midline)

```
VideoSet.frames
    |
    v
DetectionStage (YOLO-OBB backend)
    -> list[Detection]  # Detection.angle, Detection.obb_points now set
    |
    v  context.detections
TrackingStage (OC-SORT, unchanged)
    -> dict[str, list[Tracklet2D]]
    |
    v  context.tracks_2d
AssociationStage (Leiden, unchanged)
    -> list[TrackletGroup]
    |
    v  context.tracklet_groups
MidlineStage (direct_pose backend)
    -> compute_obb_crop_region(det.bbox, det.angle)
    -> extract_affine_crop(frame, obb_region)
    -> KeypointModel.infer(rotated_crop)  -> (points, confidence)
    -> untransform keypoints to frame coords
    -> Midline2D(points=..., point_confidence=...)  # new field
    |
    v  context.annotated_detections
ReconstructionStage (triangulation backend)
    -> triangulate_midlines(midline_set)
        -> weight RANSAC votes by point_confidence
        -> produce Midline3D(is_low_confidence=...)
    |
    v  context.midlines_3d
Observers (HDF5, Overlay, Animation)
```

### Midline2D Contract: Before vs After

| Field | v2.1 | v2.2 |
|-------|------|------|
| `points` | `(N, 2)` float32, N=15 hardcoded | `(N, 2)` float32, N from config |
| `half_widths` | `(N,)` float32 | `(N,)` float32, may be zeros for keypoint backend |
| `point_confidence` | not present | `(N,)` float32 or `None` (None = uniform) |
| `fish_id` | int | int (unchanged) |
| `camera_id` | str | str (unchanged) |
| `frame_index` | int | int (unchanged) |
| `is_head_to_tail` | bool | bool (unchanged) |

Both backends remain interchangeable at the `process_frame()` boundary. The `MidlineStage` does not inspect `point_confidence`; it is transparent to the stage orchestrator. Only reconstruction backends consume it.

---

## Component Boundaries: New vs Modified

### New Components

| Component | Path | Layer | Depends On |
|-----------|------|-------|------------|
| `YOLOOBBDetector` | `segmentation/detector.py` extension | Segmentation | `ultralytics` YOLO OBB API |
| `compute_obb_crop_region()` | `segmentation/crop.py` extension | Segmentation | numpy, cv2 |
| `extract_affine_crop()` | `segmentation/crop.py` extension | Segmentation | `cv2.warpAffine` |
| `untransform_points_from_obb()` | `segmentation/crop.py` extension | Segmentation | numpy |
| `KeypointModel` | `segmentation/keypoint_model.py` | Segmentation | PyTorch, U-Net encoder |
| `YOLOOBBBackend` | `core/detection/backends/yolo_obb.py` | Core | `segmentation/detector.py` |
| `DirectPoseBackend` (full impl) | `core/midline/backends/direct_pose.py` | Core | `segmentation/keypoint_model.py`, `segmentation/crop.py` |
| `training/` package | `src/aquapose/training/` | Training | `segmentation/`, `core/` |
| `training/unet.py` | `src/aquapose/training/unet.py` | Training | wraps `segmentation/training.py` |
| `training/keypoint.py` | `src/aquapose/training/keypoint.py` | Training | `segmentation/keypoint_model.py` |
| `training/dataset.py` | `src/aquapose/training/dataset.py` | Training | `segmentation/crop.py` |
| `training/config.py` | `src/aquapose/training/config.py` | Training | stdlib only |
| `aquapose train` CLI group | `cli.py` extension | CLI | `training/` |
| `aquapose init-config` command | `cli.py` extension | CLI | `engine/config.py` |

### Modified Components

| Component | Path | Change |
|-----------|------|--------|
| `Detection` dataclass | `segmentation/detector.py` | Add `angle: float | None = None`, `obb_points: np.ndarray | None = None` |
| `Midline2D` dataclass | `reconstruction/midline.py` | Add `point_confidence: np.ndarray | None = None` |
| `get_backend()` (detection) | `core/detection/backends/__init__.py` | Register `"yolo_obb"` kind |
| `TriangulationBackend` | `core/reconstruction/backends/triangulation.py` | Consume `point_confidence` for weighted RANSAC; accept `n_points` param |
| `CurveOptimizerBackend` | `core/reconstruction/backends/curve_optimizer.py` | Consume `point_confidence` for weighted Chamfer |
| `DetectionConfig` | `engine/config.py` | Promote OBB params from `extra` to first-class fields |
| `MidlineConfig` | `engine/config.py` | Add `device: str = "cuda"` |
| `ReconstructionConfig` | `engine/config.py` | Add `n_points: int = 15`, `device: str = "cuda"` |
| `load_config()` | `engine/config.py` | Propagate top-level `device` to sub-configs |
| `overlay_observer.py` | `engine/overlay_observer.py` | Add OBB bbox drawing when `Detection.angle is not None` |
| `cli.py` | `src/aquapose/cli.py` | Add `train` command group, `init-config` command |

---

## Build Order: Dependency-Driven Sequencing

The critical constraint is that each task's dependencies must exist before it can be written and tested.

### Phase A: Foundation (no dependencies on new features)

**A1. Config cleanup** (`engine/config.py`)
- Add `device` propagation, `n_points` to `ReconstructionConfig`, promote OBB params
- Add `init-config` CLI command
- No new models needed; all existing tests still pass
- Enables: all later tasks that read config

**A2. `Detection` dataclass OBB extension** (`segmentation/detector.py`)
- Add `angle: float | None = None`, `obb_points: np.ndarray | None = None`
- Backward-compatible (None defaults)
- Enables: YOLOOBBDetector, affine crop, OBB overlay

**A3. `Midline2D` dataclass confidence extension** (`reconstruction/midline.py`)
- Add `point_confidence: np.ndarray | None = None`
- Backward-compatible (None = uniform weight in both reconstruction backends)
- Enables: DirectPoseBackend output, confidence-weighted reconstruction

A1, A2, A3 can be done in parallel (no inter-dependencies within Phase A).

### Phase B: Detection Backend

**B4. Affine crop utilities** (`segmentation/crop.py`)
- `compute_obb_crop_region()`, `extract_affine_crop()`, `untransform_points_from_obb()`
- Depends on: A2 (OBB-aware `Detection`)
- Pure geometry; testable with synthetic bboxes

**B5. `YOLOOBBDetector`** (`segmentation/detector.py`)
- Wraps `ultralytics` YOLO OBB inference
- Depends on: A2, B4
- Produces `Detection` with `angle` set

**B6. `YOLOOBBBackend`** (`core/detection/backends/yolo_obb.py`)
- Wraps `YOLOOBBDetector` with eager-load and fail-fast
- Register in `core/detection/backends/__init__.py`
- Depends on: B5

**B7. OBB overlay** (`engine/overlay_observer.py`)
- Draw rotated bbox when `Detection.angle is not None`
- Depends on: A2
- Can be done in parallel with B5/B6

### Phase C: Keypoint Midline Backend

**C8. `KeypointModel`** (`segmentation/keypoint_model.py`)
- U-Net encoder + keypoint regression head (N*2 coordinates + N confidence logits)
- Depends on: A3 (so training output maps to `Midline2D.point_confidence`)
- Can be developed in parallel with Phase B

**C9. `DirectPoseBackend` implementation** (`core/midline/backends/direct_pose.py`)
- Replaces `NotImplementedError` stub
- Uses `compute_obb_crop_region()` + `extract_affine_crop()` + `KeypointModel`
- Untransforms keypoints from crop space to frame space
- Produces `AnnotatedDetection` with `Midline2D(point_confidence=...)`
- Depends on: A2, A3, B4, C8

### Phase D: Confidence-Weighted Reconstruction

**D10. `TriangulationBackend` confidence weighting**
- Consume `Midline2D.point_confidence` in RANSAC body-point voting
- Threshold near-zero confidence points as "not observed"
- Accept `n_points` constructor param (from `ReconstructionConfig.n_points`)
- Depends on: A3 only — can run in parallel with Phases B and C

**D11. `CurveOptimizerBackend` confidence weighting**
- Weight Chamfer distance contributions by point confidence
- Depends on: A3, pattern from D10

### Phase E: Training Infrastructure

**E12. `training/config.py`** — `TrainingConfig` dataclass (no internal imports)
- Depends on: nothing except stdlib; can start immediately

**E13. `training/dataset.py`** — `KeypointDataset`
- OBB crop loading + keypoint annotation parsing
- Depends on: B4 (affine crop utilities), E12

**E14. `training/unet.py`** — wrap U-Net training
- Thin wrapper or move of `segmentation/training.py`
- Depends on: E12

**E15. `training/keypoint.py`** — keypoint model training loop
- Depends on: C8, E12, E13

**E16. `training/__init__.py`** — public exports
- Depends on: E14, E15

**E17. `aquapose train` CLI commands** (`cli.py`)
- `aquapose train unet`, `aquapose train keypoint`, `aquapose train yolo`
- Depends on: E16; scripts/train_yolo.py for YOLO subcommand

### Recommended Order Summary

```
Phase A (config + dataclass contracts) → unblocks everything
    |
    +---> Phase B (OBB detection)   <--> Phase C.8 (KeypointModel)   [parallel]
    |         |                              |
    +---> Phase D.10 (confidence triangulation)                       [parallel with B,C]
              |
              v
          Phase C.9 (DirectPoseBackend) — needs B4 and C8
              |
              v
          Phase E (training infrastructure) — needs B4, C8; otherwise independent
```

---

## Architectural Patterns

### Pattern 1: Backend Registry (existing — extend for OBB)

**What:** `get_backend(kind, **kwargs)` factory in each stage's `backends/__init__.py`. New backends registered by adding an `if kind == "..."` branch.

**When to use:** Any new inference backend. Never subclass stages.

**Example:**
```python
# core/detection/backends/__init__.py
def get_backend(kind: str, **kwargs: Any) -> YOLOBackend | YOLOOBBBackend:
    if kind == "yolo":
        return YOLOBackend(**kwargs)
    if kind == "yolo_obb":
        return YOLOOBBBackend(**kwargs)
    raise ValueError(f"Unknown detector kind: {kind!r}")
```

### Pattern 2: Fail-Fast Construction (existing — follow for new backends)

**What:** All model weights and file paths are validated in `__init__()`, not `run()`. Missing files raise `FileNotFoundError` immediately.

**When to use:** Every new backend class. Catches configuration errors before the pipeline starts processing frames.

```python
class YOLOOBBBackend:
    def __init__(self, model_path: str | Path, ...) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO-OBB weights not found: {model_path}")
        self._detector = YOLOOBBDetector(model_path=model_path, ...)
```

### Pattern 3: None-Default Optional Fields (backward-compat dataclass extension)

**What:** New optional fields on existing dataclasses default to `None`. Downstream code checks for `None` before using the field.

**When to use:** Extending `Detection` or `Midline2D` without breaking existing code paths.

```python
@dataclass
class Midline2D:
    points: np.ndarray
    half_widths: np.ndarray
    # ... existing fields ...
    point_confidence: np.ndarray | None = None  # None = uniform weight
```

Reconstruction backends:
```python
weights = (
    midline.point_confidence
    if midline.point_confidence is not None
    else np.ones(len(midline.points), dtype=np.float32)
)
```

### Pattern 4: Layer Classification Before First Import

**What:** Before writing any new `.py` file, decide which layer it belongs to. Use the import boundary table to validate no violations would occur.

**When to use:** Always. The AST pre-commit hook rejects commits with violations.

```
New file question: "Does it import from engine/?"
  YES -> it is engine/ or cli/ layer
  NO  -> it is core/, segmentation/, reconstruction/, or training/ layer
```

### Pattern 5: Lazy Imports for Heavy Dependencies

**What:** `ultralytics`, `torch`, `boxmot` are imported inside `__init__()` or factory functions, not at module top-level. This prevents import failures when optional deps are absent.

**When to use:** Any new backend that imports heavy optional deps.

```python
class YOLOOBBDetector:
    def __init__(self, model_path: ...) -> None:
        from ultralytics import YOLO  # lazy — not at top of file
        self._model = YOLO(str(model_path), task="obb")
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Putting Training Logic in `engine/`

**What people do:** Add `training/` inside `engine/` since the CLI entry point is there.

**Why it's wrong:** Training is a batch utility (same level as `segmentation/`), not an orchestration concern. Placing it in `engine/` would force training code to depend on `PosePipeline` and `PipelineConfig`, creating circular initialization paths and making the package non-importable without all engine deps.

**Do this instead:** `src/aquapose/training/` is a peer of `segmentation/` and `reconstruction/`. The CLI layer (`cli.py`) imports both `engine/` and `training/` — that is the correct fan-in point.

### Anti-Pattern 2: Hardcoding N_SAMPLE_POINTS in Reconstruction

**What people do:** Leave `N_SAMPLE_POINTS: int = 15` as a module-level constant and add the new keypoint backend that produces a different N.

**Why it's wrong:** Both reconstruction backends derive array shapes from N. If the keypoint backend produces 20 points and the triangulation backend expects 15, the RANSAC body-point loop silently misaligns.

**Do this instead:** `MidlineConfig.n_points` drives N end-to-end. `ReconstructionConfig.n_points` inherits or mirrors it. Both backends derive N from config at construction time. The module constant `N_SAMPLE_POINTS` becomes documentation only.

### Anti-Pattern 3: Implementing OBB Crop Inside the Detection Stage

**What people do:** Put `extract_affine_crop()` inside `core/detection/` because OBB comes from the detector.

**Why it's wrong:** The OBB crop geometry is also needed by `core/midline/backends/direct_pose.py` when it processes the crop for keypoint extraction. Two copies in different packages diverge.

**Do this instead:** Affine crop utilities belong in `segmentation/crop.py` — the shared geometry module. Both the detection overlay and the midline backend import from the same location.

### Anti-Pattern 4: Confidence as a Stage Filter

**What people do:** Have `MidlineStage` drop `AnnotatedDetection` objects where `midline.point_confidence.mean() < threshold`.

**Why it's wrong:** Partial midlines (some keypoints visible, some occluded) carry usable information for triangulation — the high-confidence subset is still valid. Filtering at the stage level discards this.

**Do this instead:** Pass all `AnnotatedDetection` objects downstream. Let reconstruction backends weight by `point_confidence`. Only log and count partial midlines for diagnostic purposes.

### Anti-Pattern 5: Subclassing Detection for OBB

**What people do:** Create `OBBDetection(Detection)` to carry OBB-specific fields.

**Why it's wrong:** Breaks all `isinstance(det, Detection)` checks and `list[Detection]` type annotations throughout the pipeline. Every stage would need OBB-aware type guards.

**Do this instead:** Extend `Detection` with `angle: float | None = None` and `obb_points: np.ndarray | None = None`. Existing code paths continue working; OBB-aware paths branch on `angle is not None`.

---

## Integration Points

### Internal Boundaries

| Boundary | Communication | Constraint |
|----------|---------------|------------|
| `core/detection/backends/` to `segmentation/detector.py` | Direct import | OK — core imports segmentation |
| `core/midline/backends/direct_pose.py` to `segmentation/keypoint_model.py` | Direct import | OK — core imports segmentation |
| `core/midline/backends/` to `segmentation/crop.py` | Direct import for affine crop | OK |
| `training/` to `segmentation/` | Direct import | OK — peer packages |
| `training/` to `engine/` | FORBIDDEN | Training must not import PosePipeline |
| `cli.py` to `training/` | Direct import | OK — CLI is the fan-in layer |
| `Midline2D.point_confidence` to `TriangulationBackend` | Via `MidlineSet` dict | No new imports needed; field added to existing type |

### External Integrations

| Integration | Library | API Surface |
|-------------|---------|-------------|
| YOLO-OBB inference | `ultralytics` YOLO OBB | `model.predict()` returns `OBBBoxes`; extract `.xywhr` (x, y, w, h, rotation) |
| Keypoint model | PyTorch `nn.Module` | Custom regression head on top of existing U-Net encoder |
| Affine crop | OpenCV | `cv2.getRotationMatrix2D()`, `cv2.warpAffine()` |

---

## Sources

- Live codebase analysis: `src/aquapose/` (all modules read directly)
- `engine/config.py`: frozen dataclass hierarchy and `load_config()` implementation
- `core/context.py`: `PipelineContext` field types and stage contract
- `core/midline/types.py`: `AnnotatedDetection` definition
- `reconstruction/midline.py`: `Midline2D` definition and extraction helpers
- `reconstruction/triangulation.py`: `N_SAMPLE_POINTS`, `MidlineSet`, `Midline3D` contract
- `core/detection/backends/__init__.py`: backend registry pattern
- `core/midline/backends/__init__.py`: backend registry pattern
- `segmentation/crop.py`: `CropRegion`, `compute_crop_region()`, `extract_crop()`
- `.planning/PROJECT.md`: v2.2 Backends milestone scope

---

*Architecture research for: AquaPose v2.2 Backends milestone*
*Researched: 2026-02-28*
