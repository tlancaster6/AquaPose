# AquaPose Pipeline — Stage Structure v1.0

## Pre-Pipeline: Input Materialization

Resolved before the pipeline loop starts. Not stages — these are input preparation.

**Frame Source**
- Loads and decodes video frames lazily via `FrameSource.batches()`
- Applies undistortion using calibration intrinsics
- Yields `FrameBatch` — all stages receive clean, undistorted frames

**Calibration**
- Loads intrinsics and extrinsics from AquaCal calibration file
- Produces structured `CalibrationData` (projection models, camera parameters)
- Available to all stages via `PipelineContext`

---

## Pipeline Stages

### Stage 1 — Detection
*Swappable backend: model-based (YOLO, DETR, etc.)*

- **In:** frames, detection config
- **Out:** `detections_per_frame` — `list[FrameDetections]` per camera per frame. Empty lists for cameras with no fish.

### Stage 2 — Midline
*Swappable backend: segment-then-extract / direct keypoint*

- **In:** detections, frames
- **Out:** `annotated_detections` — each detection enriched with its 15-point 2D midline + half-widths. Midlines travel with their parent detection from this point forward. Segment-then-extract runs segmentation model (U-Net, SAM, etc.) → skeletonize → BFS internally. Direct keypoint produces midlines in a single model call.

### Stage 3 — Cross-View Association

- **In:** annotated detections, projection models
- **Out:** `associated_bundles` — groups of annotated detections matched across cameras for each physical fish per frame, with camera assignments. Currently matches on bbox centroids; midline data is carried through and available for future association methods.

### Stage 4 — Tracking
*Swappable backend: Hungarian matching (default)*

- **In:** associated bundles, CarryForward (previous batch's track state)
- **Out:** `tracks_per_frame` — associated bundles promoted to persistent fish IDs via temporal matching with population constraints. Lifecycle states (probationary → confirmed → coasting → dead). Updates CarryForward for next batch.

### Stage 5 — Reconstruction
*Swappable backend: triangulation / curve optimizer*

- **In:** tracked bundles (annotated detections with persistent IDs), projection models
- **Out:** `midlines_3d` — per-frame `dict[fish_id, Spline3D]` (7-control-point 3D B-splines) + `dropped: dict[fish_id, DropReason]` for fish that failed reconstruction.

---

## Invariants

- Stages are pure computation — no file I/O, no event emission, no visualization
- Empty results are zero-length typed collections, not sentinels
- Preconditions validated cheaply via assert at top of `run()`
- Swappable backends resolved at pipeline construction time from config
- Backends share the same input/output contract for their stage slot

## Swappable Backends vs Configurable Models

A **backend** represents a fundamentally different approach to a stage's task — different algorithm, different execution pattern, or different intermediate representations. Backends are registered in a stage-level registry and resolved from config at pipeline construction time. Examples: segment-then-extract vs direct keypoint detection, triangulation vs curve optimization.

A **configurable model** is a choice within a backend — a different trained model, architecture, or parameter set that doesn't change the backend's structure or data flow. Model selection is handled via backend config, not by registering a new backend. Examples: YOLO vs DETR within the model-based detection backend, U-Net vs SAM within the segment-then-extract midline backend.

The test: if swapping it in changes the internal data flow or intermediate representations, it's a new backend. If it just changes what model gets loaded, it's config.
