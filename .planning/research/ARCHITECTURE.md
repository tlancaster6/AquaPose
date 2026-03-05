# Architecture Research

**Domain:** Performance optimization of multi-camera 3D pose estimation pipeline
**Researched:** 2026-03-04
**Confidence:** HIGH (based on direct codebase analysis)

## Existing Architecture Overview

The AquaPose pipeline has a strict 3-layer architecture with a fourth coordination layer above it:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ChunkOrchestrator                             │
│  Owns: chunk loop, ChunkHandoff, identity stitching, HDF5 output    │
├──────────────────────────────────────────────────────────────────────┤
│                         PosePipeline                                  │
│  Owns: stage ordering, EventBus, lifecycle events, stage timing      │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Detection  │→ │  Tracking   │→ │ Association  │→ │ Midline   │  │
│  │  Stage 1   │  │   Stage 2   │  │   Stage 3    │  │  Stage 4  │  │
│  └────────────┘  └─────────────┘  └──────────────┘  └─────┬─────┘  │
│                                                             ↓        │
│                                         ┌──────────────────────────┐ │
│                                         │   Reconstruction Stage 5 │ │
│                                         └──────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────┤
│                          Core Computation                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Backends: YOLOOBBBackend, SegmentationBackend,               │  │
│  │           PoseEstimationBackend, DltBackend                   │  │
│  │  Scoring: score_all_pairs(), score_tracklet_pair()            │  │
│  │  Types: Detection, Midline2D, Midline3D, Tracklet2D, etc.    │  │
│  └────────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│                    FrameSource / VideoFrameSource                     │
│  Owns: OpenCV video I/O, undistortion, chunk windowing              │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Key Interface |
|-----------|---------------|---------------|
| ChunkOrchestrator | Chunk loop, identity stitching, HDF5 flush | Creates PosePipeline per chunk |
| PosePipeline | Stage ordering, event emission, timing | `run(initial_context) -> PipelineContext` |
| DetectionStage | Read frames, run per-frame per-camera YOLO-OBB | `run(context) -> context` |
| TrackingStage | Per-camera OC-SORT across frames | `run(context, carry) -> (context, carry)` |
| AssociationStage | Cross-camera Leiden clustering via ray-ray scores | `run(context) -> context` |
| MidlineStage | Crop-extract + YOLO-seg/pose per detection | `run(context) -> context` |
| ReconstructionStage | DLT triangulation + B-spline per fish per frame | `run(context) -> context` |
| YOLOOBBBackend | Single-frame OBB detection | `detect(frame) -> list[Detection]` |
| SegmentationBackend | Single-frame multi-detection seg | `process_frame(frame_idx, frame_dets, frames, camera_ids) -> dict[str, list[AnnotatedDetection]]` |
| PoseEstimationBackend | Single-frame multi-detection pose | `process_frame(frame_idx, frame_dets, frames, camera_ids) -> dict[str, list[AnnotatedDetection]]` |
| DltBackend | Single-frame multi-fish DLT triangulation | `reconstruct_frame(frame_idx, midline_set) -> dict[int, Midline3D]` |
| VideoFrameSource | Sequential frame I/O, undistortion | `__iter__() -> (frame_idx, {cam_id: np.ndarray})` |
| ChunkFrameSource | Windowed view of VideoFrameSource | `__iter__()` delegates to `read_frame()` — random-access seek per frame, no prefetch |

## Four Optimization Targets: Integration Analysis

### Target 1: Batched YOLO Inference (~70% of wall time, GPU util ~30%)

**Where the work happens:**

Detection — `YOLOOBBBackend.detect(frame)` called once per camera per frame in `DetectionStage.run()`:

```python
for _frame_idx, frames in self._frame_source:
    for cam_id in camera_ids:
        frame_dets[cam_id] = self._detector.detect(frames[cam_id])  # one frame at a time
```

Midline — `SegmentationBackend._process_detection()` or `PoseEstimationBackend._process_detection()` called once per detection in a nested camera × detection loop:

```python
for cam_id in camera_ids:
    for det in cam_dets:
        ann = self._process_detection(det, frame, cam_id, frame_idx)  # one crop at a time
```

**Interface changes required:**

Detection batching lives inside `YOLOOBBBackend`. The backend's public interface must grow a `detect_batch(frames: list[np.ndarray]) -> list[list[Detection]]` method. The stage loop restructures to collect all camera frames first, then call the batch method. Ultralytics supports `model.predict(source=[img1, img2, ...])` natively — each `Results` object in the returned list corresponds to one input image.

Midline batching requires both backends to add a `process_batch()` method. The stage loop changes to collect all crops for all cameras and detections within a frame, run one `model.predict()`, then distribute results back to per-detection annotations.

**What changes:**
- `YOLOOBBBackend`: add `detect_batch(frames) -> list[list[Detection]]`
- `DetectionStage.run()`: restructure inner loop — collect all frames, call `detect_batch()`, distribute
- `SegmentationBackend` and `PoseEstimationBackend`: add `process_batch(crops) -> list[AnnotatedDetection | None]`
- `MidlineStage.run()`: restructure inner loop — collect all crops, call `process_batch()`, distribute

**What stays the same:** `Stage` protocol (`run(context) -> context`), all `PipelineContext` fields, crop extraction logic, post-processing (skeletonization, keypoint interpolation), output data structures.

**Batching dimensions:** Detection: up to 13 cameras × N frames simultaneously. Midline: ~9 fish × ~5 confirmed cameras = ~45 crops per frame, significantly reduced by association filtering.

**Isolation:** Changes stay inside `DetectionStage`, `YOLOOBBBackend`, `SegmentationBackend`, and `PoseEstimationBackend`. Nothing upstream or downstream changes.

### Target 2: Frame I/O Optimization (~12% of wall time)

**Where the work happens:**

`ChunkFrameSource.__iter__()` calls `self._source.read_frame(global_idx)` for every frame. `VideoFrameSource.read_frame()` issues `cap.set(CAP_PROP_POS_FRAMES, idx)` (seek) then `cap.read()` for every camera, sequentially. The seek is the bottleneck — `ChunkFrameSource` wraps `VideoFrameSource` and forces random-access seek even though chunk processing is always sequential.

```python
# ChunkFrameSource.__iter__ — seek-based, no prefetch
for local_idx in range(self.end_frame - self.start_frame):
    global_idx = self.start_frame + local_idx
    frames = self._source.read_frame(global_idx)  # seek + sequential-per-camera read
    yield local_idx, frames
```

**Interface changes required:**

The `FrameSource` protocol does not change. Detection and midline stages iterate with `for frame_idx, frames in self._frame_source:` which continues to work.

Option A (minimal): Add prefetch threading inside `VideoFrameSource.read_frame()` — a background thread pool reads all cameras in parallel for the current frame.

Option B (recommended): Add a `BatchFrameSource` new class implementing `FrameSource` that streams frames sequentially (no seek) with a background prefetch queue. It is injected at `build_stages()` instead of `ChunkFrameSource` when batched inference is active, so detection and midline see N frames available at once rather than one.

**What changes:** New class `BatchFrameSource` in `core/types/frame_source.py`, or internal threading added to `VideoFrameSource`.

**What stays the same:** `FrameSource` protocol, all stage code, `ChunkOrchestrator` (it already opens `VideoFrameSource` once for the run).

**Isolation:** Fully isolated to `core/types/frame_source.py`. No stage code changes if `BatchFrameSource` satisfies the `FrameSource` protocol.

### Target 3: Vectorized DLT Reconstruction (~9% of wall time)

**Where the work happens:**

`DltBackend._reconstruct_fish()` loops over body points (15 by default), calling `_triangulate_body_point()` for each:

```python
for i in range(n_body_points):
    result = self._triangulate_body_point(i, cam_midlines, water_z)
```

`_triangulate_body_point()` constructs per-camera tensors one at a time, calls `cast_ray()` per camera per point, then `triangulate_rays()`. That is 15 body points × ~5 cameras × 2 triangulation passes ≈ 150+ Python iterations per fish per frame. The spline residual loop afterward also iterates per camera per body point.

**Key observation:** `RefractiveProjectionModel.cast_ray()` and `.project()` already accept batched tensor inputs — the existing spline residual code calls `self._models[cid].project(spline_pts_3d)` where `spline_pts_3d` is shape `(n_body_points, 3)`. This means the vectorized call pattern is already used within `DltBackend`; extending it to the triangulation loop is a natural continuation.

**Interface changes required:** None. The public `reconstruct_frame(frame_idx, midline_set) -> dict[int, Midline3D]` interface is unchanged. The vectorization is entirely inside `_triangulate_body_point()` (replaced with `_triangulate_fish_vectorized()`):

```python
# Vectorized approach sketch
def _triangulate_fish_vectorized(self, cam_midlines, water_z):
    # Stack all body points for all cameras: {cam_id: (N, 2)} pixel tensors
    for cam_id in active_cams:
        px_batch = torch.from_numpy(cam_midlines[cam_id].points).float()  # (N, 2)
        origins[cam_id], dirs[cam_id] = self._models[cam_id].cast_ray(px_batch)  # (N, 3)
    # Stack across cameras: (C, N, 3)
    all_origins = torch.stack([origins[c] for c in active_cams])
    all_dirs    = torch.stack([dirs[c]    for c in active_cams])
    # Vectorized DLT over all N body points simultaneously -> (N, 3)
    pts_3d = triangulate_rays_batched(all_origins, all_dirs)
    # Vectorized reprojection residuals -> (C, N) for outlier masking
    ...
```

**What changes:** `DltBackend._triangulate_body_point()` replaced by `_triangulate_fish_vectorized()`. Potentially a new `triangulate_rays_batched()` function in `calibration/projection.py`.

**What stays the same:** `DltBackend.reconstruct_frame()`, `ReconstructionStage`, `PipelineContext`, all evaluation infrastructure.

**Isolation:** Fully inside `DltBackend` internals (and optionally `calibration/projection.py` for the batched triangulation primitive).

**Numerical equivalence:** The vectorized path must produce results within float tolerance of the scalar path. The reconstruction evaluator (`aquapose eval --stage reconstruction`) validates this automatically against cached data.

### Target 4: Vectorized Association Scoring (~5% of wall time)

**Where the work happens:**

`score_all_pairs()` has three nested loops — camera pairs → tracklet_a × tracklet_b → shared frames. The inner `score_tracklet_pair()` processes shared frames one at a time:

```python
for frame_idx_in_shared, frame in enumerate(shared_frames):
    pix_a = torch.tensor([[centroid_a[0], centroid_a[1]]], dtype=torch.float32)
    pix_b = torch.tensor([[centroid_b[0], centroid_b[1]]], dtype=torch.float32)
    origins_a, dirs_a = lut_a.cast_ray(pix_a)   # one pixel at a time
    origins_b, dirs_b = lut_b.cast_ray(pix_b)   # one pixel at a time
    dist, _ = ray_ray_closest_point(o_a, d_a, o_b, d_b)  # scalar numpy
```

**Interface changes required:** None. Function signatures stay identical. The internal implementation of `score_tracklet_pair()` is restructured:

```python
# Vectorized approach sketch
def score_tracklet_pair(ta, tb, forward_luts, config, ...):
    # Collect all shared-frame centroids at once
    pix_a = torch.tensor([ta.centroids[frames_a[f]] for f in shared_frames])  # (T, 2)
    pix_b = torch.tensor([tb.centroids[frames_b[f]] for f in shared_frames])  # (T, 2)
    # One batched cast_ray call per tracklet
    origins_a, dirs_a = lut_a.cast_ray(pix_a)  # (T, 3), (T, 3)
    origins_b, dirs_b = lut_b.cast_ray(pix_b)  # (T, 3), (T, 3)
    # Vectorized ray-ray distance computation
    dists = ray_ray_closest_point_batch(origins_a, dirs_a, origins_b, dirs_b)  # (T,)
    # Vectorized score computation (masking, summing)
    ...
```

A new `ray_ray_closest_point_batch(origins_a, dirs_a, origins_b, dirs_b) -> torch.Tensor` function is needed, implementing the same analytic formula but operating on `(T, 3)` inputs.

**What changes:** `score_tracklet_pair()` internals, new `ray_ray_closest_point_batch()` function in `scoring.py`.

**What stays the same:** `score_all_pairs()` signature, `AssociationStage`, `PipelineContext`, all downstream stages.

**Isolation:** Fully inside `core/association/scoring.py`.

## Data Flow: Baseline vs Optimized

### Baseline

```
ChunkFrameSource (seek-based, per-frame random access)
    ↓ one frame at a time
DetectionStage
    ↓ for each camera: detector.detect(single_frame)  [1 GPU call / camera / frame]
    ↓ → context.detections
TrackingStage (unchanged)
    ↓ → context.tracks_2d
AssociationStage
    ↓ score_all_pairs() → nested loops, per-frame scalar ray-ray
    ↓ → context.tracklet_groups
MidlineStage
    ↓ for each frame, each camera, each detection:
    ↓     extract crop → model.predict(single_crop) [1 GPU call / detection]
    ↓ → context.annotated_detections
ReconstructionStage
    ↓ for each fish: for each body point: _triangulate_body_point() scalar loops
    ↓ → context.midlines_3d
```

### Optimized

```
BatchFrameSource (sequential streaming, prefetch queue, parallel camera reads)
    ↓ batches of N frames at a time
DetectionStage (batched)
    ↓ detector.detect_batch([frame_cam1, ..., frame_cam13])
    ↓     → one model.predict(13 images) call per frame (or N×13 per batch)
    ↓ → context.detections (same structure)
TrackingStage (unchanged)
    ↓ → context.tracks_2d (unchanged)
AssociationStage (vectorized scoring)
    ↓ score_all_pairs() → batched cast_ray, vectorized ray-ray distances
    ↓ → context.tracklet_groups (same structure)
MidlineStage (batched)
    ↓ collect all crops for a frame (all cameras, all detections)
    ↓ backend.process_batch(all_crops) → one model.predict(~45 crops) per frame
    ↓ → context.annotated_detections (same structure)
ReconstructionStage (vectorized)
    ↓ for each fish: _triangulate_fish_vectorized() — all body points in one tensor pass
    ↓ → context.midlines_3d (same structure)
```

## Component Boundary Summary

### No Interface Changes — Internals Only

| Component | What changes internally | Public interface unchanged |
|-----------|------------------------|---------------------------|
| `DltBackend` | Body-point scalar loop → vectorized tensor pass | `reconstruct_frame(frame_idx, midline_set)` |
| `score_tracklet_pair()` | Frame loop → batched `cast_ray()` + vectorized distance | Same signature |
| `score_all_pairs()` | Possibly restructured outer pair loop | Same signature |

### New Methods Added to Existing Classes

| Class | New Method | Caller |
|-------|-----------|--------|
| `YOLOOBBBackend` | `detect_batch(frames: list[np.ndarray]) -> list[list[Detection]]` | `DetectionStage.run()` |
| `SegmentationBackend` | `process_batch(crops: list[np.ndarray], ...) -> list[AnnotatedDetection | None]` | `MidlineStage.run()` |
| `PoseEstimationBackend` | `process_batch(crops: list[np.ndarray], ...) -> list[AnnotatedDetection | None]` | `MidlineStage.run()` |

### Stage-Level Loop Restructure (same `run()` signature)

| Stage | What changes |
|-------|-------------|
| `DetectionStage.run()` | Inner loop collects frames first, then calls `detect_batch()` |
| `MidlineStage.run()` | Inner loop collects all crops per frame, calls `process_batch()`, distributes results |

### New Components

| Component | Type | File | Purpose |
|-----------|------|------|---------|
| `BatchFrameSource` | New class, implements `FrameSource` | `core/types/frame_source.py` | Sequential streaming + prefetch; no seek overhead |
| `ray_ray_closest_point_batch()` | New function | `core/association/scoring.py` | Vectorized ray-ray distance for `(T, 3)` tensor inputs |
| `triangulate_rays_batched()` (optional) | New function | `calibration/projection.py` | Batched DLT triangulation over N body points simultaneously |

### Unchanged Components

| Component | Why unchanged |
|-----------|--------------|
| `PipelineContext` field types | Same field names, same data structures — only how values are computed changes |
| `Stage` protocol | `run(context) -> context` unchanged |
| `TrackingStage` | OC-SORT is per-camera sequential by design; not a bottleneck |
| `AssociationStage` | Calls `score_all_pairs()` which is optimized internally |
| `ReconstructionStage` | Calls `backend.reconstruct_frame()` which is optimized internally |
| `ChunkOrchestrator` | Frame source swap is transparent via `FrameSource` protocol |
| `PosePipeline` | No awareness of batching strategy |
| Observers, EventBus | Unchanged |
| Per-chunk pickle caching | Same `PipelineContext` structure — caching is unaffected |
| `aquapose eval`, `aquapose tune`, `aquapose viz` | Read cached context — unaffected by how it was computed |

## Suggested Build Order

The four optimization targets are nearly independent. Dependencies are:

- Association and reconstruction vectorization have no dependencies on each other or on I/O / batching.
- Frame I/O optimization enables maximum gain from batched inference (prefetched frames available for batch collection), but batched inference yields gains even without it.
- Batched inference is the highest complexity and highest impact, so it is built last.

### Phase Order

**Phase 1: Association vectorization** — 5% gain, lowest complexity, no dependencies

`score_tracklet_pair()` frame loop → batched `cast_ray()` + new `ray_ray_closest_point_batch()`. Verify output is numerically equivalent to scalar path. Good warm-up: establishes the batched `cast_ray()` call pattern used again in reconstruction.

**Phase 2: Reconstruction vectorization** — 9% gain, medium complexity, no dependencies

`_triangulate_body_point()` scalar loop → `_triangulate_fish_vectorized()` tensor pass. The existing code already uses `cast_ray(batch)` in the spline residual step, confirming `RefractiveProjectionModel` supports this. Use the reconstruction evaluator (`aquapose eval --stage reconstruction`) on cached data to validate numerical equivalence before and after.

**Phase 3: Frame I/O optimization** — 12% gain, low-to-medium complexity, should precede Phase 4

`BatchFrameSource` replaces `ChunkFrameSource` in `build_stages()`. Background prefetch thread eliminates seek overhead. Parallel camera reads within each frame address the sequential-per-camera bottleneck. Must be built before Phase 4 so detection and midline stages can collect full frame batches rather than single frames.

**Phase 4: Batched YOLO inference** — ~70% of wall time, highest complexity, benefits from Phase 3

`YOLOOBBBackend.detect_batch()` + `DetectionStage` loop restructure, then `SegmentationBackend.process_batch()` / `PoseEstimationBackend.process_batch()` + `MidlineStage` loop restructure.

### Dependency Graph

```
Phase 1: Association vectorization    (no dependencies — start immediately)
    │
    ↓ (patterns established)
Phase 2: Reconstruction vectorization (no dependencies — can run in parallel with Phase 1)
    │
Phase 3: Frame I/O (BatchFrameSource) (no dependencies — enables Phase 4)
    │
    ↓
Phase 4: Batched YOLO inference       (benefits from Phase 3 prefetch)
```

Phases 1 and 2 can be done in either order or in parallel. Phase 3 should precede Phase 4 for maximum benefit, but Phase 4 produces gains even without Phase 3 (batching across cameras within a single frame is already a significant improvement over one-at-a-time inference).

## Architectural Patterns

### Pattern 1: Internal Vectorization

**What:** Replace Python loops over small independent collections (body points, shared frames) with vectorized NumPy/PyTorch tensor operations. Public interface unchanged.

**When to use:** When the loop body is pure math operating on independent elements (no per-element Python control flow mid-loop), and all inputs can be pre-collected into arrays. Both body-point triangulation and frame-level ray-ray scoring satisfy this.

**Trade-offs:** Outlier rejection (filter cameras below residual threshold) requires a mask-based approach rather than an early `continue`. For DLT: compute initial residuals vectorized, apply mask, re-triangulate inliers — this is a two-pass design but still far fewer Python iterations than the scalar loop.

### Pattern 2: Backend Batch Extension

**What:** Add a batch method alongside the existing single-item method. The stage loop is restructured into two passes — collect all inputs, then call the batch method once, then distribute results back to the original output structure.

**When to use:** When the expensive operation (GPU inference) supports batched input natively (Ultralytics `model.predict(list_of_images)`), and surrounding logic (crop extraction, post-processing) is lightweight CPU work.

**Trade-offs:** Stage loop structure changes significantly. The existing single-item methods can be kept as thin wrappers over the batch methods for backward compatibility in tests. The output structure (per-frame per-camera lists) does not change, so downstream stages see no difference.

### Pattern 3: Protocol-Transparent I/O Replacement

**What:** New class implementing the same `FrameSource` protocol, with different internal I/O strategy. Drop in at `build_stages()`.

**When to use:** When the bottleneck is in I/O and the consumer only uses the protocol interface. The `FrameSource` protocol provides clean separation — detection and midline stages do not know or care whether frames come from `ChunkFrameSource` or `BatchFrameSource`.

**Trade-offs:** Prefetch adds memory pressure. At 1600×1200×3 bytes per frame × 13 cameras, buffering N=5 frames is ~375 MB. Buffer size is a tunable parameter; default should be modest (3-5 frames). Background thread errors must be propagated to the main thread — use a sentinel value or exception re-raise pattern.

## Anti-Patterns

### Anti-Pattern 1: Changing PipelineContext Field Types

**What people do:** Change `detections` to a batched tensor format to avoid per-stage re-assembly overhead.

**Why it's wrong:** All downstream stages, the diagnostic observer, cache serialization, evaluators, and tuning infrastructure depend on the existing field structure. GPU tensors also cannot be pickled across process boundaries.

**Do this instead:** Keep context field types unchanged. Batch processing is an internal implementation detail of each stage. The existing `.cpu().numpy()` pattern at backend boundaries is correct.

### Anti-Pattern 2: Cross-Stage Batching (Detection + Midline in One Pass)

**What people do:** Combine detection and midline into a single GPU pass to avoid re-reading frames.

**Why it's wrong:** Detection runs across all frames first; tracking and association run next; midline runs with filtered detections only. You cannot skip the intermediate stages. Stage result caching depends on stage independence.

**Do this instead:** Batch within each stage independently. Detection batches 13-camera frames per invocation; midline batches all crops for a frame per invocation. Frame re-reading is avoided by `BatchFrameSource` buffering.

### Anti-Pattern 3: Vectorization That Changes Numerical Output

**What people do:** Vectorize DLT but change the numerical result due to floating-point reduction order (e.g., different summation order in weighted DLT A-matrix construction).

**Why it's wrong:** The reconstruction evaluator validates against cached ground truth. Numerical non-equivalence fails the regression tests and invalidates comparisons with cached tuning sweep results.

**Do this instead:** Ensure vectorized path produces results within 1e-5 relative tolerance vs the scalar path. Run the reconstruction evaluator before and after the change. The evaluation harness is specifically designed for this validation workflow.

### Anti-Pattern 4: Prefetch Buffer Too Large

**What people do:** Prefetch all chunk frames upfront (e.g., 1000 frames × 13 cameras × 1600×1200×3 bytes ≈ 75 GB) to minimize I/O latency.

**Why it's wrong:** Exceeds GPU VRAM and system RAM for reasonable chunk sizes. The frame buffer is a pipeline stage, not a cache — it should buffer just enough frames to keep GPU busy.

**Do this instead:** Tune prefetch buffer depth as a configuration parameter. Default 3-5 frames provides sufficient overlap between I/O and GPU compute without excessive memory pressure. For batch inference of N frames at once, buffer depth should be at least N+1.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Optimization Impact |
|----------|---------------|---------------------|
| `DetectionStage` ↔ `YOLOOBBBackend` | `detect(frame)` | New `detect_batch()` added; stage loop restructured |
| `MidlineStage` ↔ `SegmentationBackend`/`PoseEstimationBackend` | `process_frame()` | New `process_batch()` added; stage loop restructured |
| `ReconstructionStage` ↔ `DltBackend` | `reconstruct_frame()` | Unchanged; DltBackend internals vectorized |
| `AssociationStage` ↔ `score_all_pairs()` | Function call | Unchanged; scoring internals vectorized |
| `ChunkOrchestrator` ↔ `VideoFrameSource` | `FrameSource` protocol | `BatchFrameSource` replaces `ChunkFrameSource` in `build_stages()` |
| Stage ↔ `PipelineContext` | Direct field read/write | Fields unchanged |
| `DltBackend` ↔ `RefractiveProjectionModel` | `.cast_ray()` / `.project()` | Already supports batched tensors — vectorization exploits existing capability |

### Evaluation Infrastructure Compatibility

| Infrastructure | Impact |
|----------------|--------|
| Per-chunk pickle caching | None — same `PipelineContext` structure |
| `aquapose eval` | None — reads cached context, evaluators unchanged |
| `aquapose tune` | None — evaluators unchanged; optimizations only reduce pipeline runtime |
| `aquapose viz` | None — reads cached context |
| Unit tests | Backend tests need coverage of batch methods; stage tests need coverage of restructured loops |

## Recommended Project Structure Changes

No new modules needed for association and reconstruction vectorization — changes are within existing files.

New additions for inference batching and I/O:

```
src/aquapose/core/types/frame_source.py
    + BatchFrameSource class (or inline in existing file)

src/aquapose/core/detection/backends/yolo_obb.py
    + detect_batch() method

src/aquapose/core/midline/backends/segmentation.py
    + process_batch() method

src/aquapose/core/midline/backends/pose_estimation.py
    + process_batch() method

src/aquapose/core/association/scoring.py
    + ray_ray_closest_point_batch() function (local helper)

src/aquapose/calibration/projection.py  (optional)
    + triangulate_rays_batched() function
```

No changes to `engine/`, `evaluation/`, or `io/`.

## Sources

- Direct analysis of `src/aquapose/core/detection/stage.py` — frame-by-frame detection loop structure
- Direct analysis of `src/aquapose/core/detection/backends/yolo_obb.py` — single-frame `detect()` pattern
- Direct analysis of `src/aquapose/core/midline/stage.py` — per-detection inference loop
- Direct analysis of `src/aquapose/core/midline/backends/segmentation.py` and `pose_estimation.py` — per-crop `model.predict()` pattern
- Direct analysis of `src/aquapose/core/reconstruction/backends/dlt.py` — body-point scalar loop, existing batched `cast_ray()` usage in spline residual step confirms `RefractiveProjectionModel` supports batched input
- Direct analysis of `src/aquapose/core/association/scoring.py` — per-frame scalar ray-ray scoring loop
- Direct analysis of `src/aquapose/core/types/frame_source.py` — `ChunkFrameSource` seek-based random-access pattern
- Direct analysis of `src/aquapose/core/context.py` — `PipelineContext` field types and `Stage` protocol
- Direct analysis of `src/aquapose/engine/pipeline.py` — `build_stages()` factory, stage ordering

---
*Architecture research for: AquaPose v3.4 Performance Optimization*
*Researched: 2026-03-04*
