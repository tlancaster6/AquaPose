# Project Research Summary

**Project:** AquaPose v3.4 Performance Optimization
**Domain:** Performance optimization of a 12-camera synchronous CV pipeline (batching, vectorization, async I/O)
**Researched:** 2026-03-05
**Confidence:** HIGH

## Executive Summary

AquaPose v3.4 is a focused performance optimization milestone targeting a synchronous multi-camera YOLO inference pipeline that currently runs far below GPU utilization capacity. The core problem is architectural: every GPU-accelerated operation is invoked one image or one data element at a time, leaving the GPU idle roughly 70% of the time. The recommended approach is four independent optimizations applied in a dependency-ordered sequence: vectorize association scoring (5% of wall time, lowest risk), vectorize DLT triangulation (9% of wall time, medium risk), replace the seek-based frame source with a streaming prefetch source (12% of wall time, design-critical for correctness), and finally introduce batched YOLO inference for both detection and midline stages (approximately 70% of wall time, highest impact and complexity). All four changes are correctness-neutral — they must produce numerically equivalent output to the baseline, verified against cached ground truth using the existing `aquapose eval` harness.

The critical insight from combined research is that each optimization is isolated to a specific component boundary and leaves all downstream interfaces unchanged. `PipelineContext` field types, `Stage` protocol signatures, the evaluation infrastructure, and the HDF5 output format are entirely untouched. This isolation is the feature that makes the four phases independently executable and low-risk. The architectural anti-patterns to avoid are well-documented: do not change context field types, do not attempt cross-stage batching, do not share `VideoCapture` objects across threads, and do not store Ultralytics `Results` objects rather than immediately extracting CPU numpy arrays.

The primary risks are correctness risks, not performance risks. Batch result-to-input index mapping errors silently corrupt all downstream reconstruction. OpenCV `VideoCapture` thread safety issues produce non-deterministic frame identity corruption. Vectorized DLT can produce numerically different inlier sets due to floating-point reordering and TF32 precision differences on Ampere+ GPUs. Each of these has a known mitigation pattern: lockstep batch index construction, single-threaded sequential frame reader feeding a queue, and numerical equivalence tests on real cached data (not synthetic). The existing `aquapose eval` stage-by-stage evaluation harness is the primary validation tool for all four phases.

## Key Findings

### Recommended Stack

No new dependencies are required for v3.4. All four optimizations use libraries already present: PyTorch batched tensor operations (`torch.linalg.svd` with batch dimension, confirmed in official docs), Ultralytics batch predict API (`model.predict(list_of_numpy_arrays)` — confirmed in official docs), NumPy broadcasting for vectorized ray-ray distance computation, and Python's `threading.Thread` + `queue.Queue` for producer-consumer I/O overlap. PyTorch can be upgraded freely (the PyTorch3D version-pinning constraint was removed in the v3.0 pivot to direct triangulation).

**Core technologies:**
- **PyTorch (any recent version):** GPU inference and batched linear algebra — `torch.linalg.svd` confirmed to support batch dimensions natively; TF32 default on Ampere+ GPUs requires explicit management
- **Ultralytics >= 8.1:** Batch predict API accepts list of numpy arrays; result ordering guaranteed to match input order
- **OpenCV / VideoCapture:** Frame I/O — NOT thread-safe; the prefetch design must use a single-threaded reader feeding a queue; `CAP_PROP_POS_FRAMES` seek inaccurate on H.264 content
- **NumPy:** Vectorized ray-ray distance computation for association scoring
- **Python `threading` + `queue`:** Producer-consumer pattern for frame prefetch overlap with GPU inference

### Expected Features

**Must have (table stakes — all four required for v3.4 milestone success):**
- **Batched YOLO detection inference** — replace 12 single-image `model.predict()` calls per frame with one `detect_batch([frame_cam1, ..., frame_cam12])` call; expected 2-5x detection throughput
- **Batched YOLO midline inference** — collect all crops across all cameras for a frame, call `model.predict(crops_list)` once, redistribute keypoints by (cam_id, det_idx); expected 3-8x midline throughput
- **Vectorized DLT triangulation** — replace 15-iteration per-body-point loop with batched `torch.linalg.svd` over all body points simultaneously; expected 5-15x reconstruction throughput
- **Frame I/O overlap with GPU inference** — `BatchFrameSource` sequential streaming with prefetch queue eliminates seek overhead and GPU idle time between frames; expected to eliminate ~12% wall-time gap

**Should have (add after v3.4 validation):**
- **Vectorized pairwise association scoring** — vectorize per-frame loop in `score_tracklet_pair()` using batched numpy ops; only ~5% of wall time, add if profiling post-v3.4 warrants it

**Defer (v3.5+):**
- TensorRT / ONNX export — high maintenance cost, batching alone likely delivers comparable gains
- Multiprocessing pipeline parallelism — requires architectural restructuring, out of scope
- GPU-accelerated video decode via decord/PyAV — add only after profiling confirms decode is still dominant post-threading
- Parallel per-camera decode via `ProcessPoolExecutor` — IPC overhead likely negative ROI

### Architecture Approach

The four optimizations fit into the existing 4-layer architecture (ChunkOrchestrator → PosePipeline → Stage implementations → FrameSource) without changing any public interfaces. The dominant pattern is "internal vectorization" — public method signatures remain identical, only the implementation changes. The exceptions are `YOLOOBBBackend` and the midline backends, which receive new `detect_batch()` and `process_batch()` methods respectively (the stage loops call these new methods instead of the single-item methods). One new class, `BatchFrameSource`, replaces `ChunkFrameSource` at the `build_stages()` factory injection point — it satisfies the existing `FrameSource` protocol so all stage code is unaffected.

**Major components and changes:**
1. **`YOLOOBBBackend`** — add `detect_batch(frames: list[np.ndarray]) -> list[list[Detection]]`; `DetectionStage.run()` restructured to collect frames then call batch method
2. **`SegmentationBackend` / `PoseEstimationBackend`** — add `process_batch(crops: list[np.ndarray]) -> list[AnnotatedDetection | None]`; `MidlineStage.run()` restructured to collect crops then call batch method
3. **`DltBackend`** — `_triangulate_body_point()` scalar loop replaced by `_triangulate_fish_vectorized()` tensor pass; `reconstruct_frame()` signature unchanged
4. **`BatchFrameSource`** (new class in `core/types/frame_source.py`) — sequential streaming + background prefetch queue; replaces `ChunkFrameSource` in `build_stages()`
5. **`score_tracklet_pair()` internals** — frame loop replaced with batched `cast_ray()` + `ray_ray_closest_point_batch()`; function signature unchanged
6. **`triangulate_rays_batched()`** (optional new function in `calibration/projection.py`) — batched DLT primitive for the vectorized reconstruction path

**What does not change:** `PipelineContext` field types, `Stage` protocol, `TrackingStage`, `AssociationStage` (calls `score_all_pairs()` optimized internally), `ReconstructionStage` (calls `backend.reconstruct_frame()` optimized internally), `ChunkOrchestrator`, `PosePipeline`, per-chunk pickle caching, `aquapose eval`, `aquapose tune`, `aquapose viz`.

### Critical Pitfalls

1. **Batch predict result-to-input mapping errors (P1)** — building the image batch list and the identity index list out of sync silently corrupts all downstream reconstruction. Mitigation: build `batch_index: list[tuple[str, int]]` in lockstep with the `images` list; assert `len(results) == len(batch_index)` immediately after every predict call; unit test with deliberate camera-skip to verify index round-trip.

2. **OpenCV VideoCapture thread safety + seek inaccuracy (P3 + P4)** — `VideoCapture` objects are not thread-safe for concurrent access, and `CAP_PROP_POS_FRAMES` on H.264 video seeks to the nearest I-frame, not the exact requested frame. Both issues produce non-deterministic frame identity corruption. Mitigation: one dedicated reader thread owns all captures; sequential reads (no seeking) feed frames into a `queue.Queue`; inference thread only dequeues.

3. **GPU tensor leak from storing Ultralytics Results (P8)** — `Results` objects hold references to live CUDA tensors; storing a list of Results before processing them accumulates GPU memory across batches until OOM. Mitigation: immediately extract to CPU numpy (`Detection` objects) for each result after predict; never store `Results` objects in a list.

4. **Vectorized DLT numerical drift and TF32 (P5 + P6)** — batched SVD may produce different floating-point results than the per-point loop due to reduction order changes and TF32 precision (10 mantissa bits vs 23 for float32 on Ampere+ GPUs), which can flip outlier rejection decisions for body points near the residual threshold. Mitigation: test vectorized path against scalar path on real YH chunk cache data; check that inlier camera sets are identical (not just final 3D positions); document and preserve the TF32 state used when `outlier_threshold=10.0` was calibrated.

5. **CUDA OOM from over-batching (P2)** — 12 cameras at 1600x1200 is ~70 MB per frame batch; dense fish scenes can trigger OOM mid-chunk, aborting the entire chunk. Mitigation: make batch sizes configurable (`detection_batch_frames`, `midline_batch_crops`); wrap `model.predict()` in a try/except that catches `OutOfMemoryError`, halves batch, and retries; never hardcode batch size.

## Implications for Roadmap

Based on the architecture research's explicit build-order recommendation, combined with pitfall severity and feature dependencies, the following phase structure is strongly recommended.

### Phase 1: Vectorized Association Scoring

**Rationale:** Lowest complexity, no dependencies on other phases, establishes the batched `cast_ray()` call pattern that reconstruction vectorization also uses. Good warm-up phase: only 5% of wall time but risk is low and the vectorized ray-ray distance primitive (`ray_ray_closest_point_batch`) is a clean standalone deliverable. Early termination semantics must be replicated exactly (P7 pitfall).
**Delivers:** `score_tracklet_pair()` inner loop replaced with batched numpy ops; new `ray_ray_closest_point_batch()` function in `scoring.py`; score dict equivalence test on real YH chunk cache.
**Addresses:** FEATURES.md vectorized association (differentiator, P2 priority)
**Avoids:** P7 (early termination semantics change), P5 (numerical drift)

### Phase 2: Vectorized DLT Reconstruction

**Rationale:** No dependencies on I/O or inference batching phases; moderate complexity. The existing `RefractiveProjectionModel.cast_ray()` already accepts batched tensor inputs (confirmed by the spline residual step already using batched calls), making vectorization a natural extension. Architecture research confirms `reconstruct_frame()` public interface is unchanged.
**Delivers:** `_triangulate_body_point()` scalar loop replaced by `_triangulate_fish_vectorized()`; optional `triangulate_rays_batched()` primitive in `calibration/projection.py`; numerical equivalence test on real YH chunk cache showing inlier sets identical and 3D points within 1e-4 m.
**Addresses:** FEATURES.md vectorized DLT (P1 table stakes, 5-15x reconstruction throughput)
**Avoids:** P5 (numerical drift from batched SVD), P6 (TF32 precision baseline documentation)

### Phase 3: Frame I/O Optimization (BatchFrameSource)

**Rationale:** Should precede inference batching because prefetched frames available as a full batch enable maximum gain from Phase 4. The I/O path is correctness-critical: OpenCV thread safety and seek inaccuracy pitfalls (P3, P4) mean the design must be settled before any code is written. Sequential-read-into-queue is the only safe design.
**Delivers:** `BatchFrameSource` class implementing `FrameSource` protocol with background decode thread and bounded queue; replaces `ChunkFrameSource` in `build_stages()`; eliminates seek overhead and GPU idle gaps; memory-bounded prefetch (3-5 frame buffer depth, configurable).
**Addresses:** FEATURES.md frame I/O overlap (P1 table stakes, ~12% wall-time elimination)
**Avoids:** P3 (VideoCapture thread safety), P4 (CAP_PROP_POS_FRAMES seek inaccuracy)

### Phase 4: Batched YOLO Inference (Detection + Midline)

**Rationale:** Highest impact (~70% of wall time) and highest complexity. Benefits from Phase 3 prefetch (frames available as full-batch input), but delivers meaningful gains even without it (batching across 12 cameras for one frame is already a major improvement over one-at-a-time). Detection batching should be implemented and validated before midline batching because detection is simpler (fixed input count = n_cameras per frame; all full frames). Midline batching is more complex (variable crop count, per-detection affine transforms, (cam_id, det_idx) index reconstruction).
**Delivers:** `YOLOOBBBackend.detect_batch()`; restructured `DetectionStage.run()`; `SegmentationBackend.process_batch()` / `PoseEstimationBackend.process_batch()`; restructured `MidlineStage.run()`; configurable `detection_batch_frames` and `midline_batch_crops` config fields; OOM recovery via try/except batch-size halving.
**Addresses:** FEATURES.md batched YOLO detection (P1) and batched YOLO midline (P1) — together the dominant bottleneck
**Avoids:** P1 (batch index mapping), P2 (CUDA OOM), P8 (GPU tensor leak)

### Phase Ordering Rationale

- Phases 1 and 2 are independent and could run in parallel but are sequenced to keep each phase focused and its correctness tests clean.
- Phase 3 precedes Phase 4 because the prefetch source design directly enables the "collect full frame batch" loop structure in `DetectionStage`. Phase 4 delivers gains even without Phase 3, but Phase 3 maximizes those gains.
- The sequence from lowest-risk to highest-risk matches the sequence from smallest impact to largest: each phase is a validated foundation for the next.
- The correctness validation harness (`aquapose eval`) is available for every phase; no phase should be declared complete without a passing stage-level evaluation run on real YH chunk data.

### Research Flags

Phases requiring careful in-codebase verification before coding (not deeper external research — the patterns are well-established):
- **Phase 3 (BatchFrameSource):** Confirm that `ChunkFrameSource`'s `read_frame(global_idx)` seek pattern is the sole I/O path exercised during a chunk run. Verify that `VideoFrameSource` has no internal position-tracking state that would conflict with sequential background-thread reading.
- **Phase 4 (Batched YOLO):** Confirm Ultralytics result ordering guarantee on the exact pinned version in `pyproject.toml`. Verify batch predict behavior when all input images are the same resolution (no internal padding asymmetry across cameras of same size).

Phases with standard, well-documented patterns (skip research-phase, implement directly):
- **Phase 1 (Association vectorization):** NumPy broadcasting for ray-ray distance is a standard geometric computation. The only non-standard requirement is replicating early-termination semantics — document this as an explicit design constraint in the implementation plan.
- **Phase 2 (DLT vectorization):** Batched SVD via `torch.linalg.svd` is documented and confirmed. Two-pass outlier rejection with masking is a standard pattern; the existing code already uses batched `cast_ray()` confirming the infrastructure supports it.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | No new dependencies; all libraries verified against official docs. Ultralytics batch predict API confirmed. PyTorch batched linalg confirmed. |
| Features | HIGH | Codebase read directly; profiling targets from PROJECT.md; Ultralytics batch API verified against docs. All four table-stakes features have confirmed implementation paths. |
| Architecture | HIGH | Based on direct codebase analysis of all affected components. All interface boundaries explicitly mapped. Isolation of changes verified at each component boundary. |
| Pitfalls | HIGH | OpenCV thread safety verified against multiple GitHub issues (4890, 9053, 20227, 24229). TF32 behavior verified against PyTorch official docs. GPU tensor leak pattern verified against Ultralytics issue history. |

**Overall confidence:** HIGH

### Gaps to Address

- **TF32 baseline audit:** The hardware used to generate the original `outlier_threshold=10.0` tuning (from the YH grid search runs) has not been identified. Whether TF32 was enabled during that calibration determines the correct default for the vectorized path. Resolve by checking the machine configuration in the YH run metadata before Phase 2 coding begins.
- **Dense-scene OOM boundary:** The CUDA OOM threshold for the 12-camera x 1600x1200 setup is not profiled. The conservative default for `detection_batch_frames` should be 1 (current behavior). Establish the safe working batch size empirically at the start of Phase 4 on real hardware.
- **YH AVI codec:** Whether the YH rig records H.264 or MJPEG (all-keyframes) determines whether the `CAP_PROP_POS_FRAMES` seek inaccuracy pitfall (P4) applies to the existing recordings. Verify codec before finalizing Phase 3 design — thread safety is a concern regardless, but seek accuracy is codec-dependent.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `YOLOOBBBackend`, `DetectionStage`, `MidlineStage`, `PoseEstimationBackend`, `DltBackend`, `score_all_pairs`, `VideoFrameSource`, `ChunkFrameSource`, `context.py`, `pipeline.py`
- [Ultralytics Predict Docs](https://docs.ultralytics.com/modes/predict/) — batch inference API, result ordering guarantee
- [torch.linalg.svd PyTorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html) — batched SVD support confirmed
- [torch.linalg.lstsq PyTorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) — batched least-squares, CUDA driver limitation noted
- [PyTorch Numerical Accuracy Notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — TF32 precision, batched matmul non-determinism
- `/home/tlancaster6/Projects/AquaPose/.planning/PROJECT.md` — v3.4 milestone definition and profiling targets

### Secondary (MEDIUM confidence)
- [Ultralytics YOLO11 Batch Inference Blog](https://www.ultralytics.com/blog/using-ultralytics-yolo11-to-run-batch-inferences) — batch arg behavior, default size=1
- [OpenCV multithreading with VideoCapture](https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/) — threading pattern, GIL behavior for `cap.read()`
- [YOLOv8 Batch Inference Speed Analysis](https://dev-kit.io/blog/machine-learning/yolov8-batch-inference-speed-and-efficiency) — throughput scaling characteristics
- [NumPy pairwise vectorization](https://towardsdatascience.com/how-to-vectorize-pairwise-dis-similarity-metrics-5d522715fb4e/) — broadcasting patterns for distance matrices
- [PyTorch CUDA Memory Documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — `memory_summary`, cache behavior

### Tertiary (verified issue reports)
- [OpenCV Issue #4890](https://github.com/opencv/opencv/issues/4890) — CAP_PROP_POS_FRAMES seek inaccuracy (open since 2015)
- [OpenCV Issue #9053](https://github.com/opencv/opencv/issues/9053) — frame seek not exact with ffmpeg backend
- [OpenCV Issue #20227](https://github.com/opencv/opencv/issues/20227) — set+read differs from sequential read
- [OpenCV Issue #24229](https://github.com/opencv/opencv/issues/24229) — VideoCapture thread lock absence
- [Ultralytics Issue #4057](https://github.com/ultralytics/ultralytics/issues/4057) — CUDA OOM during inference

---
*Research completed: 2026-03-05*
*Ready for roadmap: yes*
