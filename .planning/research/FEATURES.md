# Feature Research

**Domain:** Performance optimization of a multi-camera YOLO inference pipeline (AquaPose v3.4)
**Researched:** 2026-03-05
**Confidence:** HIGH — codebase was read directly; Ultralytics batch inference API was verified against official docs; PyTorch batched linalg is documented; threading patterns are well-established.

---

## Context: What Already Exists and Must Be Preserved

| Existing Component | What It Does | Constraint |
|-------------------|--------------|------------|
| `DetectionStage.run()` | Iterates frames, calls `detector.detect(frame)` per camera per frame | Must preserve Result type: `list[dict[str, list[Detection]]]` |
| `YOLOOBBBackend.detect(frame)` | Calls `model.predict(frame, ...)` for a single image | Hot path — currently 1 image per GPU call |
| `MidlineStage.run()` | Iterates frames, calls `backend.process_frame(...)` per frame | Must preserve Result type: `list[dict[str, list[AnnotatedDetection]]]` |
| `PoseEstimationBackend._process_detection()` | Extracts crop, calls `model.predict(crop, ...)` for a single crop | Hot path — currently 1 crop per GPU call |
| `DltBackend._reconstruct_fish()` | Loops over `n_body_points` (15), triangulating one point at a time | Per-point Python loop calling `_triangulate_body_point()` per point |
| `DltBackend._triangulate_body_point()` | Casts rays, stacks tensors, calls `triangulate_rays()` / `weighted_triangulate_rays()` | Called 15x per fish per frame — loop overhead adds up |
| `score_tracklet_pair()` | Loops over shared frames; per-frame numpy ops; single pair at a time | Python loop inside; called O(tracklets^2) times |
| `score_all_pairs()` | Double loop over camera pairs, tracklet-A × tracklet-B | Outer Python loop; inner calls `score_tracklet_pair()` |
| `VideoFrameSource.__iter__()` | Sequential `cap.read()` per camera per frame; undistorts synchronously | CPU-bound decode blocks main thread; no overlap with GPU |
| `ChunkFrameSource.__iter__()` | Calls `source.read_frame(global_idx)` which seeks and reads each camera | Seek-per-frame is expensive; no prefetching |

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features a researcher expects from a "performance optimization" milestone. Missing these means the milestone has not delivered on its stated goal.

| Feature | Why Expected | Complexity | Expected Speedup | Notes |
|---------|--------------|------------|-----------------|-------|
| Batched YOLO detection inference | 12-camera rig calling predict() 12× per frame is the obvious GPU bottleneck; batch inference is the canonical first fix for unbatched YOLO pipelines | MEDIUM | 2–5× detection throughput (GPU util goes from ~30% to ~80%+) | Ultralytics `model.predict([img1, img2, ...])` accepts a list of numpy arrays — confirmed in official docs. Batch of 12 images per frame call replaces 12 single-image calls. DetectionStage collects all 12 camera frames, calls predict once, redistributes Results list by camera index. Breaking change to `YOLOOBBBackend.detect()` signature — needs `detect_batch(frames)` variant or signature change to accept list. |
| Batched YOLO midline inference (crop batching) | Same root cause as detection: each fish crop is a separate GPU call. A frame with 9 fish visible across 12 cameras generates ~108 predict() calls — all unbatched | HIGH | 3–8× midline throughput | Crops from all cameras for one frame (or multiple frames) are stacked into a single `model.predict([crop1, crop2, ...])` call. Complexity is HIGHER than detection batch because crops are variable-count (fish may be missing in some cameras), and the Results list must be redistributed back to (cam_id, detection_index) pairs. Affine crop extraction is still per-detection (CPU) — only the GPU predict() call is batched. |
| Vectorized DLT triangulation across all body points | Currently 15 sequential Python-loop iterations per fish per frame, each constructing and solving a small linear system. With 9 fish and ~12 cameras, this is 1,620+ small solve calls per frame. Vectorized solves over all body points at once is the standard fix. | MEDIUM | 5–15× DLT reconstruction (est. 9% of wall time → <1%) | The DLT system per body point is `A @ x = 0` with A of shape (2*n_cams, 4). Stacking all 15 body points gives a batched A of shape (15, 2*n_cams, 4). `torch.linalg.svd` and `torch.linalg.lstsq` both support batched inputs natively (batch dim = first dim). Outlier rejection still requires per-point residual computation but can also be vectorized with batched `project()`. The two-pass structure (triangulate all → reject → re-triangulate inliers) complicates batching of the second pass because inlier sets differ per point — this is manageable with masking. |
| Frame I/O overlap with GPU inference | VideoFrameSource decodes synchronously on the main thread, blocking GPU between frames. This is the standard producer-consumer problem. A prefetch thread reading the next frame batch while GPU processes the current one eliminates this dead time. | MEDIUM | Eliminate ~12% I/O wall-time overhead; actual net speedup depends on how much GPU was idle waiting for frames | Standard pattern: background thread fills a `queue.Queue(maxsize=N)` with pre-decoded frame dicts; main inference thread consumes from the queue. OpenCV `cap.read()` releases the GIL so the background thread genuinely runs concurrently. Constraint: GPU inference must happen on main thread (ultralytics requirement confirmed). The prefetch thread only does decode + undistort (CPU work). |

### Differentiators (Nice to Have, Real Value)

Features that go beyond the baseline four bottlenecks and could provide additional speedup, but are not required to declare the milestone successful.

| Feature | Value Proposition | Complexity | Expected Speedup | Notes |
|---------|-------------------|------------|-----------------|-------|
| Vectorized pairwise association scoring (score_all_pairs) | Association scoring is ~5% of wall time. The inner loop over shared frames per pair does LUT lookups and ray-ray distance, all numpy. Vectorizing the shared-frame loop into a matrix operation eliminates per-frame Python overhead. | HIGH | 2–4× association scoring speed; net effect small (5% of total → ~1–2%) | The ray-ray distance formula is analytic and vectorizable: given matrices of origins O (T, 3) and directions D (T, 3) for both cameras, the full shared-frame batch can be solved in one numpy broadcasting call. Difficulty: early termination (stop after `early_k` frames with zero score) is inherently sequential — must decide whether to drop it or approximate with threshold on first-K-frames score. Removing early termination simplifies vectorization at the cost of slightly more compute for clearly-zero pairs. This is a judgment call. |
| Parallel per-camera video decode (ProcessPoolExecutor) | Each camera file is an independent decode. With 12 cameras, 12 cap.read() calls could theoretically run in parallel. | HIGH | Unclear — cv2.VideoCapture is not fork-safe; multiprocessing adds IPC overhead for large frame arrays. May not improve over threaded prefetch. | LOW priority. The threading prefetch approach (table stakes #4) already parallelizes decode with inference. True parallel decode via subprocesses is complex and IPC-expensive. Do not implement unless profiling post-threading-prefetch shows decode is still a bottleneck. |
| Batch frame processing across multiple frames in detection stage | Instead of collecting 12 cameras × 1 frame per predict() call, collect 12 cameras × N frames (e.g., N=4) per call, making the batch size 48. | MEDIUM | Marginal — memory bandwidth bound beyond batch ~16. Pipeline's chunk processing already amortizes scheduling overhead. | LOW priority. Batch size of 12 (one full frame across all cameras) is likely sufficient to saturate GPU. Larger batches add latency (must buffer N frames before calling predict) and complicate downstream result redistribution. Profile first. |

### Anti-Features (Do Not Build)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| TensorRT / ONNX export for YOLO models | Promises 2–4× inference speedup over PyTorch | Custom YOLO training workflow with Ultralytics must remain compatible with `aquapose train`. TensorRT export requires separate `.engine` file management, breaks the single-weights-path config decision, and Ultralytics export is version-sensitive. Speedup benefit is real but the maintenance cost and workflow disruption are not justified when batching alone likely delivers comparable gains. | Achieve GPU efficiency through batching first; revisit TensorRT if profiling post-batching still shows GPU as bottleneck. |
| Multiprocessing pipeline parallelism (stage overlap) | Run detection and tracking in parallel for different frames | The 5-stage pipeline is strictly sequential per chunk (each stage writes context consumed by the next). Stage-level parallelism requires either per-frame streaming (breaking the current batch-per-chunk model) or complex double-buffering. Architectural risk is high; not justified when the bottleneck is within-stage GPU utilization, not between-stage latency. | Fix within-stage efficiency first. If E2E time is still unsatisfactory after the four table-stakes optimizations, reconsider. |
| Async/await (asyncio) for video I/O | Modern async I/O promises cleaner code than threading | cv2.VideoCapture is not async-compatible. Ultralytics inference is synchronous. Asyncio provides no benefit for CPU-bound C extension work. Threading with a Queue is the correct primitive here. | Use `threading.Thread` + `queue.Queue` for producer-consumer overlap. |
| Replacing OpenCV VideoCapture with PyAV or decord | Faster GPU-accelerated decoders | GPU-accelerated decode (NVDEC via PyAV or decord) would help only if decode is the bottleneck after threading. Currently decode time is ~12% — large but manageable with threading. New decode library adds a dependency, changes frame format, and may affect undistortion pipeline. | Profile after threading prefetch. Add GPU-decode library only if profiling reveals decode is still dominant. |
| Numba JIT compilation for triangulation | JIT compile the inner loop for speed | The triangulation bottleneck is not per-iteration Python overhead — it is the sequential structure (15 separate small matrix solves). PyTorch batched SVD is the correct fix, not JIT. Numba introduces a compilation step, type restrictions, and debugging complexity. | Use `torch.linalg.svd` with a batch dimension of n_body_points. |
| Caching triangulation results across frames | Avoid re-triangulating stationary fish | Fish are moving continuously at 30fps. Body points change each frame. No valid reuse opportunity without approximate matching that would introduce correctness risk. | Not applicable. |

---

## Feature Dependencies

```
[Batched YOLO detection inference]
    └──modifies──> YOLOOBBBackend: detect() → detect_batch(frames: list[np.ndarray])
    └──modifies──> DetectionStage.run(): collect frames, call detect_batch, redistribute Results
    └──no dependency on──> [Frame I/O overlap] (independent)

[Batched YOLO midline inference]
    └──modifies──> PoseEstimationBackend: _process_detection() → process_frame_batch()
    └──modifies──> MidlineStage.run(): collect crops across cameras, call batch predict, redistribute
    └──depends on knowledge of──> [Batched YOLO detection inference] (same API pattern)
    └──independent of──> [Frame I/O overlap]

[Vectorized DLT triangulation]
    └──modifies──> DltBackend._reconstruct_fish(): replace 15-iteration loop with batched SVD
    └──modifies──> DltBackend._triangulate_body_point(): replaced by vectorized equivalent
    └──requires──> torch.linalg.svd batched input support (confirmed in PyTorch docs)
    └──independent of──> inference batching features

[Frame I/O overlap with GPU inference]
    └──modifies──> VideoFrameSource or ChunkFrameSource: add background decode thread + Queue
    └──or──> new PrefetchingFrameSource wrapper satisfying FrameSource protocol
    └──independent of──> inference batching features
    └──independent of──> DLT vectorization

[Vectorized pairwise association scoring] (differentiator)
    └──modifies──> score_tracklet_pair(): replace per-frame loop with batched numpy ops
    └──modifies──> score_all_pairs(): optionally parallelize pair scoring
    └──independent of all other features
    └──lowest priority: only 5% of wall time
```

### Dependency Notes

- **Batched detection and batched midline share the same API pattern** but are independent changes. Detection batch is simpler (all inputs are full frames, fixed per-frame count = n_cameras). Midline batch is harder (variable crop count per frame, per-detection affine transforms must still run sequentially on CPU before the batched GPU call).

- **Vectorized DLT has a two-pass structure constraint.** The first triangulation (all cameras) can be fully batched (shape: `[n_body_points, 2*n_cams, 4]`). The second triangulation (inlier cameras only) cannot use a uniform batch because inlier sets differ per body point. Use padded masked tensors or fall back to a masked loop for the second pass. The first pass alone (eliminating 15 separate small solves) is the majority of the gain.

- **Frame I/O prefetch must keep GPU on the main thread.** Ultralytics inference is not thread-safe for GPU execution. Only decode + undistort moves to the background thread. The main thread does inference and consumes from the queue.

- **Prefetching changes the iteration contract for ChunkFrameSource.** The current sequential `read_frame(global_idx)` seek pattern is the most expensive possible access pattern for prefetching. The prefetch thread should decode sequentially (no seeking) ahead of the main thread's consumption.

---

## MVP Definition

### Launch With (v3.4 — all four are required for milestone success)

- [ ] **Batched YOLO detection inference** — collect all n_cameras frames per frame-tick, call `model.predict([frame_1, ..., frame_12])` once, redistribute Results by index. Requires `detect_batch()` variant in `YOLOOBBBackend`. Expected speedup: 2–5× detection throughput.

- [ ] **Batched YOLO midline inference (crop batching)** — collect all crops across all cameras for a frame (or frame-window), call `model.predict([crop_1, ..., crop_N])` once, redistribute keypoints back to (cam_id, detection_index). Expected speedup: 3–8× midline throughput.

- [ ] **Vectorized DLT triangulation** — replace per-body-point loop in `_reconstruct_fish()` with batched `torch.linalg.svd` or `torch.linalg.lstsq` over all body points simultaneously. Minimum viable: vectorize the first-pass triangulation (all cameras); second-pass inlier re-triangulation can remain looped or be masked. Expected speedup: 5–15× reconstruction throughput.

- [ ] **Frame I/O overlap with GPU inference** — background thread prefetches decoded frames into a bounded queue. Main inference thread consumes. Eliminates the ~12% wall-time gap where GPU is idle waiting for the next decoded frame batch.

### Add After Validation (v3.4.x)

- [ ] **Vectorized pairwise association scoring** — vectorize the per-frame loop in `score_tracklet_pair()` using batched numpy ops. Add only if profiling post-v3.4 shows association is a meaningful share of remaining wall time.

### Future Consideration (v3.5+)

- [ ] Per-frame streaming pipeline with stage-level parallelism (requires pipeline architecture change — out of scope)
- [ ] TensorRT / ONNX export (revisit if batching delivers insufficient speedup)
- [ ] GPU-accelerated video decode via decord or PyAV (revisit if decode remains dominant after threading)

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Estimated Wall-Time Reduction |
|---------|------------|---------------------|----------|-------------------------------|
| Batched YOLO detection inference | HIGH | MEDIUM | P1 | Large (detection is dominant bottleneck at ~74% GPU-bound time) |
| Batched YOLO midline inference | HIGH | HIGH | P1 | Large (midline uses same model.predict path, same bottleneck) |
| Vectorized DLT triangulation | HIGH | MEDIUM | P1 | Medium (reconstruction ~9% of wall time, 5–15× speedup within that) |
| Frame I/O overlap | HIGH | MEDIUM | P1 | Medium-small (~12% of wall time, most eliminated) |
| Vectorized association scoring | MEDIUM | HIGH | P2 | Small (~5% of wall time, 2–4× within that) |
| TensorRT / ONNX export | LOW | HIGH | P3 | Potentially large but high risk/complexity |
| GPU-accelerated video decode | LOW | HIGH | P3 | Small after threading overlap |
| Parallel-frame decode (multiprocessing) | LOW | HIGH | P3 | Unclear, likely negative ROI |

**Priority key:**
- P1: Required for v3.4 milestone declaration
- P2: Add if profiling post-P1 shows residual bottleneck
- P3: Future milestone, revisit only after profiling confirms need

---

## Implementation Complexity Notes

### Batched YOLO Detection (MEDIUM complexity)
The Ultralytics API accepts `model.predict(list_of_numpy_arrays)` — confirmed in official docs. The primary complexity is in redistribution: `model.predict([f1, f2, ..., f12])` returns a list of 12 `Results` objects in the same order as inputs. The existing `detect()` method processes one frame and returns `list[Detection]`. A new `detect_batch(frames: dict[str, np.ndarray]) -> dict[str, list[Detection]]` method collects camera_ids, builds an ordered list of frames, calls predict once, and reconstructs the per-camera dict from the Results list. `DetectionStage.run()` changes to call `detect_batch` instead of `detect` per camera.

### Batched YOLO Midline (HIGH complexity)
The crop-batching approach requires: (1) extract affine crops for all detections across all cameras for the current frame (CPU, sequential, unchanged); (2) collect all crops into a single list with a parallel index mapping `[(cam_id, det_idx), ...]`; (3) call `model.predict(crops_list)` once; (4) redistribute Results back to `(cam_id, det_idx)` pairs using the index map. The keypoint extraction and spline interpolation post-predict remain unchanged and sequential. The higher complexity vs. detection batching comes from variable crop count per frame (some cameras may have 0–9 crops) and the need to maintain the (cam_id, det_idx) index.

### Vectorized DLT (MEDIUM complexity)
The key mathematical structure: for body point `i`, the current code builds a system `A_i @ x_i ≈ 0` where `A_i` has shape `(2*n_cams, 4)`. Stacking gives `A` of shape `(n_body_points, 2*n_cams, 4)`. `torch.linalg.svd(A)` returns `V` of shape `(n_body_points, 4, 4)`; the solution for each point is `V[:, :, -1]` (last column = smallest singular value). The outlier rejection second pass is harder to vectorize uniformly because inlier sets differ per point. Acceptable approach: vectorize first pass fully (eliminates ~half the compute); for second pass, use a masked approach or a short loop only over points that had outliers in the first pass (typically a minority).

### Frame I/O Prefetch (MEDIUM complexity)
Standard producer-consumer pattern. Create a `PrefetchingFrameSource` wrapper that satisfies the `FrameSource` protocol. On `__enter__`, start a `threading.Thread` that sequentially decodes frames and pushes `(frame_idx, frames_dict)` to a `queue.Queue(maxsize=4)` (4-frame buffer). Main thread consumes from the queue in `__iter__`. The thread sets a sentinel value when exhausted. Key constraint: the existing `ChunkFrameSource.__iter__()` uses `source.read_frame(global_idx)` with seeks — the prefetch thread should use sequential `cap.read()` (no seeks) for maximum efficiency. This may require the prefetch thread to operate on the underlying `VideoFrameSource` directly rather than through `ChunkFrameSource`.

### Vectorized Association Scoring (HIGH complexity, LOW priority)
The `score_tracklet_pair()` function iterates over `shared_frames` and computes a LUT lookup + ray-ray distance per frame. Vectorizing requires: (1) collecting all shared-frame centroid tensors into matrices `O_a (T, 3)`, `D_a (T, 3)`, `O_b (T, 3)`, `D_b (T, 3)`; (2) computing ray-ray distances as a batched numpy op — the analytic formula `ray_ray_closest_point` is vectorizable with broadcasting; (3) aggregating scores. The early-termination heuristic (bail after `early_k` frames with zero score) is incompatible with full vectorization — it would need to be dropped or replaced with a first-K-frames pre-filter. Given that association is only ~5% of wall time, the complexity/benefit ratio is poor. Do not implement in v3.4.

---

## Correctness Preservation Requirements

All optimizations are correctness-neutral: the mathematical result must be numerically identical (or within floating-point tolerance) to the pre-optimization baseline. This is enforced by the existing golden-data verification framework and `aquapose eval`.

| Optimization | Correctness Risk | Mitigation |
|--------------|-----------------|------------|
| Batched detection | LOW — same predict call, just batched | Verify per-camera Results ordering matches input ordering |
| Batched midline | LOW — same predict call, variable crop count | Verify (cam_id, det_idx) index round-trip |
| Vectorized DLT | MEDIUM — batched SVD numerically equivalent, but outlier rejection second pass needs care | Compare Tier1 reprojection metrics before/after using `aquapose eval` |
| Frame I/O prefetch | LOW — same frames, different delivery timing | Verify frame_idx ordering is preserved; assert on frame content in unit tests |

---

## Sources

- `/home/tlancaster6/Projects/AquaPose/.planning/PROJECT.md` — v3.4 milestone definition and profiling targets (HIGH confidence — primary source)
- Direct codebase inspection: `YOLOOBBBackend`, `DetectionStage`, `MidlineStage`, `PoseEstimationBackend`, `DltBackend`, `score_all_pairs`, `VideoFrameSource`, `ChunkFrameSource` (HIGH confidence)
- [Ultralytics Predict Docs](https://docs.ultralytics.com/modes/predict/) — confirmed batch inference API accepts list of numpy arrays (HIGH confidence)
- [Ultralytics YOLO11 Batch Inference Blog](https://www.ultralytics.com/blog/using-ultralytics-yolo11-to-run-batch-inferences) — batch arg behavior, default size=1 (MEDIUM confidence)
- [YOLOv8 Batch Inference Speed Analysis](https://dev-kit.io/blog/machine-learning/yolov8-batch-inference-speed-and-efficiency) — throughput scaling characteristics (MEDIUM confidence)
- [torch.linalg.svd PyTorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html) — confirmed batched SVD support (HIGH confidence)
- [torch.linalg.lstsq PyTorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) — confirmed batched least-squares, CUDA limitation (gels driver = full-rank only) (HIGH confidence)
- [OpenCV multithreading with VideoCapture](https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/) — threading pattern, GIL behavior for cap.read() (MEDIUM confidence)
- [PyTorch DataLoader prefetch tactics](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8) — pin_memory, non-blocking transfer, producer-consumer patterns (MEDIUM confidence)
- [NumPy pairwise vectorization](https://towardsdatascience.com/how-to-vectorize-pairwise-dis-similarity-metrics-5d522715fb4e/) — broadcasting patterns for distance matrices (MEDIUM confidence)

---

*Feature research for: AquaPose v3.4 Performance Optimization*
*Researched: 2026-03-05*
