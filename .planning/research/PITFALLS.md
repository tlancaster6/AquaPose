# Pitfalls Research

**Domain:** Adding batching, async I/O, and vectorization to an existing synchronous CV pipeline (v3.4 Performance Optimization milestone)
**Researched:** 2026-03-05
**Confidence:** HIGH — codebase directly inspected for all integration points; pitfalls verified against PyTorch official docs, Ultralytics GitHub issues, and OpenCV known issues.

> **Scope note:** This file covers pitfalls specific to v3.4 Performance Optimization: adding batched YOLO inference (detection and midline stages), optimized frame I/O via look-ahead decode, vectorized DLT reconstruction, and vectorized association scoring to the existing synchronous per-frame/per-point pipeline. Prior milestone pitfalls (v2.2, v3.0, v3.1, v3.2, v3.3) are preserved below the milestone-specific section.

---

## v3.4 Performance Optimization Pitfalls

These pitfalls are specific to retrofitting batching, prefetching, and vectorization into the existing synchronous AquaPose pipeline. The primary risk sources are:

1. **Batch predict result-to-frame mapping** — multiple frames batched to one YOLO call, results must be re-indexed back to the right frame and camera.
2. **CUDA OOM from over-batching** — 12 cameras × N frames × 1600×1200 images overwhelm GPU memory at even modest N.
3. **OpenCV VideoCapture thread safety** — the existing `ChunkFrameSource` uses `read_frame()` with `CAP_PROP_POS_FRAMES` seeks; neither is thread-safe under concurrent access.
4. **Frame seek inaccuracy under async decode** — `CAP_PROP_POS_FRAMES` seeks to keyframes, not always the requested frame, causing frame index corruption when decoding is parallelized.
5. **Numerical drift after vectorization** — replacing scalar-loop DLT and ray-ray scoring with batched numpy/torch operations introduces floating-point reordering that can shift results outside the existing golden-data tolerance window.
6. **TF32 precision on Ampere+ GPUs** — PyTorch enables TensorFloat32 by default on Ampere+ GPUs, silently reducing mantissa precision in matmuls and creating spurious numerical differences from the float32 baseline.

---

### Pitfall P1: Batch Predict Result-to-Frame Mapping Errors

**What goes wrong:**
When detection or midline stages are modified to call `model.predict([frame_cam_A_t0, frame_cam_A_t1, frame_cam_A_t2, ...])` across multiple frames or cameras, the returned `results` list has `len(results) == len(inputs)`. If the batch is built by flattening across cameras and frames (e.g., `[cam0_frame0, cam1_frame0, cam2_frame0, cam0_frame1, ...]`), then `results[k]` must be correctly demapped to `(camera_id, frame_idx)`. Any off-by-one in how the batch was constructed — a frame dropped on EOF, a camera skipped due to calibration miss — silently shifts all subsequent result assignments: `results[3]` ends up attributed to `cam1_frame1` when it actually came from `cam2_frame0`.

**Why it happens:**
The existing per-frame loop in `DetectionStage.run()` and `SegmentationBackend.process_frame()` processes one image at a time; the `(cam_id, frame_idx)` identity is implicit in the loop variable. When batching is added, that identity must be made explicit and carried through as a parallel index structure. Developers typically build the flat image list first, run predict, then try to reconstruct identity from position — any asymmetry between construction and reconstruction corrupts the mapping. The Ultralytics docs confirm result ordering is preserved relative to the input list, but do not defend against construction errors.

**How to avoid:**
- Build a `batch_index: list[tuple[str, int]]` in lockstep with the `images: list[np.ndarray]` list: `batch_index.append((cam_id, frame_idx))` exactly when `images.append(frame)` is called.
- After `results = model.predict(images, ...)`, iterate `for (cam_id, frame_idx), r in zip(batch_index, results)`.
- Assert `len(results) == len(batch_index)` immediately after every predict call; treat a mismatch as a fatal error, not a warning.
- Write a unit test that batches a known 3-frame × 3-camera sequence, injects a deliberately skipped camera on frame 1, and verifies the result attributions are correct.

**Warning signs:**
- Detection bounding boxes appear in the wrong camera's coordinate system (e.g., top-view centroid at a position impossible for that camera's field of view).
- Per-camera detection counts are transposed (camera B has N fish while camera A shows 0, then reverses the next frame).
- Stage timing is correct but downstream reconstruction produces `nan` or wildly incorrect 3D points for specific fish.

**Phase to address:**
Inference batching phase (detection and midline). The batch-index-in-lockstep pattern must be the design contract before any batching code is written.

---

### Pitfall P2: CUDA OOM from Over-Batching at 12-Camera × Multi-Frame Scale

**What goes wrong:**
AquaPose processes 12 cameras at 1600×1200 resolution. A single undistorted frame is 1600×1200×3 bytes = ~5.8 MB. Batching all 12 cameras for a single frame yields ~70 MB input. At 8-frame batches that becomes ~560 MB input tensors on top of model activations, NMS workspace, and mask outputs. For the segmentation backend, each predicted mask is an additional (H×W) tensor per detection. YOLO internally pads images to the model's `imgsz` before batching; if cameras have different effective sizes post-undistortion, Ultralytics pads to the largest, multiplying memory usage. OOM is triggered silently mid-batch with a CUDA exception that aborts the entire chunk, discarding all in-flight results.

**Why it happens:**
Developers test batching at small `imgsz` (e.g., 640) in development but the pipeline uses the model's native resolution (typically 640 or 1280). The 12-camera scenario is unusual — most YOLO users batch 8–32 images of 640×640 consumer photos, not 12 synchronized high-resolution scientific images. GPU memory profiling is skipped because throughput looks fine on small batches.

**How to avoid:**
- Profile GPU memory with `torch.cuda.memory_summary()` at the start of a batched detection call before committing to a batch size. The target budget is: GPU VRAM total minus 2 GB headroom for model weights and reconstruction tensors.
- Use a configurable `detection_batch_frames` parameter (default 1, not `len(chunk)`). Tune upward empirically on the target hardware.
- Wrap every `model.predict(batch)` call in a try/except that catches `torch.cuda.OutOfMemoryError`, halves the batch, and retries. This prevents a single large chunk from crashing a multi-hour run.
- Add `torch.cuda.empty_cache()` between batches if memory pressure is detected.
- For the segmentation backend, the mask tensor size scales with the number of detections × crop resolution. Batching N crops through seg is safer than batching full frames; measure both paths.

**Warning signs:**
- `torch.cuda.OutOfMemoryError` mid-run after the pipeline appeared to work on smaller test clips.
- GPU memory grows monotonically across frames without releasing (leaked tensors not moved off GPU after predict).
- VRAM utilization visible in `nvidia-smi` spikes then crashes for specific frame batches (dense fish scenes with more detections than average).

**Phase to address:**
Inference batching phase. Batch size must be a tunable config parameter with a safe conservative default, not a hardcoded value.

---

### Pitfall P3: OpenCV VideoCapture Is Not Thread-Safe for Concurrent Reads

**What goes wrong:**
The existing `VideoFrameSource` uses a shared `dict[str, cv2.VideoCapture]` where each camera has one `VideoCapture` object. `ChunkFrameSource.__iter__` calls `self._source.read_frame(global_idx)`, which calls `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)` followed by `cap.read()` on each capture. If async I/O is added by spinning up a prefetch thread or thread-pool that calls `read_frame` concurrently, two threads may interleave their `set+read` on the same capture object: thread A seeks to frame 50, thread B seeks to frame 51 on the same capture, thread A reads frame 51 (thread B's frame). The OpenCV `VideoCapture` API has no internal lock and is documented as not thread-safe.

**Why it happens:**
The existing sequential `read_frame` is obviously safe. When a prefetch thread is added to overlap I/O with GPU inference, the implementation typically calls `read_frame` from the background thread without realizing the captures are shared mutable state. Python's GIL does not help here because `cap.set` and `cap.read` release the GIL for the underlying C++ I/O operations, allowing true concurrent access to the C++ object.

**How to avoid:**
- Never share a `VideoCapture` object across threads. The only correct threading model for the existing captures is: one thread owns all captures and enqueues decoded frames into a `queue.Queue`. The inference thread dequeues frames.
- Alternatively, open one `VideoCapture` per thread per camera (each thread has its own `cap = cv2.VideoCapture(path)` for each camera). This is safe but increases file handle count (12 cameras × N threads = 12N handles).
- If a prefetch queue is used, decode and enqueue complete multi-camera frame dicts `{cam_id: frame}` as atomic units so inference always sees a consistent frame.
- Add a `threading.Lock` per camera capture if the single-capture-per-camera model must be preserved; acquire before `set+read`, release after.

**Warning signs:**
- Occasional frames from the wrong time position decoded into the chunk (visible as a single frame where fish positions jump discontinuously then return).
- Intermittent `cv2.error: (-215:Assertion failed)` or segfaults in the OpenCV backend (C++ assertion on corrupted capture state).
- Race condition is non-deterministic — appears only under load with certain chunk boundary alignments.

**Phase to address:**
Frame I/O optimization phase. Thread safety must be the design invariant before any prefetch thread is introduced.

---

### Pitfall P4: CAP_PROP_POS_FRAMES Seek Inaccuracy Corrupts Frame Identity

**What goes wrong:**
`ChunkFrameSource.__iter__` calls `read_frame(global_idx)` which calls `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)`. For H.264/H.265 encoded video (the AVI files in the AquaPose rig), `CAP_PROP_POS_FRAMES` seeks to the nearest preceding keyframe (I-frame), not the exact requested frame. For videos encoded with default keyframe intervals (typically every 30–250 frames), requesting frame 47 may seek to frame 30 (the last keyframe) and then decode forward to frame 47. When this is parallelized — e.g., thread A requests frame 47 and thread B requests frame 48 on the same capture — the P-frame decode for frame 48 depends on the state of frame 47 already being decoded. This is a second, independent way async access to a single capture corrupts frame identity, separate from the thread-safety issue in P3.

**Why it happens:**
Sequential `read()` calls on a video file that is played forward are always exact. Random access via `set(CAP_PROP_POS_FRAMES)` is a fundamentally different operation that relies on the codec's seek tables. The known OpenCV issue #4890 (open since 2015) documents that this seek is not always accurate. The `ChunkFrameSource` was designed for sequential playback but exposes `read_frame(idx)` for random access; async prefetch that uses random access on H.264 content will encounter this.

**How to avoid:**
- Use sequential playback — advance through the chunk linearly. Buffer frames into a deque or `queue.Queue` from a single reader thread. Do not use `CAP_PROP_POS_FRAMES` for async prefetch.
- If random access is required (e.g., for selective frame loading), re-encode the source videos with `ffmpeg -i input.avi -g 1 output.avi` to set keyframe interval to 1 (all frames are keyframes). This eliminates seek error at the cost of larger file size.
- Add a frame verification check: after `cap.set(CAP_PROP_POS_FRAMES, idx); cap.read()`, also read `int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1` and assert it equals `idx`. Log a warning and read forward if there is a mismatch.
- The safest async I/O pattern for chunk processing: read sequentially in one thread, buffer up to N frames ahead, yield from buffer. Prefetch gains come from pipelining, not random access.

**Warning signs:**
- Frame-to-frame fish position jumps that occur only in async mode, not sequential mode.
- Per-chunk reprojection error statistics that are occasionally inflated (suggests the wrong frame was matched to cached detection results).
- Frame seek errors are codec- and file-dependent — present for H.264 .avi but potentially absent for raw .mjpeg containers.

**Phase to address:**
Frame I/O optimization phase. The prefetch design must be based on sequential read + buffer, not parallel random access.

---

### Pitfall P5: Vectorized DLT Produces Numerically Inequivalent Results Due to Floating-Point Reordering

**What goes wrong:**
The current DLT triangulation in `_triangulate_body_point` processes one body point at a time in a Python loop. The operations inside (`torch.stack`, `weighted_triangulate_rays`, per-camera residual computation) use float32 tensors. When vectorized to process all N body points simultaneously — stacking ray origins/directions for all body points into a `(N, K, 3)` tensor and calling a batched SVD or least-squares — the order of floating-point additions changes. Floating-point addition is not associative: `(a + b) + c != a + (b + c)` in float32. Batched matmul in PyTorch is not guaranteed to produce bitwise-identical results to per-element matmul. The result is that vectorized DLT produces 3D points that differ from the per-point loop by small amounts (~1e-5 to 1e-3 in world coordinates), which can flip outlier rejection decisions for body points near the residual threshold (10.0 px), producing different inlier sets and therefore different final reconstructions.

**Why it happens:**
Developers test vectorized reconstruction by checking that "most points match" without establishing a tolerance-aware equivalence test. The per-point loop is the reference; the vectorized version is the candidate. Without a test that checks all-inlier-sets-match (not just the final 3D points), small numerical differences propagate invisibly through outlier rejection and produce structurally different reconstructions that look visually similar but differ in which cameras contributed.

**How to avoid:**
- Define numerical equivalence as: vectorized and per-point-loop produce identical inlier camera sets for every body point, and final 3D points agree within `1e-4` metres absolute (well within reconstruction noise). Spline control points must agree within `1e-3` metres.
- Run the vectorized path against the per-point-loop path on the same cached `MidlineSet` inputs (from a real YH run cache) before declaring the vectorized path correct.
- If TF32 is enabled on the GPU (Ampere or later), disable it for the triangulation path: `torch.backends.cuda.matmul.allow_tf32 = False` within the DLT backend. TF32 reduces mantissa precision from 23 bits to 10 bits and is the most likely source of larger-than-expected numerical differences.
- Use float64 for the SVD step if numerical stability is a concern; the per-body-point tensor is small enough that float64 cost is negligible.
- Keep the per-point-loop as a `debug_mode` fallback path behind a config flag so the golden reference is always available.

**Warning signs:**
- Vectorized reconstruction produces identical per-camera reprojection errors on unit test data but different outlier rejection outcomes on real data near threshold boundaries.
- `is_low_confidence` flags differ between vectorized and loop paths for a small fraction of fish (outlier sets differ by one camera).
- Mean residual statistics between vectorized and loop differ by more than 0.1 px on real data (indicates a structural difference, not just rounding).

**Phase to address:**
Vectorized reconstruction phase. A numerical equivalence test on real cached data must pass before the vectorized path replaces the loop.

---

### Pitfall P6: TF32 Silently Reduces Precision on Ampere+ GPUs

**What goes wrong:**
PyTorch enables TensorFloat32 (TF32) by default on NVIDIA Ampere and later GPUs (A100, RTX 30xx, RTX 40xx). TF32 uses the full float32 exponent range but only 10 mantissa bits (vs. 23 for float32), giving ~3 decimal digits of precision instead of ~7. This affects all `torch.matmul`, `@`, and convolution operations. For DLT triangulation, the normal equations matrix `A^T A` is computed as a matmul; TF32 precision loss there can produce 3D point offsets of 1e-3 to 1e-2 metres — large enough to shift reconstruction quality metrics. This is not a new feature introduced by vectorization; it is a pre-existing GPU configuration issue that becomes visible when the vectorized path runs more matmuls than the per-point loop.

**Why it happens:**
TF32 is a compile-time-invisible default. There is no warning when it activates. The existing per-point loop uses `triangulate_rays` which relies on `torch.linalg.lstsq`; TF32 affects this. However, because the per-point results were used to tune thresholds (outlier_threshold=10.0 from the empirical grid search), TF32 is baked into the calibrated baseline. A vectorized path that disables TF32 for stability produces slightly different results than the tuned baseline — both paths may be numerically correct but neither is a drop-in replacement for the other without recalibration.

**How to avoid:**
- Audit the existing baseline: determine whether the real-data runs that produced `outlier_threshold=10.0` were executed with TF32 enabled or disabled. The YH run configuration (hardware: unknown from code, needs hardware check) determines which is the baseline.
- If TF32 was enabled during baseline tuning: keep TF32 enabled in the vectorized path to preserve equivalence. If TF32 was disabled: disable it in the vectorized path.
- Document the TF32 setting as a DltBackend config parameter (`allow_tf32: bool`, default matches baseline).
- Add the TF32 setting to the evaluation runner so parameter sweeps can control it explicitly.

**Warning signs:**
- Large batch of tuning sweep results shift uniformly in one direction when run on a new machine with a different GPU generation.
- Two physically different machines produce subtly different `mean_residual` distributions from identical code.
- `torch.backends.cuda.matmul.allow_tf32` returns `True` when you expected it to be `False`.

**Phase to address:**
Vectorized reconstruction phase, but also relevant to any phase that runs matmuls with the existing tuned threshold.

---

### Pitfall P7: Association Vectorization Breaks Existing Early-Termination Semantics

**What goes wrong:**
`score_tracklet_pair` implements an early-termination shortcut: if after `early_k` frames the `score_sum == 0.0`, the pair is immediately scored 0 and the remaining shared frames are skipped. This is both a performance optimization and a semantic contract: pairs that score 0 on the first `early_k` frames are definitionally weak and skipped. When association scoring is vectorized to compute all pairwise scores in a batched tensor operation (e.g., building an `(M, N, T)` ray-distance tensor for M tracklets × N tracklets × T shared frames), early termination cannot be implemented without masking or short-circuit logic. A naive vectorized implementation removes early termination, changes the effective scoring formula for weak pairs, and may elevate some low-confidence pairs above `score_min`, changing the Leiden graph structure.

**Why it happens:**
Early termination is a sequential loop optimization that has no direct tensor equivalent. Developers replacing the loop with a batched operation naturally drop the early termination as a "performance detail that doesn't affect correctness." In this case it does affect correctness: `score_tracklet_pair` with `early_k=10` and `score_sum=0.0` at frame 10 returns exactly `0.0`; a vectorized version computing all T frames would return a small positive score if any later frame happens to pass threshold.

**How to avoid:**
- Treat early termination as a semantic contract that the vectorized path must replicate. After computing the batched distance tensor, apply the masking rule: for any pair where `distances[:early_k].min() > threshold` (no inliers in first `early_k` frames), zero out the entire row before aggregation.
- Alternatively, retain the sequential pair-scoring loop but vectorize the inner ray-ray distance computation within each pair. This provides speedup (eliminating the inner Python loop over shared frames) without changing pair-level semantics.
- Run the vectorized scorer against the sequential scorer on the cached `Tracklet2D` data from a real YH chunk and assert identical output dicts (same keys, same scores within 1e-6).

**Warning signs:**
- More edges survive `score_min` filtering in the vectorized path than in the sequential path for the same chunk.
- Leiden clustering produces more or fewer groups in the vectorized path (different graph inputs).
- Singleton rate changes between vectorized and sequential paths (more edges → fewer singletons, or vice versa depending on how the graph changes).

**Phase to address:**
Vectorized association phase. Equivalence test on real chunk data must pass before the vectorized scorer is enabled.

---

### Pitfall P8: GPU Memory Not Released Between Batch Predict Calls (Tensor Leak)

**What goes wrong:**
Ultralytics `model.predict()` returns `Results` objects that hold references to GPU tensors (e.g., `r.obb.xywhr` is a CUDA tensor). The existing per-detection loop in `YOLOOBBBackend.detect()` calls `.cpu().numpy()` on each tensor immediately, releasing the GPU reference. When batching is added and results are stored in a list before processing, the CUDA tensors in all `Results` objects remain live on GPU until the list is garbage-collected. For 12-camera × 8-frame batches with ~9 detections each, this is ~864 result objects all holding live CUDA tensors simultaneously. Combined with the model's own activation buffers still present from the forward pass, GPU memory may not be released between batches, causing OOM on the second or third batch.

**Why it happens:**
The current code pattern of immediately calling `.cpu().numpy()` inside the detection loop is incidentally correct for memory management. When refactoring to store results for later processing, the immediate-release guarantee is lost. Python reference counting does not guarantee immediate deallocation when CUDA tensors are involved — `torch.cuda.empty_cache()` must be called explicitly or the results must be moved to CPU immediately after predict.

**How to avoid:**
- Process each result object to extract CPU numpy arrays before storing to the detection list. Do not store `Results` objects; store `Detection` objects (which contain only numpy arrays). This preserves the existing memory safety pattern.
- If `Results` objects must be kept temporarily, explicitly call `del results` and `torch.cuda.empty_cache()` after extraction.
- Add `torch.cuda.memory_allocated()` assertions in the test suite to verify that GPU memory returns to baseline after each detect call.

**Warning signs:**
- `nvidia-smi` shows GPU memory growing monotonically across frames during detection, never releasing between batches.
- OOM errors that occur only after the pipeline has been running for several minutes (progressive memory accumulation).
- Memory issues reproducible only when batch size > 1 (the per-detection code path is fine but the batch code path leaks).

**Phase to address:**
Inference batching phase. Memory release must be an explicit design requirement, not an accidental property of the existing code.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoded batch_size=8 | Simpler code, works on development GPU | OOM on smaller GPUs; can't tune without code change | Never — must be config |
| Skipping numerical equivalence test | Faster iteration, "looks right" | Vectorized path passes unit tests but fails on real data near threshold boundaries; regression discovered weeks later | Never for paths that affect outlier rejection |
| Sharing VideoCapture across threads without lock | Simpler implementation | Silent frame identity corruption, hard to reproduce race condition | Never |
| Using CAP_PROP_POS_FRAMES for async random access | Obvious API | Frame position errors on H.264 video, non-deterministic bugs | Never for async access; acceptable for single-threaded sequential |
| Storing Ultralytics Results objects instead of extracted numpy | Fewer conversion calls | GPU memory accumulation across batch, OOM after minutes | Never — always extract to CPU immediately |
| Disabling TF32 globally for stability | Eliminates precision class of bugs | Performance regression on all GPU operations including inference | Only as a debug fallback, not default |

---

## Integration Gotchas

Common mistakes when connecting to existing pipeline infrastructure.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Ultralytics batch predict | Pass flat list of images; assume results align with (cam_id, frame_idx) by position | Build `batch_index` list in lockstep with image list; zip index and results after predict |
| Ultralytics OBB batch predict | Iterate `for r in results` without checking which image each result came from | Use `enumerate(results)` and `batch_index[i]` to recover `(cam_id, frame_idx)` |
| ChunkFrameSource + async thread | Call `read_frame(idx)` from background thread while main thread also calls it | Dedicate one thread to sequential reads into a `queue.Queue`; inference thread dequeues |
| DLT vectorization + existing cache | Compare vectorized results to sequential results using fresh synthetic data only | Compare on real YH chunk cache where body points near threshold boundary are present |
| Association vectorization + Leiden | Replace score dict with vectorized scores; check cluster count is same | Assert full score dict equivalence (keys and values) before enabling vectorized path |
| GPU batch size tuning | Fix batch size in code; test on development machine | Add `detection_batch_frames` to config with conservative default; document hardware limits |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Batch size too large for dense scenes | OOM on frames where all 9 fish are detected (vs. typical 3-4) | Cap memory usage, not just image count; measure at max-detection scenes | First dense interaction frame in a long recording |
| Sequential prefetch queue too small | Prefetch thread stalls waiting for inference to consume; no speedup | Queue depth >= 2× expected GPU inference time in frames | When GPU inference is slower than expected (dense frames) |
| Vectorized DLT allocates (N×K×T) intermediates | Memory spike during reconstruction; CUDA OOM on body-point × camera × arc-length tensor | Profile intermediate tensor sizes before committing to batch layout | When N_body_points × N_cameras × N_body_points exceeds GPU memory |
| ray_ray_closest_point vectorized into (M×N×T×3) | Memory explosion: 13 cameras, ~50 tracklets each → 650×650×1000×3 float32 = 5+ GB | Vectorize inner loop (per shared frame) not outer loop (per tracklet pair) | First run on a full-length chunk with many tracklets |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Batch detection:** Batch index in lockstep with image list — verify `assert len(results) == len(batch_index)` is present at every predict call site.
- [ ] **CUDA memory:** GPU memory returns to baseline after each batch — verify with `torch.cuda.memory_allocated()` in tests, not just "no OOM observed."
- [ ] **Frame source threading:** Background reader thread owns all captures exclusively — verify no `read_frame` calls from inference thread.
- [ ] **Numerical equivalence:** Vectorized path tested against per-point loop on real YH chunk data (not just synthetic) — verify test uses actual cached MidlineSet and checks inlier camera sets, not just final 3D positions.
- [ ] **Association early termination:** Vectorized scorer replicates early termination semantics — verify output score dict is identical to sequential scorer on real chunk data.
- [ ] **TF32 baseline documented:** Whether original tuned threshold (10.0 px) was calibrated with TF32 on or off — verify by checking hardware of the machine that ran the original grid search.
- [ ] **Batch size is configurable:** `detection_batch_frames`, `midline_batch_crops` exist as config fields — verify they appear in the config dataclass and can be overridden via YAML.

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Batch index mapping error discovered post-run | HIGH | Identify which frames are affected (correlation analysis of detection positions across cameras), re-run affected chunks with batch_size=1 |
| CUDA OOM mid-chunk | LOW | Add try/except around predict, halve batch size, retry; chunk caching means completed chunks are not re-run |
| Thread safety corruption of frame data | HIGH | Cannot easily identify corrupted frames post-facto; must disable async, re-run, compare outputs |
| Vectorized DLT numerical drift beyond tolerance | MEDIUM | Disable vectorized path via config flag, run sequential path; re-run evaluation sweep to verify recovery |
| TF32 mismatch with tuned thresholds | MEDIUM | Set `allow_tf32` to match baseline; re-run affected evaluation metrics |
| GPU memory leak (tensor accumulation) | MEDIUM | Add `torch.cuda.empty_cache()` between batches; restart run from last committed chunk cache |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1: Batch result mapping | Inference batching (detection + midline) | Unit test: 3-frame × 3-camera batch with deliberate camera skip; assert all result attributions correct |
| P2: CUDA OOM from over-batching | Inference batching | Integration test: run on dense frame (9 detections per camera) at various batch sizes; confirm no OOM and configurable limit |
| P3: VideoCapture thread safety | Frame I/O optimization | Code review: assert no `read_frame` calls from non-reader thread; document threading model |
| P4: CAP_PROP_POS_FRAMES inaccuracy | Frame I/O optimization | Design review: confirm prefetch uses sequential read+queue, not random access |
| P5: Vectorized DLT numerical drift | Vectorized reconstruction | Numerical equivalence test on real YH chunk cache: inlier sets identical, 3D points within 1e-4 m |
| P6: TF32 precision | Vectorized reconstruction | Document baseline TF32 state; add `allow_tf32` config; assert metric parity with documented baseline |
| P7: Association early termination | Vectorized association | Score dict equivalence test on real chunk data: identical keys and values within 1e-6 |
| P8: GPU tensor leak | Inference batching | Memory allocation test: `torch.cuda.memory_allocated()` returns to baseline after each batch call |

---

## Sources

- [PyTorch Numerical Accuracy Notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — batched matmul non-identical to per-element, TF32 precision
- [PyTorch CUDA Memory Documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — memory fragmentation, `memory_summary`
- [Ultralytics Predict Docs](https://docs.ultralytics.com/modes/predict/) — result ordering, batch inference behavior
- [OpenCV Issue #4890](https://github.com/opencv/opencv/issues/4890) — CAP_PROP_POS_FRAMES seek inaccuracy, long-standing known issue
- [OpenCV Issue #9053](https://github.com/opencv/opencv/issues/9053) — frame seek not exact with ffmpeg backend
- [OpenCV Issue #20227](https://github.com/opencv/opencv/issues/20227) — set+read differs from sequential read
- [OpenCV Issue #24229](https://github.com/opencv/opencv/issues/24229) — VideoCapture thread lock absence
- [Ultralytics Issue #4057](https://github.com/ultralytics/ultralytics/issues/4057) — CUDA OOM during inference
- AquaPose codebase direct inspection: `VideoFrameSource.read_frame`, `ChunkFrameSource.__iter__`, `YOLOOBBBackend.detect`, `SegmentationBackend.process_frame`, `DltBackend._triangulate_body_point`, `score_tracklet_pair` — all call patterns verified from source

---

> **Previous milestone pitfalls (v2.2, v3.0, v3.1, v3.2, v3.3) were in the previous version of this file. They have been superseded by this v3.4-specific document. If prior pitfall context is needed, consult git history or the RETROSPECTIVE.md.**

---
*Pitfalls research for: Adding batching, async I/O, and vectorization to existing synchronous CV pipeline (v3.4 Performance Optimization)*
*Researched: 2026-03-05*
