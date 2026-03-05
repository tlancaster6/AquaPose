# Phase 59: Batched YOLO Inference - Context

**Gathered:** 2026-03-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace per-camera/per-crop serial YOLO `predict()` calls with batched inference in both the detection and midline stages. Detection batches all 12 camera frames per timestep into a single `predict()` call. Midline batches all crops across all cameras per frame into a single `predict()` call. CUDA OOM triggers automatic retry with halved batch size. Goal is GPU utilization improvement from ~30% toward its practical ceiling with numerically identical results.

</domain>

<decisions>
## Implementation Decisions

### Batching granularity
- Detection: all 12 camera frames for a timestep go into a single `predict()` call (no sub-batching)
- Midline: all crops across all cameras for a single frame go into one `predict()` call
- Both midline backends (segmentation + pose_estimation) get batched — not just the active one
- Crop preparation (extract_affine_crop) is done in bulk (CPU) before the single batch `predict()` call (GPU)

### OOM retry strategy
- On CUDA OOM, halve batch size and retry; keep halving down to batch_size=1, then fail with the real exception
- Warn (logger.warning) at each halving step
- Once a reduced batch size succeeds, persist it for the rest of the run (no re-attempting full batch each frame)
- Catch both `torch.cuda.OutOfMemoryError` and `RuntimeError` with 'CUDA out of memory' in message
- Shared utility function used by both detection and midline stages (DRY)
- At end of run, if batch size was reduced, log a recommendation for the user's config

### Backend interface
- Stage collects images/crops, then calls new `detect_batch()` / `process_batch()` on the backend
- Existing single-image methods (`detect()`, `process_frame()`) are kept alongside new batch methods
- `detect_batch(frames: list[ndarray])` takes a list of ndarrays — backend doesn't see camera IDs
- `process_batch(crops: list[...])` takes a list of crops, returns results in positional correspondence (same order)
- Stage handles mapping results back to `(cam_id, det_idx)` by tracking input order

### Config design
- `detection_batch_frames` lives on DetectionConfig (per-stage config pattern)
- `midline_batch_crops` lives on MidlineConfig (per-stage config pattern)
- Default value: 0, meaning "no limit — batch everything"
- No config toggle for OOM retry — always on; batch_size=1 surfaces the real exception if needed
- End-of-run log recommendation if OOM retry reduced the batch size

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `YOLOOBBBackend.detect(frame)` at `core/detection/backends/yolo_obb.py:67` — current single-frame detection, needs `detect_batch()` sibling
- `SegmentationBackend.process_frame()` at `core/midline/backends/segmentation.py:108` — loops cameras then detections, each calling `_process_detection()` which calls `model.predict()` on one crop
- `PoseEstimationBackend.process_frame()` at `core/midline/backends/pose_estimation.py:171` — identical loop structure to segmentation
- `extract_affine_crop()` at `core/midline/crop.py:13` — CPU crop extraction, will be called in bulk before batch predict
- Ultralytics `model.predict()` natively accepts `list[ndarray]` for batched inference

### Established Patterns
- Backends are registered via `get_backend()` factory in each stage's `backends/__init__.py`
- Stages receive `PipelineContext`, read inputs, write outputs — stateless per-run
- Config is frozen dataclass hierarchy: `PipelineConfig` > per-stage configs
- `core/` has no engine imports — shared OOM utility should live in `core/` (e.g. `core/inference.py`)

### Integration Points
- `DetectionStage.run()` at `core/detection/stage.py:60` — inner loop iterates cameras × `detect()`, needs refactor to collect frames then call `detect_batch()`
- `MidlineStage.run()` at `core/midline/stage.py:140` — delegates to `backend.process_frame()`, needs refactor to collect crops then call `process_batch()`
- `engine/config.py` — needs `detection_batch_frames` and `midline_batch_crops` fields added to respective config dataclasses

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 59-batched-yolo-inference*
*Context gathered: 2026-03-04*
