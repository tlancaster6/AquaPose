---
phase: 59-batched-yolo-inference
verified: 2026-03-05T04:00:00Z
status: human_needed
score: 4/5 must-haves verified
human_verification:
  - test: "Run `aquapose eval` detection and midline metrics on a real YH chunk with batched inference and compare to pre-batching baseline"
    expected: "Metrics are identical (or within floating-point tolerance) to pre-batching baseline"
    why_human: "Requires GPU hardware, trained model weights, and real video data -- cannot verify programmatically in CI"
---

# Phase 59: Batched YOLO Inference Verification Report

**Phase Goal:** Detection and midline YOLO models receive batched inputs instead of one image at a time, increasing GPU utilization from ~30% toward its practical ceiling
**Verified:** 2026-03-05T04:00:00Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | DetectionStage.run() collects all 12 camera frames for a timestep and calls detect_batch() once per timestep instead of 12 times | VERIFIED | `stage.py` lines 98-119: iterates camera_ids, builds `ordered_items` list, then calls `predict_with_oom_retry(self._detector.detect_batch, frame_list, ...)` once per timestep. Tests `test_batched_detection_calls_detect_batch_once` and `test_batched_detection_handles_missing_cameras` confirm. |
| 2 | MidlineStage.run() collects all crops across all cameras for a frame and calls process_batch() once, correctly redistributing results by (cam_id, det_idx) | VERIFIED | `midline/stage.py` lines 222-318: collects crops/metadata per frame, calls `predict_with_oom_retry(_batch_predict, ...)` once, then redistributes via `cam_det_indices` and `failed_entries` back to per-camera dict. Tests `test_batched_run_calls_process_batch` and `test_batched_run_redistributes_correctly` confirm. |
| 3 | `detection_batch_frames` and `midline_batch_crops` config fields exist and control batch sizes | VERIFIED | `config.py` line 58: `detection_batch_frames: int = 0` on DetectionConfig. Line 129: `midline_batch_crops: int = 0` on MidlineConfig. Both documented in docstrings. `pipeline.py` lines 388 and 405 wire them through `build_stages()` to stage constructors. |
| 4 | A CUDA OOM during model.predict() triggers an automatic retry with halved batch size rather than crashing the pipeline | VERIFIED | `inference.py` lines 84-110: catches `torch.cuda.OutOfMemoryError` and RuntimeError with "CUDA out of memory", halves effective batch size, calls `torch.cuda.empty_cache()`, persists in BatchState, retries from scratch. Propagates when effective <= 1. 9 unit tests in `test_inference.py` cover all OOM scenarios. |
| 5 | `aquapose eval` detection and midline metrics on a real YH chunk are identical to the pre-batching baseline | UNCERTAIN | Cannot verify without GPU, trained weights, and real video data. Requires human testing. |

**Score:** 4/5 truths verified (1 needs human)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/core/inference.py` | BatchState + predict_with_oom_retry | VERIFIED | 111 lines, exports both. BatchState is mutable dataclass with effective_batch_size, original_batch_size, oom_occurred. predict_with_oom_retry handles chunking, OOM catch, halving, retry. |
| `tests/unit/core/test_inference.py` | Unit tests for OOM retry | VERIFIED | 203 lines, 9 test cases covering normal batching, batch_size=0, CUDA OOM halving, RuntimeError OOM, non-OOM propagation, batch_size=1 propagation, state persistence, oom_occurred flag, empty inputs. |
| `src/aquapose/core/__init__.py` | Exports BatchState, predict_with_oom_retry | VERIFIED | Line 25: imports both; lines 31,43: in __all__. |
| `src/aquapose/engine/config.py` | detection_batch_frames, midline_batch_crops fields | VERIFIED | Lines 58 and 129. Both default to 0 (no limit). Documented in class docstrings. |
| `src/aquapose/core/detection/backends/yolo_obb.py` | detect_batch() on YOLOOBBBackend | VERIFIED | Lines 88-109: detect_batch() calls model.predict with batch=len(frames). Shared _parse_results() helper (lines 111-175). |
| `src/aquapose/core/detection/backends/yolo.py` | detect_batch() on YOLOBackend | VERIFIED | Lines 188-209: detect_batch() calls model.predict with batch=len(frames). Shared _parse_box_results() module function (lines 21-60). |
| `src/aquapose/core/detection/stage.py` | Batched detection with OOM retry | VERIFIED | Lines 76-153: run() collects frames, calls predict_with_oom_retry wrapping detect_batch, maps results back to camera IDs, logs OOM recommendation. |
| `src/aquapose/core/midline/backends/segmentation.py` | process_batch() on SegmentationBackend | VERIFIED | Lines 148-225: process_batch(crops, metadata) runs batched model.predict with batch=len(crop_images), processes each result with _extract_mask and _skeletonize_and_project. |
| `src/aquapose/core/midline/backends/pose_estimation.py` | process_batch() on PoseEstimationBackend | VERIFIED | Lines 211-314: process_batch(crops, metadata) runs batched model.predict with batch=len(crop_images), processes each result with _extract_keypoints and _keypoints_to_midline. |
| `src/aquapose/core/midline/stage.py` | Batched midline extraction with OOM retry | VERIFIED | Lines 144-351: run() collects crops per frame (CPU), calls predict_with_oom_retry wrapping process_batch (GPU), redistributes results back to per-camera dict. |
| `src/aquapose/engine/pipeline.py` | Wiring through build_stages() | VERIFIED | Lines 388 and 405 pass config fields to stage constructors. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| detection/stage.py | core/inference.py | predict_with_oom_retry wrapping detect_batch | WIRED | Line 21: imports BatchState and predict_with_oom_retry. Lines 114-119: calls predict_with_oom_retry(self._detector.detect_batch, ...). |
| detection/backends/yolo_obb.py | ultralytics | model.predict(frames, batch=len(frames)) | WIRED | Lines 102-108: self._model.predict(frames, ..., batch=len(frames)). |
| detection/backends/yolo.py | ultralytics | model.predict(frames, batch=len(frames)) | WIRED | Lines 202-209: self._detector._model.predict(frames, ..., batch=len(frames)). |
| midline/stage.py | core/inference.py | predict_with_oom_retry wrapping process_batch | WIRED | Line 23: imports BatchState and predict_with_oom_retry. Lines 291-296: calls predict_with_oom_retry(_batch_predict, ...). |
| midline/backends/segmentation.py | ultralytics | model.predict(crop_images, batch=len(crop_images)) | WIRED | Lines 183-185: self._model.predict(crop_images, ..., batch=len(crop_images)). |
| midline/backends/pose_estimation.py | ultralytics | model.predict(crop_images, batch=len(crop_images)) | WIRED | Lines 246-248: self._model.predict(crop_images, ..., batch=len(crop_images)). |
| engine/pipeline.py | engine/config.py | detection_batch_frames, midline_batch_crops passed to stages | WIRED | Lines 388 and 405 read from config and pass to stage constructors. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| BATCH-01 | 59-02 | Detection stage batches all camera frames per timestep into a single predict() call | SATISFIED | DetectionStage.run() collects frames and calls detect_batch() once per timestep via predict_with_oom_retry. |
| BATCH-02 | 59-03 | Midline stage batches all crops per frame into a single predict() call | SATISFIED | MidlineStage.run() collects crops and calls process_batch() once per frame via predict_with_oom_retry. |
| BATCH-03 | 59-01 | Batch sizes are configurable via pipeline config fields | SATISFIED | detection_batch_frames and midline_batch_crops fields on frozen config dataclasses, wired through build_stages(). |
| BATCH-04 | 59-01 | Inference gracefully retries with halved batch size on CUDA OOM | SATISFIED | predict_with_oom_retry catches OOM, halves batch, retries, persists state. 9 unit tests. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | No anti-patterns found in any phase-modified files. No TODOs, FIXMEs, placeholders, empty implementations, or console.log-only handlers. |

### Human Verification Required

### 1. Eval Baseline Comparison

**Test:** Run `aquapose eval` detection and midline metrics on a real YH chunk using the batched inference path and compare against the pre-batching baseline metrics.
**Expected:** Detection and midline metrics (precision, recall, IoU, endpoint error) are identical to the pre-batching baseline, confirming batching does not alter inference results.
**Why human:** Requires GPU hardware, trained YOLO model weights, and real video data from the YH project. Cannot be verified in automated CI without these resources.

### Gaps Summary

All four automated success criteria are fully verified. The codebase correctly implements:
- Batched detection with detect_batch() called once per timestep instead of 12 serial detect() calls
- Batched midline with process_batch() called once per frame with correct crop collection and result redistribution
- Config fields wired end-to-end from YAML through pipeline to stage constructors
- OOM retry utility with comprehensive test coverage

The only unverified item is Success Criterion 5 (eval metrics identity), which requires real data and GPU hardware to confirm that batched inference produces numerically identical results to the serial path.

---

_Verified: 2026-03-05T04:00:00Z_
_Verifier: Claude (gsd-verifier)_
