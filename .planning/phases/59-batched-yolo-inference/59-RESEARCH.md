# Phase 59: Batched YOLO Inference - Research

**Researched:** 2026-03-04
**Domain:** Ultralytics YOLO batched inference, CUDA OOM handling, pipeline stage refactoring
**Confidence:** HIGH

## Summary

This phase replaces per-camera/per-crop serial `model.predict()` calls with batched inference in both the detection and midline stages. The Ultralytics YOLO API natively supports batched inference via `model.predict(list_of_ndarrays)`, returning results in input order. The detection stage currently loops 12 cameras calling `detect()` once per camera; the midline stage loops cameras x detections calling `_process_detection()` once per crop. Both can be restructured to collect inputs, call `predict()` once with a list, and redistribute results.

The key architectural insight is that crop preparation (CPU) and post-processing (CPU) must be separated from the GPU predict call. The stage collects all inputs, calls the backend's new batch method once, and maps results back by positional correspondence. A shared OOM retry utility halves the batch on CUDA OOM, persists the reduced size for the rest of the run, and logs a config recommendation at shutdown.

**Primary recommendation:** Add `detect_batch()` to detection backends and `process_batch()` to midline backends. Stages collect inputs, call batch methods, redistribute results. Shared `_retry_with_oom_halving()` utility in `core/inference.py`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Detection: all 12 camera frames for a timestep go into a single `predict()` call (no sub-batching)
- Midline: all crops across all cameras for a single frame go into one `predict()` call
- Both midline backends (segmentation + pose_estimation) get batched -- not just the active one
- Crop preparation (extract_affine_crop) is done in bulk (CPU) before the single batch `predict()` call (GPU)
- On CUDA OOM, halve batch size and retry; keep halving down to batch_size=1, then fail with the real exception
- Warn (logger.warning) at each halving step
- Once a reduced batch size succeeds, persist it for the rest of the run (no re-attempting full batch each frame)
- Catch both `torch.cuda.OutOfMemoryError` and `RuntimeError` with 'CUDA out of memory' in message
- Shared utility function used by both detection and midline stages (DRY)
- At end of run, if batch size was reduced, log a recommendation for the user's config
- Stage collects images/crops, then calls new `detect_batch()` / `process_batch()` on the backend
- Existing single-image methods (`detect()`, `process_frame()`) are kept alongside new batch methods
- `detect_batch(frames: list[ndarray])` takes a list of ndarrays -- backend doesn't see camera IDs
- `process_batch(crops: list[...])` takes a list of crops, returns results in positional correspondence (same order)
- Stage handles mapping results back to `(cam_id, det_idx)` by tracking input order
- `detection_batch_frames` lives on DetectionConfig (per-stage config pattern)
- `midline_batch_crops` lives on MidlineConfig (per-stage config pattern)
- Default value: 0, meaning "no limit -- batch everything"
- No config toggle for OOM retry -- always on; batch_size=1 surfaces the real exception if needed
- End-of-run log recommendation if OOM retry reduced the batch size

### Claude's Discretion
No specific areas marked for Claude's discretion.

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BATCH-01 | Detection stage batches all camera frames per timestep into a single `predict()` call | Ultralytics `model.predict(list[ndarray])` natively supports this; `DetectionStage.run()` inner loop at stage.py:80-83 needs refactoring to collect frames then call `detect_batch()` |
| BATCH-02 | Midline stage batches all crops per frame into a single `predict()` call | Both `SegmentationBackend` and `PoseEstimationBackend` loop cameras x detections calling `_process_detection()` one at a time; need `process_batch()` that separates crop extraction from predict |
| BATCH-03 | Batch sizes are configurable via pipeline config fields | `DetectionConfig` and `MidlineConfig` are frozen dataclasses in `engine/config.py`; add `detection_batch_frames: int = 0` and `midline_batch_crops: int = 0` with `_filter_fields` validation |
| BATCH-04 | Inference gracefully retries with halved batch size on CUDA OOM | Shared utility in `core/inference.py` catches `torch.cuda.OutOfMemoryError` and `RuntimeError` with OOM message; calls `torch.cuda.empty_cache()` before retry |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ultralytics | 8.4.19 (installed) | YOLO model inference | Already the project's inference engine; `model.predict()` natively accepts `list[ndarray]` for batching |
| torch | (project dep) | CUDA OOM handling | `torch.cuda.OutOfMemoryError` is the exception class; `torch.cuda.empty_cache()` for memory cleanup |

### Supporting
No new libraries needed. All batch inference is handled by existing ultralytics/torch APIs.

## Architecture Patterns

### Recommended Project Structure
```
src/aquapose/
├── core/
│   ├── inference.py              # NEW: shared OOM retry utility
│   ├── detection/
│   │   ├── backends/
│   │   │   ├── yolo_obb.py       # MODIFY: add detect_batch()
│   │   │   └── yolo.py           # MODIFY: add detect_batch()
│   │   └── stage.py              # MODIFY: collect frames, call detect_batch()
│   └── midline/
│       ├── backends/
│       │   ├── segmentation.py   # MODIFY: add process_batch()
│       │   └── pose_estimation.py # MODIFY: add process_batch()
│       └── stage.py              # MODIFY: collect crops, call process_batch()
├── engine/
│   └── config.py                 # MODIFY: add batch size fields
```

### Pattern 1: Batch Method on Backend
**What:** Each backend gets a new batch method alongside the existing single-item method.
**When to use:** Always -- the batch method is the primary inference path; single-item methods remain for backward compatibility.
**Example:**
```python
# In YOLOOBBBackend
def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
    """Detect fish in multiple frames via a single batched predict call.

    Args:
        frames: List of BGR images as uint8 arrays of shape (H, W, 3).

    Returns:
        List of detection lists, one per input frame, in positional
        correspondence with the input.
    """
    results = self._model.predict(
        frames, conf=self._conf, iou=self._iou, verbose=False, batch=len(frames)
    )
    return [self._parse_result(r) for r in results]
```

### Pattern 2: Stage Collects Then Dispatches
**What:** Stage collects all inputs for a timestep/frame into a flat list with provenance tracking, calls the backend batch method once, then redistributes results.
**When to use:** Both detection and midline stages.
**Example:**
```python
# In DetectionStage.run() -- detection batching
for _frame_idx, frames in self._frame_source:
    # Collect all camera frames for this timestep
    frame_list = [frames[cam_id] for cam_id in camera_ids]

    # Single batch predict call
    batch_results = self._detector.detect_batch(frame_list)

    # Redistribute by camera ID
    frame_dets = {
        cam_id: batch_results[i]
        for i, cam_id in enumerate(camera_ids)
    }
    detections_per_frame.append(frame_dets)
```

### Pattern 3: OOM Retry with Batch Halving
**What:** A shared utility function wraps `model.predict()` calls with automatic OOM recovery.
**When to use:** Both detection and midline batch predict calls.
**Example:**
```python
# In core/inference.py
def predict_with_oom_retry(
    predict_fn: Callable[[list], list],
    inputs: list,
    batch_size: int,  # 0 means no limit
    batch_state: BatchState,
) -> list:
    """Call predict_fn on inputs with automatic OOM retry.

    On CUDA OOM, halves effective batch size, clears cache, retries.
    Persists reduced batch_size in batch_state for subsequent calls.
    """
    effective_batch = batch_state.effective_batch_size or batch_size or len(inputs)

    while effective_batch >= 1:
        try:
            results = []
            for i in range(0, len(inputs), effective_batch):
                chunk = inputs[i:i + effective_batch]
                results.extend(predict_fn(chunk))
            return results
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "CUDA out of memory" not in str(e):
                raise
            if effective_batch <= 1:
                raise
            torch.cuda.empty_cache()
            effective_batch = effective_batch // 2
            batch_state.effective_batch_size = effective_batch
            logger.warning(
                "CUDA OOM: halving batch size to %d", effective_batch
            )
    return []  # unreachable
```

### Pattern 4: Midline Crop-Predict Separation
**What:** Midline `process_batch()` separates CPU crop extraction from GPU predict. The stage prepares all crops in a loop, hands them to `process_batch()` which calls `model.predict(list_of_crop_images)` once, then each result is post-processed individually.
**When to use:** Both segmentation and pose_estimation backends.
**Key insight:** The current `_process_detection()` interleaves crop extraction and predict. The batch version must split these phases:
1. **Collect phase (CPU):** Extract all affine crops, tracking metadata (det, frame, cam_id, AffineCrop.M)
2. **Predict phase (GPU):** Single `model.predict([crop.image for crop in crops])`
3. **Post-process phase (CPU):** Match each result to its metadata, run skeletonize/interpolate

### Anti-Patterns to Avoid
- **Passing camera IDs to backend:** Backend should not know about camera topology. Stage handles the mapping.
- **Re-attempting full batch after a reduced batch succeeds:** Once OOM triggers a halving, persist that reduced size for the rest of the run.
- **Modifying existing single-image methods:** Keep `detect()` and `process_frame()` as-is for backward compatibility and tests.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Batched YOLO inference | Custom tensor stacking + model.forward() | `model.predict(list[ndarray], batch=N)` | Ultralytics handles preprocessing, resizing, padding, NMS internally |
| CUDA memory cleanup | Manual tensor deallocation | `torch.cuda.empty_cache()` | PyTorch's cache allocator handles the details |
| Image preprocessing for batched predict | Manual letterboxing/padding | Ultralytics internal `pre_transform` | Ultralytics automatically handles same-resolution rect padding vs square padding |

**Key insight:** Ultralytics `model.predict()` already handles all the complexity of batching different-resolution images. Since all 12 camera frames have the same resolution (same rig), rect padding will be used automatically which is optimal.

## Common Pitfalls

### Pitfall 1: Result Ordering Assumption
**What goes wrong:** Assuming `model.predict(list)` returns results in a different order than input.
**Why it happens:** Documentation doesn't explicitly state ordering guarantee.
**How to avoid:** The Ultralytics implementation constructs Results by zipping predictions with original images in order. Verified via source code analysis. Results maintain input order.
**Warning signs:** Detections appearing on wrong cameras in output.

### Pitfall 2: Batch Parameter vs List Length
**What goes wrong:** Passing a list of 12 images but `batch=1` (default), resulting in 12 separate forward passes instead of 1.
**Why it happens:** Ultralytics default `batch=1` for predict mode. Passing a list of images is treated as a dataset, batched according to the `batch` parameter.
**How to avoid:** Explicitly pass `batch=len(frames)` when calling `model.predict()` to ensure all images go through in a single forward pass. Or use `batch=N` where N is the configured batch size.
**Warning signs:** No GPU utilization improvement despite batching code changes.

### Pitfall 3: CUDA OOM on Midline Crops
**What goes wrong:** Midline batches can be large (12 cameras x N fish per camera = potentially 60-100+ crops per frame).
**Why it happens:** Unlike detection (always 12 frames), midline crop count scales with fish count.
**How to avoid:** The OOM retry with halving handles this automatically. Config `midline_batch_crops` allows users to cap batch size proactively.
**Warning signs:** First frame processes fine but later frames with more detections OOM.

### Pitfall 4: Segmentation Mask Index Mismatch
**What goes wrong:** When `model.predict()` receives a batch of N crop images, each result may contain multiple masks (multiple fish detected in one crop). Current code takes `results[0].masks.data[0]` -- the first mask from the first result.
**Why it happens:** In batched mode, `results` is a list of N Results objects (one per input image). Each Result's `.masks.data[0]` is still correct -- it's the first mask for that specific crop.
**How to avoid:** Iterate `results` (one per crop), not `results[0]` for all crops. Current `_extract_mask` takes a Results list and uses `results[0]` -- in batch mode, pass each individual Result wrapped in a list, or refactor to accept a single Result.
**Warning signs:** All crops returning the same mask or None.

### Pitfall 5: Frozen Dataclass Field Addition
**What goes wrong:** Adding a new field to a frozen dataclass with a mutable default (list, dict).
**Why it happens:** `@dataclass(frozen=True)` requires `field(default_factory=...)` for mutable defaults.
**How to avoid:** `detection_batch_frames: int = 0` is fine (immutable default). Just add the field and update `_filter_fields` will auto-validate.
**Warning signs:** `TypeError` at config construction time.

### Pitfall 6: torch.cuda.empty_cache() Missing Before Retry
**What goes wrong:** Retry with smaller batch fails because fragmented GPU memory from the failed allocation is not released.
**Why it happens:** PyTorch's caching allocator holds failed allocation fragments.
**How to avoid:** Always call `torch.cuda.empty_cache()` between the caught OOM exception and the retry attempt.
**Warning signs:** Even batch_size=1 fails after an OOM.

## Code Examples

### Detection Backend: detect_batch()
```python
# Source: verified against ultralytics 8.4.19 predict API
def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
    """Detect fish in multiple frames via a single batched predict call.

    Results are returned in positional correspondence with input frames.
    """
    if not frames:
        return []

    results = self._model.predict(
        frames, conf=self._conf, iou=self._iou, verbose=False,
        batch=len(frames),
    )

    all_detections: list[list[Detection]] = []
    for r in results:
        detections = []
        if r.obb is not None:
            xywhr = r.obb.xywhr.cpu().numpy()
            corners_all = r.obb.xyxyxyxy.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()
            for i in range(len(xywhr)):
                # ... same parsing as existing detect() ...
                pass
        all_detections.append(detections)

    return all_detections
```

### Midline Backend: process_batch() Structure
```python
# Source: project codebase patterns
def process_batch(
    self,
    crops: list[AffineCrop],
    metadata: list[tuple[str, int, Detection]],  # (cam_id, frame_idx, det)
) -> list[AnnotatedDetection]:
    """Process multiple crops in a single batched predict call.

    Args:
        crops: Pre-extracted affine crops.
        metadata: Per-crop (cam_id, frame_idx, detection) tuples.

    Returns:
        AnnotatedDetection list in positional correspondence with inputs.
    """
    if not crops or self._model is None:
        return [self._null_result(m) for m in metadata]

    crop_images = [c.image for c in crops]
    results = self._model.predict(
        crop_images, conf=self._conf, verbose=False,
        batch=len(crop_images),
    )

    annotated = []
    for crop, result, (cam_id, frame_idx, det) in zip(crops, results, metadata):
        ann = self._postprocess_single(result, crop, cam_id, frame_idx, det)
        annotated.append(ann)
    return annotated
```

### OOM Retry Utility
```python
# Source: pattern synthesis from torch docs + project conventions
import logging
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")

@dataclass
class BatchState:
    """Mutable state tracking effective batch size across a run."""
    effective_batch_size: int | None = None
    original_batch_size: int | None = None
    oom_occurred: bool = False

def predict_with_oom_retry(
    predict_fn: Callable[[list], list],
    inputs: list,
    max_batch_size: int,
    state: BatchState,
) -> list:
    """Execute predict_fn with automatic OOM retry and batch halving."""
    if not inputs:
        return []

    effective = state.effective_batch_size or max_batch_size or len(inputs)
    if state.original_batch_size is None:
        state.original_batch_size = effective

    while effective >= 1:
        try:
            results = []
            for i in range(0, len(inputs), effective):
                chunk = inputs[i:i + effective]
                results.extend(predict_fn(chunk))
            return results
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "CUDA out of memory" not in str(e):
                raise
            if effective <= 1:
                raise
            torch.cuda.empty_cache()
            effective //= 2
            state.effective_batch_size = effective
            state.oom_occurred = True
            logger.warning(
                "CUDA OOM during inference: halving batch size to %d",
                effective,
            )
    return []
```

### Config Field Addition
```python
# In engine/config.py DetectionConfig
@dataclass(frozen=True)
class DetectionConfig:
    detector_kind: str = "yolo"
    weights_path: str | None = None
    crop_size: list[int] = field(default_factory=lambda: [128, 64])
    detection_batch_frames: int = 0  # NEW: 0 = no limit
    extra: dict[str, Any] = field(default_factory=dict)

# In engine/config.py MidlineConfig
@dataclass(frozen=True)
class MidlineConfig:
    # ... existing fields ...
    midline_batch_crops: int = 0  # NEW: 0 = no limit
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-image `model.predict()` | `model.predict(list, batch=N)` | Always available in ultralytics | Single GPU forward pass for N images |
| `RuntimeError` only for OOM | `torch.cuda.OutOfMemoryError` added | PyTorch 2.0 (2023) | Dedicated exception class, cleaner catching |

**Deprecated/outdated:**
- None relevant to this phase.

## Open Questions

1. **Ultralytics batch parameter with list input**
   - What we know: `model.predict(list_of_images)` works. The `batch` parameter controls how many images are sent through the model in a single forward pass.
   - What's unclear: Whether passing `batch=len(frames)` is strictly necessary when input is already a list, or if ultralytics auto-batches the list. The default `batch=1` would process sequentially despite receiving a list.
   - Recommendation: Explicitly pass `batch=len(frames)` to guarantee true batched GPU inference. This is the conservative correct approach.

2. **Mixed-resolution images in detection batching**
   - What we know: All 12 cameras in the AquaPose rig produce same-resolution frames. Ultralytics uses rect padding when all images in a batch are same-size (more efficient), square padding otherwise.
   - What's unclear: Whether undistortion changes resolution across cameras.
   - Recommendation: Same-resolution assumption is safe for this rig. No special handling needed.

## Sources

### Primary (HIGH confidence)
- Ultralytics predict documentation: https://docs.ultralytics.com/modes/predict/
- Ultralytics predictor source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py
- Project codebase: `core/detection/backends/yolo_obb.py`, `core/detection/stage.py`, `core/midline/backends/segmentation.py`, `core/midline/backends/pose_estimation.py`, `core/midline/stage.py`, `engine/config.py`

### Secondary (MEDIUM confidence)
- Ultralytics batch inference blog: https://www.ultralytics.com/blog/using-ultralytics-yolo11-to-run-batch-inferences
- Ultralytics GitHub issues on batch processing: https://github.com/ultralytics/ultralytics/issues/1310

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - ultralytics 8.4.19 installed, API verified via docs and source
- Architecture: HIGH - existing codebase patterns are clear, refactoring path is straightforward
- Pitfalls: HIGH - batch parameter default, OOM handling, result ordering all verified
- OOM retry pattern: MEDIUM - standard PyTorch pattern, but exact `torch.cuda.empty_cache()` interaction with ultralytics internal state not fully verified

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable -- ultralytics API unlikely to change for installed version)
