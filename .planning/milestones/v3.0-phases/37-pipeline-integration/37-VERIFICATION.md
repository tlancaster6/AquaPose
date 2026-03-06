---
phase: 37-pipeline-integration
verified: 2026-03-01T23:30:00Z
status: human_needed
score: 6/6 success criteria verified
gaps:
  - truth: "Setting midline.backend: segmentation runs the full pipeline end-to-end; the MidlineStage produces binary masks per detection that feed skeletonization the same way U-Net masks did"
    status: resolved
    reason: "The SegmentationBackend implementation is complete and wired correctly. Stale docstrings were updated in commit 23996d0."
    artifacts:
      - path: "src/aquapose/core/midline/backends/__init__.py"
        issue: "Lines 4-5, 18, 20, 26, 29: docstring says 'no-op stubs ... pending Phase 37 YOLO model integration' — backends are now fully implemented"
      - path: "src/aquapose/engine/config.py"
        issue: "Lines 73-74, 82: MidlineConfig docstring says 'Both are currently no-op stubs returning midline=None pending Phase 37 YOLO model integration' — this is now false"
    missing:
      - "Update backends/__init__.py module docstring and get_backend() docstring to remove 'no-op stubs' and 'Phase 37' references"
      - "Update MidlineConfig class docstring and backend field docstring in engine/config.py to remove stub/pending language"
human_verification:
  - test: "Run the full PosePipeline with midline.backend='segmentation' and a real YOLO-seg model weights file against sample video data"
    expected: "MidlineStage produces Midline2D objects (non-None midline) for detections where mask area exceeds min_area; annotated_detections is populated in the PipelineContext"
    why_human: "Real end-to-end execution requires physical video files and trained YOLO-seg weights not available in the repo. Unit tests mock YOLO inference — true end-to-end behavior cannot be verified programmatically."
  - test: "Run the full PosePipeline with midline.backend='pose_estimation' and a real YOLO-pose model weights file against sample video data"
    expected: "MidlineStage produces Midline2D objects with point_confidence arrays for detections where >= min_observed_keypoints are above confidence_floor"
    why_human: "Same reason: requires real YOLO-pose weights and video. Spline interpolation path verified via mocked inference; true model output behavior needs human testing."
---

# Phase 37: Pipeline Integration Verification Report

**Phase Goal:** The pipeline supports `segmentation` and `pose_estimation` as selectable midline backends; running either end-to-end produces `Midline2D` objects compatible with the reconstruction stages
**Verified:** 2026-03-01T23:30:00Z
**Status:** gaps_found (stale docstrings + human verification needed for true end-to-end)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Setting `midline.backend: segmentation` in pipeline config runs end-to-end; MidlineStage produces binary masks per detection that feed skeletonization | PARTIAL | SegmentationBackend fully implemented with YOLO-seg + skeletonization + affine back-projection. build_stages passes `config.midline.backend` directly to MidlineStage. All 9 unit tests pass. Stale docstrings claim stubs. Real end-to-end needs human test. |
| 2 | Setting `midline.backend: pose_estimation` in pipeline config runs end-to-end; MidlineStage produces Midline2D with 6-keypoint coordinates resampled to n_sample_points and per-point confidence | VERIFIED | PoseEstimationBackend fully implemented with YOLO-pose + confidence filtering + spline interpolation + affine back-projection. point_confidence populated. All 15 unit tests pass. |
| 3 | Both backends produce Midline2D instances with identical shape and field structure — reconstruction stages require no backend-specific branching | VERIFIED | Both produce `Midline2D(points=(N,2), half_widths=(N,), fish_id, camera_id, frame_index, is_head_to_tail, point_confidence)`. PoseEstimationBackend uses zeros for half_widths (no distance transform); SegmentationBackend uses None for point_confidence. Structure is identical — the union of fields is compatible. Reconstruction stage uses Midline2D without branching on backend type. |

**Score:** 2/3 fully verified, 1/3 partial (docstring staleness + real e2e needs human)

---

## Plan 01 Must-Haves (PIPE-03 — Backend Renaming)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | `MidlineConfig(backend='segmentation')` constructs without error | VERIFIED | `hatch run python -c "from aquapose.engine.config import MidlineConfig; MidlineConfig(backend='segmentation')"` — confirmed, default backend is "segmentation" |
| 2 | `MidlineConfig(backend='pose_estimation')` constructs without error | VERIFIED | Same test confirmed both names valid |
| 3 | `MidlineConfig(backend='segment_then_extract')` raises ValueError | VERIFIED | Tests in `test_config.py` lines 436-441 confirm old names raise ValueError; passes in test suite |
| 4 | `get_backend('segmentation')` returns SegmentationBackend instance | VERIFIED | Registry in `backends/__init__.py` lines 41-46: imports and returns `SegmentationBackend(**kwargs)` |
| 5 | `get_backend('pose_estimation')` returns PoseEstimationBackend instance | VERIFIED | Registry lines 48-51: imports and returns `PoseEstimationBackend(**kwargs)` |
| 6 | `build_stages` passes `backend='segmentation'` to MidlineStage by default | VERIFIED | `pipeline.py` line 330: `backend=config.midline.backend`; `MidlineConfig.backend` defaults to `"segmentation"` |

---

## Plan 02 Must-Haves (PIPE-01, PIPE-02 — Real Inference)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | SegmentationBackend loads a YOLO-seg model and produces binary masks from OBB crops | VERIFIED | `segmentation.py`: `YOLO(str(weights_path))` loaded in `__init__`; `_extract_mask()` calls `model.predict()`, extracts `masks.data[0]`, converts with `.cpu().numpy()`, resizes with `cv2.INTER_NEAREST` |
| 2 | SegmentationBackend skeletonizes masks and produces Midline2D in full-frame coordinates | VERIFIED | `_skeletonize_and_project()` calls `_adaptive_smooth`, `_skeleton_and_widths`, `_longest_path_bfs`, `_resample_arc_length` from `reconstruction.midline`; back-projects via `invert_affine_points` |
| 3 | SegmentationBackend returns midline=None when mask area is below threshold or skeletonization fails | VERIFIED | Lines 197-201: area check `np.count_nonzero(mask_np) < self._min_area`; empty BFS path check `if not path_yx: return None`; skeleton length check vs `n_points` |
| 4 | PoseEstimationBackend loads a YOLO-pose model and extracts keypoint coordinates from OBB crops | VERIFIED | `pose_estimation.py`: `YOLO(str(weights_path))` in `__init__`; `_extract_keypoints()` accesses `res.keypoints.xy[0].cpu().numpy()` (pixel coords, not normalized) |
| 5 | PoseEstimationBackend fits a spline through visible keypoints and produces Midline2D in full-frame coordinates | VERIFIED | `_keypoints_to_midline()` uses `scipy.interpolate.interp1d` with `kind="linear"`, `fill_value="extrapolate"`; back-projects via `invert_affine_points` |
| 6 | PoseEstimationBackend returns midline=None when fewer than 3 keypoints are visible | VERIFIED | Lines 258-262: `visible_mask = kpts_conf >= self._confidence_floor`; `if n_visible < self._min_observed_keypoints: return _null` |
| 7 | Both backends produce AnnotatedDetection objects with identical structure | VERIFIED | Both return `AnnotatedDetection(detection=det, mask=..., crop_region=None, midline=..., camera_id=cam_id, frame_index=frame_idx)`. Structure is identical. |

---

## Required Artifacts

| Artifact | Status | Details |
|---------|--------|---------|
| `src/aquapose/core/midline/backends/segmentation.py` | VERIFIED (336 lines) | SegmentationBackend with full YOLO-seg pipeline. Exports `SegmentationBackend`. |
| `src/aquapose/core/midline/backends/pose_estimation.py` | VERIFIED (371 lines) | PoseEstimationBackend with full YOLO-pose pipeline. Exports `PoseEstimationBackend` and `_keypoints_to_midline`. |
| `src/aquapose/core/midline/backends/__init__.py` | VERIFIED (57 lines) | Registry resolves `"segmentation"` and `"pose_estimation"` correctly. Docstring is stale (says no-op stubs). |
| `src/aquapose/core/midline/stage.py` | VERIFIED (580 lines) | Default `backend="segmentation"`. Branches on `"pose_estimation"` to pass pose-specific kwargs. Uses `get_backend()`. |
| `src/aquapose/engine/config.py` | VERIFIED | `MidlineConfig._valid_backends = {"segmentation", "pose_estimation"}`. Default `backend="segmentation"`. `keypoint_confidence_floor=0.3`. Docstring stale. |
| `src/aquapose/engine/pipeline.py` | VERIFIED | `build_stages` passes `backend=config.midline.backend` at line 330. Fully wired. |
| `tests/unit/core/midline/test_segmentation_backend.py` | VERIFIED (335 lines, 9 tests all passing) | Covers instantiation, no-model path, mocked inference, area threshold, angle=None, import boundary. |
| `tests/unit/core/midline/test_pose_estimation_backend.py` | VERIFIED (423 lines, 15 tests all passing) | Covers instantiation, _keypoints_to_midline, no-model path, mocked inference, confidence filtering, angle=None, import boundary. |
| `src/aquapose/core/midline/backends/segment_then_extract.py` | DELETED (expected) | File absent — confirmed |
| `src/aquapose/core/midline/backends/direct_pose.py` | DELETED (expected) | File absent — confirmed |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `segmentation.py` | `aquapose.reconstruction.midline` | imports `_adaptive_smooth, _skeleton_and_widths, _longest_path_bfs, _resample_arc_length, Midline2D` | WIRED | Lines 17-23: confirmed all 5 names imported directly |
| `segmentation.py` | `aquapose.segmentation.crop` | imports `extract_affine_crop, invert_affine_points` | WIRED | Line 24: confirmed |
| `pose_estimation.py` | `aquapose.segmentation.crop` | imports `extract_affine_crop, invert_affine_points` | WIRED | Line 18: confirmed |
| `pose_estimation.py` | `scipy.interpolate` | `interp1d` for spline resampling | WIRED | Line 14: `from scipy.interpolate import interp1d`; used at lines 48, 54, 60 |
| `engine/config.py` | `core/midline/backends/__init__.py` | backend string validation matches registry | WIRED | Both use `{"segmentation", "pose_estimation"}`; config rejects unknown names before registry is reached |
| `core/midline/stage.py` | `core/midline/backends/__init__.py` | `get_backend()` call with new backend names | WIRED | Lines 157, 173: `get_backend("pose_estimation", ...)` and `get_backend(backend, ...)` |
| `engine/pipeline.py` | `core/midline/stage.py` | `build_stages` passes `backend=config.midline.backend` | WIRED | Line 330: `backend=config.midline.backend` confirmed |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| PIPE-01 | 37-02-PLAN.md | YOLOSegBackend produces binary masks per detection for midline extraction via skeletonization | SATISFIED | `SegmentationBackend._extract_mask()` + `_skeletonize_and_project()` implement the full pipeline. 9 unit tests verify behavior. Note: requirement uses old name "YOLOSegBackend"; implemented as `SegmentationBackend` per plan renaming decision. |
| PIPE-02 | 37-02-PLAN.md | YOLOPoseBackend produces keypoint coordinates with per-point confidence for direct midline construction | SATISFIED | `PoseEstimationBackend._extract_keypoints()` + `_keypoints_to_midline()` implement confidence-filtered spline interpolation with `point_confidence` in output Midline2D. 15 unit tests verify behavior. Note: requirement uses old name "YOLOPoseBackend". |
| PIPE-03 | 37-01-PLAN.md | Config system supports backend selection (yolo_seg, yolo_pose) via midline.backend field | SATISFIED | `MidlineConfig.backend` accepts `"segmentation"` and `"pose_estimation"` (renamed per plan decision from `yolo_seg`/`yolo_pose`). Old names raise ValueError. Validated via `hatch run python` + 6 tests in `test_config.py`. Note: requirement uses old candidate names; actual names differ by plan decision. |

**Note on requirement name mismatches:** REQUIREMENTS.md was authored before the Plan 01 renaming decision and uses "YOLOSegBackend"/"YOLOPoseBackend" and "yolo_seg"/"yolo_pose". The plans explicitly renamed these to "segmentation"/"pose_estimation" (documented in 37-01-PLAN.md and CONTEXT.md). The substance of all three requirements is fully satisfied — only the candidate names differ.

---

## Anti-Patterns Found

| File | Lines | Pattern | Severity | Impact |
|------|-------|---------|----------|--------|
| `src/aquapose/core/midline/backends/__init__.py` | 4-5, 18, 20, 26, 29 | "no-op stubs ... pending Phase 37 YOLO model integration" | Warning | Misleading — backends are now fully implemented. Will confuse future developers reading the registry. |
| `src/aquapose/engine/config.py` | 73-74, 82 | "Both are currently no-op stubs returning midline=None pending Phase 37 YOLO model integration" | Warning | Same issue — MidlineConfig docstring says backends are stubs when they are not. |

No blocker anti-patterns found. No placeholder returns, no TODO/FIXME in implementation paths, no stub `return {}` / `return None` in the actual inference logic.

---

## Human Verification Required

### 1. SegmentationBackend end-to-end with real model

**Test:** Configure PipelineConfig with `midline.backend = "segmentation"` and point `midline.weights_path` to a trained YOLO-seg model. Run PosePipeline on sample video. Inspect `context.annotated_detections` after MidlineStage completes.
**Expected:** Detections with fish present produce non-None midlines; midline.points has shape `(n_points, 2)` in full-frame pixel coordinates; midline.half_widths has shape `(n_points,)` with physically reasonable values (0-20px for fish width).
**Why human:** Real YOLO-seg model weights are not committed to the repo. Unit tests mock model.predict() — actual skeletonization quality from real mask tensors cannot be verified programmatically.

### 2. PoseEstimationBackend end-to-end with real model

**Test:** Configure PipelineConfig with `midline.backend = "pose_estimation"` and point `midline.keypoint_weights_path` to a trained YOLO-pose model. Run PosePipeline on sample video. Inspect `context.annotated_detections`.
**Expected:** Detections produce Midline2D with `points.shape == (n_points, 2)`, `point_confidence.shape == (n_points,)` with values in [0, 1], and `half_widths` are zeros.
**Why human:** Same reason — requires real YOLO-pose weights. Also validates that `.cpu().numpy()` works correctly on actual CUDA tensors from model inference.

---

## Gaps Summary

The phase is functionally complete — both backends are implemented with real YOLO inference, full coordinate back-projection, and comprehensive unit tests (24 passing). The pipeline wiring is correct end-to-end: `load_config` → `MidlineConfig.backend` validation → `build_stages` → `MidlineStage(backend=...)` → `get_backend(...)` → `{Segmentation|PoseEstimation}Backend.process_frame()` → `Midline2D`.

The single gap is documentation staleness: `backends/__init__.py` and `engine/config.py` still describe the backends as "no-op stubs pending Phase 37 YOLO model integration." This is false — Phase 37 is complete. This is a Warning-severity issue (not a blocker) but should be corrected to avoid confusion.

Two items require human verification with real model weights, as unit tests mock the YOLO inference layer.

---

*Verified: 2026-03-01T23:30:00Z*
*Verifier: Claude (gsd-verifier)*
