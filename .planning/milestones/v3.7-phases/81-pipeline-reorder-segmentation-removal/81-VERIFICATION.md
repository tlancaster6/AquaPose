---
phase: 81-pipeline-reorder-segmentation-removal
verified: 2026-03-11T00:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 81: Pipeline Reorder / Segmentation Removal Verification Report

**Phase Goal:** Pose estimation runs immediately after detection (before tracking), and the segmentation midline backend is fully removed from the codebase
**Verified:** 2026-03-11
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|---------|
| 1  | PoseStage runs as Stage 2 (after Detection, before Tracking) in build_stages() | VERIFIED | `build_stages()` constructs list `[detection_stage, pose_stage, tracking_stage, AssociationStage, reconstruction_stage]`; `_STAGE_OUTPUT_FIELDS` has `PoseStage` at expected position; confirmed in pipeline.py lines 431-439 |
| 2  | PoseStage enriches Detection objects in-place with raw 6-keypoint data (no upsampling to 15 points) | VERIFIED | `stage.py` lines 185-186: `det.keypoints = kpts_xy` / `det.keypoint_conf = kpts_conf`; no `_keypoints_to_midline` call in stage.py; docstring explicitly states "Does not upsample to 15-point midlines" |
| 3  | PipelineContext has no annotated_detections field | VERIFIED | Confirmed via runtime import: `PipelineContext` fields are `['frame_count', 'camera_ids', 'detections', 'tracks_2d', 'tracklet_groups', 'midlines_3d', 'stage_timing', 'carry_forward']` |
| 4  | Pipeline config uses 'pose' key everywhere, not 'midline' | VERIFIED | `PoseConfig` class exists; `config.pose.weights_path` used in `_check_model_weights()`; CLI `stop_after` choices are `["detection", "pose", "tracking", "association"]`; 'midline' accepted as deprecated alias only |
| 5  | core/pose/ directory exists with PoseStage class, core/midline/ directory no longer exists | VERIFIED | `src/aquapose/core/pose/` contains `stage.py`, `crop.py`, `backends/`, `types.py`, `__init__.py`; `src/aquapose/core/midline/` does not exist |
| 6  | ReconstructionStage reads keypoints from context.detections (not annotated_detections) and interpolates 6->15 points | VERIFIED | `reconstruction/stage.py` line 228: guards on `context.detections is None`; line 284: `det.keypoints is not None` check; line 290: calls `_keypoints_to_midline(det.keypoints, ...)` — function defined at line 48 |
| 7  | backends/segmentation.py, orientation.py, and midline.py deleted | VERIFIED | All three paths confirmed absent: `src/aquapose/core/pose/backends/segmentation.py` DELETED; `src/aquapose/core/pose/orientation.py` DELETED; `src/aquapose/core/pose/midline.py` DELETED |
| 8  | All unit tests pass | VERIFIED | `hatch run test`: 1105 passed, 3 skipped, 14 deselected, 20 warnings |
| 9  | No imports of MidlineStage, AnnotatedDetection, or annotated_detections in production code | VERIFIED | No class imports found; remaining references in `tuning.py`/`runner.py`/`synthetic.py`/`diagnostic_observer.py` are in comments, docstrings, or backward-compat `getattr(ctx, "annotated_detections", None)` guards — not functional imports of the removed class |

**Score:** 9/9 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/core/pose/stage.py` | PoseStage class with in-place Detection enrichment | VERIFIED | 205 lines; `PoseStage.run()` writes `det.keypoints`/`det.keypoint_conf` in-place; `__all__ = ["PoseStage"]` |
| `src/aquapose/core/types/detection.py` | Detection dataclass with keypoints and keypoint_conf fields | VERIFIED | Runtime confirmed: fields include `keypoints` and `keypoint_conf` |
| `src/aquapose/core/context.py` | PipelineContext without annotated_detections | VERIFIED | Runtime confirmed: 8 fields, no `annotated_detections` |
| `src/aquapose/engine/pipeline.py` | build_stages() with Detection->Pose->Tracking->Association->Reconstruction order | VERIFIED | Stage list at lines 431-439; `_STAGE_OUTPUT_FIELDS` keys: `['DetectionStage', 'SyntheticDataStage', 'PoseStage', 'TrackingStage', 'AssociationStage', 'ReconstructionStage']` — no `MidlineStage` |
| `src/aquapose/core/reconstruction/stage.py` | Updated reconstruction reading keypoints from detections with 6->15 interpolation | VERIFIED | `_keypoints_to_midline` defined at line 48; called from `_run_with_tracklet_groups()` |
| `tests/unit/core/pose/test_pose_stage.py` | Tests for PoseStage in new location | VERIFIED | File exists in `tests/unit/core/pose/`; tests verify `no annotated_detections field` and in-place keypoint enrichment |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/aquapose/engine/pipeline.py` | `src/aquapose/core/pose/stage.py` | `build_stages()` constructs PoseStage at position 1 | WIRED | `PoseStage` imported from `aquapose.core`; `pose_stage` constructed and inserted at index 1 in the stage list |
| `src/aquapose/core/pose/stage.py` | `src/aquapose/core/types/detection.py` | PoseStage writes det.keypoints and det.keypoint_conf on each Detection | WIRED | Lines 185-186: `det.keypoints = kpts_xy` and `det.keypoint_conf = kpts_conf` inside the batch results loop |
| `src/aquapose/core/reconstruction/stage.py` | `src/aquapose/core/types/detection.py` | Reads det.keypoints from context.detections and interpolates to Midline2D | WIRED | Lines 284-291: `det.keypoints is not None` guard, `_keypoints_to_midline(det.keypoints, ...)` call |
| `src/aquapose/evaluation/tuning.py` | `src/aquapose/core/context.py` | Reads context.detections instead of context.annotated_detections | WIRED | Dual-path at line 140: v3.7 path reads `ctx.detections`; backward-compat fallback uses `getattr(ctx, "annotated_detections", None)` for legacy runs |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| PIPE-01 | 81-01-PLAN | Pose estimation runs immediately after detection (Stage 2), before tracking | SATISFIED | `build_stages()` places `pose_stage` at index 1 between DetectionStage and TrackingStage |
| PIPE-02 | 81-02-PLAN | Segmentation midline backend removed (backends/segmentation.py, skeletonization code, orientation resolution) | SATISFIED | All three files deleted: `segmentation.py`, `orientation.py`, `midline.py`; corresponding tests deleted |
| PIPE-03 | 81-01-PLAN, 81-02-PLAN | PipelineContext and stage interfaces updated for new stage ordering | SATISFIED | `annotated_detections` removed from PipelineContext; `_STAGE_OUTPUT_FIELDS` updated to PoseStage; all downstream consumers (DiagnosticObserver, tuning.py, runner.py, overlay.py) updated |

No orphaned requirements found — all three PIPE requirements are mapped in plan frontmatter and verified in the codebase.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/unit/core/midline/__init__.py` | 1 | Empty test directory with stale docstring "Unit tests for aquapose.core.midline (Stage 2)" left in repo | Info | No functional impact; test files were moved to `tests/unit/core/pose/` but the empty directory was not cleaned up |
| `tests/unit/engine/test_timing.py` | 29, 57, 70 | `"MidlineStage"` used as a string in test data (not an import) | Info | TimingObserver tests pass any string as stage name — these are testing observer mechanics, not pipeline stage names. Tests pass. No functional impact. |
| `tests/unit/engine/test_console_observer.py` | 38, 47 | `"MidlineStage"` string in test data | Info | Same pattern as above — testing ConsoleObserver output format with arbitrary string. No functional impact. |
| `tests/unit/evaluation/test_viz.py` | 36 | `ctx.annotated_detections = annotated_detections or [None] * frame_count` — dynamic attribute assignment to test legacy context loading | Warning | PipelineContext is not frozen; dynamic assignment works at runtime. This test exercises backward-compat code path for reading legacy cached contexts. No impact on phase goal. |

No blocker anti-patterns found.

---

## Human Verification Required

None. All phase goals are verifiable through static analysis and the test suite.

---

## Gaps Summary

No gaps. All 9 must-haves verified. Three observations noted:

1. **Empty `tests/unit/core/midline/` directory**: The test files inside were moved to `tests/unit/core/pose/` but the empty `tests/unit/core/midline/` package directory (containing only `__init__.py` with a stale docstring) was not removed. This is cosmetic only — no tests live there.

2. **"MidlineStage" string in timing/console observer tests**: These test files use the string `"MidlineStage"` as test data for observer mechanics tests. They are not imports of the removed class and do not prevent the goal. They represent incomplete cleanup of test data strings but do not affect any passing tests.

3. **Backward-compat `annotated_detections` paths in tuning.py and runner.py**: The dual-path pattern using `getattr(ctx, "annotated_detections", None)` is an intentional decision documented in 81-02-SUMMARY to support re-reading old diagnostic cache files that predate v3.7. This is correct behavior.

---

_Verified: 2026-03-11_
_Verifier: Claude (gsd-verifier)_
