---
phase: 82-association-upgrade-keypoint-centroid
plan: "01"
subsystem: tracking
tags: [tracking, association, keypoint-centroid, ocsort, config]
dependency_graph:
  requires: []
  provides: [keypoint-centroid-tracklets]
  affects: [association-stage, ocsort-tracker, tracking-stage]
tech_stack:
  added: []
  patterns:
    - Keypoint centroid with OBB fallback in _TrackletBuilder.add_frame
    - Config threading via AssociationConfig -> TrackingStage -> OcSortTracker -> _TrackletBuilder
key_files:
  created:
    - .planning/phases/82-association-upgrade-keypoint-centroid/82-NOTES.md
  modified:
    - src/aquapose/engine/config.py
    - src/aquapose/core/tracking/ocsort_wrapper.py
    - src/aquapose/core/tracking/stage.py
    - src/aquapose/engine/pipeline.py
    - tests/unit/tracking/test_ocsort_wrapper.py
    - tests/unit/core/tracking/test_tracking_stage.py
decisions:
  - spine1 (index 2) selected as default centroid keypoint — mid-body, most stable under clipping and occlusion
  - confidence floor 0.3 matches pose backend default; interior keypoints reliably exceed this in production
  - Detection recovered from boxmot result column 7 (det_idx) to link tracker output to source Detection keypoints
  - get_state/from_state preserve centroid config with defaults for backward compat with old state blobs
metrics:
  duration: ~5 minutes
  completed_date: "2026-03-11"
  tasks_completed: 2
  files_modified: 6
  files_created: 1
---

# Phase 82 Plan 01: Keypoint Centroid Extraction Summary

**One-liner:** Spine1 (index 2) keypoint replaces OBB centroid in Tracklet2D with configurable confidence floor and silent OBB fallback.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add config fields and implement keypoint centroid in tracker | cf1e0e3 (test), 05997c8 (impl) | config.py, ocsort_wrapper.py, stage.py, pipeline.py, test_ocsort_wrapper.py, test_tracking_stage.py |
| 2 | Document keypoint selection rationale | 4f9eea3 | 82-NOTES.md |

## What Was Built

Swapped the centroid source in `Tracklet2D` from OBB geometric center to mid-body keypoint (spine1, index 2) with a configurable fallback.

**Implementation chain:**

1. `AssociationConfig` gains two new YAML-tunable fields: `centroid_keypoint_index=2` and `centroid_confidence_floor=0.3`
2. `_TrackletBuilder.add_frame()` now accepts `detection`, `centroid_keypoint_index`, `centroid_confidence_floor` — when status is `"detected"` and detection carries a confident keypoint at the configured index, the keypoint position overrides the OBB center; all other cases fall back silently
3. `_TrackletBuilder` gains `keypoint_centroid_count` field; `to_tracklet2d()` emits a debug log of keypoint usage rate per tracklet
4. `OcSortTracker` accepts centroid config params, stores them, threads to builders via detection column-7 index recovery; `get_state`/`from_state` preserve them with backward-compat defaults
5. `TrackingStage.__init__()` accepts centroid params and threads them to `OcSortTracker` at construction time
6. `pipeline.py` passes `config.association.centroid_keypoint_index` and `centroid_confidence_floor` to `TrackingStage`

**Association stage, LUT generation, ray-ray scoring, and Leiden clustering are completely untouched.** The change is fully encapsulated in the tracklet centroid source.

## Decisions Made

1. **spine1 (index 2) as default**: Mid-body keypoint equidistant from nose and tail. Most geometrically stable under frame-edge clipping (extremity keypoints are lost first) and partial occlusion. The OBB centroid drifts when the bounding box is clipped, but spine1 remains stable as long as the fish body center is visible.

2. **Confidence floor = 0.3**: Matches `PoseConfig.keypoint_confidence_floor` default. Interior keypoints (spine1/spine2) consistently exceed this in the production model (mAP50-95=0.974).

3. **Detection recovery via column 7**: BoxMot's OcSort returns the original detection index in column 7 of the result array. Using this to recover `detections[det_idx]` avoids storing per-frame detection objects and keeps the integration minimal.

4. **Backward-compat state defaults**: `from_state()` uses `.get()` with defaults so old state blobs (without centroid fields) continue to work unchanged.

## Test Coverage

6 new tests in `TestKeypointCentroid` (test_ocsort_wrapper.py):
- High-confidence keypoint at index 2 overrides OBB centroid
- Low-confidence keypoint (below floor) falls back to OBB centroid
- Missing keypoints (`None`) falls back to OBB centroid
- Coasted frames (no detection) produce valid float centroids
- Custom keypoint index (index 0 = nose) is applied correctly
- `OcSortTracker` constructor accepts centroid config params

2 new tests in `TestCentroidConfigThreading` (test_tracking_stage.py):
- `TrackingStage` accepts centroid config params and stores them
- Default values are correct (index=2, floor=0.3)

All 1113 existing tests pass unchanged.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

Files created/modified:
- `src/aquapose/engine/config.py` — centroid fields added to AssociationConfig
- `src/aquapose/core/tracking/ocsort_wrapper.py` — keypoint extraction implemented
- `src/aquapose/core/tracking/stage.py` — centroid params threaded
- `src/aquapose/engine/pipeline.py` — config threading updated
- `tests/unit/tracking/test_ocsort_wrapper.py` — new TestKeypointCentroid class
- `tests/unit/core/tracking/test_tracking_stage.py` — new TestCentroidConfigThreading class
- `.planning/phases/82-association-upgrade-keypoint-centroid/82-NOTES.md` — rationale doc

Commits:
- cf1e0e3: test(82-01): add failing tests for keypoint centroid extraction
- 05997c8: feat(82-01): implement keypoint centroid extraction in tracker
- 4f9eea3: docs(82-01): add keypoint selection rationale notes

## Self-Check: PASSED
