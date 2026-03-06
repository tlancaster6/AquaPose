---
phase: 24-per-camera-2d-tracking
plan: 01
subsystem: tracking
tags: [boxmot, ocsort, kalman-filter, tracklet2d, carry-forward, pipeline-stage]

# Dependency graph
requires:
  - phase: 22-pipeline-scaffolding
    provides: TrackingStubStage, TrackingConfig stub, CarryForward, PipelineContext.tracks_2d, build_stages factory
  - phase: 22-pipeline-scaffolding
    provides: Tracklet2D dataclass in core/tracking/types.py

provides:
  - OcSortTracker wrapper in tracking/ocsort_wrapper.py — boxmot fully isolated
  - TrackingStage (Stage 2) in core/tracking/stage.py — per-camera OC-SORT tracking
  - TrackingConfig expanded with tracker_kind, n_init, iou_threshold, det_thresh
  - 25 unit tests for wrapper and stage contracts

affects:
  - 25-cross-view-association — consumes context.tracks_2d dict[str, list[Tracklet2D]]
  - 22-pipeline-scaffolding — TrackingStubStage removed, test references updated

# Tech tracking
tech-stack:
  added:
    - boxmot>=11.0 (OC-SORT tracker via OcSort class; 6-col input format x1,y1,x2,y2,conf,cls)
  patterns:
    - boxmot fully isolated in single wrapper module (ocsort_wrapper.py) — no other module imports from boxmot
    - OcSortTracker.get_state()/from_state() pattern for opaque cross-batch CarryForward serialization
    - TrackingStage uses deferred import for OcSortTracker to respect ENG-07 import boundary
    - Any-typed config in TrackingStage.__init__ avoids circular engine->core import

key-files:
  created:
    - src/aquapose/tracking/ocsort_wrapper.py
    - src/aquapose/core/tracking/stage.py
    - tests/unit/tracking/test_ocsort_wrapper.py
    - tests/unit/core/tracking/test_tracking_stage.py
  modified:
    - pyproject.toml (added boxmot>=11.0)
    - src/aquapose/tracking/__init__.py (export OcSortTracker)
    - src/aquapose/core/tracking/__init__.py (export TrackingStage)
    - src/aquapose/engine/pipeline.py (remove TrackingStubStage, use TrackingStage)
    - src/aquapose/engine/config.py (expand TrackingConfig)
    - tests/unit/engine/test_build_stages.py (updated for TrackingStage)
    - tests/unit/core/reconstruction/test_reconstruction_stage.py (remove TrackingStubStage refs)

key-decisions:
  - "boxmot OcSort takes 6-column input [x1,y1,x2,y2,conf,cls] — not 5-column as the plan suggested; cls=0.0 (single class) appended automatically"
  - "OcSort does NOT output coasting tracks in its update() result — coasting positions are captured separately from active_tracks with time_since_update>0"
  - "TrackingStage.__init__ accepts Any-typed config to avoid circular engine->core import (same pattern as other core stages)"
  - "type: ignore[arg-type] added for stage.run(context, carry) call in pipeline.py — TrackingStage signature differs from Stage Protocol but isinstance check is correct at runtime"
  - "Pre-existing type errors reduced from 17 to 12 — stage.py Any-typed config eliminated false positives"

patterns-established:
  - "Boxmot isolation: all boxmot imports confined to tracking/ocsort_wrapper.py; downstream code only sees Tracklet2D"
  - "Coasting capture pattern: after each update(), iterate active_tracks for time_since_update>0 to record coasted frame positions"
  - "Carry state is opaque dict containing the live tracker object — no serialization to disk, only in-memory carry across batches"

requirements-completed:
  - TRACK-01

# Metrics
duration: 14min
completed: 2026-02-27
---

# Phase 24 Plan 01: Per-Camera 2D Tracking Summary

**OC-SORT per-camera 2D tracking via boxmot wrapper producing typed Tracklet2D objects, replacing TrackingStubStage with a real TrackingStage that carries state between batches**

## Performance

- **Duration:** 14 min
- **Started:** 2026-02-27T18:58:45Z
- **Completed:** 2026-02-27T19:12:54Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments

- OcSortTracker wrapper fully isolates boxmot — single module boundary, clean Tracklet2D output
- TrackingStage (Stage 2) replaces TrackingStubStage; per-camera OC-SORT tracking with carry-forward
- 25 unit tests covering empty inputs, single track, coasting, multi-track, state roundtrip, stage contract
- TrackingConfig expanded with 4 OC-SORT parameters; _filter_fields() prevents stale YAML key errors
- Full test suite: 491 tests pass, lint clean, type errors reduced from 17 to 12

## Task Commits

Each task was committed atomically:

1. **Task 1: OcSortTracker wrapper and tests** - `04b36cf` (feat)
2. **Task 2: TrackingStage, config expansion, pipeline rewire** - `e303255` (feat)

## Files Created/Modified

- `src/aquapose/tracking/ocsort_wrapper.py` - OcSortTracker wrapping boxmot OcSort; local ID mapping; get_state/from_state
- `src/aquapose/core/tracking/stage.py` - TrackingStage (Stage 2); per-camera delegation to OcSortTracker
- `src/aquapose/engine/config.py` - TrackingConfig expanded with tracker_kind, n_init, iou_threshold, det_thresh
- `src/aquapose/engine/pipeline.py` - TrackingStubStage removed; build_stages uses TrackingStage; isinstance dispatch updated
- `src/aquapose/tracking/__init__.py` - exports OcSortTracker
- `src/aquapose/core/tracking/__init__.py` - exports TrackingStage
- `pyproject.toml` - boxmot>=11.0 dependency added
- `tests/unit/tracking/test_ocsort_wrapper.py` - 13 unit tests for wrapper
- `tests/unit/core/tracking/test_tracking_stage.py` - 12 unit tests for stage
- `tests/unit/engine/test_build_stages.py` - updated for TrackingStage
- `tests/unit/core/reconstruction/test_reconstruction_stage.py` - removed TrackingStubStage references

## Decisions Made

- **boxmot API deviation**: OcSort requires 6-column input `[x1,y1,x2,y2,conf,cls]` (not 5-column as plan assumed). cls=0.0 appended for single-class tracking.
- **Coasting capture**: OcSort does not output coasting tracks in `update()` result — coasting positions must be captured by iterating `active_tracks` where `time_since_update > 0`. This was discovered during API exploration and handled correctly.
- **Any-typed config**: `TrackingStage.__init__` accepts `Any` typed config to avoid a circular `engine -> core` import. The same deferred-import pattern is used for the OcSortTracker inside `run()`.
- **type: ignore on carry dispatch**: The `isinstance(stage, TrackingStage)` branch in pipeline.py calls `stage.run(context, carry)` — basedpyright flags this because the `Stage` Protocol only declares `run(context)`. Added `# type: ignore[arg-type]` since runtime behavior is correct.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] boxmot OcSort requires 6-column input, not 5-column**
- **Found during:** Task 1 (wrapper implementation)
- **Issue:** Plan specified `[x1, y1, x2, y2, confidence]` (5-col). boxmot asserts shape[1]==6 with format `[x1,y1,x2,y2,conf,cls]`
- **Fix:** Added `cls=0.0` column to all detection arrays in ocsort_wrapper.py
- **Files modified:** src/aquapose/tracking/ocsort_wrapper.py
- **Verification:** All wrapper tests pass; `update()` no longer raises AssertionError
- **Committed in:** 04b36cf (Task 1)

**2. [Rule 1 - Bug] OcSort coasting tracks not in update() output — must capture from active_tracks**
- **Found during:** Task 1 (coasting behavior exploration)
- **Issue:** Plan assumed coasting tracks appear in update() output. boxmot OcSort returns empty when no detections; coasting tracks only accessible via `tracker.active_tracks` with `time_since_update > 0`
- **Fix:** Added post-update loop over `active_tracks` to capture coasting frame positions from Kalman state
- **Files modified:** src/aquapose/tracking/ocsort_wrapper.py
- **Verification:** `test_coasting_detection_gap` passes — gap frames appear as "coasted" in frame_status
- **Committed in:** 04b36cf (Task 1)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug, both discovered during API exploration)
**Impact on plan:** Both fixes essential for correct boxmot integration. No scope creep.

## Issues Encountered

- Pre-existing test `test_active_stages_importable` and `test_build_stages_returns_stages` in test_reconstruction_stage.py imported `TrackingStubStage` — updated to use `TrackingStage` as part of Task 2 cleanup.

## Next Phase Readiness

- Phase 25 (Association) can now receive `context.tracks_2d: dict[str, list[Tracklet2D]]` with real per-camera tracklets
- Phase 25 still hard-depends on Phase 23 (LUTs) completion — coordinate if needed
- boxmot coasting is in-memory only (state object carries live tracker) — no disk serialization concern for v2.1

---
*Phase: 24-per-camera-2d-tracking*
*Completed: 2026-02-27*

## Self-Check: PASSED

- src/aquapose/tracking/ocsort_wrapper.py: FOUND
- src/aquapose/core/tracking/stage.py: FOUND
- tests/unit/tracking/test_ocsort_wrapper.py: FOUND
- tests/unit/core/tracking/test_tracking_stage.py: FOUND
- .planning/phases/24-per-camera-2d-tracking/24-01-SUMMARY.md: FOUND
- commit 04b36cf: FOUND
- commit e303255: FOUND
