---
phase: 20-post-refactor-loose-ends
plan: "04"
subsystem: tracking
tags: [tracking, pipeline, stage-coupling, association, hungarian, refactor]

requires:
  - phase: 20-01
    provides: PipelineContext and Stage Protocol in core/context.py
  - phase: 15-04
    provides: TrackingStage, HungarianBackend, FishTracker wrapping v1.0 behavior
  - phase: 15-03
    provides: AssociationBundle type and AssociationStage (Stage 3) producing bundles

provides:
  - TrackingStage reads context.associated_bundles as primary input (Stage 3 hard dependency)
  - HungarianBackend.track_frame() consumes AssociationBundle objects directly
  - FishTracker.update_from_bundles() — greedy nearest-3D-centroid matching with full lifecycle
  - No internal RANSAC re-association in tracking pipeline path
  - discover_births() preserved in tracking/associate.py as utility (no longer called during pipeline)

affects: [reconstruction-stage, pipeline-integration-tests, e2e-tests]

tech-stack:
  added: []
  patterns:
    - "Bundle-first tracking: Stage 4 consumes Stage 3 AssociationBundle centroids via greedy 3D nearest-neighbor assignment — no camera projection needed"
    - "FishTracker.update_from_bundles() as the canonical pipeline-facing API; update() preserved for backward compat"

key-files:
  created: []
  modified:
    - src/aquapose/core/tracking/stage.py
    - src/aquapose/core/tracking/backends/__init__.py
    - src/aquapose/core/tracking/backends/hungarian.py
    - src/aquapose/tracking/tracker.py
    - tests/unit/core/tracking/test_tracking_stage.py

key-decisions:
  - "TrackingStage.run() reads context.associated_bundles as primary input; raises ValueError if None — Stage 3 is now a hard dependency for Stage 4"
  - "HungarianBackend.track_frame() no longer accepts detections_per_camera; bundles are the sole data input"
  - "HungarianBackend no longer loads calibration models — update_from_bundles() uses pre-computed 3D centroids, not camera projection"
  - "FishTracker.update_from_bundles() performs greedy nearest-3D-centroid assignment (distance-sorted candidates, priority tie-breaking for confirmed/coasting vs probationary tracks)"
  - "Birth from unmatched bundles: n_cameras < min_cameras_birth bundles are skipped; proximity check (birth_proximity_distance) still applied; TRACK-04 dead-ID recycling preserved"
  - "discover_births() in tracking/associate.py kept as utility function — no longer called during normal pipeline operation"

patterns-established:
  - "Pipeline stage precondition pattern: check context.{input_field} is None, raise ValueError naming the producing stage"

requirements-completed: [REMEDIATE]

duration: 25min
completed: 2026-02-27
---

# Phase 20 Plan 04: Stage 3/4 Coupling Fix Summary

**TrackingStage and HungarianBackend refactored to consume AssociationBundle objects from Stage 3 directly — eliminating redundant internal RANSAC re-association (AUD-005/AUD-006)**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-02-27T02:12:00Z
- **Completed:** 2026-02-27T02:27:52Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- TrackingStage.run() now reads context.associated_bundles as primary input; raises ValueError naming Stage 3 as missing dependency
- FishTracker.update_from_bundles() added — greedy nearest-3D-centroid assignment (no camera projection), same lifecycle semantics (probationary/confirmed/coasting/dead), same population constraint (TRACK-04 dead-ID recycling)
- HungarianBackend simplified: no longer loads calibration models or takes detections_per_camera; pure bundle consumer
- discover_births() preserved as utility but no longer called during pipeline run
- 514 unit tests pass (513 original + 1 new test for Stage 3 precondition)

## Task Commits

1. **Task 1: Refactor TrackingStage to read bundles from Stage 3** - `0fe1ebd` (refactor)
2. **Task 2: Update tracking tests for bundle-based input** - `586f4b1` (test)

## Files Created/Modified

- `src/aquapose/core/tracking/stage.py` - Primary input changed from context.detections to context.associated_bundles; precondition ValueError updated
- `src/aquapose/core/tracking/backends/__init__.py` - track_frame() signature docs updated (removed detections_per_camera param)
- `src/aquapose/core/tracking/backends/hungarian.py` - track_frame() signature simplified; _load_models() removed; module docstring cleaned
- `src/aquapose/tracking/tracker.py` - Added update_from_bundles() method with full greedy assignment, birth logic, lifecycle management
- `tests/unit/core/tracking/test_tracking_stage.py` - All tests updated to use associated_bundles; new test_tracking_requires_bundles added; _make_bundle() helper added

## Decisions Made

- TrackingStage.run() reads context.associated_bundles as primary input; raises ValueError if None — Stage 3 is now a hard dependency for Stage 4
- HungarianBackend.track_frame() no longer accepts detections_per_camera; bundles are the sole data input
- HungarianBackend no longer loads calibration models — update_from_bundles() uses pre-computed 3D centroids, not camera projection
- FishTracker.update_from_bundles() performs greedy nearest-3D-centroid assignment (distance-sorted candidates, priority tie-breaking for confirmed/coasting vs probationary tracks)
- Birth from unmatched bundles: n_cameras < min_cameras_birth bundles are skipped; proximity check (birth_proximity_distance) still applied; TRACK-04 dead-ID recycling preserved
- discover_births() in tracking/associate.py kept as utility function — no longer called during normal pipeline operation

## Deviations from Plan

None - plan executed exactly as written. The refactor scope was well-defined: update stage.py, backends/, and tracker.py; update tests.

## Issues Encountered

Two pre-commit hook failures (lint/format) on first commit attempt — ruff flagged B007 (unused `dist` loop variable renamed to `_dist`) and auto-formatted two files. Fixed inline and committed on second attempt. No code logic changes needed.

## Next Phase Readiness

- Stage 3/4 coupling (AUD-005, AUD-006) resolved — pipeline data flow is now honest
- Plans 20-05 remains (final phase wrap-up or pending items)
- All 514 unit tests pass with clean lint/format

---
*Phase: 20-post-refactor-loose-ends*
*Completed: 2026-02-27*
