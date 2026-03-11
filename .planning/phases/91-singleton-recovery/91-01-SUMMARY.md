---
phase: 91-singleton-recovery
plan: 01
subsystem: association
tags: [singleton-recovery, ray-geometry, greedy-assignment, split-assign, tracklet-group]

# Dependency graph
requires:
  - phase: 90-group-validation
    provides: validate_groups() that produces singletons (evicted/split tracklets)
  - phase: 88-multi-keypoint-scoring
    provides: per-keypoint ray-ray scoring infrastructure in scoring.py
provides:
  - recovery.py with RecoveryConfigLike protocol + recover_singletons() entry point
  - greedy whole-assignment pass with same-camera detected-frame overlap constraint
  - binary split-assign sweep for swap recovery (both segments must match DIFFERENT groups)
  - per-keypoint ray-to-3D residual scoring with centroid fallback for keypoints=None
  - 4 new AssociationConfig fields (recovery_enabled, recovery_residual_threshold, recovery_min_shared_frames, recovery_min_segment_length)
  - recovery wired in AssociationStage.run() as Step 5 after validate_groups()
affects:
  - phase: 92-association-evaluation
    note: evaluation should measure singleton reduction; recovered groups have per_frame_confidence=None and consensus_centroids=None

# Tech tracking
tech-stack:
  added: []
  patterns:
    - RecoveryConfigLike protocol pattern (runtime_checkable, structural, IB-003 compliant)
    - standalone _point_to_ray_distance() copied from refinement.py (no cross-module import)
    - staleness invalidation: per_frame_confidence and consensus_centroids set to None on modified groups
    - TDD execution: test file committed (RED) before implementation (GREEN)

key-files:
  created:
    - src/aquapose/core/association/recovery.py
    - tests/unit/core/association/test_recovery.py
  modified:
    - src/aquapose/core/association/__init__.py
    - src/aquapose/core/association/stage.py
    - src/aquapose/engine/config.py

key-decisions:
  - "Module independence: recovery.py copies _point_to_ray_distance() standalone — no imports from validation.py or refinement.py"
  - "Staleness invalidation: per_frame_confidence and consensus_centroids set to None on groups that gain new tracklets — no re-triangulation within recovery pass"
  - "Separate recovery_ config fields (recovery_residual_threshold=0.025, recovery_min_shared_frames=3, recovery_min_segment_length=10) for independent tuning vs validation fields"
  - "Split-assign requires both segments to match DIFFERENT groups — single-segment match leaves singleton unchanged"
  - "Coasted frames never block assignment — only detected frames count for same-camera overlap constraint"

patterns-established:
  - "RecoveryConfigLike: structural Protocol pattern for core/->engine/ config boundary"
  - "On-demand group triangulation per (singleton, group) pair — no cache needed at ~9 fish"
  - "Greedy single-pass assignment with re-check of overlap against already-absorbed tracklets"

requirements-completed:
  - RECOV-01
  - RECOV-02
  - RECOV-03
  - RECOV-04

# Metrics
duration: 13min
completed: 2026-03-11
---

# Phase 91 Plan 01: Singleton Recovery Summary

**Singleton recovery module with greedy whole-assignment, binary split-assign sweep, same-camera overlap constraint, and per-keypoint ray-to-3D residual scoring**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-11T20:16:05Z
- **Completed:** 2026-03-11T20:29:11Z
- **Tasks:** 2 (Task 1: TDD implementation; Task 2: lint/typecheck)
- **Files modified:** 5 created/modified

## Accomplishments
- Created `recovery.py` with `RecoveryConfigLike` protocol + `recover_singletons()` entry point implementing greedy whole-assignment and binary split-assign sweep
- 13 unit tests passing covering all 4 requirements (RECOV-01 through RECOV-04)
- Wired `recover_singletons()` into `AssociationStage.run()` as Step 5 after `validate_groups()`
- Added 4 new `AssociationConfig` fields with defaults matching existing thresholds for zero-diff baseline behavior

## Task Commits

Each task was committed atomically (TDD pattern: test → impl):

1. **Task 1 RED: Add failing tests** - `89caba7` (test)
2. **Task 1 GREEN: Implement recovery module** - `f30ccf3` (feat)

_Note: Task 2 (lint/typecheck) had no separate commit since all issues were fixed inline with Task 1 GREEN._

## Files Created/Modified
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/association/recovery.py` - RecoveryConfigLike + recover_singletons() + all internal helpers
- `/home/tlancaster6/Projects/AquaPose/tests/unit/core/association/test_recovery.py` - 13 unit tests for all recovery behaviors
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/association/__init__.py` - Added RecoveryConfigLike and recover_singletons exports
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/association/stage.py` - Wired Step 5 singleton recovery after validation
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/config.py` - Added 4 recovery_ fields to AssociationConfig

## Decisions Made
- Module independence: `_point_to_ray_distance()` copied standalone from `refinement.py` — no cross-module imports per research recommendation
- Staleness invalidation: `per_frame_confidence` and `consensus_centroids` set to `None` on groups absorbing new tracklets — recomputation would add complexity beyond phase scope; downstream consumers already guard against `None`
- Separate `recovery_*` config fields allow independent tuning vs Phase 90 validation thresholds
- Split-assign requires BOTH segments to match DIFFERENT groups — if only one matches, singleton stays unchanged

## Deviations from Plan

None — plan executed exactly as written. The TDD test for split-assign required multiple test redesigns to establish geometrically valid scenarios (where whole-assign fails but per-segment assign succeeds), but this was problem-solving within the planned TDD flow, not unplanned work.

## Issues Encountered

During the TDD RED-to-GREEN cycle, the split-assign test required careful geometric reasoning:

1. First design: singleton's whole score against a group was the same as the segment score (whole-assign succeeded, split-assign never ran). Fixed by redesigning to ensure the singleton's mean residual against any single group exceeded the threshold when scored across all frames.
2. Second design: the "after" segment had divergent rays, so it couldn't match any group. Fixed by using a pixel_fn that generates converging-to-point_b rays for the second half of the singleton.
3. Final design: two groups converging to different 3D points (z=0.5 and z=1.5), singleton with pixel_fn that generates rays through point_a for first half and point_b for second half.

## Next Phase Readiness
- Recovery module complete and integrated into association pipeline
- `recovery_enabled=True` by default — active in all runs immediately
- Phase 92 evaluation can measure reduction in singleton rate with this recovery active
- Singleton rate expected to decrease from ~27% baseline; Phase 92 to measure actual improvement

---
*Phase: 91-singleton-recovery*
*Completed: 2026-03-11*
