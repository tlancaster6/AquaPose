---
phase: 15-stage-migrations
plan: 04
subsystem: core
tags: [tracking, hungarian, fish-tracker, stage-protocol, temporal-association]

# Dependency graph
requires:
  - phase: 15-03
    provides: AssociationStage (Stage 3) cross-camera bundles and AssociationBundle types
  - phase: 13-engine-core
    provides: Stage Protocol, PipelineContext, TrackingConfig in engine/

provides:
  - TrackingStage in core/tracking/stage.py satisfying Stage Protocol
  - HungarianBackend wrapping v1.0 FishTracker for exact behavioral equivalence
  - Backend registry (get_backend) supporting 'hungarian' kind
  - FishTrack/TrackState re-exports in core/tracking/types.py
  - Extended TrackingConfig with full tracker parameter set in engine/config.py
  - 9 interface tests verifying Stage Protocol compliance and import boundary

affects: [15-05-reconstruction, verification, observers, CLI]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - HungarianBackend wraps existing FishTracker — port behavior, not rewrite
    - Stage 4 reads context.detections (Stage 1 output), NOT context.associated_bundles (Stage 3) — v1.0 equivalence debt documented
    - Stateful backend: FishTracker created once at construction, persists across frames
    - TYPE_CHECKING guard for engine/ annotations enforces ENG-07 import boundary

key-files:
  created:
    - src/aquapose/core/tracking/__init__.py
    - src/aquapose/core/tracking/types.py
    - src/aquapose/core/tracking/backends/__init__.py
    - src/aquapose/core/tracking/backends/hungarian.py
    - src/aquapose/core/tracking/stage.py
    - tests/unit/core/tracking/__init__.py
    - tests/unit/core/tracking/test_tracking_stage.py
  modified:
    - src/aquapose/engine/config.py

key-decisions:
  - "TrackingStage reads context.detections (Stage 1 raw output) NOT context.associated_bundles (Stage 3) — FishTracker.update() re-derives cross-camera association internally, preserving exact v1.0 numerical equivalence. Stage 3 output is a data product for future backends/observers but not consumed by Stage 4 in v1.0-equivalent mode. Documented as known design debt."
  - "HungarianBackend passes raw detections to FishTracker.update() directly — no reimplementation of association logic, all existing FishTracker behavior preserved exactly"
  - "FishTracker created once at construction (not per-frame) — tracker is stateful, maintaining track identities (fish_id) and lifecycle state (PROBATIONARY -> CONFIRMED -> COASTING -> DEAD) across all frames"
  - "TrackingConfig extended with full tracker parameter set (min_hits, max_age, reprojection_threshold, birth_interval, min_cameras_birth, velocity_damping, velocity_window) — was just max_fish before"

patterns-established:
  - "Backend registry pattern: get_backend(kind, **kwargs) returns backend with track_frame() method — mirrors association get_backend() pattern"
  - "Calibration loaded eagerly in backend constructor via local imports — fail-fast on missing files, avoids circular imports"
  - "HungarianBackend.track_frame(frame_idx, bundles, detections_per_camera) — bundles param included for future backends even though current implementation passes detections directly to FishTracker"

requirements-completed: [STG-04]

# Metrics
duration: 9min
completed: 2026-02-26
---

# Phase 15 Plan 04: TrackingStage Summary

**TrackingStage wraps v1.0 FishTracker as Stage 4 in core/tracking/, preserving exact behavioral equivalence via stateful Hungarian backend with persistent fish identity across frames**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-26T00:47:16Z
- **Completed:** 2026-02-26T00:56:30Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- TrackingStage in core/tracking/ satisfies Stage Protocol via structural typing (isinstance check passes)
- HungarianBackend wraps existing FishTracker with no reimplementation — exact v1.0 behavioral equivalence preserved
- Tracker state persists across frames: FishTracker constructed once at stage construction, maintaining fish_id continuity
- Extended TrackingConfig with full tracker parameter set (7 new fields vs single max_fish)
- 9 interface tests all pass: protocol conformance, context population, state persistence, attribute validation, registry, import boundary

## Task Commits

Each task was committed atomically:

1. **Task 1: Create core/tracking/ module with Hungarian backend and stage** - `609b2cf` (feat)
2. **Task 2: Interface tests for TrackingStage** - `0cbeaf1` (test)

**Plan metadata:** (created next)

## Files Created/Modified

- `src/aquapose/core/tracking/__init__.py` - Package public API: TrackingStage, FishTrack, TrackState
- `src/aquapose/core/tracking/types.py` - Re-exports FishTrack/TrackState from aquapose.tracking.tracker
- `src/aquapose/core/tracking/backends/__init__.py` - Backend registry: get_backend() for 'hungarian'
- `src/aquapose/core/tracking/backends/hungarian.py` - HungarianBackend wrapping FishTracker, loads calibration, exposes track_frame()
- `src/aquapose/core/tracking/stage.py` - TrackingStage satisfying Stage Protocol, reads context.detections
- `src/aquapose/engine/config.py` - Extended TrackingConfig with 7 tracker parameters (min_hits, max_age, etc.)
- `tests/unit/core/tracking/__init__.py` - Test package init
- `tests/unit/core/tracking/test_tracking_stage.py` - 9 interface tests

## Decisions Made

- **Stage 4 reads detections not bundles (v1.0 equivalence debt):** FishTracker.update() performs its own cross-camera association internally. For exact v1.0 numerical equivalence, Stage 4 passes raw detections directly rather than consuming Stage 3 associated_bundles. Stage 3 output is a data product for future backends/observers but is not consumed by Stage 4 in v1.0-equivalent mode. Documented as known design debt.
- **HungarianBackend stateful by construction:** FishTracker instance created once in HungarianBackend.__init__() and persists across all track_frame() calls — enables fish_id continuity (TRACK-04 population constraint preserved).
- **TrackingConfig extended:** Added 7 new tracker parameter fields to TrackingConfig frozen dataclass — min_hits, max_age, reprojection_threshold, birth_interval, min_cameras_birth, velocity_damping, velocity_window. These were previously hard-coded defaults in FishTracker.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing strict= on zip() call in stage.py**
- **Found during:** Task 1 (stage.py creation)
- **Issue:** Ruff B905 lint error — zip() without explicit strict= parameter
- **Fix:** Added `strict=False` to zip(context.detections, bundles_per_frame)
- **Files modified:** src/aquapose/core/tracking/stage.py
- **Verification:** hatch run lint passes
- **Committed in:** 609b2cf (Task 1 commit, after fix)

**2. [Rule 1 - Bug] Fixed import ordering in test file**
- **Found during:** Task 2 (test file creation)
- **Issue:** Ruff I001 import ordering error in test file
- **Fix:** Ran ruff --fix on test file to correct import block ordering
- **Files modified:** tests/unit/core/tracking/test_tracking_stage.py
- **Verification:** hatch run lint passes
- **Committed in:** 0cbeaf1 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — lint/style corrections caught by pre-commit hooks)
**Impact on plan:** Both trivial lint fixes, no functional impact or scope creep.

## Issues Encountered

None beyond the auto-fixed lint issues above.

## Next Phase Readiness

- TrackingStage ready for Stage 5 (ReconstructionStage) which reads context.tracks
- FishTrack/TrackState exported from core/tracking/ for downstream stage access
- All 9 interface tests pass, import boundary enforced
- TrackingConfig extended — YAML configs with tracking.min_hits etc. will now parse correctly

## Self-Check: PASSED

- FOUND: src/aquapose/core/tracking/__init__.py
- FOUND: src/aquapose/core/tracking/stage.py
- FOUND: src/aquapose/core/tracking/backends/hungarian.py
- FOUND: tests/unit/core/tracking/test_tracking_stage.py
- FOUND: commit 609b2cf (feat: core/tracking/ module)
- FOUND: commit 0cbeaf1 (test: interface tests)

---
*Phase: 15-stage-migrations*
*Completed: 2026-02-26*
