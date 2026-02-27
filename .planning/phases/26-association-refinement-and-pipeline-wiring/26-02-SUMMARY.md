---
phase: 26-association-refinement-and-pipeline-wiring
plan: 02
subsystem: midline
tags: [orientation, head-tail, velocity, geometric-vote, temporal-prior]

requires:
  - phase: 26-association-refinement-and-pipeline-wiring
    provides: TrackletGroup with per_frame_confidence from 26-01
provides:
  - Head-tail orientation resolver with 3-signal combination
  - MidlineStage filtering by tracklet group membership
  - Orientation-corrected annotated_detections
affects: [26-03-reconstruction]

tech-stack:
  added: []
  patterns: [3-signal weighted vote for orientation, per-camera flip correction]

key-files:
  created:
    - src/aquapose/core/midline/orientation.py
    - tests/unit/core/midline/test_orientation.py
  modified:
    - src/aquapose/core/midline/stage.py
    - src/aquapose/core/midline/__init__.py
    - src/aquapose/engine/config.py
    - src/aquapose/engine/pipeline.py

key-decisions:
  - "_DefaultOrientationConfig in stage.py avoids IB-003 engine import in core/"
  - "Velocity vote: head_dir = -(tail-head), alignment with velocity -> keep if positive, flip if negative"
  - "Robust consensus from best-half pairwise midpoints reused from 26-01 pattern"

patterns-established:
  - "OrientationConfigLike: Protocol pattern for orientation config"
  - "_DefaultOrientationConfig: frozen dataclass fallback in core/ avoiding engine import"

requirements-completed: [PIPE-02]

duration: 10min
completed: 2026-02-27
---

# Plan 26-02: Midline Orientation and Tracklet-Group Filtering Summary

**Head-tail orientation resolver combining geometric vote, velocity alignment, and temporal prior with MidlineStage tracklet-group filtering**

## Performance

- **Duration:** 10 min
- **Tasks:** 2
- **Files modified:** 4
- **Files created:** 2

## Accomplishments
- resolve_orientation() combines 3 weighted signals for head-tail disambiguation
- Velocity signal gated by speed_threshold (default 2.0 px/frame)
- MidlineStage.run() filters detections to only process tracklet group members
- Orientation resolution applied per fish per frame with temporal tracking
- Backward-compatible fallback when tracklet_groups unavailable

## Task Commits

1. **Task 1: Add orientation config and implement orientation resolver** - `dd4f553` (feat)
2. **Task 2: Update MidlineStage for tracklet-group filtering** - `8546017` (feat)

## Files Created/Modified
- `src/aquapose/core/midline/orientation.py` - 3-signal orientation resolver
- `src/aquapose/core/midline/stage.py` - Tracklet-group filtering, orientation wiring
- `src/aquapose/core/midline/__init__.py` - Export resolve_orientation, OrientationConfigLike
- `src/aquapose/engine/config.py` - MidlineConfig orientation weight fields
- `src/aquapose/engine/pipeline.py` - Pass lut_config, midline_config to MidlineStage
- `tests/unit/core/midline/test_orientation.py` - 11 unit tests

## Decisions Made
- Used _DefaultOrientationConfig dataclass in core/ to avoid IB-003 engine import boundary violation
- Velocity vote uses head_dir = -(tail - head) convention: fish swim head-first, so head direction should align with velocity

## Deviations from Plan
None significant.

## Issues Encountered
- Import boundary check caught `from aquapose.engine.config import MidlineConfig` in core/. Fixed with _DefaultOrientationConfig fallback.

## Next Phase Readiness
- Orientation-corrected midlines ready for reconstruction
- Plan 26-03 (Reconstruction rewrite) can proceed

---
*Phase: 26-association-refinement-and-pipeline-wiring*
*Completed: 2026-02-27*
