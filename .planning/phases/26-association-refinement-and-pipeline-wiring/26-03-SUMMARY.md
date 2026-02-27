---
phase: 26-association-refinement-and-pipeline-wiring
plan: 03
subsystem: reconstruction, hdf5
tags: [tracklet-groups, camera-membership, gap-interpolation, fish-first-hdf5]

requires:
  - phase: 26-association-refinement-and-pipeline-wiring
    provides: Orientation-corrected midlines from 26-02
provides:
  - ReconstructionStage consuming tracklet_groups with known camera membership
  - Fish-first HDF5 layout with spline_controls and confidence arrays
  - Gap interpolation for short dropped-frame gaps
affects: [27-diagnostic-visualization]

tech-stack:
  added: []
  patterns: [fish-first HDF5 layout, tracklet-group camera membership, linear gap interpolation]

key-files:
  created:
    - .planning/phases/26-association-refinement-and-pipeline-wiring/26-03-SUMMARY.md
  modified:
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/engine/hdf5_observer.py
    - src/aquapose/engine/config.py
    - src/aquapose/engine/pipeline.py
    - src/aquapose/core/association/types.py
    - src/aquapose/core/tracking/types.py
    - tests/unit/core/reconstruction/test_reconstruction_stage.py
    - tests/unit/engine/test_hdf5_observer.py

key-decisions:
  - "Legacy FishTrack/AssociationBundle types deprecated but not deleted -- still imported by visualization modules"
  - "Fish-first HDF5 layout gated on tracklet_groups presence -- frame-major preserved as backward compat"
  - "Coasted frames excluded from camera count -- only detected frames contribute"
  - "Linear interpolation of control points for short gaps -- confidence=0 and is_low_confidence=True"

patterns-established:
  - "_find_matching_annotated: centroid-proximity detection lookup (reused from 26-02)"
  - "Fish-first HDF5: /fish_{id}/spline_controls[T,N,3] + confidence[T]"

requirements-completed: [PIPE-03]

duration: 15min
completed: 2026-02-27
---

# Plan 26-03: Reconstruction Rewrite and Fish-First HDF5 Summary

**ReconstructionStage consuming tracklet_groups with known camera membership, gap interpolation, and fish-first HDF5 output**

## Performance

- **Duration:** 15 min
- **Tasks:** 2
- **Files modified:** 8
- **Files created:** 1

## Accomplishments
- ReconstructionStage reads tracklet_groups for per-frame camera membership (no RANSAC)
- Frames below min_cameras are dropped with reason "insufficient_views"
- Short gaps (<=max_interp_gap) filled by linear interpolation of control points
- Interpolated frames flagged with is_low_confidence=True and confidence=0
- HDF5ExportObserver writes fish-first layout when tracklet_groups present
- Frame-major layout preserved as backward-compatible fallback
- Root attributes include config_hash, run_timestamp, calibration_path
- ReconstructionConfig extended with min_cameras, max_interp_gap, n_control_points
- Legacy FishTrack/AssociationBundle deprecated with comments

## Task Commits

1. **Tasks 1+2: ReconstructionStage rewrite + HDF5 fish-first layout** - `fda7fae` (feat)

## Files Created/Modified
- `src/aquapose/core/reconstruction/stage.py` - Full rewrite: tracklet-group-driven reconstruction
- `src/aquapose/engine/hdf5_observer.py` - Fish-first + frame-major dual layout
- `src/aquapose/engine/config.py` - ReconstructionConfig new fields
- `src/aquapose/engine/pipeline.py` - Pass new config fields to ReconstructionStage
- `src/aquapose/core/association/types.py` - AssociationBundle deprecation comment
- `src/aquapose/core/tracking/types.py` - FishTrack deprecation comment
- `tests/unit/core/reconstruction/test_reconstruction_stage.py` - Rewritten: 16 tests
- `tests/unit/engine/test_hdf5_observer.py` - Extended: 14 tests

## Decisions Made
- Kept legacy types with deprecation comments rather than deleting (visualization imports them)
- Fish-first HDF5 uses gzip compression (level 4) for datasets
- Coasted frames in tracklets do not contribute to camera count

## Deviations from Plan
- Combined Task 1 and Task 2 into a single commit (both were straightforward)
- Did not remove legacy types (FishTrack, AssociationBundle) -- deprecated instead

## Issues Encountered
- Ruff format auto-fixed 2 files on first commit attempt; re-staged and committed
- Ruff flagged `datetime.timezone.utc` -> `datetime.UTC` alias and `list()[0]` -> `next(iter())`

## Next Phase Readiness
- Full 5-stage pipeline wired: Detection -> Tracking -> Association -> Midline -> Reconstruction
- Phase 26 complete; Phase 27 (Diagnostic Visualization) can proceed

---
*Phase: 26-association-refinement-and-pipeline-wiring*
*Completed: 2026-02-27*
