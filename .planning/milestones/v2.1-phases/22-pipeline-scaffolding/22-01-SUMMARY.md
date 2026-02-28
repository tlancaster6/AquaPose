---
phase: 22-pipeline-scaffolding
plan: 01
subsystem: core
tags: [domain-types, pipeline, dataclasses, v2.1, tracking, association]

requires: []

provides:
  - "Tracklet2D frozen dataclass in core/tracking/types.py (per-camera temporal tracklet)"
  - "TrackletGroup frozen dataclass in core/association/types.py (cross-camera identity cluster)"
  - "CarryForward frozen dataclass in core/context.py (cross-batch 2D tracker state)"
  - "PipelineContext updated: tracks_2d + tracklet_groups replace associated_bundles + tracks"
  - "Legacy code deleted: TrackingStage, AssociationStage, FishTracker, ransac_centroid_cluster"
  - "FishTrack/TrackState/TrackHealth moved to core/tracking/types.py (reconstruction compatibility)"

affects:
  - 22-02-pipeline-scaffolding
  - 24-tracking-stage
  - 25-association-stage
  - 26-reconstruction-stage

tech-stack:
  added: []
  patterns:
    - "Frozen dataclasses for immutable pipeline data contracts"
    - "Import boundary preserved: association types use generic tuple (not Tracklet2D) to avoid cross-package imports within core/"
    - "TYPE_CHECKING imports updated from deleted tracker.py to aquapose.core.tracking"

key-files:
  created: []
  modified:
    - src/aquapose/core/tracking/types.py
    - src/aquapose/core/association/types.py
    - src/aquapose/core/context.py
    - src/aquapose/core/tracking/__init__.py
    - src/aquapose/core/association/__init__.py
    - src/aquapose/core/__init__.py
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/engine/pipeline.py
    - src/aquapose/tracking/__init__.py
    - src/aquapose/reconstruction/midline.py
    - src/aquapose/visualization/midline_viz.py
    - src/aquapose/visualization/triangulation_viz.py
    - tests/unit/core/reconstruction/test_reconstruction_stage.py
    - tests/unit/engine/test_build_stages.py
    - tests/unit/engine/test_stages.py

key-decisions:
  - "FishTrack/TrackState/TrackHealth moved to core/tracking/types.py instead of being deleted — reconstruction and visualization stages still reference them via TYPE_CHECKING imports until Phase 26"
  - "TrackletGroup.tracklets uses generic tuple at runtime (not tuple[Tracklet2D]) to preserve core/ import boundary per ENG-07"
  - "CarryForward.tracks_2d_state uses dict (not dict[str, Any]) to avoid importing Any — element types documented in docstring only"
  - "build_stages() temporarily returns 3 stages (detection/midline/reconstruction) until Phase 22-02 adds TrackingStage and AssociationStage stubs"
  - "reconstruction/stage.py iterates annotated_detections directly (not ctx.tracks) in v2.1 transition — Phase 26 wires TrackletGroups"

patterns-established:
  - "v2.1 pipeline ordering: Detection -> 2D Tracking -> Association -> Midline -> Reconstruction"
  - "Frozen dataclasses for all pipeline domain types (Tracklet2D, TrackletGroup, CarryForward)"
  - "AssociationBundle retained alongside TrackletGroup for reconstruction compatibility until Phase 26"

requirements-completed:
  - PIPE-01

duration: 11min
completed: 2026-02-27
---

# Phase 22 Plan 01: Domain Types and Legacy Deletion Summary

**Tracklet2D/TrackletGroup/CarryForward frozen dataclasses established as v2.1 data contracts; PipelineContext reordered to Detection -> 2D Tracking -> Association -> Midline -> Reconstruction; all legacy tracking/association source deleted (13 files, 5500 lines)**

## Performance

- **Duration:** 11 min
- **Started:** 2026-02-27T11:17:06Z
- **Completed:** 2026-02-27T11:28:55Z
- **Tasks:** 2
- **Files modified:** 23 (9 deleted, 14 modified)

## Accomplishments

- Defined Tracklet2D (per-camera temporal tracklet), TrackletGroup (cross-camera identity cluster), and CarryForward (cross-batch 2D tracker state) as frozen dataclasses
- Updated PipelineContext to v2.1 stage ordering — `tracks_2d` and `tracklet_groups` replace `associated_bundles` and `tracks`
- Deleted all legacy tracking/association code: TrackingStage, AssociationStage, HungarianBackend, RansacCentroidBackend, FishTracker, ransac_centroid_cluster, TrackingWriter plus their 5 test files (5500 lines removed)
- Moved FishTrack/TrackState/TrackHealth into core/tracking/types.py for reconstruction/visualization compatibility

## Task Commits

1. **Task 1: Define Tracklet2D, TrackletGroup, CarryForward and update PipelineContext** - `3a014fa` (feat)
2. **Task 2: Delete legacy tracking/association code and update consumers** - `634efc7` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/aquapose/core/tracking/types.py` - Added Tracklet2D frozen dataclass; moved FishTrack/TrackState/TrackHealth from tracking/tracker.py
- `src/aquapose/core/association/types.py` - Added TrackletGroup frozen dataclass; kept AssociationBundle for reconstruction compatibility
- `src/aquapose/core/context.py` - Added CarryForward frozen dataclass; updated PipelineContext with v2.1 fields (tracks_2d, tracklet_groups)
- `src/aquapose/core/tracking/__init__.py` - Updated exports (Tracklet2D, FishTrack, TrackState, TrackHealth); removed TrackingStage
- `src/aquapose/core/association/__init__.py` - Updated exports (TrackletGroup, AssociationBundle); removed AssociationStage
- `src/aquapose/core/__init__.py` - Added CarryForward; removed deleted stage classes
- `src/aquapose/core/reconstruction/stage.py` - Updated run() to iterate annotated_detections (ctx.tracks removed)
- `src/aquapose/engine/pipeline.py` - Removed AssociationStage/TrackingStage from build_stages; stubs in Phase 22-02
- `src/aquapose/tracking/__init__.py` - Replaced with minimal stub (all contents deleted)
- `src/aquapose/reconstruction/midline.py` - Updated TYPE_CHECKING import to aquapose.core.tracking
- `src/aquapose/visualization/midline_viz.py` - Updated TYPE_CHECKING import to aquapose.core.tracking
- `src/aquapose/visualization/triangulation_viz.py` - Updated TYPE_CHECKING imports to aquapose.core.tracking
- **Deleted:** src/aquapose/core/tracking/stage.py, src/aquapose/core/tracking/backends/*, src/aquapose/core/association/stage.py, src/aquapose/core/association/backends/*, src/aquapose/tracking/tracker.py, src/aquapose/tracking/associate.py, src/aquapose/tracking/writer.py
- **Deleted:** tests/unit/core/tracking/test_tracking_stage.py, tests/unit/core/association/test_association_stage.py, tests/unit/tracking/test_tracker.py, tests/unit/tracking/test_associate.py, tests/unit/tracking/test_writer.py

## Decisions Made

- FishTrack/TrackState/TrackHealth moved into core/tracking/types.py (not deleted): reconstruction/stage.py and visualization still reference them via TYPE_CHECKING imports, and moving rather than deleting avoids breaking them until Phase 26 fully updates reconstruction
- TrackletGroup.tracklets uses generic `tuple` (not `tuple[Tracklet2D]`) at runtime to preserve the import boundary — association types cannot import tracking types within core/ per ENG-07; actual element type documented in docstring only
- build_stages() in engine/pipeline.py temporarily returns 3 stages (detection/midline/reconstruction) — TrackingStage and AssociationStage stubs are added in Plan 22-02

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated TYPE_CHECKING imports in reconstruction and visualization**
- **Found during:** Task 2 (deleting tracker.py)
- **Issue:** reconstruction/midline.py and visualization/*.py had TYPE_CHECKING imports from `aquapose.tracking.tracker` which was being deleted; these would break typecheck
- **Fix:** Updated 4 TYPE_CHECKING import statements to import from `aquapose.core.tracking` instead
- **Files modified:** src/aquapose/reconstruction/midline.py, src/aquapose/visualization/midline_viz.py, src/aquapose/visualization/triangulation_viz.py
- **Verification:** hatch run test — 438 passed
- **Committed in:** 634efc7 (Task 2 commit)

**2. [Rule 3 - Blocking] Updated reconstruction/stage.py to remove ctx.tracks reference**
- **Found during:** Task 2 (removing tracks field from PipelineContext)
- **Issue:** reconstruction/stage.py read ctx.tracks which was removed from PipelineContext; would raise AttributeError at runtime
- **Fix:** Updated run() to iterate annotated_detections directly with empty frame_tracks stub; updated docstring to reflect v2.1 transition state
- **Files modified:** src/aquapose/core/reconstruction/stage.py
- **Verification:** 438 tests pass; reconstruction/stage.py importable and functional
- **Committed in:** 634efc7 (Task 2 commit)

**3. [Rule 3 - Blocking] Updated engine/pipeline.py to remove deleted stage references**
- **Found during:** Task 2 (deleting AssociationStage and TrackingStage)
- **Issue:** build_stages() imported AssociationStage and TrackingStage from aquapose.core which were deleted
- **Fix:** Removed deleted stage instantiation from build_stages(); function now returns 3 stages (production) or 2 (synthetic) until stubs land in Plan 22-02
- **Files modified:** src/aquapose/engine/pipeline.py, tests/unit/engine/test_build_stages.py, tests/unit/core/reconstruction/test_reconstruction_stage.py, tests/unit/engine/test_stages.py
- **Verification:** 438 tests pass; build_stages() importable and functional
- **Committed in:** 634efc7 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (all Rule 3 - blocking issues caused by deleting legacy modules)
**Impact on plan:** All auto-fixes were necessary to keep the codebase in a buildable state after deleting the legacy modules. No scope creep — these are the expected cascading cleanup changes from the planned deletion.

## Issues Encountered

- Ruff auto-fixed import ordering on first commit attempt (re-staged and committed successfully on second attempt)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Domain types established: Tracklet2D, TrackletGroup, CarryForward ready for downstream stages
- PipelineContext v2.1 field layout locked (tracks_2d, tracklet_groups)
- Plan 22-02 (stub stages) can proceed immediately — needs TrackingStage and AssociationStage stubs
- Phase 24 (OC-SORT Tracking) ready to implement against the CarryForward interface
- Phase 25 (Association) blocked until Phase 24 complete

---
*Phase: 22-pipeline-scaffolding*
*Completed: 2026-02-27*
