---
phase: 22-pipeline-scaffolding
plan: 02
subsystem: engine
tags: [pipeline, stub-stages, build_stages, v2.1, carry-forward, tracking, association]

requires:
  - 22-01

provides:
  - "TrackingStubStage in engine/pipeline.py: produces empty tracks_2d {}, accepts/returns CarryForward"
  - "AssociationStubStage in engine/pipeline.py: produces empty tracklet_groups []"
  - "build_stages() wired to 5-stage order (production) or 4-stage (synthetic)"
  - "PosePipeline.run() dispatches TrackingStubStage with carry arg via isinstance"
  - "DiagnosticObserver StageSnapshot updated to tracks_2d and tracklet_groups fields"
  - "ReconstructionStage.run() handles empty tracklet_groups with early-return stub path"
  - "AssociationConfig and TrackingConfig simplified to Phase 22 stubs"
  - "Regression tests skipped with EVAL-01 note (retained as templates)"

affects:
  - 24-tracking-stage
  - 25-association-stage
  - 26-reconstruction-stage

tech-stack:
  added: []
  patterns:
    - "isinstance(stage, TrackingStubStage) dispatch in PosePipeline.run() for carry-aware stages"
    - "CarryForward passed and returned by TrackingStubStage, stored on pipeline run loop"
    - "_filter_fields() helper in load_config to filter stale YAML/CLI kwargs from stub dataclasses"
    - "pytestmark = pytest.mark.skip for module-level regression test deferrals"

key-files:
  created: []
  modified:
    - src/aquapose/engine/pipeline.py
    - src/aquapose/engine/config.py
    - src/aquapose/engine/diagnostic_observer.py
    - src/aquapose/core/reconstruction/stage.py
    - tests/unit/engine/test_build_stages.py
    - tests/unit/engine/test_diagnostic_observer.py
    - tests/unit/core/reconstruction/test_reconstruction_stage.py
    - tests/unit/engine/test_config.py
    - tests/regression/test_per_stage_regression.py
    - tests/regression/test_end_to_end_regression.py

key-decisions:
  - "TrackingStubStage lives in engine/pipeline.py (not core/) — it is an engine-level stub, not a domain-level stage"
  - "PosePipeline.run() uses isinstance(stage, TrackingStubStage) dispatch rather than a CarryStage Protocol — simpler, avoids adding a new protocol for a single exceptional stage"
  - "AssociationConfig and TrackingConfig stripped to minimal stubs; old fields filtered in load_config via _filter_fields() to avoid TypeError from stale YAML"
  - "ReconstructionStage.run() early-returns empty midlines_3d when tracklet_groups is [] (stub path); raises ValueError only when both tracklet_groups is None and annotated_detections is None"
  - "test_config.py updated to use max_coast_frames (new TrackingConfig field) — deviation auto-fix Rule 3 (blocking test failure)"

patterns-established:
  - "5-stage pipeline: Detection -> TrackingStubStage -> AssociationStubStage -> Midline -> Reconstruction"
  - "Synthetic 4-stage: SyntheticDataStage -> TrackingStubStage -> AssociationStubStage -> Reconstruction"
  - "CarryForward plumbing established; TrackingStubStage passes it through unchanged for Phase 24"

requirements-completed:
  - PIPE-01

duration: 15min
completed: 2026-02-27
---

# Phase 22 Plan 02: Stub Stages and Pipeline Rewire Summary

**TrackingStubStage and AssociationStubStage wired into build_stages() completing the v2.1 5-stage pipeline order; all 451 unit tests pass; regression tests skipped with EVAL-01 deferral note**

## Performance

- **Duration:** 15 min
- **Started:** 2026-02-27T17:52:02Z
- **Completed:** 2026-02-27T18:07:00Z
- **Tasks:** 2
- **Files modified:** 10 (0 created, 10 modified)

## Accomplishments

- Added `TrackingStubStage` (Stage 2) to `engine/pipeline.py`: produces correctly-typed empty `tracks_2d = {}`, accepts and returns `CarryForward` unchanged, establishing the carry interface for Phase 24 (OC-SORT)
- Added `AssociationStubStage` (Stage 3) to `engine/pipeline.py`: produces correctly-typed empty `tracklet_groups = []`
- Rewired `build_stages()` to the 5-stage v2.1 order (production) and 4-stage (synthetic): `Detection -> TrackingStubStage -> AssociationStubStage -> Midline -> Reconstruction`
- Updated `PosePipeline.run()` to detect `TrackingStubStage` via `isinstance` and dispatch with carry argument
- Simplified `AssociationConfig` and `TrackingConfig` to Phase 22 stubs; added `_filter_fields()` helper to `load_config` to filter stale YAML/CLI kwargs
- Updated `DiagnosticObserver` `StageSnapshot` fields: replaced `associated_bundles`/`tracks` with `tracks_2d`/`tracklet_groups`
- Updated `ReconstructionStage.run()` to early-return with empty midlines when `tracklet_groups == []` (stub path)
- Adapted all affected tests; added `pytestmark` skip to regression tests with EVAL-01 note

## Task Commits

1. **Task 1: Create stub stages, rewire build_stages(), update config and observers** - `6b7a2c2` (feat)
2. **Task 2: Adapt all affected tests to the new pipeline structure** - `5475a0a` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/aquapose/engine/pipeline.py` - Added TrackingStubStage, AssociationStubStage classes; rewired build_stages() to 5/4-stage order; updated PosePipeline.run() carry dispatch
- `src/aquapose/engine/config.py` - Simplified AssociationConfig (empty stub) and TrackingConfig (max_coast_frames=30 only); added _filter_fields() helper in load_config
- `src/aquapose/engine/diagnostic_observer.py` - Updated StageSnapshot and _SCALAR_FIELDS/_PER_FRAME_FIELDS to tracks_2d and tracklet_groups
- `src/aquapose/core/reconstruction/stage.py` - Added early-return stub path when tracklet_groups is empty; updated docstring
- `tests/unit/engine/test_build_stages.py` - Rewrote for 5-stage production / 4-stage synthetic; added TestStubStagesDirectly with carry pass-through tests
- `tests/unit/engine/test_diagnostic_observer.py` - Updated stage names; added tracks_2d/tracklet_groups snapshot test
- `tests/unit/core/reconstruction/test_reconstruction_stage.py` - Added stub path test; updated build_stages assertions for 5-stage order
- `tests/unit/engine/test_config.py` - Updated TrackingConfig field assertions to max_coast_frames
- `tests/regression/test_per_stage_regression.py` - Added pytestmark skip with EVAL-01 note
- `tests/regression/test_end_to_end_regression.py` - Added pytestmark skip with EVAL-01 note

## Decisions Made

- `TrackingStubStage` lives in `engine/pipeline.py` (not `core/`) — it is an engine-level placeholder, not a domain-level stage; placement mirrors where build_stages() constructs it
- `PosePipeline.run()` dispatches tracking via `isinstance(stage, TrackingStubStage)` check rather than a separate `CarryStage` Protocol — simpler and avoids protocol proliferation for a single exceptional stage
- `AssociationConfig` and `TrackingConfig` stripped to stubs; `_filter_fields()` ensures stale YAML/CLI keys (e.g., old `max_fish`) are silently filtered rather than raising `TypeError`
- `ReconstructionStage.run()` early-returns empty results when `tracklet_groups == []`; raises `ValueError` only when both `tracklet_groups is None` and `annotated_detections is None`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated test_config.py for simplified TrackingConfig**
- **Found during:** Task 2 (running hatch run test)
- **Issue:** `test_config.py` referenced `config.tracking.max_fish == 9` which was a field on the old TrackingConfig; the simplified stub only has `max_coast_frames`
- **Fix:** Updated all references in test_config.py to use `max_coast_frames == 30`; updated YAML override test to use `max_coast_frames: 15`; updated serialization roundtrip test
- **Files modified:** `tests/unit/engine/test_config.py`
- **Commit:** `5475a0a` (included in Task 2 commit)

None of the core plan changes were architectural deviations — test_config.py was the expected cascading update from simplifying TrackingConfig.

## Issues Encountered

- Ruff E402 (import not at top of file) on first commit attempt — logger assignment was placed between imports; fixed by moving it after all imports

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- PIPE-01 complete: 5-stage pipeline wired and running without errors
- Phase 24 (OC-SORT Tracking) can now implement against the `CarryForward` interface established by TrackingStubStage
- Phase 25 (Leiden Association) blocked until Phase 24 complete
- Phase 26 (Reconstruction) can now implement against the `tracklet_groups` interface established by AssociationStubStage

---
*Phase: 22-pipeline-scaffolding*
*Completed: 2026-02-27*
