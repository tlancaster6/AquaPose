---
phase: 15-stage-migrations
plan: "03"
subsystem: core-pipeline
tags: [association, ransac, centroid-clustering, cross-view, stage-protocol, structural-typing]

requires:
  - phase: 15-02
    provides: "MidlineStage in core/midline/ with AnnotatedDetection output"
  - phase: 13-engine-core
    provides: "Stage Protocol, PipelineContext, engine/config.py AssociationConfig"
provides:
  - "AssociationStage in core/association/stage.py satisfying Stage Protocol"
  - "RansacCentroidBackend in core/association/backends/ransac_centroid.py delegating to discover_births()"
  - "AssociationBundle dataclass — per-fish cross-camera detection grouping for Stage 3 output"
  - "Backend registry get_backend() for association backends"
  - "Interface tests for all association stage contracts (9 tests, all passing)"
affects: [15-04-tracking, engine-integration]

tech-stack:
  added: []
  patterns:
    - "Same TYPE_CHECKING guard pattern for PipelineContext annotation (ENG-07)"
    - "Eager calibration loading at construction (fail-fast)"
    - "Backend registry factory pattern matching midline/detection stage structure"
    - "All detections unclaimed at Stage 3 — tracking has not run yet"

key-files:
  created:
    - src/aquapose/core/association/__init__.py
    - src/aquapose/core/association/types.py
    - src/aquapose/core/association/stage.py
    - src/aquapose/core/association/backends/__init__.py
    - src/aquapose/core/association/backends/ransac_centroid.py
    - tests/unit/core/association/__init__.py
    - tests/unit/core/association/test_association_stage.py
  modified:
    - src/aquapose/engine/config.py
    - src/aquapose/engine/stages.py

key-decisions:
  - "AssociationBundle uses fish_idx (0-indexed per-frame) not fish_id — persistent IDs assigned by Tracking (Stage 4)"
  - "RansacCentroidBackend delegates to existing discover_births() — port behavior, not rewrite"
  - "ALL detections are unclaimed at Stage 3 — no tracks exist yet; v1.0 discover_births was called only on unclaimed remainder"
  - "AssociationStage reads annotated_detections preferentially over detections — unwraps AnnotatedDetection to get Detection for RANSAC"
  - "AssociationConfig gains expected_count, min_cameras, reprojection_threshold — no longer an empty placeholder"

patterns-established:
  - "core/association/ mirrors core/detection/ and core/midline/ structure: types.py + stage.py + backends/__init__.py + backends/<name>.py"
  - "Backend eagerly loads calibration in __init__ for fail-fast behavior"

requirements-completed: [STG-03]

duration: 15min
completed: "2026-02-26"
---

# Phase 15 Plan 03: Association Stage Migration Summary

**AssociationStage (Stage 3) extracted as core/association/ package, delegating RANSAC centroid clustering to v1.0 discover_births() with all detections treated as unclaimed**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-02-26T01:49:00Z
- **Completed:** 2026-02-26T02:04:03Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- AssociationStage implemented as a pure Stage Protocol implementor in core/association/ with no runtime engine/ imports
- RansacCentroidBackend loads calibration models eagerly at construction and delegates directly to the v1.0 discover_births() RANSAC algorithm
- AssociationBundle dataclass introduces fish_idx (per-frame 0-indexed position) as distinct from the persistent fish_id assigned by Tracking
- AssociationConfig extended with expected_count, min_cameras, reprojection_threshold fields (was empty placeholder)
- 9 interface tests verify Protocol conformance, context population, import boundary, empty detection handling, and backend registry

## Task Commits

1. **Task 1: Create core/association/ module** - `9d1937d` (feat)
2. **Task 2: Interface tests for AssociationStage** - `551d5c9` (test)

## Files Created/Modified

- `src/aquapose/core/association/__init__.py` - Package exports AssociationStage, AssociationBundle
- `src/aquapose/core/association/types.py` - AssociationBundle dataclass
- `src/aquapose/core/association/stage.py` - AssociationStage reading context.detections/annotated_detections, writing associated_bundles
- `src/aquapose/core/association/backends/__init__.py` - Backend registry factory
- `src/aquapose/core/association/backends/ransac_centroid.py` - RansacCentroidBackend wrapping discover_births()
- `src/aquapose/engine/config.py` - AssociationConfig extended with 3 RANSAC parameters
- `src/aquapose/engine/stages.py` - associated_bundles docstring updated to reference AssociationBundle type
- `tests/unit/core/association/__init__.py` - Test package init
- `tests/unit/core/association/test_association_stage.py` - 9 interface tests

## Decisions Made

- **fish_idx vs fish_id:** Association uses fish_idx (per-frame ordering) not persistent fish_id. Persistent identity is assigned by Tracking (Stage 4), which consumes these bundles as input.
- **Delegation over reimplementation:** RansacCentroidBackend calls `discover_births()` from `aquapose.tracking.associate` directly. The existing algorithm is correct and battle-tested; no reimplementation needed.
- **ALL detections unclaimed at Stage 3:** In v1.0, `discover_births()` was called only on detections not claimed by existing tracks. In Stage 3, tracking has not yet run, so every detection is unclaimed. This is an architectural shift but the algorithm is identical.
- **Prefer annotated_detections:** Stage.run() reads `context.annotated_detections` (Stage 2 output) preferentially; falls back to `context.detections` (Stage 1 output). The AnnotatedDetection wrapper is unwrapped to extract the underlying Detection for RANSAC input.
- **AssociationConfig no longer empty:** Gains `expected_count`, `min_cameras`, `reprojection_threshold` — the three core RANSAC parameters for centroid clustering.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- Pre-commit ruff format reformatted `stage.py` and `test_association_stage.py` (long lines). Re-staged and committed successfully on second attempt. Not a deviation — standard pre-commit workflow.
- One pre-existing failing test in `tests/unit/tracking/test_tracker.py::test_near_claim_penalty_suppresses_ghost` — unrelated to new code, out of scope.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- AssociationStage complete and tested; ready for Phase 15-04 (TrackingStage migration)
- TrackingStage will consume associated_bundles produced by this stage
- No blockers
