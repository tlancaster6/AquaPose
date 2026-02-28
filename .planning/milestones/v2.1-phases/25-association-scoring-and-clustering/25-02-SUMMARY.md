---
phase: 25-association-scoring-and-clustering
plan: 02
subsystem: association
tags: [leiden-clustering, fragment-merging, must-not-link, association-stage]

requires:
  - phase: 25-01
    provides: score_all_pairs(), AssociationConfig with scoring/clustering fields
  - phase: 23-refractive-lookup-tables
    provides: load_forward_luts(), load_inverse_luts()
  - phase: 24-per-camera-2d-tracking
    provides: Tracklet2D, tracks_2d per-camera dict
  - phase: 22-pipeline-scaffolding
    provides: TrackletGroup, PipelineContext, build_stages()
provides:
  - build_must_not_link() for same-camera detected-overlap constraints
  - cluster_tracklets() via Leiden algorithm with must-not-link enforcement
  - merge_fragments() with linear interpolation for gap frames
  - AssociationStage replacing AssociationStubStage in pipeline
  - ClusteringConfigLike Protocol for IB-003 boundary
  - HandoffState dataclass for future inter-frame state
affects: [26-midline, 27-reconstruction]

tech-stack:
  added: [igraph>=0.11, leidenalg>=0.10]
  patterns: [ClusteringConfigLike Protocol, deferred imports in stage.run()]

key-files:
  created:
    - src/aquapose/core/association/clustering.py
    - src/aquapose/core/association/stage.py
    - tests/unit/core/association/test_clustering.py
  modified:
    - src/aquapose/core/association/__init__.py
    - src/aquapose/core/association/types.py
    - src/aquapose/engine/pipeline.py
    - tests/unit/engine/test_build_stages.py
    - tests/unit/core/reconstruction/test_reconstruction_stage.py
    - tests/unit/engine/test_diagnostic_observer.py
    - pyproject.toml

key-decisions:
  - "igraph + leidenalg for community detection — mature, well-tested Leiden implementation"
  - "Must-not-link enforcement via eviction of lower-affinity tracklet to singleton"
  - "Fragment merging uses linear centroid interpolation; gap frames tagged 'interpolated'"
  - "AssociationStage gracefully degrades to empty groups when LUTs unavailable"

patterns-established:
  - "ClusteringConfigLike Protocol: 4-field subset of AssociationConfig for clustering"
  - "Deferred imports in AssociationStage.run() to break circular engine->core dependency"
  - "pytest.importorskip('leidenalg') guard for clustering tests"

requirements-completed: [ASSOC-02]

duration: 15min
completed: 2026-02-27
---

# Plan 25-02: Leiden Clustering and Fragment Merging Summary

**Leiden-based tracklet clustering with must-not-link constraints, fragment merging, and full AssociationStage integration**

## Performance

- **Duration:** 15 min
- **Tasks:** 3 (clustering, stage, tests)
- **Files modified:** 11

## Accomplishments
- Implemented build_must_not_link() detecting same-camera detection-backed temporal overlaps
- Implemented cluster_tracklets() using igraph connected components + Leiden RBConfigurationVertexPartition with must-not-link enforcement
- Implemented merge_fragments() with same-camera non-overlapping fragment merging and linear centroid/bbox interpolation for gap frames
- Created AssociationStage replacing AssociationStubStage, with LUT loading, scoring, clustering, and fragment merging pipeline
- Added HandoffState dataclass to types.py for future inter-frame state handoff
- Added igraph and leidenalg to project dependencies
- Updated all existing tests referencing AssociationStubStage across 4 test files
- 12 unit tests all passing (with leidenalg importorskip guard)
- Full test suite: 525 passed, 7 skipped

## Task Commits

1. **All tasks combined:** `d9cbab7` (feat) — clustering, stage, tests, pipeline integration

## Files Created/Modified
- `src/aquapose/core/association/clustering.py` — Leiden clustering, must-not-link, fragment merging
- `src/aquapose/core/association/stage.py` — AssociationStage with LUT loading and graceful degradation
- `src/aquapose/core/association/types.py` — Added HandoffState dataclass
- `src/aquapose/core/association/__init__.py` — Updated exports for all new symbols
- `src/aquapose/engine/pipeline.py` — Replaced AssociationStubStage with AssociationStage
- `pyproject.toml` — Added igraph>=0.11, leidenalg>=0.10
- `tests/unit/core/association/test_clustering.py` — 12 tests
- `tests/unit/engine/test_build_stages.py` — Updated stub references
- `tests/unit/core/reconstruction/test_reconstruction_stage.py` — Updated stub references
- `tests/unit/engine/test_diagnostic_observer.py` — Updated stage name string

## Decisions Made
- igraph + leidenalg chosen over custom graph code for Leiden community detection
- Must-not-link enforcement via iterative eviction (lower affinity tracklet becomes singleton)
- Fragment merging discards coasted frames, keeps only detected frames, interpolates gaps
- AssociationStage uses deferred imports pattern to avoid circular dependencies
- Combined all tasks into single commit since they are tightly coupled

## Deviations from Plan
None significant — plan executed as specified.

## Issues Encountered
- Pre-commit lint errors (B007 unused loop vars, F401 unused imports, I001 import ordering, E402 import after importorskip, SIM108 ternary) — all fixed
- Subagent spawning not possible within Claude Code session — executed directly

## User Setup Required
None — igraph and leidenalg install automatically via pip.

## Next Phase Readiness
- AssociationStage produces TrackletGroup list with fish_id, tracklets, confidence
- Pipeline fully wired: Detection -> Tracking -> Association -> Midline -> Reconstruction
- HandoffState ready for future inter-frame state management

---
*Phase: 25-association-scoring-and-clustering*
*Completed: 2026-02-27*
