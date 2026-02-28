---
phase: 26-association-refinement-and-pipeline-wiring
plan: 01
subsystem: association
tags: [triangulation, ray-ray, refinement, eviction, confidence]

requires:
  - phase: 25-cross-view-association
    provides: Leiden clustering, merge_fragments, TrackletGroup
provides:
  - refine_clusters() with per-frame 3D triangulation and tracklet eviction
  - Per-frame confidence on TrackletGroup
  - RefinementConfigLike protocol for IB-003 boundary
affects: [26-02-midline, 26-03-reconstruction]

tech-stack:
  added: []
  patterns: [robust consensus via best-half pairwise midpoints, point-to-ray distance eviction metric]

key-files:
  created:
    - src/aquapose/core/association/refinement.py
    - tests/unit/core/association/test_refinement.py
  modified:
    - src/aquapose/core/association/types.py
    - src/aquapose/core/association/stage.py
    - src/aquapose/core/association/__init__.py
    - src/aquapose/engine/config.py

key-decisions:
  - "Robust consensus: use best 50% of pairwise midpoints (sorted by distance) to resist outlier corruption"
  - "Eviction metric is point-to-ray distance in metres (not pixel reprojection) since only ForwardLUTs are available"
  - "Test threshold 0.5m for synthetic unit-distance geometry; production default 0.025m"

patterns-established:
  - "RefinementConfigLike: Protocol pattern for refinement config (same as AssociationConfigLike, ClusteringConfigLike)"

requirements-completed: [ASSOC-03]

duration: 12min
completed: 2026-02-27
---

# Plan 26-01: Cluster Refinement Summary

**Per-frame 3D triangulation cluster refinement with robust consensus, tracklet eviction, and per-frame confidence scoring**

## Performance

- **Duration:** 12 min
- **Tasks:** 2
- **Files modified:** 6
- **Files created:** 2

## Accomplishments
- refine_clusters() validates tracklet membership via ray-ray consensus triangulation
- Outlier tracklets evicted to singleton groups with confidence=0.1
- Per-frame confidence computed from ray convergence quality (1.0 - mean_dist/threshold)
- Robust consensus uses best 50% of pairwise midpoints to resist outlier corruption
- AssociationStage.run() calls refine_clusters() after merge_fragments()

## Task Commits

1. **Task 1: Add refinement config fields and update TrackletGroup** - `b3285da` (feat)
2. **Task 2: Implement refine_clusters() and wire into AssociationStage** - `cd783de` (feat)

## Files Created/Modified
- `src/aquapose/core/association/refinement.py` - Core refinement module with refine_clusters(), consensus, eviction, confidence
- `src/aquapose/core/association/types.py` - per_frame_confidence field on TrackletGroup
- `src/aquapose/core/association/stage.py` - Wire refine_clusters() call after merge_fragments()
- `src/aquapose/core/association/__init__.py` - Export refine_clusters, RefinementConfigLike
- `src/aquapose/engine/config.py` - eviction_reproj_threshold, min_cameras_refine, refinement_enabled
- `tests/unit/core/association/test_refinement.py` - 10 unit tests

## Decisions Made
- Used robust consensus (best 50% of pairwise midpoints sorted by distance) instead of simple mean/median to resist outlier tracklets corrupting the consensus point
- Eviction metric uses point-to-ray distance in metres rather than pixel reprojection since only ForwardLUTs (pixel->ray) are available, not inverse projection
- Test mock threshold set to 0.5m for synthetic geometry where cameras are at unit distances

## Deviations from Plan
None significant - algorithm core matches plan; robust consensus approach added for correctness.

## Issues Encountered
- Initial consensus computation (mean of all pairwise midpoints) was corrupted by outlier rays, causing all tracklets to be evicted. Fixed by using best-half median approach.

## Next Phase Readiness
- refine_clusters() ready for downstream consumers
- Per-frame confidence available for reconstruction and HDF5 output
- Plan 26-02 (Midline orientation) can proceed

---
*Phase: 26-association-refinement-and-pipeline-wiring*
*Completed: 2026-02-27*
