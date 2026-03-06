---
phase: 25-association-scoring-and-clustering
plan: 01
subsystem: association
tags: [ray-geometry, ghost-penalty, affinity-scoring, camera-overlap]

requires:
  - phase: 23-refractive-lookup-tables
    provides: ForwardLUT.cast_ray(), InverseLUT, ghost_point_lookup(), camera_overlap_graph()
  - phase: 24-per-camera-2d-tracking
    provides: Tracklet2D with frames, centroids, frame_status
  - phase: 22-pipeline-scaffolding
    provides: AssociationConfig stub, TrackletGroup type, PipelineContext.tracks_2d
provides:
  - ray_ray_closest_point() for analytic skew-line closest approach
  - score_tracklet_pair() with ghost penalty, early termination, overlap reliability
  - score_all_pairs() with camera overlap graph filtering
  - AssociationConfigLike Protocol for IB-003 import boundary
  - Expanded AssociationConfig with 10 scoring/clustering thresholds
affects: [25-02-clustering, 26-midline, 27-reconstruction]

tech-stack:
  added: []
  patterns: [AssociationConfigLike Protocol, deferred calibration imports]

key-files:
  created:
    - src/aquapose/core/association/scoring.py
    - tests/unit/core/association/test_scoring.py
  modified:
    - src/aquapose/engine/config.py
    - src/aquapose/core/association/__init__.py

key-decisions:
  - "Pure numpy for ray_ray_closest_point — no torch dependency for geometry math"
  - "AssociationConfigLike Protocol with 7 fields consumed by scoring (not all 10 from AssociationConfig)"
  - "Ghost penalty counts 'negative' cameras (no supporting detection) as fraction of visible-excluding-pair"

patterns-established:
  - "AssociationConfigLike Protocol: same IB-003 pattern as LutConfigLike"
  - "Deferred imports for calibration.luts inside scoring functions"

requirements-completed: [ASSOC-01]

duration: 8min
completed: 2026-02-27
---

# Plan 25-01: Pairwise Association Scoring Summary

**Ray-ray closest-point scoring with ghost-point penalties for cross-camera tracklet affinity computation**

## Performance

- **Duration:** 8 min
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Expanded AssociationConfig from empty stub to 10 YAML-tunable scoring/clustering fields
- Implemented ray_ray_closest_point() with analytic skew-line formula handling intersecting, skew, and parallel rays
- Implemented score_tracklet_pair() with full SPECSEED Step 1 algorithm (inlier fraction + ghost penalty + early termination + overlap reliability weighting)
- Implemented score_all_pairs() using camera_overlap_graph for efficient pair pruning
- 10 unit tests all passing covering geometry, edge cases, and ghost penalty suppression

## Task Commits

1. **Task 1: Expand AssociationConfig and implement scoring** - `670ab3b` (feat)
2. **Task 2: Unit tests** - included in same commit (TDD-style)

## Files Created/Modified
- `src/aquapose/core/association/scoring.py` - Pairwise scoring with ray-ray distance, ghost penalty, early termination
- `src/aquapose/engine/config.py` - AssociationConfig with 10 fields
- `src/aquapose/core/association/__init__.py` - Updated exports
- `tests/unit/core/association/test_scoring.py` - 10 unit tests

## Decisions Made
- Pure numpy for ray_ray_closest_point (no torch dependency for geometry)
- Mock LUT strategy with precise grid coordinate mapping for ghost penalty tests
- Combined both tasks into single commit since tests depend on implementation

## Deviations from Plan
None - plan executed as specified.

## Issues Encountered
- Ghost penalty test initially failed due to mock InverseLUT grid index not mapping to the expected midpoint coordinates; fixed by computing exact grid coords for the (0.5, 0.0, 0.5) midpoint.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Scoring functions ready for Plan 25-02 (Leiden clustering)
- score_all_pairs() returns the weighted edge dict that cluster_tracklets() will consume
- AssociationConfig has all fields needed by both scoring and clustering

---
*Phase: 25-association-scoring-and-clustering*
*Completed: 2026-02-27*
