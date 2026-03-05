---
phase: 65-frame-selection-and-dataset-assembly
plan: 01
subsystem: training
tags: [frame-selection, kmeans, curvature, subsampling, scipy]

# Dependency graph
requires:
  - phase: 63-pseudo-labeling-pipeline
    provides: "Midline3D objects with control_points for curvature computation"
provides:
  - "temporal_subsample: select every Kth frame from sorted indices"
  - "filter_empty_frames: remove frames with no 3D reconstructions"
  - "compute_curvature: mean absolute curvature from B-spline control points"
  - "diversity_sample: K-means curvature binning with per-bin frame sampling"
affects: [65-02, dataset-assembly, training-data-prep]

# Tech tracking
tech-stack:
  added: [scipy.cluster.vq.kmeans2]
  patterns: [finite-difference curvature from control points, curvature-binned diversity sampling]

key-files:
  created:
    - src/aquapose/training/frame_selection.py
    - tests/unit/training/test_frame_selection.py
  modified:
    - src/aquapose/training/__init__.py

key-decisions:
  - "Used scipy kmeans2 instead of sklearn KMeans to avoid heavy dependency"
  - "Finite-difference curvature from tangent vectors (no scipy.interpolate needed)"

patterns-established:
  - "Curvature computation: T[i] = cp[i+1] - cp[i], k[i] = |dT| / avg(|T|)"
  - "Diversity sampling: per-fish curvature clustering with union of selected frames"

requirements-completed: [FRAME-01, FRAME-02, FRAME-03]

# Metrics
duration: 4min
completed: 2026-03-05
---

# Phase 65 Plan 01: Frame Selection Summary

**Frame selection utilities with temporal subsampling, empty-frame filtering, and curvature-based K-means diversity sampling using scipy**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-05T20:08:05Z
- **Completed:** 2026-03-05T20:12:00Z
- **Tasks:** 1 (TDD: RED + GREEN + REFACTOR)
- **Files modified:** 3

## Accomplishments
- Implemented temporal_subsample for every-Kth-frame selection with automatic sorting
- Implemented filter_empty_frames to remove frames with no 3D reconstructions
- Implemented compute_curvature using finite differences of tangent vectors from B-spline control points
- Implemented diversity_sample using scipy kmeans2 to cluster per-fish curvatures and sample equally from bins
- Full test coverage with 12 unit tests covering edge cases, determinism, and correctness

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests** - `a9efef6` (test)
2. **Task 1 (GREEN): Implementation** - `115f288` (feat)
3. **Task 1 (REFACTOR): Typecheck fix** - `e83718d` (fix)

## Files Created/Modified
- `src/aquapose/training/frame_selection.py` - Frame selection functions: temporal_subsample, filter_empty_frames, compute_curvature, diversity_sample
- `tests/unit/training/test_frame_selection.py` - 12 unit tests covering all functions and edge cases
- `src/aquapose/training/__init__.py` - Added exports for all 4 new functions

## Decisions Made
- Used scipy.cluster.vq.kmeans2 instead of sklearn KMeans to avoid heavy dependency (as suggested in plan)
- Used finite-difference curvature computation from tangent vectors rather than scipy.interpolate B-spline evaluation -- simpler and avoids needing the knot vector
- Added pyright ignore comment for scipy kmeans2 seed parameter (valid at runtime, missing from type stubs)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test expectation for temporal_subsample sorting**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Test expected `temporal_subsample([9,3,0,6], step=3)` to return all 4 elements `[0,3,6,9]`, but step=3 correctly returns only `[0,9]`
- **Fix:** Changed test to use step=1 for the sorting verification test
- **Verification:** All tests pass
- **Committed in:** 115f288

**2. [Rule 1 - Bug] Suppressed false-positive pyright error on scipy kmeans2**
- **Found during:** Task 1 (verification with `hatch run check`)
- **Issue:** basedpyright reports "No parameter named seed" for scipy.cluster.vq.kmeans2, but the parameter exists at runtime
- **Fix:** Added `# pyright: ignore[reportCallIssue]` comment
- **Verification:** `hatch run check` passes cleanly
- **Committed in:** e83718d

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Minor test correction and type-stub workaround. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Frame selection functions ready for use by 65-02 (dataset assembly pipeline)
- All functions are pure (no side effects), easy to compose in CLI workflow
- diversity_sample accepts the same midlines_3d structure used by pseudo_label_cli

---
*Phase: 65-frame-selection-and-dataset-assembly*
*Completed: 2026-03-05*
