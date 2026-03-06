---
phase: 09-curve-based-optimization-as-a-replacement-for-triangulation
plan: 02
subsystem: reconstruction
tags: [curve-optimizer, b-spline, triangulation, pipeline, diagnostics]

# Dependency graph
requires:
  - phase: 09-01
    provides: CurveOptimizer implementation with coarse-to-fine B-spline optimization
provides:
  - --method flag in diagnose_pipeline.py dispatching to curve or triangulation
  - CurveOptimizer and CurveOptimizerConfig exported from aquapose.reconstruction
  - Side-by-side comparison capability for both reconstruction methods
affects: [phase 10, curve optimizer validation, triangulation replacement decision]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Method dispatch via --method CLI flag, single stage 5 block with if/else branching

key-files:
  created: []
  modified:
    - scripts/diagnose_pipeline.py
    - src/aquapose/reconstruction/__init__.py

key-decisions:
  - "--method flag defaults to triangulation to preserve existing behavior; curve opt-in"
  - "CurveOptimizer dispatch is a single call (optimizer.optimize_midlines) — no sub-timing breakdown needed"
  - "Both methods produce identical midlines_3d list format — downstream HDF5 and viz unchanged"

patterns-established:
  - "Method dispatch pattern: args.method == 'curve' branch before existing triangulation block"

requirements-completed: [RECON-03, RECON-04, RECON-05]

# Metrics
duration: 5min
completed: 2026-02-23
---

# Phase 9 Plan 02: Integration and Diagnostic Wiring Summary

**CurveOptimizer wired into diagnose_pipeline.py via --method flag, enabling side-by-side comparison of curve vs triangulation reconstruction on real data**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-23T04:59:00Z
- **Completed:** 2026-02-23T05:04:00Z
- **Tasks:** 1 of 2 (Task 2 is checkpoint:human-verify — awaiting user verification)
- **Files modified:** 2

## Accomplishments

- `--method {triangulation,curve}` flag added to diagnose_pipeline.py with full dispatch logic
- CurveOptimizer and CurveOptimizerConfig re-exported from aquapose.reconstruction public API
- Stage 5 dispatches to CurveOptimizer.optimize_midlines() when --method curve is specified
- Both methods produce identical midlines_3d output format — HDF5 writer and all visualizations work unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire curve optimizer into package exports and diagnose_pipeline.py** - `8557f4b` (feat) + `8631c94` (fix: contiguous control points for L-BFGS)

**Plan metadata:** (pending — awaiting checkpoint completion)

## Files Created/Modified

- `scripts/diagnose_pipeline.py` - Added --method argument, Stage 5 curve/triangulation dispatch
- `src/aquapose/reconstruction/__init__.py` - Added CurveOptimizer, CurveOptimizerConfig, optimize_midlines exports

## Decisions Made

- `--method` defaults to `"triangulation"` to preserve existing behavior; users must explicitly opt into curve method
- Curve optimizer dispatch is a single call with no sub-timing (the optimizer handles all internal stages)
- Both methods produce `list[dict[int, Midline3D]]` — no downstream changes required

## Deviations from Plan

None — Task 1 was already implemented in the prior session (commits 8557f4b and 8631c94). Verified all done criteria satisfied:
- `--help` shows `--method {triangulation,curve}` flag
- `hatch run lint` passes
- `CurveOptimizer` and `CurveOptimizerConfig` importable from `aquapose.reconstruction`

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Awaiting Task 2: user runs both `--method triangulation` and `--method curve` on real data and evaluates:
  - Speed (curve method should be faster than ~76s for 30 frames)
  - Visual quality of spline overlays
  - Residual values and arc length distributions
  - Reconstruction success rate
- If curve method validated: Phase 10 will replace triangulation with curve optimizer as the default

---
*Phase: 09-curve-based-optimization-as-a-replacement-for-triangulation*
*Completed: 2026-02-23*
