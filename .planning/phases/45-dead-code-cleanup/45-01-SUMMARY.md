---
phase: 45-dead-code-cleanup
plan: 01
subsystem: reconstruction
tags: [dead-code, dlt, backend-registry, config]

requires:
  - phase: 44-evaluation-harness
    provides: DLT backend as primary reconstruction backend
provides:
  - DLT-only backend registry with old triangulation/curve_optimizer deleted
  - Simplified ReconstructionConfig defaulting to backend='dlt'
  - N_SAMPLE_POINTS and SPLINE_KNOTS migrated to utils.py
affects: [reconstruction, evaluation, synthetic, visualization]

tech-stack:
  added: []
  patterns: [single-backend-registry]

key-files:
  created: []
  modified:
    - src/aquapose/core/reconstruction/utils.py
    - src/aquapose/core/reconstruction/backends/__init__.py
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/engine/config.py
    - src/aquapose/evaluation/harness.py
    - src/aquapose/synthetic/fish.py
    - src/aquapose/io/midline_writer.py
    - src/aquapose/visualization/__init__.py
    - scripts/tune_threshold.py
    - scripts/measure_baseline.py

key-decisions:
  - "Deleted triangulation_viz.py entirely -- no production callers found outside visualization __init__.py re-exports"
  - "Removed inlier_threshold, snap_threshold, max_depth from ReconstructionConfig since they were triangulation-backend-specific"
  - "Changed stage.py constructor to accept **backend_kwargs for forward compatibility"

patterns-established:
  - "Single-backend registry: get_backend() only supports 'dlt'"

requirements-completed: [CLEAN-01, CLEAN-02, CLEAN-03]

duration: 8min
completed: 2026-03-03
---

# Plan 45-01: Dead Code Cleanup Summary

**Deleted 5 dead modules (4776 lines), migrated shared constants to utils.py, simplified backend registry and config to DLT-only**

## Performance

- **Duration:** 8 min
- **Tasks:** 2
- **Files deleted:** 5
- **Files modified:** 10

## Accomplishments
- Deleted four dead reconstruction modules (triangulation.py, curve_optimizer.py, and their backend wrappers) plus triangulation_viz.py
- Migrated N_SAMPLE_POINTS and SPLINE_KNOTS constants to utils.py with proper __all__ exports
- Simplified backend registry to DLT-only, updated ReconstructionConfig to default to 'dlt' with old fields removed
- Updated all production-code import sites (stage.py, harness.py, synthetic/fish.py, io/midline_writer.py, scripts)

## Task Commits

1. **Tasks 1+2: Delete dead modules and update production code** - `8f9983b` (feat)

## Files Created/Modified
- `src/aquapose/core/reconstruction/utils.py` - Added N_SAMPLE_POINTS and SPLINE_KNOTS constants
- `src/aquapose/core/reconstruction/backends/__init__.py` - DLT-only registry
- `src/aquapose/core/reconstruction/stage.py` - Removed old backend wiring
- `src/aquapose/engine/config.py` - Simplified ReconstructionConfig
- `src/aquapose/evaluation/harness.py` - Removed TriangulationBackend, default to 'dlt'
- `src/aquapose/synthetic/fish.py` - Import from utils instead of triangulation
- `src/aquapose/io/midline_writer.py` - Import from utils instead of triangulation
- `src/aquapose/visualization/__init__.py` - Removed triangulation_viz re-exports
- `scripts/tune_threshold.py` - Updated baseline backend reference
- `scripts/measure_baseline.py` - Updated default backend to 'dlt'

## Decisions Made
- Deleted triangulation_viz.py entirely since no production callers existed
- Combined both tasks into a single atomic commit since deletions and import updates are tightly coupled

## Deviations from Plan
- Also updated scripts/tune_threshold.py and scripts/measure_baseline.py which still referenced "triangulation" backend
- Combined tasks 1 and 2 into a single commit for atomicity

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Production code is clean; test files still need updating (Plan 45-02)
- GUIDEBOOK.md and project config need updating (Plan 45-02)

---
*Phase: 45-dead-code-cleanup*
*Completed: 2026-03-03*
