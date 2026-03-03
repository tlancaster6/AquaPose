---
phase: 44-validation-and-tuning
plan: 01
subsystem: reconstruction
tags: [dlt, triangulation, evaluation, grid-search, tuning]

requires:
  - phase: 43-triangulation-rebuild
    provides: DltBackend with outlier_threshold parameter
provides:
  - ReconstructionConfig.outlier_threshold field for config-driven threshold
  - run_evaluation outlier_threshold pass-through for DLT backend
  - scripts/tune_threshold.py grid search and comparison tool
affects: [44-02-PLAN]

tech-stack:
  added: []
  patterns: [grid-search-with-composite-scoring, yield-penalized-optimization]

key-files:
  created:
    - scripts/tune_threshold.py
  modified:
    - src/aquapose/engine/config.py
    - src/aquapose/evaluation/harness.py

key-decisions:
  - "Composite score formula: mean_error * (1 + max(0, 1 - yield_ratio) * 10) penalizes yield drops by 10x the deficit"
  - "Grid search uses tempfile.TemporaryDirectory to avoid polluting fixture directory with per-threshold eval_results.json"
  - "Re-uses grid search EvalResults for top-N comparison rather than re-running evaluations"

patterns-established:
  - "Tuning scripts in scripts/ follow argparse + run_evaluation pattern from measure_baseline.py"

requirements-completed: [RECON-08]

duration: 5min
completed: 2026-03-02
---

# Plan 44-01: Config Wiring + Grid Search Summary

**outlier_threshold wired through ReconstructionConfig and run_evaluation; grid search script scores candidates on error-vs-yield composite with per-fish comparison against triangulation baseline**

## Performance

- **Duration:** 5 min
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added outlier_threshold field (default 50.0) to ReconstructionConfig with DLT backend documentation
- Wired outlier_threshold parameter through run_evaluation to DltBackend.from_models
- Created scripts/tune_threshold.py with full grid search, composite scoring, and side-by-side comparison reporting

## Task Commits

1. **Task 1: Add outlier_threshold to config and wire through harness** - `83631bc` (feat)
2. **Task 2: Create scripts/tune_threshold.py** - `ca560f8` (feat)

## Files Created/Modified
- `src/aquapose/engine/config.py` - Added outlier_threshold field to ReconstructionConfig
- `src/aquapose/evaluation/harness.py` - Added outlier_threshold parameter to run_evaluation
- `scripts/tune_threshold.py` - Grid search + comparison script (296 lines)

## Decisions Made
None - followed plan as specified

## Deviations from Plan
None - plan executed exactly as written

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- tune_threshold.py ready for user to run against real fixture data
- Plan 44-02 depends on user running the script and choosing a threshold value

---
*Phase: 44-validation-and-tuning*
*Completed: 2026-03-02*
