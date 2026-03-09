---
phase: 74-round-1-evaluation-decision
plan: 02
subsystem: evaluation
tags: [pipeline, metrics, pseudo-labels, eval-compare]

# Dependency graph
requires:
  - phase: 74-01
    provides: eval-compare CLI for metric comparison
  - phase: 73-round-1-pseudo-labels-retraining
    provides: Round 1 winner models (OBB curated, Pose curated+aug)
  - phase: 72-baseline-pipeline-run-metrics
    provides: Baseline pipeline run (run_20260307_140127) for comparison
provides:
  - Round 1 pipeline evaluation (run_20260309_175421) with eval_results.json
  - eval_comparison.json with metric deltas vs baseline
  - Go/no-go decision recorded in 74-DECISION.md
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created:
    - .planning/phases/74-round-1-evaluation-decision/74-DECISION.md
  modified: []

key-decisions:
  - "Skip round 2 -- accept round 1 models as final production models based on strong metric improvements"
  - "Phase 75 (round 2 pseudo-labels & retraining) is skipped"

patterns-established: []

requirements-completed: [ITER-04]

# Metrics
duration: ~20min
completed: 2026-03-09
---

# Phase 74 Plan 02: Round 1 Pipeline Evaluation & Decision Summary

**Round 1 models reduced p50 reprojection error 28% (3.02 to 2.16 px) and singleton rate 12.5%; accepted as final -- round 2 skipped**

## Performance

- **Duration:** ~20 min (includes pipeline execution time)
- **Started:** 2026-03-09T21:50:00Z
- **Completed:** 2026-03-09T22:10:00Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- Full pipeline re-run with round 1 models on 9000 frames (30 chunks x 300)
- Metric comparison via eval-compare showing improvements across all primary and most secondary metrics
- Decision recorded: skip round 2, accept round 1 models as final

## Task Commits

Each task was committed atomically:

1. **Task 1: Pre-flight validation and pipeline re-run** - (no code changes -- pipeline execution only)
2. **Task 2: Run eval-compare and prepare decision document** - `52186fd` (docs)
3. **Task 3: Decision checkpoint -- record go/no-go verdict** - `698fabc` (docs)

## Key Metrics

| Metric | Baseline | Round 1 | Delta |
|--------|----------|---------|-------|
| singleton_rate | 31.3% | 27.4% | -12.5% |
| p50 reprojection error | 3.02 px | 2.16 px | -28.4% |
| p90 reprojection error | 5.20 px | 4.17 px | -19.8% |
| fish_yield_ratio | 85.7% | 91.2% | +6.4% |
| continuity_ratio | 94.7% | 98.2% | +3.6% |
| fish_reconstructed | 52,578 | 59,433 | +13.0% |
| detection jitter | 0.257 | 0.084 | -67.3% |
| tail keypoint error (mean) | 7.32 px | 4.51 px | -38.4% |

## Decisions Made

- **Skip round 2, accept round 1 as final:** All primary metrics showed clear directional improvement. Singleton rate down 12.5%, p50 reproj error down 28.4%, p90 reproj error down 19.8%. Improvements are substantial enough that further pseudo-label iteration is unlikely to yield proportional gains.
- **Phase 75 skipped:** No round 2 pseudo-label generation or retraining needed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Round 1 models are the final production models for this milestone
- Phase 75 (round 2) is skipped
- Current model paths in config.yaml are the final weights

## Self-Check: PASSED

All files and commits verified.

---
*Phase: 74-round-1-evaluation-decision*
*Completed: 2026-03-09*
