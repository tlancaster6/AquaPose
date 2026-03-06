---
phase: 70-metrics-comparison-infrastructure
plan: 01
subsystem: evaluation
tags: [numpy, percentile, fragmentation, metrics, dataclass]

requires:
  - phase: 46-50
    provides: Evaluation system with stage evaluators, EvalRunner, output formatters
provides:
  - ReconstructionMetrics with p50/p90/p95 reprojection error percentiles
  - MidlineMetrics with p10/p50/p90 confidence percentiles
  - AssociationMetrics with p50/p90 camera count percentiles
  - FragmentationMetrics dataclass and evaluate_fragmentation() pure function
  - Updated EvalRunner with fragmentation wiring
  - Updated text and JSON output formatters with all new metrics
affects: [70-02, evaluation, pipeline-metrics]

tech-stack:
  added: []
  patterns: [percentile-augmented frozen dataclasses, frame-level gap analysis]

key-files:
  created:
    - src/aquapose/evaluation/stages/fragmentation.py
    - tests/unit/evaluation/test_stage_fragmentation.py
  modified:
    - src/aquapose/evaluation/stages/reconstruction.py
    - src/aquapose/evaluation/stages/midline.py
    - src/aquapose/evaluation/stages/association.py
    - src/aquapose/evaluation/stages/__init__.py
    - src/aquapose/evaluation/runner.py
    - src/aquapose/evaluation/output.py
    - tests/unit/evaluation/test_stage_reconstruction.py
    - tests/unit/evaluation/test_stage_midline.py
    - tests/unit/evaluation/test_stage_association.py
    - tests/unit/evaluation/test_eval_output.py

key-decisions:
  - "All new percentile fields use optional defaults (None) for backward compatibility"
  - "Fragmentation uses Midline3D.frame_index for global frame indexing, not list position"
  - "Track births/deaths counted relative to global first/last observed frame"

patterns-established:
  - "Percentile augmentation: add optional float | None fields to frozen dataclasses with np.percentile computation"
  - "Fragmentation evaluator: pure function taking list[dict[int, Midline3D] | None] and n_animals"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, EVAL-06]

duration: 15min
completed: 2026-03-06
---

# Plan 70-01: Percentile Metrics and Track Fragmentation Summary

**Reprojection/confidence/camera percentiles added to 3 existing metrics dataclasses, plus new FragmentationMetrics evaluator with frame-level gap analysis and track-level birth/death statistics, all wired into EvalRunner and text/JSON output**

## Performance

- **Duration:** 15 min
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- Added p50/p90/p95 reprojection error percentiles to ReconstructionMetrics (EVAL-01)
- Added p10/p50/p90 confidence percentiles to MidlineMetrics (EVAL-02)
- Added p50/p90 camera count percentiles to AssociationMetrics (EVAL-03)
- Created FragmentationMetrics with gap counts, continuity ratios, births/deaths (EVAL-06)
- Wired fragmentation into EvalRunner.run() and both output formatters

## Task Commits

1. **Task 1: Percentile fields and fragmentation evaluator** - `a8c8d26`
2. **Task 2: Wire into EvalRunner and output formatters** - `e687026`

## Decisions Made
None - followed plan as specified

## Deviations from Plan
None - plan executed exactly as written

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All percentile and fragmentation metrics ready for Plan 70-02 to build upon
- ReconstructionMetrics extended with optional fields, ready for per_point_error and curvature_stratified

---
*Phase: 70-metrics-comparison-infrastructure*
*Completed: 2026-03-06*
