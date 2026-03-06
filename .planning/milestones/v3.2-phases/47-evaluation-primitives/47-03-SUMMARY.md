---
phase: 47-evaluation-primitives
plan: "03"
subsystem: evaluation
tags: [reconstruction, metrics, dataclass, stage-evaluator, tier1, tier2, reprojection-error]

# Dependency graph
requires:
  - phase: 47-evaluation-primitives (plans 01-02)
    provides: detection, tracking, association, midline evaluators; stages/__init__.py placeholder
  - phase: 46-cache-primitives
    provides: PipelineContext, stage cache envelopes, DiagnosticObserver
provides:
  - ReconstructionMetrics frozen dataclass with tier2_stability field
  - evaluate_reconstruction() function wrapping compute_tier1() internally
  - DEFAULT_GRID for reconstruction parameter sweeps
  - stages/__init__.py with all 5 evaluators, 5 metric classes, 2 DEFAULT_GRIDs
  - evaluation/__init__.py updated to export all new stage symbols
affects: [48-context-loader, 49-tuning-orchestrator]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ReconstructionMetrics wraps Tier1Result fields without inheriting — fresh dataclass"
    - "tier2_stability extracted as max of all non-None per_fish_dropout values"
    - "evaluate_reconstruction() uses keyword-only tier2_result param for optional Tier 2 pass-through"
    - "DEFAULT_GRID stores float values for outlier_threshold and n_points sweep ranges"

key-files:
  created:
    - src/aquapose/evaluation/stages/reconstruction.py
    - tests/unit/evaluation/test_stage_reconstruction.py
  modified:
    - src/aquapose/evaluation/stages/__init__.py
    - src/aquapose/evaluation/__init__.py

key-decisions:
  - "ReconstructionMetrics is a fresh frozen dataclass, NOT a subclass of Tier1Result"
  - "tier2_stability = max of all non-None displacement values in Tier2Result.per_fish_dropout (None if all None or empty)"
  - "evaluate_reconstruction() wraps compute_tier1() internally for reprojection error aggregation"
  - "tier2_result is keyword-only param to avoid positional confusion with fish_available"

patterns-established:
  - "Stage evaluator wraps lower-level metric function internally (evaluate_reconstruction wraps compute_tier1)"
  - "to_dict() converts int fish keys to str for JSON compatibility"
  - "Optional pass-through of pre-computed Tier2Result via keyword-only param"

requirements-completed: [EVAL-05, TUNE-06]

# Metrics
duration: 6min
completed: "2026-03-03"
---

# Phase 47 Plan 03: Reconstruction Stage Evaluator Summary

**ReconstructionMetrics frozen dataclass with tier2_stability, evaluate_reconstruction() wrapping compute_tier1(), and full evaluation package export wiring for all 5 stage evaluators**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-03T18:52:14Z
- **Completed:** 2026-03-03T18:58:28Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created `ReconstructionMetrics` frozen dataclass with mean/max reprojection error, inlier ratio, low-confidence flag rate, `tier2_stability: float | None`, per-camera/per-fish error dicts, and `to_dict()`
- Implemented `evaluate_reconstruction()` that wraps `compute_tier1()` internally, accepts optional keyword-only `tier2_result: Tier2Result | None`, and extracts overall max displacement as `tier2_stability`
- Defined `DEFAULT_GRID` with `outlier_threshold` (19 values: 10–100 step 5) and `n_points` (4 values)
- Wired all 5 evaluators, 5 metric classes, and 2 aliased DEFAULT_GRIDs through `stages/__init__.py` and `evaluation/__init__.py`
- 26 unit tests covering all behavior, serialization, tier2 extraction edge cases, and no-engine-imports check

## Task Commits

Each task was committed atomically:

1. **Task 1: Create reconstruction evaluator with DEFAULT_GRID (TDD RED)** - `c0e914f` (test)
2. **Task 1: Create reconstruction evaluator with DEFAULT_GRID (TDD GREEN)** - `0d57e98` (feat)
3. **Task 2: Wire __init__.py exports for stages/ and evaluation/** - `9692f71` (feat)

_Note: TDD task has two commits (test RED then feat GREEN)_

## Files Created/Modified

- `src/aquapose/evaluation/stages/reconstruction.py` - ReconstructionMetrics dataclass, evaluate_reconstruction(), DEFAULT_GRID
- `tests/unit/evaluation/test_stage_reconstruction.py` - 26 unit tests for reconstruction evaluator
- `src/aquapose/evaluation/stages/__init__.py` - Re-exports all 5 evaluators, 5 metric classes, 2 DEFAULT_GRIDs
- `src/aquapose/evaluation/__init__.py` - Updated to export all new stage symbols alongside existing exports

## Decisions Made

- `ReconstructionMetrics` is a fresh frozen dataclass, not a subclass of `Tier1Result` — keeps evaluation types independent from metric result types
- `tier2_stability` is the max of all non-None displacement values in `Tier2Result.per_fish_dropout`; None when all values are None or the dropout dict is empty
- `tier2_result` is a keyword-only parameter to prevent positional confusion with `fish_available`
- `evaluate_reconstruction()` wraps `compute_tier1()` to avoid duplicating aggregation logic

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Ruff pre-commit hook required fixing 3 lint issues in the test file: using `AttributeError` instead of bare `Exception` in `pytest.raises`, using `itertools.pairwise()` instead of `zip()` for pairwise iteration, and correcting import sort order. All fixed before final commit.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 5 stage evaluators (detection, tracking, association, midline, reconstruction) are fully implemented and importable from `aquapose.evaluation`
- Phase 48 (ContextLoader) can import and use all evaluators from the evaluation package
- Phase 49 (TuningOrchestrator) has both DEFAULT_GRIDs available as `ASSOCIATION_DEFAULT_GRID` and `RECONSTRUCTION_DEFAULT_GRID`
- No blockers

---
*Phase: 47-evaluation-primitives*
*Completed: 2026-03-03*

## Self-Check: PASSED

- FOUND: src/aquapose/evaluation/stages/reconstruction.py
- FOUND: tests/unit/evaluation/test_stage_reconstruction.py
- FOUND: src/aquapose/evaluation/stages/__init__.py
- FOUND: src/aquapose/evaluation/__init__.py
- FOUND: .planning/phases/47-evaluation-primitives/47-03-SUMMARY.md
- FOUND: commit c0e914f (test: failing tests)
- FOUND: commit 0d57e98 (feat: reconstruction evaluator)
- FOUND: commit 9692f71 (feat: __init__.py wiring)
