---
phase: 47-evaluation-primitives
plan: "02"
subsystem: evaluation
tags: [evaluation, association, midline, metrics, dataclass, pure-function, tdd]

# Dependency graph
requires:
  - phase: 47-evaluation-primitives-01
    provides: stages/__init__.py subpackage and detection evaluator pattern

provides:
  - AssociationMetrics frozen dataclass with fish_yield_ratio, singleton_rate, camera_distribution, to_dict()
  - evaluate_association() pure function accepting list[MidlineSet] and n_animals
  - DEFAULT_GRID with 5 keys matching tune_association.py SWEEP_RANGES + SECONDARY_RANGES
  - MidlineMetrics frozen dataclass with confidence stats, completeness, temporal_smoothness, to_dict()
  - evaluate_midline() pure function accepting list[dict[int, Midline2D]]

affects:
  - 47-evaluation-primitives-03 (reconstruction evaluator, completes stage evaluator suite)
  - 48-context-loader (consumes stage evaluators)
  - 49-tuning-orchestrator (uses DEFAULT_GRID for association sweeps)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stage evaluators as pure functions with frozen dataclass return types"
    - "to_dict() converts camera_distribution int keys to str keys for JSON"
    - "point_confidence=None treated as all-1.0 confidence for unified metric computation"
    - "Temporal smoothness via per-fish centroid L2 distances across consecutive frames"
    - "DEFAULT_GRID as flat dict[str, list[float]] merging SWEEP_RANGES + SECONDARY_RANGES"
    - "TDD: failing import test -> implementation -> lint fix -> pass"

key-files:
  created:
    - src/aquapose/evaluation/stages/association.py
    - src/aquapose/evaluation/stages/midline.py
    - tests/unit/evaluation/test_stage_association.py
    - tests/unit/evaluation/test_stage_midline.py
  modified: []

key-decisions:
  - "DEFAULT_GRID early_k values stored as floats [5.0, 10.0, ...] to match dict[str, list[float]] type (source has ints)"
  - "camera_distribution int keys converted to str in to_dict() for JSON compatibility"
  - "temporal_smoothness is 0.0 for fish with only a single frame observation (no consecutive pairs)"
  - "point_confidence=None treated as all-1.0 for both confidence stats and completeness"

patterns-established:
  - "Stage evaluator: pure function + frozen dataclass + to_dict() + no engine imports"
  - "AST import check in tests verifies engine isolation at the source level"
  - "Parents[3] resolves test file to project root for AST path construction"

requirements-completed: [EVAL-03, EVAL-04, TUNE-06]

# Metrics
duration: 5min
completed: "2026-03-03"
---

# Phase 47 Plan 02: Association and Midline Stage Evaluators Summary

**AssociationMetrics + evaluate_association() with DEFAULT_GRID (5-param tuning grid from tune_association.py), and MidlineMetrics + evaluate_midline() with confidence/completeness/temporal-smoothness metrics**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-03T18:38:18Z
- **Completed:** 2026-03-03T18:49:48Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Association evaluator with fish_yield_ratio, singleton_rate, camera_distribution, and to_dict()
- DEFAULT_GRID flat dict with 5 keys matching SWEEP_RANGES + SECONDARY_RANGES from tune_association.py verbatim
- Midline evaluator with mean/std confidence, completeness (treating None as 1.0), and temporal smoothness via centroid L2
- 29 unit tests total across both evaluators, all passing

## Task Commits

Each task was committed atomically (TDD: test -> feat):

1. **Task 1: Association evaluator RED** - `6a19fc7` (test: failing tests for association evaluator and DEFAULT_GRID)
2. **Task 1: Association evaluator GREEN** - `8fcf8c0` (feat: implement AssociationMetrics, evaluate_association(), DEFAULT_GRID)
3. **Task 2: Midline evaluator RED** - `f27ade7` (test: failing tests for MidlineMetrics and evaluate_midline())
4. **Task 2: Midline evaluator GREEN** - `c69254b` (feat: implement MidlineMetrics and evaluate_midline())
5. **Task 2: Lint fix** - `6280f60` (fix: import order in test_stage_midline.py)

_Note: TDD tasks have test -> feat commits; lint auto-fix committed separately._

## Files Created/Modified

- `src/aquapose/evaluation/stages/association.py` - AssociationMetrics, evaluate_association(), DEFAULT_GRID
- `src/aquapose/evaluation/stages/midline.py` - MidlineMetrics, evaluate_midline()
- `tests/unit/evaluation/test_stage_association.py` - 14 unit tests for association evaluator
- `tests/unit/evaluation/test_stage_midline.py` - 15 unit tests for midline evaluator

## Decisions Made

- DEFAULT_GRID early_k values stored as floats [5.0, 10.0, ...] rather than ints to match the `dict[str, list[float]]` type annotation; source (tune_association.py) uses ints but the typed constant needs floats for type consistency.
- camera_distribution int keys converted to str in to_dict() for JSON compatibility (JSON object keys must be strings).
- temporal_smoothness returns 0.0 for fish seen in only one frame (no consecutive pairs to compare).
- point_confidence=None treated as uniform 1.0 confidence array for both confidence stats and completeness.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect path depth in AST engine-import test**
- **Found during:** Task 1 (Association evaluator GREEN phase)
- **Issue:** Test used `Path(__file__).parents[4]` to reach project root, but test file is only 3 levels deep from root (parents[3] = AquaPose/). parents[4] resolved to /home/tlancaster6/Projects/.
- **Fix:** Changed parents[4] to parents[3] in test_stage_association.py
- **Files modified:** tests/unit/evaluation/test_stage_association.py
- **Verification:** Test passes after fix; `hatch run test tests/unit/evaluation/test_stage_association.py -x` all 14 pass
- **Committed in:** `8fcf8c0` (part of feat commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in path depth)
**Impact on plan:** Minor path correction in test. No scope creep.

## Issues Encountered

- Ruff pre-commit hook reorganized imports in both test files (isort ordering). Re-staged and committed after auto-fix on each TDD commit.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Association and midline evaluators complete; 4 of 5 stage evaluators now implemented (detection, tracking, association, midline).
- Plan 47-03 (reconstruction evaluator) can proceed immediately.
- DEFAULT_GRID is ready for Phase 49 (TuningOrchestrator) to consume.
- No blockers.

---
*Phase: 47-evaluation-primitives*
*Completed: 2026-03-03*
