---
phase: 49-tuningorchestrator-and-aquapose-tune-cli
plan: "01"
subsystem: evaluation
tags: [tuning, grid-sweep, two-tier-validation, parameter-optimization, orchestrator]

# Dependency graph
requires:
  - phase: 47-per-stage-evaluators
    provides: evaluate_association, evaluate_reconstruction, DEFAULT_GRID constants
  - phase: 48-evalrunner-and-aquapose-eval-cli
    provides: EvalRunner pattern with load_stage_cache and _build_midline_sets

provides:
  - TuningOrchestrator class with sweep_association() and sweep_reconstruction()
  - Joint 2D grid sweep over ray_distance_threshold x score_min with sequential carry-forward
  - Two-tier validation at configurable frame counts
  - format_comparison_table, format_yield_matrix, format_config_diff output formatters
  - TuningResult frozen dataclass

affects:
  - 49-02 (aquapose tune CLI — will import and call TuningOrchestrator)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Joint 2D grid + sequential carry-forward for efficient parameter sweep
    - Two-tier frame sampling (fast n_frames then top-N at n_frames_validate)
    - Inline engine imports inside methods to avoid top-level coupling
    - Shallow copy (copy.copy) for context isolation in sweep loops

key-files:
  created:
    - src/aquapose/evaluation/tuning.py
    - tests/unit/evaluation/test_tuning.py
  modified:
    - src/aquapose/evaluation/__init__.py

key-decisions:
  - "early_k grid values (stored as float) are cast to int before dataclasses.replace() on AssociationConfig"
  - "n_points grid key maps to n_sample_points in ReconstructionConfig via _patch_reconstruction_config()"
  - "winner_params dict normalizes n_points -> n_sample_points in output for consistency"
  - "TuningOrchestrator exports TuningResult and formatting functions via evaluation/__init__.py"

patterns-established:
  - "Sweep pattern: copy.copy(ctx) per combo, run stage, evaluate, score, sort, validate top-N"
  - "Baseline computed once before sweep loop for comparison baseline_metrics dict"

requirements-completed:
  - TUNE-01
  - TUNE-02
  - TUNE-03
  - TUNE-04
  - TUNE-05

# Metrics
duration: 30min
completed: 2026-03-03
---

# Phase 49 Plan 01: TuningOrchestrator Summary

**Cache-backed parameter sweep engine with joint 2D grid, sequential carry-forward, two-tier validation, and formatted output for association and reconstruction tuning**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-03-03T19:30:00Z
- **Completed:** 2026-03-03T19:59:04Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments

- TuningOrchestrator class with sweep_association() sweeping ray_distance_threshold x score_min jointly then eviction_reproj_threshold, leiden_resolution, early_k sequentially
- sweep_reconstruction() with 1D sequential carry-forward over outlier_threshold then n_points (mapped to n_sample_points)
- Two-tier validation re-evaluates top-N candidates at n_frames_validate frame count for final winner selection
- format_comparison_table, format_yield_matrix, format_config_diff output formatters for CLI use
- 22 unit tests covering scoring, formatting, parameter casting, and frame count behavior — all passing

## Task Commits

1. **Task 1: Create TuningOrchestrator with unit tests** - `16d167f` (feat)

## Files Created/Modified

- `src/aquapose/evaluation/tuning.py` - TuningOrchestrator class, TuningResult dataclass, scoring helpers, formatting functions
- `tests/unit/evaluation/test_tuning.py` - 22 unit tests covering all public behavior
- `src/aquapose/evaluation/__init__.py` - Added TuningOrchestrator, TuningResult, and formatting exports

## Decisions Made

- The files already existed in the git untracked state — implemented and complete before this execution. Fixed lint issues: removed unused imports in tuning.py (two unused `ReconstructionStage` imports, one unused `baseline_score` assignment), added `strict=True` to `zip()`, fixed import ordering in test file and evaluation __init__.py.
- Added TuningOrchestrator to evaluation/__init__.py per project code-style rules requiring new public classes be exported from parent package.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed lint errors in tuning.py and test_tuning.py**
- **Found during:** Task 1 (lint verification)
- **Issue:** Two unused `ReconstructionStage` imports in tuning.py, one unused `baseline_score` variable, `zip()` without `strict=`, unused `MagicMock, patch` imports in test file, unsorted import blocks
- **Fix:** Removed unused imports, dropped unused variable assignment, added `strict=True` to `zip()`, auto-fixed import ordering with `ruff check --fix`
- **Files modified:** src/aquapose/evaluation/tuning.py, tests/unit/evaluation/test_tuning.py, src/aquapose/evaluation/__init__.py
- **Verification:** `hatch run lint` passes with "All checks passed!"
- **Committed in:** 16d167f (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (lint errors)
**Impact on plan:** All fixes required for CI compliance. No scope creep.

## Issues Encountered

None — implementation was already complete in untracked files. Focused on lint cleanup and adding package exports.

## Next Phase Readiness

- TuningOrchestrator fully implemented and tested, ready for 49-02 (aquapose tune CLI)
- All output formatting functions available for CLI integration

---
*Phase: 49-tuningorchestrator-and-aquapose-tune-cli*
*Completed: 2026-03-03*
