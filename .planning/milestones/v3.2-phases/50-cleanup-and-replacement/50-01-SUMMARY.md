---
phase: 50-cleanup-and-replacement
plan: 01
subsystem: evaluation
tags: [cleanup, dead-code-removal, evaluation, diagnostic-observer, npz, pickle-cache]

# Dependency graph
requires:
  - phase: 49-tuning-orchestrator
    provides: TuningOrchestrator and EvalRunner as replacement evaluation stack
  - phase: 46-pickle-cache
    provides: per-stage pickle cache system replacing NPZ export
provides:
  - DiagnosticObserver with only pickle cache output (no NPZ methods)
  - Cleaned evaluation public API without legacy harness symbols
  - Cleaned io public API without midline fixture NPZ symbols
  - All tests passing with legacy code removed
affects: [future evaluation phases]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Legacy dead code fully deleted (not shimmed) — harness.py, midline_fixture.py, compute_tier2, format_summary_table, write_regression_json"
    - "DiagnosticObserver._on_pipeline_complete is now a no-op; pickle cache is the sole data output"

key-files:
  created: []
  modified:
    - src/aquapose/engine/diagnostic_observer.py
    - src/aquapose/engine/observer_factory.py
    - src/aquapose/evaluation/__init__.py
    - src/aquapose/evaluation/metrics.py
    - src/aquapose/evaluation/output.py
    - src/aquapose/evaluation/runner.py
    - src/aquapose/io/__init__.py
    - tests/unit/engine/test_diagnostic_observer.py
    - tests/unit/evaluation/test_output.py
    - tests/unit/evaluation/test_metrics.py

key-decisions:
  - "Legacy evaluation code (harness.py, midline_fixture.py, compute_tier2, format_summary_table, write_regression_json) fully deleted, not shimmed"
  - "DiagnosticObserver.__init__ calibration_path parameter removed — no longer needed without NPZ export"
  - "Pre-existing typecheck failures (40 errors in unrelated files) documented as out of scope; lint (ruff) passes cleanly"
  - "Pre-existing test failure (test_pose_dataset_structure) confirmed pre-existing before this plan's changes"

patterns-established:
  - "When deleting a module, also delete its test file and update parent __init__.py and __all__"

requirements-completed:
  - CLEAN-04
  - CLEAN-05

# Metrics
duration: 9min
completed: 2026-03-03
---

# Phase 50 Plan 01: Legacy Evaluation Cleanup Summary

**Deleted harness.py, midline_fixture.py, and all NPZ export machinery from DiagnosticObserver; per-stage pickle cache is now the sole evaluation data output**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-03T21:55:11Z
- **Completed:** 2026-03-03T22:04:00Z
- **Tasks:** 2
- **Files modified:** 12 (2 deleted, 2 test files deleted, 8 modified)

## Accomplishments

- Deleted `evaluation/harness.py` and `io/midline_fixture.py` entirely
- Stripped all NPZ export methods from `DiagnosticObserver` — removed `export_pipeline_diagnostics`, `export_midline_fixtures`, `_match_annotated_by_centroid`, `_build_projection_models`, `_collect_*_section`, and `_write_calib_arrays`; `_on_pipeline_complete` is now a no-op
- Removed `compute_tier2`, `format_summary_table`, `write_regression_json` from `evaluation/metrics.py` and `evaluation/output.py`
- Updated `evaluation/__init__.py` and `io/__init__.py` to remove all deleted symbol exports
- Removed `calibration_path` parameter from `DiagnosticObserver.__init__` and updated `observer_factory.py` accordingly
- Deleted `test_harness.py` and `test_midline_fixture.py`; rewrote `test_diagnostic_observer.py`, `test_output.py`, and `test_metrics.py` to cover only surviving functionality
- 788 tests pass (1 pre-existing unrelated failure in `test_build_yolo_training_data.py`)

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete legacy source files and prune orphaned code** - `ba47aa8` (chore)
2. **Task 2: Clean up tests and verify full suite** - `fc03147` (chore)

## Files Created/Modified

- `src/aquapose/engine/diagnostic_observer.py` - Removed all NPZ export methods, calibration_path param; _on_pipeline_complete is now no-op
- `src/aquapose/engine/observer_factory.py` - Removed calibration_path from DiagnosticObserver constructor calls
- `src/aquapose/evaluation/__init__.py` - Removed EvalResults, generate_fixture, run_evaluation from public API
- `src/aquapose/evaluation/metrics.py` - Removed compute_tier2 function
- `src/aquapose/evaluation/output.py` - Removed format_summary_table and write_regression_json functions
- `src/aquapose/evaluation/runner.py` - Removed stale docstring reference to deleted method
- `src/aquapose/io/__init__.py` - Removed NPZ_VERSION, CalibBundle, MidlineFixture, load_midline_fixture from public API
- `tests/unit/engine/test_diagnostic_observer.py` - Rewrote: removed NPZ export tests, kept StageSnapshot/on_event tests
- `tests/unit/evaluation/test_output.py` - Rewrote: removed format_summary_table/write_regression_json tests, kept flag_outliers/format_baseline_report tests
- `tests/unit/evaluation/test_metrics.py` - Rewrote: removed compute_tier2 tests, kept select_frames/compute_tier1 tests
- *(deleted)* `src/aquapose/evaluation/harness.py`
- *(deleted)* `src/aquapose/io/midline_fixture.py`
- *(deleted)* `tests/unit/evaluation/test_harness.py`
- *(deleted)* `tests/unit/io/test_midline_fixture.py`

## Decisions Made

- Legacy evaluation code fully deleted (not shimmed): harness.py, midline_fixture.py, compute_tier2, format_summary_table, write_regression_json
- DiagnosticObserver `calibration_path` parameter removed — no longer needed without NPZ export
- 40 pre-existing basedpyright typecheck errors in unrelated files are out of scope (ruff lint passes cleanly)
- Pre-existing test failure (`test_pose_dataset_structure` in `test_build_yolo_training_data.py`) confirmed pre-existing before this plan

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated stale docstring reference in runner.py**
- **Found during:** Task 2 (grep sweep)
- **Issue:** Docstring in `runner.py` referenced `DiagnosticObserver.export_midline_fixtures` which no longer exists
- **Fix:** Removed the reference from the docstring
- **Files modified:** `src/aquapose/evaluation/runner.py`
- **Verification:** grep -rn "export_midline_fixtures" src/ returns no source hits
- **Committed in:** fc03147 (Task 2 commit)

**2. [Rule 1 - Bug] Corrected test data in test_flag_outliers_identifies_high_value**
- **Found during:** Task 2 (test run)
- **Issue:** My replacement test used values {1.0, 1.1, 10.0} where 10.0 does not exceed mean+2*std=12.33
- **Fix:** Used 9 values at 1.0 and 1 at 1000.0 where the outlier clearly exceeds mean+2*std≈700
- **Files modified:** `tests/unit/evaluation/test_output.py`
- **Verification:** test passes
- **Committed in:** fc03147 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 stale docstring, 1 wrong test data)
**Impact on plan:** Both minor correctness fixes. No scope creep.

## Issues Encountered

- Ruff formatter reformatted files on first commit attempt; re-staged and recommitted successfully both times.
- `hatch run check` typecheck command has 40 pre-existing basedpyright errors in files not touched by this plan (yolo backends, animation observers, etc.). Ruff lint passes cleanly.

## Next Phase Readiness

- Phase 50 Plan 01 complete. Codebase has no remaining NPZ export machinery.
- Phase 50 Plan 02 (if any) can proceed immediately.
- CLEAN-04 and CLEAN-05 requirements satisfied.

---
*Phase: 50-cleanup-and-replacement*
*Completed: 2026-03-03*
