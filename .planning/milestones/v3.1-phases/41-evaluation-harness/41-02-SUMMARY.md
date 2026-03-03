---
phase: 41-evaluation-harness
plan: "02"
subsystem: evaluation
tags: [evaluation, metrics, harness, tdd, ascii-table, json, numpy, torch]

# Dependency graph
requires:
  - phase: 41-01-calib-bundle
    provides: "CalibBundle frozen dataclass, NPZ v2.0 format with calib/ keys"
  - phase: 40-diagnostic-capture
    provides: "MidlineFixture dataclass, load_midline_fixture, NPZ format"
  - "src/aquapose/core/reconstruction/triangulation.py"
  - "src/aquapose/calibration/projection.py"
provides:
  - "run_evaluation orchestrator for offline reconstruction quality metrics"
  - "select_frames deterministic frame sampling via np.linspace"
  - "compute_tier1 reprojection error aggregation (per-camera, per-fish, overall)"
  - "compute_tier2 leave-one-out displacement aggregation with None/N/A handling"
  - "format_summary_table ASCII human-readable output"
  - "write_regression_json machine-diffable JSON output with numpy-safe encoder"
  - "EvalResults frozen dataclass as return type"
affects: [42-pipeline-integration, evaluation-cli]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD: RED (failing tests) → GREEN (implementation) per task"
    - "Mock triangulate_midlines at module level for harness unit tests"
    - "NumpySafeEncoder: json.JSONEncoder subclass converting np.floating/np.integer"
    - "select_frames: np.linspace(0, n-1, k, dtype=int) for deterministic endpoint-inclusive sampling"
    - "Leave-one-out: build new dicts excluding dropout_cam for ALL fish, never mutate fixture"

key-files:
  created:
    - src/aquapose/evaluation/__init__.py
    - src/aquapose/evaluation/metrics.py
    - src/aquapose/evaluation/harness.py
    - src/aquapose/evaluation/output.py
    - tests/unit/evaluation/__init__.py
    - tests/unit/evaluation/test_metrics.py
    - tests/unit/evaluation/test_harness.py
    - tests/unit/evaluation/test_output.py
  modified: []

key-decisions:
  - "Harness calls triangulate_midlines() directly (not TriangulationBackend) to avoid calibration file dependency"
  - "_build_models_from_calib builds RefractiveProjectionModel from CalibBundle using torch.from_numpy().float()"
  - "Tier 2 reduced midline_set excludes dropout_cam for ALL fish, not just the current fish"
  - "Tier 2 displacements displayed in mm in ASCII table; metres in JSON"
  - "Per-fish Tier 1 aggregates use mean_residual/max_residual from Midline3D directly"
  - "JSON uses str fish_id keys for JSON compatibility (int keys become strings in JSON)"

patterns-established:
  - "EvalResults as frozen dataclass return type for run_evaluation"
  - "Two-tier metric separation: metrics.py (pure computation) + output.py (formatting)"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05]

# Metrics
duration: 11min
completed: 2026-03-02
---

# Phase 41 Plan 02: Evaluation Harness Core Implementation Summary

**Offline evaluation harness with select_frames, Tier 1 reprojection error, Tier 2 leave-one-out displacement, ASCII summary table, and JSON regression output built via TDD**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-02T20:20:39Z
- **Completed:** 2026-03-02T20:32:00Z
- **Tasks:** 3 (TDD — 6 commits: 3 RED + 3 GREEN)
- **Files created:** 8 (4 source + 4 test)

## Accomplishments

- Created `src/aquapose/evaluation/` package with `metrics.py`, `harness.py`, `output.py`, `__init__.py`
- `select_frames` implements deterministic np.linspace sampling with edge-case handling (empty, fewer-than-requested)
- `compute_tier1` aggregates per-camera and per-fish reprojection errors from `Midline3D.per_camera_residuals`; handles `None` gracefully
- `compute_tier2` aggregates max control-point displacement from leave-one-out runs; records `None` for failed dropouts
- `format_summary_table` produces multi-line ASCII output with Tier 1 per-camera rows, OVERALL aggregate, and Tier 2 per-fish dropout table with N/A for missing entries
- `write_regression_json` produces machine-diffable JSON with `_NumpySafeEncoder` handling numpy scalar types
- `run_evaluation` orchestrator loads fixture, builds `RefractiveProjectionModel` from `CalibBundle`, selects frames, runs Tier 1 + Tier 2, writes JSON, returns `EvalResults`
- `from aquapose.evaluation import run_evaluation, EvalResults, select_frames` works
- All 708 tests pass (27 new, 0 regressions)

## Task Commits

Each task was committed atomically (RED → GREEN):

1. **Task 1 RED: Failing tests for select_frames and metrics** - `491ace4` (test)
2. **Task 1 GREEN: Implement metrics.py** - `a81993d` (feat)
3. **Task 2 RED: Failing tests for output formatting** - `5882338` (test)
4. **Task 2 GREEN: Implement output.py** - `855f593` (feat)
5. **Task 3 RED: Failing tests for run_evaluation harness** - `9fddea0` (test)
6. **Task 3 GREEN: Implement harness.py + updated __init__.py** - `57a32b1` (feat)

## Files Created

- `src/aquapose/evaluation/__init__.py` — Public API: EvalResults, Tier1Result, Tier2Result, run_evaluation, select_frames
- `src/aquapose/evaluation/metrics.py` — select_frames, Tier1Result, Tier2Result, compute_tier1, compute_tier2
- `src/aquapose/evaluation/harness.py` — EvalResults, _build_models_from_calib, run_evaluation
- `src/aquapose/evaluation/output.py` — _NumpySafeEncoder, format_summary_table, write_regression_json
- `tests/unit/evaluation/__init__.py` — Package init (empty)
- `tests/unit/evaluation/test_metrics.py` — 13 tests for frame selection and metric computation
- `tests/unit/evaluation/test_output.py` — 9 tests for ASCII table and JSON output
- `tests/unit/evaluation/test_harness.py` — 5 tests for run_evaluation with mocked triangulation

## Decisions Made

- `triangulate_midlines()` called directly (not via `TriangulationBackend`) — avoids needing a calibration JSON file
- `_build_models_from_calib` uses `torch.from_numpy(...).float()` to ensure float32 CPU tensors
- Leave-one-out reduced midline_set excludes dropout_cam for **all fish** (not just the current fish) — matches RESEARCH.md pseudocode
- Tier 2 displacements shown in mm in ASCII table (multiply by 1000); stored as metres in JSON
- JSON fish_id keys are strings (JSON spec requires string keys)
- `NumpySafeEncoder` converts `np.floating` → `float` and `np.integer` → `int` for clean JSON serialization

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `hatch run test -- tests/unit/evaluation/ -x` — 27 tests pass
- `hatch run test` — 708 tests pass (0 regressions)
- `hatch run check` — ruff: all checks passed; basedpyright: no new errors (pre-existing errors in unrelated modules are out of scope)
- `from aquapose.evaluation import run_evaluation, EvalResults, select_frames` — import verified

## Self-Check: PASSED

- FOUND: src/aquapose/evaluation/__init__.py
- FOUND: src/aquapose/evaluation/metrics.py
- FOUND: src/aquapose/evaluation/harness.py
- FOUND: src/aquapose/evaluation/output.py
- FOUND: tests/unit/evaluation/test_metrics.py
- FOUND: tests/unit/evaluation/test_harness.py
- FOUND: tests/unit/evaluation/test_output.py
- FOUND commit 491ace4 (test RED Task 1)
- FOUND commit a81993d (feat GREEN Task 1)
- FOUND commit 5882338 (test RED Task 2)
- FOUND commit 855f593 (feat GREEN Task 2)
- FOUND commit 9fddea0 (test RED Task 3)
- FOUND commit 57a32b1 (feat GREEN Task 3)

---
*Phase: 41-evaluation-harness*
*Completed: 2026-03-02*
