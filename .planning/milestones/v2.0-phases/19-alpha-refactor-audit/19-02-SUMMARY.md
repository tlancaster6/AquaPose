---
phase: 19-alpha-refactor-audit
plan: "02"
subsystem: tools
tags: [smoke-test, testing, e2e, verification, reproducibility]
one_liner: "Subprocess-based smoke test runner with CLI, pytest integration, and HDF5 reproducibility check"
dependency_graph:
  requires: []
  provides:
    - tools/smoke_test.py
    - tests/e2e/test_smoke.py
  affects:
    - pyproject.toml
tech_stack:
  added: []
  patterns:
    - SmokeTestRunner class with dataclass results
    - pytest @slow/@e2e marker pattern for optional integration tests
key_files:
  created:
    - tools/smoke_test.py
    - tests/e2e/test_smoke.py
  modified:
    - pyproject.toml
decisions:
  - CLI uses aquapose console script (not python -m aquapose) — aquapose has no __main__.py
  - test_synthetic_mode always runnable with calibration only — other tests skip if no full config
  - Subprocess invocation falls back to python -c import if console script not found
  - --only flag restricts to modes/backends/repro for partial runs
metrics:
  duration_minutes: 7
  tasks_completed: 2
  files_created: 2
  files_modified: 1
  completed_date: "2026-02-26"
---

# Phase 19 Plan 02: Smoke Test Script Summary

Subprocess-based smoke test runner with CLI, pytest integration, and HDF5 reproducibility check.

## What Was Built

### Task 1: `tools/smoke_test.py`

A standalone, reusable smoke test runner that exercises the AquaPose pipeline via subprocess invocation of `aquapose run`. Key design choices:

- **`SmokeTestRunner` class** with `run_mode_tests()`, `run_backend_tests()`, `run_reproducibility_test()`, and `run_all()` public methods
- **`TestResult` / `SmokeTestReport` dataclasses** — structured output with JSON and human-readable summary
- **`ALL_MODES` / `ALL_BACKENDS` constants** at top of file — adding a new mode or backend is a one-line change
- **CLI interface**: `--config`, `--calibration`, `--output-dir`, `--only`, `--frame-limit`, `--json-report`
- **Independence**: each test is independent; failure of one doesn't block others
- **Timeout**: 5-minute subprocess timeout per run
- **Reproducibility**: runs pipeline twice with `synthetic.seed=99`, compares all HDF5 arrays with `np.allclose(atol=1e-10)`
- **Synthetic mode support**: accepts `--calibration` alone (no `--config`) and auto-generates a minimal YAML

### Task 2: `tests/e2e/test_smoke.py`

Pytest wrapper for CI integration:

- **`TestSmoke` class** with 5 tests: `test_synthetic_mode`, `test_benchmark_mode`, `test_production_mode`, `test_diagnostic_mode`, `test_reproducibility`
- **`@pytest.mark.slow @pytest.mark.e2e`** — excluded from fast test runs
- **`test_synthetic_mode`** always runnable with only AquaCal calibration (verified passing in 6.56s)
- **Other tests** skip gracefully via `pytest.skip()` when `_PRODUCTION_CONFIG` is not configured
- **`e2e` marker** registered in `pyproject.toml`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `aquapose` has no `__main__.py` — `python -m aquapose` fails**

- **Found during:** Task 2 verification (synthetic mode test)
- **Issue:** `_run_aquapose()` used `python -m aquapose run` but `aquapose` is a Click entry point registered as a console script, not a runnable package
- **Fix:** Updated `_run_aquapose()` to locate the `aquapose` console script next to `sys.executable`, with fallback to `python -c "from aquapose.cli import main; main()"`
- **Files modified:** `tools/smoke_test.py`
- **Commit:** 359a396

**2. [Rule 2 - Missing functionality] `e2e` pytest marker not registered**

- **Found during:** Task 2 — `@pytest.mark.e2e` would cause pytest warnings without registration
- **Fix:** Added `e2e` to `[tool.pytest.ini_options] markers` in `pyproject.toml`
- **Files modified:** `pyproject.toml`
- **Commit:** 359a396

## Self-Check: PASSED

- tools/smoke_test.py: FOUND (commit 6b0a2e5)
- tests/e2e/test_smoke.py: FOUND (commit 359a396)
- Commit 6b0a2e5: FOUND
- Commit 359a396: FOUND
