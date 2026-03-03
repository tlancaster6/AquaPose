---
phase: 41-evaluation-harness
verified: 2026-03-02T21:00:00Z
status: passed
score: 13/13 must-haves verified
re_verification: false
---

# Phase 41: Evaluation Harness Verification Report

**Phase Goal:** An offline evaluation framework exists that loads fixtures and computes Tier 1 and Tier 2 metrics without running the full pipeline
**Verified:** 2026-03-02T21:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                         | Status     | Evidence                                                                                 |
|----|-----------------------------------------------------------------------------------------------|------------|------------------------------------------------------------------------------------------|
| 1  | MidlineFixture with calib_bundle=None loads correctly from v1.0 NPZ files (backward compat)  | VERIFIED   | load_midline_fixture sets calib_bundle=None when version=="1.0"; tests pass              |
| 2  | MidlineFixture with calib_bundle populated loads correctly from v2.0 NPZ files               | VERIFIED   | _parse_calib_bundle called for version=="2.0"; CalibBundle populated; tests pass         |
| 3  | CalibBundle contains per-camera K_new, R, t plus shared water_z, interface_normal, n_air, n_water | VERIFIED | CalibBundle frozen dataclass defined with all required fields in midline_fixture.py      |
| 4  | export_midline_fixtures writes calib/ keys when models dict is provided                      | VERIFIED   | _write_calib_arrays writes calib/{cam_id}/K_new, /R, /t plus shared keys when models!=None |
| 5  | Round-trip: export v2.0 fixture with calibration -> load -> CalibBundle fields match originals | VERIFIED | Round-trip tests in test_midline_fixture.py and test_diagnostic_observer.py pass        |
| 6  | run_evaluation loads a MidlineFixture with CalibBundle and produces metrics without the pipeline | VERIFIED | run_evaluation calls load_midline_fixture, _build_models_from_calib, triangulate_midlines; all pass |
| 7  | Frame selection returns 15 frames from a 300-frame fixture via np.linspace, deterministically | VERIFIED   | select_frames uses np.linspace(0, n-1, 15, dtype=int); determinism test passes           |
| 8  | Frame selection returns all frames with a warning when fixture has fewer than requested       | VERIFIED   | warnings.warn issued when len(available) < n_frames; test confirms                      |
| 9  | Tier 1 output contains per-fish per-camera reprojection error with mean and max aggregates   | VERIFIED   | compute_tier1 builds per_camera and per_fish dicts with mean_px/max_px; test passes     |
| 10 | Tier 2 output contains leave-one-out max control-point displacement per fish per dropout camera | VERIFIED | compute_tier2 aggregates max of non-None displacements; test passes                    |
| 11 | Tier 2 records None/N/A when dropout reconstruction fails                                    | VERIFIED   | compute_tier2 sets None when all displacements are None; format_summary_table shows N/A |
| 12 | Summary table is printed as human-readable ASCII text                                        | VERIFIED   | format_summary_table produces multi-line string with "Tier 1" and "Tier 2" sections     |
| 13 | Regression data is written as JSON with per-fish and per-camera aggregates                   | VERIFIED   | write_regression_json produces JSON with tier1/tier2/fixture/frames_evaluated keys       |

**Score:** 13/13 truths verified

### Required Artifacts

| Artifact                                          | Expected                                              | Status     | Details                                                               |
|---------------------------------------------------|-------------------------------------------------------|------------|-----------------------------------------------------------------------|
| `src/aquapose/io/midline_fixture.py`              | CalibBundle dataclass, NPZ v2.0 loading support       | VERIFIED   | 271 lines; CalibBundle frozen dataclass; _parse_calib_bundle; v1.0+v2.0 support |
| `src/aquapose/engine/diagnostic_observer.py`      | Calibration NPZ serialization in export_midline_fixtures | VERIFIED | calib/ keys written by _write_calib_arrays when models!=None         |
| `src/aquapose/io/__init__.py`                     | CalibBundle exported                                  | VERIFIED   | CalibBundle imported and in __all__                                   |
| `tests/unit/io/test_midline_fixture.py`           | Round-trip tests for v2.0 calibration-bundled fixtures | VERIFIED  | 474 lines; backward-compat + v2.0 CalibBundle tests present           |
| `tests/unit/engine/test_diagnostic_observer.py`   | v2.0 export tests with mock models                    | VERIFIED   | 741 lines; export with/without models tests present                   |
| `src/aquapose/evaluation/__init__.py`             | Public API: run_evaluation, EvalResults, select_frames | VERIFIED  | Imports and exports run_evaluation, EvalResults, Tier1Result, Tier2Result, select_frames |
| `src/aquapose/evaluation/harness.py`              | run_evaluation orchestrator, model building from CalibBundle | VERIFIED | 204 lines; EvalResults, _build_models_from_calib, run_evaluation all implemented |
| `src/aquapose/evaluation/metrics.py`              | select_frames, compute_tier1, compute_tier2           | VERIFIED   | 174 lines; all three functions plus Tier1Result and Tier2Result dataclasses |
| `src/aquapose/evaluation/output.py`               | format_summary_table, write_regression_json           | VERIFIED   | 183 lines; both functions plus _NumpySafeEncoder implemented          |
| `tests/unit/evaluation/test_metrics.py`           | Unit tests for frame selection and metric computation | VERIFIED   | 211 lines; 13 tests covering select_frames edge cases + compute_tier1/tier2 |
| `tests/unit/evaluation/test_harness.py`           | Integration test for run_evaluation with synthetic fixture | VERIFIED | 206 lines; 5 tests with mocked triangulate_midlines                  |
| `tests/unit/evaluation/test_output.py`            | Unit tests for summary table and JSON output formatting | VERIFIED  | 245 lines; 9 tests covering ASCII table and JSON output               |

### Key Link Verification

| From                                      | To                                              | Via                                      | Status  | Details                                                          |
|-------------------------------------------|-------------------------------------------------|------------------------------------------|---------|------------------------------------------------------------------|
| `src/aquapose/io/midline_fixture.py`      | `src/aquapose/engine/diagnostic_observer.py`    | NPZ key convention (calib/ prefix)       | WIRED   | _write_calib_arrays writes calib/ keys read by _parse_calib_bundle |
| `src/aquapose/io/midline_fixture.py`      | `src/aquapose/calibration/projection.py`        | CalibBundle fields match RefractiveProjectionModel args | WIRED | K_new/R/t/water_z/interface_normal/n_air/n_water all present   |
| `src/aquapose/evaluation/harness.py`      | `src/aquapose/io/midline_fixture.py`            | load_midline_fixture + CalibBundle       | WIRED   | `from aquapose.io.midline_fixture import CalibBundle, load_midline_fixture`; both called |
| `src/aquapose/evaluation/harness.py`      | `src/aquapose/core/reconstruction/triangulation.py` | triangulate_midlines() called directly | WIRED   | `from aquapose.core.reconstruction.triangulation import triangulate_midlines`; called at lines 132, 158 |
| `src/aquapose/evaluation/harness.py`      | `src/aquapose/calibration/projection.py`        | RefractiveProjectionModel built from CalibBundle | WIRED | `from aquapose.calibration.projection import RefractiveProjectionModel`; _build_models_from_calib uses it |
| `src/aquapose/evaluation/output.py`       | eval_results.json                               | json.dump with numpy-safe encoder        | WIRED   | json.dump called at line 180 with _NumpySafeEncoder; write_regression_json returns output path |

### Requirements Coverage

| Requirement | Source Plan | Description                                                                       | Status    | Evidence                                                                      |
|-------------|------------|-----------------------------------------------------------------------------------|-----------|-------------------------------------------------------------------------------|
| EVAL-01     | 41-01, 41-02 | Evaluation harness loads MidlineSet fixtures + calibration data and runs metrics without the full pipeline | SATISFIED | run_evaluation loads v2.0 fixture with CalibBundle, builds models, runs triangulation independently |
| EVAL-02     | 41-02      | Frame selection produces 15-20 frames from ~300 frame window via uniform temporal sampling | SATISFIED | select_frames uses np.linspace(0, n-1, n_frames, dtype=int); determinism confirmed |
| EVAL-03     | 41-02      | Tier 1 metric: per-fish, per-camera reprojection error (mean and max) with overall aggregates | SATISFIED | compute_tier1 produces per_camera/per_fish dicts with mean_px/max_px plus overall aggregates |
| EVAL-04     | 41-02      | Tier 2 metric: leave-one-out camera stability (max control-point displacement across dropout runs) | SATISFIED | compute_tier2 aggregates max displacement; run_evaluation accumulates leave-one-out results |
| EVAL-05     | 41-02      | Evaluation outputs human-readable summary table and machine-diffable regression data | SATISFIED | format_summary_table produces ASCII; write_regression_json produces indent=2 JSON |

All 5 requirements satisfied. REQUIREMENTS.md marks all as Complete for Phase 41. No orphaned requirements detected.

### Anti-Patterns Found

No anti-patterns found. Scanned all new and modified files in `src/aquapose/evaluation/` and `src/aquapose/io/midline_fixture.py` for TODO/FIXME/placeholder patterns and empty implementations. None found.

### Human Verification Required

None. All observable truths verified programmatically. The 708-test suite passes. No visual or real-time behavior requires human inspection at this stage.

### Gaps Summary

No gaps. All 13 truths verified, all 12 artifacts confirmed substantive and wired, all 6 key links confirmed wired, all 5 requirements satisfied.

The evaluation harness is complete and functional:
- CalibBundle frozen dataclass enables self-contained v2.0 fixtures
- load_midline_fixture handles both v1.0 (backward compat) and v2.0 formats
- export_midline_fixtures writes calib/ keys when models provided
- run_evaluation orchestrates fixture loading, model building, frame selection, Tier 1 + Tier 2 computation, ASCII output and JSON regression output — without running the full pipeline
- 708 tests pass with 0 regressions

---

_Verified: 2026-03-02T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
