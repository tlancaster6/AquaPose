---
phase: 30-config-and-contracts
plan: "02"
subsystem: engine/config
tags: [config, pipeline, device, n_sample_points, refactoring]
dependency_graph:
  requires: [30-01]
  provides: [CFG-01, CFG-02, CFG-03, CFG-04, CFG-12]
  affects: [engine/pipeline, reconstruction/triangulation, reconstruction/curve_optimizer, core/synthetic, io/midline_writer]
tech_stack:
  added: []
  patterns: [device-auto-detection, config-propagation, sentinel-validation]
key_files:
  created: []
  modified:
    - src/aquapose/engine/config.py
    - src/aquapose/engine/pipeline.py
    - src/aquapose/reconstruction/triangulation.py
    - src/aquapose/reconstruction/curve_optimizer.py
    - src/aquapose/core/synthetic.py
    - src/aquapose/io/midline_writer.py
    - tests/unit/engine/test_config.py
    - tests/unit/engine/test_build_stages.py
    - tests/unit/engine/test_pipeline.py
    - tests/unit/core/test_synthetic.py
    - tests/e2e/test_smoke.py
decisions:
  - "device auto-detected via _default_device() helper using torch.cuda.is_available(); defaults to cuda:0 when CUDA present, cpu otherwise"
  - "n_sample_points default is 10 (not 15); n_sample_points propagates to midline.n_points in load_config() unless midline.n_points is explicitly set"
  - "N_SAMPLE_POINTS=15 kept as module-level fallback constant in triangulation.py; all pipeline modules now accept n_sample_points as a constructor parameter"
  - "n_animals sentinel is 0; load_config() raises ValueError when resolved_n_animals <= 0; direct PipelineConfig() construction bypasses this check"
  - "device and stop_frame removed from DetectionConfig; _RENAME_HINTS updated with both fields to produce did-you-mean hints when used in detection sub-config"
metrics:
  duration_minutes: 21
  tasks_completed: 3
  files_modified: 11
  completed_date: "2026-02-28"
---

# Phase 30 Plan 02: Config Promotion and Propagation Summary

Promoted device, n_sample_points, stop_frame, and project_dir to top-level PipelineConfig with auto-detection and full pipeline propagation, eliminating per-stage device overrides and all hardcoded 15-point literals.

## What Was Built

### Task 1: PipelineConfig field promotion (commit d7bb339)

Added four new top-level fields to `PipelineConfig`:

- `device: str` — auto-detected via `_default_device()` helper using `torch.cuda.is_available()`. Returns `"cuda:0"` when CUDA is available, `"cpu"` otherwise.
- `n_sample_points: int = 10` — locked decision: default is 10 (not the old 15). Propagates to `midline.n_points` in `load_config()` unless `midline.n_points` is explicitly set in YAML.
- `stop_frame: int | None = None` — moved from `DetectionConfig.stop_frame`.
- `project_dir: str = ""` — empty string means no resolution.

Removed `device` and `stop_frame` from `DetectionConfig`. Added rename hints to `_RENAME_HINTS` for both fields to produce "did you mean device (top-level)?" errors.

Changed `n_animals` default from 9 to 0 (sentinel). Added validation in `load_config()`: raises `ValueError("n_animals is required and must be > 0")` when `resolved_n_animals <= 0`.

Updated all existing test calls to `load_config()` to include `n_animals` in overrides.

### Task 2: Downstream propagation (commit d4ca315)

Updated `build_stages()` in `pipeline.py`:
- `DetectionStage`: now uses `config.device` and `config.stop_frame` (top-level)
- `MidlineStage`: now uses `config.device` (top-level)
- `SyntheticDataStage`: now passes `n_points=config.n_sample_points`

Updated downstream modules to accept `n_sample_points` as a constructor/init parameter:
- `SyntheticDataStage.__init__(n_points=10)` — uses `self._n_points` instead of `n_points = 15`
- `CurveOptimizer.__init__(n_sample_points=N_SAMPLE_POINTS)` — uses `self._n_sample_points` in `optimize_midlines()` for residual evaluation and half-width array sizing
- `Midline3DWriter.__init__(n_sample_points=N_SAMPLE_POINTS)` — uses `self._n_sample_points` for HDF5 dataset sizing and buffer allocation

Updated `N_SAMPLE_POINTS = 15` comment in `triangulation.py` to indicate it is a fallback constant, not the authoritative value.

Fixed `test_synthetic.py`: `annot.midline.points.shape == (10, 2)` (was 15).

### Task 3: New tests and E2E CPU test (commit 91b32e5)

Added 8 new tests to `test_config.py`:
1. `test_device_auto_detected` — PipelineConfig() device is "cuda:0" or "cpu"
2. `test_n_sample_points_default_is_10` — PipelineConfig() n_sample_points == 10
3. `test_n_sample_points_propagates_to_midline` — YAML n_sample_points:8 → midline.n_points == 8
4. `test_midline_n_points_overrides_top_level` — explicit midline.n_points:12 wins over n_sample_points:8
5. `test_stop_frame_at_top_level` — stop_frame:100 in YAML reads correctly
6. `test_n_animals_required_raises_when_missing` — empty YAML raises ValueError
7. `test_device_in_detection_raises_with_hint` — detection.device raises "did you mean"
8. `test_stop_frame_in_detection_raises_with_hint` — detection.stop_frame raises "did you mean"

Added `TestBuildStagesConfigDevice` to `test_build_stages.py` with `test_build_stages_uses_config_device` verifying device="cpu" config constructs without error and n_sample_points is passed to SyntheticDataStage.

Updated `test_smoke.py`:
- Parametrized `test_synthetic_pipeline_completes` with `device=["cpu", "cuda:0"]` (CUDA variant auto-included when GPU available)
- Added `n_animals=3` to all synthetic config overrides
- Fixed real-data config: moved `device` and `stop_frame` from `detection` sub-config to top level

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_synthetic to expect 10 points instead of 15**
- **Found during:** Task 2 verification
- **Issue:** `test_annotated_detections_have_midlines` asserted `annot.midline.points.shape == (15, 2)` but `n_sample_points=10` is the new default
- **Fix:** Changed expected shape to `(10, 2)` to match the new default
- **Files modified:** `tests/unit/core/test_synthetic.py`
- **Commit:** d4ca315

**2. [Rule 1 - Bug] Updated test_pipeline.py to include n_animals in all load_config() calls**
- **Found during:** Task 2 verification run
- **Issue:** Multiple `load_config()` calls without n_animals raised ValueError with new sentinel validation
- **Fix:** Added `n_animals: 3` to all CLI overrides in test_pipeline.py
- **Files modified:** `tests/unit/engine/test_pipeline.py`
- **Commit:** d4ca315

**3. [Rule 1 - Bug] Fixed real-data smoke test configuration**
- **Found during:** Task 3 (updating test_smoke.py)
- **Issue:** `_build_real_config()` placed `device` and `stop_frame` inside `detection` sub-config which no longer supports them
- **Fix:** Moved both fields to top-level overrides and added `n_animals: 9`
- **Files modified:** `tests/e2e/test_smoke.py`
- **Commit:** 91b32e5

## Verification Results

- `hatch run test`: 578 passed, 0 failed (31 deselected for slow/e2e marks)
- `PipelineConfig().device` returns "cuda:0" (CUDA available on test machine)
- `PipelineConfig().n_sample_points == 10` confirmed
- YAML with `n_sample_points: 8` produces `config.midline.n_points == 8`
- YAML without `n_animals` raises `ValueError: n_animals is required and must be > 0`
- YAML with `detection: {device: cpu}` raises `ValueError` with "did you mean" hint
- E2E synthetic pipeline runs on CPU (`test_synthetic_pipeline_completes[cpu]` passes)
- E2E synthetic pipeline also runs on CUDA (`test_synthetic_pipeline_completes[cuda:0]` passes)
- No hardcoded 15 literals remain in pipeline-path modules (reconstruction, synthetic, midline_writer)

## Self-Check: PASSED

All key files present and all commits verified:

**Files:**
- FOUND: src/aquapose/engine/config.py
- FOUND: src/aquapose/engine/pipeline.py
- FOUND: src/aquapose/reconstruction/triangulation.py
- FOUND: src/aquapose/reconstruction/curve_optimizer.py
- FOUND: src/aquapose/core/synthetic.py
- FOUND: src/aquapose/io/midline_writer.py
- FOUND: .planning/phases/30-config-and-contracts/30-02-SUMMARY.md

**Commits:**
- FOUND: d7bb339 (feat: promote fields to PipelineConfig)
- FOUND: d4ca315 (feat: propagate through build_stages and downstream)
- FOUND: 91b32e5 (test: add config promotion tests and CPU E2E test)
