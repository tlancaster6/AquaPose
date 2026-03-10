---
phase: 70-metrics-comparison-infrastructure
plan: 02
subsystem: evaluation
tags: [per-keypoint, curvature, calibration, projection, metrics]

requires:
  - phase: 70-01
    provides: ReconstructionMetrics with percentile fields, evaluation system
provides:
  - compute_per_point_error() for per-keypoint reprojection error (EVAL-04)
  - compute_curvature_stratified() for curvature-binned quality (EVAL-05)
  - EvalRunner calibration loading via _load_projection_models()
  - Per-keypoint and curvature-stratified tables in text/JSON output
affects: [evaluation, pipeline-metrics]

tech-stack:
  added: []
  patterns: [BSpline reprojection, curvature quartile binning, graceful calibration degradation]

key-files:
  modified:
    - src/aquapose/evaluation/stages/reconstruction.py
    - src/aquapose/evaluation/runner.py
    - src/aquapose/evaluation/output.py
    - tests/unit/evaluation/test_stage_reconstruction.py
    - tests/unit/evaluation/test_eval_output.py

key-decisions:
  - "Per-keypoint error uses BSpline evaluation at uniform params then refractive projection per camera"
  - "Curvature stratification uses 2D curvature from best-confidence camera's midline"
  - "Calibration loaded lazily with graceful degradation (empty dict on failure)"
  - "Use compute_undistortion_maps(cam_data) for K_new, matching all other callers"

patterns-established:
  - "Lazy calibration loading: try/except import + construction, return empty dict on failure"
  - "curvature_stratified dict uses float | int | str values for mixed-type bin metadata"

requirements-completed: [EVAL-04, EVAL-05]

duration: 20min
completed: 2026-03-06
---

# Plan 70-02: Per-Keypoint Error and Curvature-Stratified Quality Summary

**Per-keypoint reprojection error and curvature-stratified reconstruction quality computed from 3D spline reprojection, wired into EvalRunner with calibration loading and rendered in text/JSON output**

## Performance

- **Duration:** 20 min
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Added compute_per_point_error() for per-body-point mean + p90 reprojection error (EVAL-04)
- Added compute_curvature_stratified() for quartile-binned reconstruction quality (EVAL-05)
- Added _load_projection_models() to EvalRunner with graceful calibration degradation
- Added per-keypoint error table and curvature-stratified quality table to text output
- Fixed typecheck errors: used compute_undistortion_maps for K_new, corrected curvature_stratified type annotations

## Task Commits

1. **Task 1: Per-keypoint and curvature-stratified computation** - `250b0d3`
2. **Task 2: Wire into EvalRunner and output formatters** - `ebe4329`

## Decisions Made
- Used compute_undistortion_maps(cam_data).K_new instead of cam_data.K for projection models, matching all other callers in the codebase

## Deviations from Plan
- Plan suggested `cal_data.camera_ids` which doesn't exist; used `cal_data.cameras.items()` instead
- Plan suggested `RefractiveProjectionModel(cal_data, cam_id)` constructor; actual constructor takes individual K/R/t/water_z/normal/n_air/n_water params

## Issues Encountered
- Typecheck errors from incorrect CalibrationData/RefractiveProjectionModel API usage in plan; fixed by consulting actual class definitions

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 6 EVAL requirements (01-06) complete
- Phase 70 ready for verification and completion

---
*Phase: 70-metrics-comparison-infrastructure*
*Completed: 2026-03-06*
