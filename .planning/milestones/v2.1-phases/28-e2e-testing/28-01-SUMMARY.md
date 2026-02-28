---
phase: 28-e2e-testing
plan: 01
subsystem: testing
tags: [pytest, e2e, synthetic, pipeline, v2.1, SyntheticDataStage, PosePipeline]

# Dependency graph
requires:
  - phase: 27-diagnostic-visualization
    provides: Full v2.1 5-stage pipeline with observer infrastructure
  - phase: 22-pipeline-scaffolding
    provides: PosePipeline, PipelineContext, build_stages factory
  - phase: 24-per-camera-2d-tracking
    provides: TrackingStage with OC-SORT
  - phase: 25-association
    provides: AssociationStage with Leiden clustering
provides:
  - "tests/e2e/test_smoke.py: Rewritten e2e tests -- TestSyntheticSmoke (fast CI) + TestRealData (@slow)"
  - "tests/e2e/conftest.py: Session-scoped fixtures for calibration, video, weights, output dir"
  - "SyntheticDataStage bug fix: fish placed below water interface (water_z + 0.05-0.35m)"
  - "tests/e2e/output/ gitignored for artifact review"
affects: [29-optimization, future-regression-testing]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "E2E test structure: TestSyntheticSmoke (no @slow) + TestRealData (@slow @e2e)"
    - "conftest.py session-scoped fixtures with pytest.skip on missing data paths"
    - "Real-data tests save artifacts to tests/e2e/output/ for human review"

key-files:
  created:
    - tests/e2e/conftest.py
    - .planning/phases/28-e2e-testing/28-BUGS.md
  modified:
    - tests/e2e/test_smoke.py
    - src/aquapose/core/synthetic.py
    - .gitignore

key-decisions:
  - "Direct in-process invocation (PosePipeline.run()) preferred over SmokeTestRunner subprocess approach -- faster, cleaner tracebacks"
  - "Synthetic tests NOT marked @slow -- serve as fast CI guard in normal test suite"
  - "Real-data tests require LUTs to produce 3D output -- documented as known limitation, not test failure"
  - "Fish x/y range kept at +/-0.08m (original) to work for both real cameras and mock unit tests"

patterns-established:
  - "E2E fixture pattern: session-scoped with pytest.skip on missing paths, separate e2e_output_dir for artifacts"
  - "Bug triage: blocking bugs fixed inline, non-blocking bugs documented in 28-BUGS.md"

requirements-completed: [EVAL-01]

# Metrics
duration: 22min
completed: 2026-02-27
---

# Phase 28 Plan 01: E2E Test Rewrite Summary

**Rewritten e2e tests exercise the v2.1 5-stage pipeline in-process: TestSyntheticSmoke (fast CI, no @slow) + TestRealData (@slow @e2e), with a blocking bug fix in SyntheticDataStage fish z-coordinate placement**

## Performance

- **Duration:** 22 min
- **Started:** 2026-02-27T22:28:00Z
- **Completed:** 2026-02-27T22:50:07Z
- **Tasks:** 1 of 2 (Task 2 is checkpoint:human-verify)
- **Files modified:** 4

## Accomplishments

- Rewrote tests/e2e/test_smoke.py: dropped legacy SmokeTestRunner subprocess approach in favor of direct in-process PosePipeline.run() invocation for cleaner tracebacks and faster execution
- Added tests/e2e/conftest.py with session-scoped fixtures for calibration_path, test_video_dir, yolo_weights, unet_weights, and e2e_output_dir; all skip gracefully when data is unavailable
- Fixed blocking bug in SyntheticDataStage: fish were placed at z=0.02-0.12m (above the water interface at z=1.03m), causing zero detections in every camera; corrected to water_z + 0.05-0.35m
- Synthetic smoke tests (2) pass in `hatch run test` without @slow marker; real-data test class exists with @slow @e2e markers

## Task Commits

1. **Task 1: Rewrite e2e tests + conftest + fix SyntheticDataStage** - `b7adf90` (feat)

## Files Created/Modified

- `tests/e2e/test_smoke.py` - Rewritten with TestSyntheticSmoke + TestRealData classes
- `tests/e2e/conftest.py` - New: session fixtures for calibration/video/weights/output paths
- `src/aquapose/core/synthetic.py` - Bug fix: fish z-coordinate (water_z + 0.05-0.35m)
- `.gitignore` - Added tests/e2e/output/ exclusion

## Decisions Made

- Direct in-process PosePipeline.run() preferred over SmokeTestRunner subprocess approach: faster, better tracebacks, simpler config construction.
- Synthetic tests not marked @slow: they serve as fast CI guard and run in ~30s with real calibration.
- AssociationStage LUT dependency documented as known limitation (NB-001 in 28-BUGS.md), not treated as test failure. Synthetic output validation only requires tracks_2d to be non-empty (tracks pass even without LUTs).
- Fish x/y range kept at +/-0.08m to satisfy both real cameras (3+ cameras visible) and mock unit tests (640px image, 1000px/m projection scale).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] SyntheticDataStage places fish above water interface**
- **Found during:** Task 1 (test writing, synthetic pipeline investigation)
- **Issue:** `_generate_fish_splines()` used `cz = rng.uniform(0.02, 0.12)` (meters), but the air-water interface in release_calibration is at `water_z = 1.03m`. Refractive projection returns `valid=False` for points above the interface, so all 12 cameras produced 0 detections.
- **Fix:** Added `water_z` parameter to `_generate_fish_splines()`, passed from `cal_data.water_z`. Fish now placed at `water_z + rng.uniform(0.05, 0.35)`. Result: 12 tracklets across 7 cameras from synthetic data.
- **Files modified:** `src/aquapose/core/synthetic.py`
- **Verification:** Synthetic pipeline produces 12 tracklets, `test_synthetic_output_validation` passes, all unit tests still pass (554 passed)
- **Committed in:** b7adf90 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Fix was necessary for any synthetic pipeline output. Without it, e2e tests would fail even though the tracking/association/reconstruction code was correct.

## Issues Encountered

- EN DASH characters in docstrings (from writing) flagged by ruff RUF002/RUF003 in pre-commit hook; replaced with hyphens.

## User Setup Required

None - no external service configuration required. Note: real-data tests require pre-built LUTs (`aquapose build-luts`) for 3D output; this is a one-time setup step documented in 28-BUGS.md.

## Checkpoint: Awaiting Human Verification

Task 2 is a `checkpoint:human-verify` requiring the user to run real-data e2e tests and review artifacts.

**To continue:** Run `hatch run test-all tests/e2e/test_smoke.py -v` and review `tests/e2e/output/` for reprojection overlays, then signal approval.

## Next Phase Readiness

- Synthetic e2e tests pass as fast CI guard
- Real-data test infrastructure ready (conftest.py fixtures, output dir)
- Real-data test execution pending user verification (Task 2 checkpoint)
- Non-blocking: AssociationStage needs LUTs for full 3D output

---
*Phase: 28-e2e-testing*
*Completed: 2026-02-27*

## Self-Check: PASSED

Files verified:
- `tests/e2e/test_smoke.py` -- FOUND
- `tests/e2e/conftest.py` -- FOUND
- `src/aquapose/core/synthetic.py` -- FOUND (modified)
- `.gitignore` -- FOUND (modified)

Commits verified:
- b7adf90 -- FOUND (`feat(28-01): rewrite e2e tests and fix synthetic fish placement`)
