---
phase: 62-prep-infrastructure
plan: 01
subsystem: training, midline
tags: [cli, calibration, keypoints, fail-fast]

requires:
  - phase: 33
    provides: PoseEstimationBackend, keypoint_t_values config field
provides:
  - calibrate-keypoints CLI with --config flag (in-place YAML update)
  - PoseEstimationBackend fail-fast on missing keypoint_t_values
  - init-config scaffold with prep reminder comments and next-steps output
affects: [63]

tech-stack:
  added: []
  patterns: [pyyaml round-trip for config update, fail-fast at construction time]

key-files:
  created:
    - tests/unit/test_calibrate_keypoints.py
  modified:
    - src/aquapose/training/prep.py
    - src/aquapose/core/midline/backends/pose_estimation.py
    - src/aquapose/cli.py

key-decisions:
  - "Simple pyyaml load/update/dump for config round-trip (comments stripped, acceptable tradeoff)"
  - "PoseEstimationBackend raises ValueError at __init__ time, not at inference time"

requirements-completed: [PREP-01, PREP-02]

duration: ~10min
completed: 2026-03-05
---

# Plan 62-01: Rework calibrate-keypoints and Add Fail-Fast

**calibrate-keypoints CLI updated to write keypoint_t_values directly into pipeline config YAML; PoseEstimationBackend fails fast when t-values missing**

## Performance

- **Duration:** ~10 min
- **Tasks:** 2
- **Files modified:** 3 production, 1 test

## Accomplishments
- Replaced --output flag with --config on calibrate-keypoints (reads and updates YAML in place)
- PoseEstimationBackend.__init__ raises ValueError when keypoint_t_values is None (replaces silent np.linspace fallback)
- init-config scaffold includes YAML comment and next-steps console output reminding users to run prep commands
- TDD approach: tests written first (RED), then implementation (GREEN)

## Task Commits

1. **Task 1: Rework calibrate-keypoints and add fail-fast** - `cde55f0` (feat)
2. **Task 2: Update init-config scaffold with prep reminders** - included in same commit

## Decisions Made
- pyyaml load/dump for config update (comments stripped but acceptable)
- ValueError rather than RuntimeError for missing keypoint_t_values (construction-time error)

## Deviations from Plan
None

## Issues Encountered
- Existing tests creating PoseEstimationBackend without keypoint_t_values needed updating to pass explicit values
- Pre-commit ruff format auto-fixed test file on first commit attempt; re-staged and committed successfully

## User Setup Required
None

## Next Phase Readiness
- Phase 63 can rely on calibrated keypoint_t_values being present in config (fail-fast guarantees it)

---
*Phase: 62-prep-infrastructure*
*Completed: 2026-03-05*
