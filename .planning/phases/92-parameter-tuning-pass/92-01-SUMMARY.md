---
phase: 92-parameter-tuning-pass
plan: 01
subsystem: association
tags: [scoring, tuning, config, cli, centroid, keypoints, grid-sweep]

# Dependency graph
requires:
  - phase: 91-singleton-recovery
    provides: recovery_enabled guard and singleton recovery pipeline wiring
  - phase: 89-association-refactor
    provides: multi-keypoint scoring infrastructure (score_tracklet_pair)
provides:
  - tune CLI fallback to config.yaml when config_exhaustive.yaml is absent
  - use_multi_keypoint_scoring toggle on AssociationConfig (v3.7 vs v3.8 comparison)
  - _score_pair_centroid_only() centroid-based scoring path for v3.7 baseline
  - keypoint_confidence_floor as 3rd joint dimension in DEFAULT_GRID
  - 27-combo 3D joint grid in sweep_association (ray_dist x score_min x kpt_floor)
affects:
  - 92-02 (parameter tuning run will use these new grid/scoring capabilities)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - AssociationConfigLike protocol updated when AssociationConfig gains fields
    - MockAssociationConfig in test_scoring.py must mirror AssociationConfigLike protocol
    - TDD with keypoints=None tracklets to distinguish centroid-only vs multi-kpt code paths

key-files:
  created: []
  modified:
    - src/aquapose/cli.py
    - src/aquapose/engine/config.py
    - src/aquapose/core/association/scoring.py
    - src/aquapose/evaluation/stages/association.py
    - src/aquapose/evaluation/tuning.py
    - tests/unit/engine/test_config.py
    - tests/unit/engine/test_cli.py
    - tests/unit/core/association/test_scoring.py
    - tests/unit/evaluation/test_stage_association.py

key-decisions:
  - "Centroid-only toggle placed BEFORE keypoints=None check in score_tracklet_pair so toggle is respected even when keypoints are populated"
  - "keypoint_confidence_floor added to joint (Phase 1) grid not carry-forward (Phase 2) because it interacts tightly with ray_dist and score_min"
  - "3D joint grid gives 27 combos (~18-27min sweep) which is acceptable"

patterns-established:
  - "AssociationConfigLike protocol must be kept in sync with AssociationConfig fields"

requirements-completed: [EVAL-01]

# Metrics
duration: 9min
completed: 2026-03-11
---

# Phase 92 Plan 01: Parameter Tuning Prep Summary

**tune CLI fallback to config.yaml, v3.7 centroid-only scoring toggle, and keypoint_confidence_floor added as 3rd joint grid dimension for 27-combo sweep**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-11T20:22:47Z
- **Completed:** 2026-03-11T20:31:11Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- tune CLI no longer crashes when config_exhaustive.yaml is absent — falls back to config.yaml
- AssociationConfig.use_multi_keypoint_scoring toggle (default True) enables apples-to-apples v3.7 vs v3.8 comparison on cached data
- _score_pair_centroid_only() implements single centroid ray per frame, matching v3.7 behavior
- DEFAULT_GRID extended with keypoint_confidence_floor [0.2, 0.3, 0.4] as 3rd joint dimension
- sweep_association now runs 3D joint grid (27 combos) covering all three key parameters simultaneously

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix tune CLI config fallback and add scoring toggle to AssociationConfig** - `7db35c4` (feat)
2. **Task 2: Implement centroid-only scoring path, extend grid, and update sweep** - `2a30a95` (feat)

_Note: Both tasks used TDD (RED → GREEN → format fix)_

## Files Created/Modified
- `src/aquapose/cli.py` - tune_cmd falls back to config.yaml when config_exhaustive.yaml absent
- `src/aquapose/engine/config.py` - AssociationConfig gains use_multi_keypoint_scoring field
- `src/aquapose/core/association/scoring.py` - _score_pair_centroid_only() + toggle branch in score_tracklet_pair; AssociationConfigLike updated
- `src/aquapose/evaluation/stages/association.py` - DEFAULT_GRID adds keypoint_confidence_floor
- `src/aquapose/evaluation/tuning.py` - sweep_association uses 3D joint grid with kpt_floor
- `tests/unit/engine/test_config.py` - tests for use_multi_keypoint_scoring field
- `tests/unit/engine/test_cli.py` - TestTuneCLI: fallback and no-config-at-all tests
- `tests/unit/core/association/test_scoring.py` - TestCentroidOnlyScoring class; MockAssociationConfig updated
- `tests/unit/evaluation/test_stage_association.py` - test_default_grid_has_exactly_six_keys; keypoint_confidence_floor values test

## Decisions Made
- Toggle placed BEFORE keypoints=None check so it's respected even when keypoints exist on the tracklet
- keypoint_confidence_floor goes into the joint (Phase 1) grid, not carry-forward, because it interacts tightly with ray_dist and score_min
- 27-combo 3D joint grid is acceptable (~18-27 min sweep time)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Updated MockAssociationConfig and test_stage_association.py to match new protocol**
- **Found during:** Task 1 and 2 (GREEN phase)
- **Issue:** MockAssociationConfig in test_scoring.py must satisfy AssociationConfigLike protocol; test_stage_association.py had hardcoded "5 keys" assertion
- **Fix:** Added use_multi_keypoint_scoring to MockAssociationConfig; updated key count test to 6 and added keypoint_confidence_floor value test
- **Files modified:** tests/unit/core/association/test_scoring.py, tests/unit/evaluation/test_stage_association.py
- **Verification:** All 1192 tests pass
- **Committed in:** 2a30a95 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 - missing critical protocol compliance)
**Impact on plan:** Essential for test suite correctness. No scope creep.

## Issues Encountered
- CLI test patching: tune_cmd uses lazy imports inside the function body so `patch("aquapose.cli.TuningOrchestrator")` doesn't work. Resolved by testing that the old error message is absent rather than mocking the full orchestrator — simpler and more targeted.
- ruff format reformatted 3 files on first commit attempt; re-staged and recommitted cleanly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- tune CLI unblocked: can now run against any run directory with config.yaml
- v3.7 baseline comparison ready: set use_multi_keypoint_scoring=False in config to compare centroid vs multi-keypoint scoring on same cached data
- 27-combo joint grid ready for phase 92-02 parameter sweep execution

---
*Phase: 92-parameter-tuning-pass*
*Completed: 2026-03-11*
