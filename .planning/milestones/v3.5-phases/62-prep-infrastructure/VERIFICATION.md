# Phase 62: Prep Infrastructure - Verification

**Verified:** 2026-03-05

## Goal
Users can prepare calibrated keypoint t-values and pre-generated LUTs before running the pipeline. The pipeline fails fast with actionable errors if prep steps haven't been run.

## Verification Checklist

### Plan 62-01: calibrate-keypoints and Fail-Fast
- [x] `aquapose prep calibrate-keypoints --config <yaml>` updates pipeline config YAML in place with keypoint_t_values
- [x] Config round-trip preserves other fields (midline section created if absent)
- [x] PoseEstimationBackend raises ValueError at __init__ when keypoint_t_values is None
- [x] PoseEstimationBackend accepts explicit keypoint_t_values without raising
- [x] init-config scaffold includes YAML comment near midline section
- [x] init-config prints next-steps console output after project creation

### Plan 62-02: generate-luts CLI and Lazy Generation Removal
- [x] `aquapose prep generate-luts --config <yaml>` generates and saves forward+inverse LUTs
- [x] Running without --force skips when LUTs already exist
- [x] `--force` flag regenerates even when LUTs exist
- [x] AssociationStage.run() raises FileNotFoundError if LUTs missing (no lazy generation)
- [x] build_stages() raises FileNotFoundError early if association will run and LUTs missing
- [x] build_stages() skips LUT check when stop_after is before association
- [x] _LutConfigFromDict avoids training->engine import boundary violation

### Testing
- [x] 5 tests in test_calibrate_keypoints.py (CLI round-trip, fail-fast)
- [x] 5 tests in test_generate_luts_cli.py (CLI, skip, force, build_stages checks)
- [x] Autouse conftest fixtures for LUT mocking in engine and reconstruction test suites
- [x] All 887 tests pass; no new failures introduced
- [x] Import boundary check passes

## Result: PASS

---
*Phase: 62-prep-infrastructure*
*Verified: 2026-03-05*
