---
phase: 66-training-run-management
plan: 01
subsystem: training
tags: [cli, yaml, csv, provenance, run-management]

# Dependency graph
requires:
  - phase: 65-dataset-assembly
    provides: assembled datasets with confidence.json and pseudo_val_metadata.json sidecars
provides:
  - run_manager module with create_run_dir, snapshot_config, write_summary, parse_best_metrics, extract_dataset_provenance, print_next_steps
  - CLI commands with --config/--tag flags replacing --output-dir
  - Timestamped run directory creation under project training tree
  - summary.json with metrics, provenance, and training config
affects: [66-02-compare-command]

# Tech tracking
tech-stack:
  added: []
  patterns: [run-directory-management, provenance-tracking, config-snapshot]

key-files:
  created: [src/aquapose/training/run_manager.py, tests/unit/training/test_run_manager.py]
  modified: [src/aquapose/training/cli.py, src/aquapose/training/__init__.py, tests/unit/training/test_training_cli.py]

key-decisions:
  - "yaml.safe_load for project config (no engine imports, boundary compliant)"
  - "xfail marker for compare command test (anticipating Plan 66-02)"
  - "summary.json schema follows RESEARCH.md recommendation with run_id, metrics, provenance"

patterns-established:
  - "Run directory pattern: {project_dir}/training/{model_type}/run_{timestamp}/"
  - "Config snapshot: cli_args as config.yaml + dataset sidecars copied to run dir"
  - "Post-training flow: parse results.csv -> write summary.json -> print next steps"

requirements-completed: [TRAIN-01, TRAIN-03]

# Metrics
duration: 6min
completed: 2026-03-05
---

# Phase 66 Plan 01: Run Manager and CLI Integration Summary

**Run manager module with timestamped directories, config snapshots, summary.json provenance, and --config/--tag CLI flags replacing --output-dir**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-05T20:51:41Z
- **Completed:** 2026-03-05T20:57:57Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Created run_manager.py with 7 functions for run directory management, config snapshotting, metrics parsing, summary generation, and provenance extraction
- Modified all three training CLI commands (yolo-obb, seg, pose) to use --config/--tag instead of --output-dir
- 14 unit tests for run_manager, updated 3 help text tests, added xfail test for compare command

## Task Commits

Each task was committed atomically:

1. **Task 1: Create run_manager module (TDD RED)** - `834931d` (test)
2. **Task 1: Create run_manager module (TDD GREEN)** - `ad3b04a` (feat)
3. **Task 2: Modify training CLI commands** - `54d878e` (feat)

_Note: Task 1 used TDD with separate RED and GREEN commits_

## Files Created/Modified
- `src/aquapose/training/run_manager.py` - Run directory creation, config snapshot, summary.json, metrics parsing, provenance extraction
- `src/aquapose/training/cli.py` - Replaced --output-dir with --config/--tag, integrated run management
- `src/aquapose/training/__init__.py` - Added run_manager exports to public API
- `tests/unit/training/test_run_manager.py` - 14 unit tests for all run_manager functions
- `tests/unit/training/test_training_cli.py` - Updated flag tests, added compare xfail test

## Decisions Made
- Used yaml.safe_load directly in run_manager to avoid engine import boundary violation
- Marked compare command test as xfail since it will be implemented in Plan 66-02
- summary.json schema includes run_id, tag, model_type, model_variant, parent_weights, dataset_path, dataset_sources, training_config, metrics, training_duration_seconds, created timestamp

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- run_manager module provides all functions needed for Plan 66-02 (compare command)
- summary.json schema established for cross-run comparison parsing
- xfail test ready to be removed when compare command is implemented

---
*Phase: 66-training-run-management*
*Completed: 2026-03-05*

## Self-Check: PASSED
