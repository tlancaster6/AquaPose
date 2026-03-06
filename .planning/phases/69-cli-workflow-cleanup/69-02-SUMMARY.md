---
phase: 69-cli-workflow-cleanup
plan: 02
subsystem: cli
tags: [click, project-resolution, run-shorthand, cli-migration]

requires:
  - phase: 69-cli-workflow-cleanup/01
    provides: "cli_utils.py with resolve_project, resolve_run, get_project_dir, get_config_path"
provides:
  - "All CLI commands use --project name-based resolution instead of --config path"
  - "Run-accepting commands support shorthand (latest, timestamp, path)"
  - "Pseudo-label inspect discovers all pseudo-label subsets from run directory"
affects: [69-cli-workflow-cleanup/03]

tech-stack:
  added: []
  patterns:
    - "@click.pass_context + get_project_dir/get_config_path for project resolution"
    - "monkeypatch_project fixture pattern for CLI test isolation"
    - "_inspect_subset helper for shared pseudo-label visualization"

key-files:
  created: []
  modified:
    - "src/aquapose/cli.py"
    - "src/aquapose/training/cli.py"
    - "src/aquapose/training/data_cli.py"
    - "src/aquapose/training/prep.py"
    - "src/aquapose/training/pseudo_label_cli.py"
    - "tests/unit/engine/test_cli.py"
    - "tests/unit/training/test_training_cli.py"
    - "tests/unit/training/test_data_cli.py"
    - "tests/unit/test_calibrate_keypoints.py"
    - "tests/unit/test_generate_luts_cli.py"
    - "tests/unit/training/test_pseudo_label_cli.py"

key-decisions:
  - "aquapose.cli_utils treated as shared utility (not engine/cli layer) for import boundary compliance"
  - "Pseudo-label inspect reworked from --data-dir to run-based auto-discovery of all subsets"
  - "Extracted _inspect_subset helper for reuse by generate --viz and inspect command"

patterns-established:
  - "monkeypatch_project fixture: patches aquapose.cli_utils.resolve_project for test isolation"
  - "All CLI commands use @click.pass_context + cli_utils helpers instead of --config"

requirements-completed: [CLI-04, CLI-05, CLI-06]

duration: 12min
completed: 2026-03-06
---

# Phase 69 Plan 02: CLI Config-to-Project Migration Summary

**Migrated all CLI commands from --config path args to --project name resolution with run shorthand on eval/viz/tune/smooth-z/pseudo-label**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-06
- **Completed:** 2026-03-06
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- Removed --config from all CLI commands that resolve project configuration (run, tune, train obb/seg/pose/compare, data import/assemble/status/list/exclude/include/remove, prep calibrate-keypoints/generate-luts, pseudo-label generate)
- Added RUN positional argument with shorthand resolution to eval, viz, tune, smooth-z, and pseudo-label generate/inspect
- Reworked pseudo-label inspect to auto-discover all pseudo-label subsets from run directory
- Updated all test files to use monkeypatch_project fixture pattern

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate top-level commands to project/run resolution** - `72b8f78` (feat)
2. **Task 2: Migrate subgroup commands to project resolution** - `80a6818` (feat)

## Files Created/Modified
- `src/aquapose/cli.py` - Migrated run/eval/tune/viz/smooth-z to project/run resolution
- `src/aquapose/training/cli.py` - Migrated yolo-obb/seg/pose/compare to project resolution
- `src/aquapose/training/data_cli.py` - Migrated all 7 data commands to project resolution
- `src/aquapose/training/prep.py` - Migrated calibrate-keypoints/generate-luts to project resolution
- `src/aquapose/training/pseudo_label_cli.py` - Migrated generate/inspect to project+run resolution
- `tests/unit/engine/test_cli.py` - Updated to use monkeypatch_project pattern
- `tests/unit/training/test_training_cli.py` - Added --config absence assertions
- `tests/unit/training/test_data_cli.py` - Rewrote all ~40 invocations for --project pattern
- `tests/unit/test_calibrate_keypoints.py` - Switched to cli invocation with --project
- `tests/unit/test_generate_luts_cli.py` - Switched to cli invocation with --project
- `tests/unit/training/test_pseudo_label_cli.py` - Rewrote for run-based interface

## Decisions Made
- Treated `aquapose.cli_utils` as shared utility layer, not part of engine/cli import boundary. Updated import boundary test to whitelist it.
- Reworked pseudo-label inspect from `--data-dir` (single directory) to RUN positional (auto-discovers obb, pose/consensus, pose/gap subsets)
- Extracted `_inspect_subset()` helper for shared visualization between generate's --viz and inspect command

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed unused variable lint errors in test_pseudo_label_cli.py**
- **Found during:** Task 2 (commit pre-commit hook)
- **Issue:** RUF059 flagged `run_dir` as unused in two test functions
- **Fix:** Renamed to `_run_dir` to indicate intentionally unused
- **Files modified:** tests/unit/training/test_pseudo_label_cli.py
- **Verification:** Pre-commit hooks pass
- **Committed in:** 80a6818 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor lint fix, no scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All commands use project resolution; Plan 03 (command renaming/cleanup) can proceed
- Run shorthand working on all applicable commands
- Test patterns established for future CLI changes

---
*Phase: 69-cli-workflow-cleanup*
*Completed: 2026-03-06*
