---
phase: 69-cli-workflow-cleanup
plan: 01
subsystem: cli
tags: [click, project-resolution, run-resolution, cli-utils]

requires:
  - phase: none
    provides: greenfield CLI utilities
provides:
  - resolve_project and resolve_run utilities for all CLI commands
  - --project top-level option on main CLI group
  - Renamed init command with expanded scaffold (training_data dirs)
affects: [69-02, 69-03]

tech-stack:
  added: []
  patterns: [ctx.obj caching for lazy project resolution, CWD walk-up for project detection]

key-files:
  created: [src/aquapose/cli_utils.py, tests/unit/test_cli_utils.py]
  modified: [src/aquapose/cli.py, tests/unit/engine/test_cli.py]

key-decisions:
  - "CWD walk-up stops at home dir to avoid scanning unrelated directories"
  - "Timestamp matching uses removeprefix('run_') for clean prefix comparison"
  - "Lazy caching in ctx.obj['_project_dir'] avoids repeated resolution"

patterns-established:
  - "Project resolution: resolve_project(name) for named, resolve_project(None) for CWD detection"
  - "Run resolution: timestamp prefix matching with sorted-name ordering"

requirements-completed: [CLI-01, CLI-02, CLI-03]

duration: 3min
completed: 2026-03-06
---

# Phase 69 Plan 01: CLI Foundation Utilities Summary

**Project and run resolution utilities with --project CLI option and renamed init command with expanded scaffold**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-06T19:36:45Z
- **Completed:** 2026-03-06T19:40:09Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created cli_utils.py with resolve_project (by name or CWD walk-up) and resolve_run (by timestamp, "latest", or path)
- Added --project/-p top-level option to main CLI group with ctx.obj storage
- Renamed init-config to init with training_data/obb/ and training_data/pose/ in scaffold
- 13 unit tests for resolution utilities, 6 updated tests for init command

## Task Commits

Each task was committed atomically:

1. **Task 1: Create cli_utils.py with project and run resolution** - `12dc82d` (feat, TDD)
2. **Task 2: Wire --project into CLI group and rename init-config to init** - `5ed22da` (feat)

## Files Created/Modified
- `src/aquapose/cli_utils.py` - Project/run resolution utilities (resolve_project, resolve_run, get_project_dir, get_config_path)
- `tests/unit/test_cli_utils.py` - 13 unit tests for resolution behaviors
- `src/aquapose/cli.py` - Added --project option, renamed init-config to init, expanded scaffold
- `tests/unit/engine/test_cli.py` - Updated init-config invocations to init, added training_data assertions

## Decisions Made
- CWD walk-up stops at home dir to avoid scanning unrelated directories
- Timestamp matching uses removeprefix('run_') for clean prefix comparison
- Lazy caching in ctx.obj['_project_dir'] avoids repeated resolution
- Negative run indices not supported (conflict with Click argument parser)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-commit hook caught RUF043 (regex metacharacter in pytest match pattern for "config.yaml") - fixed with raw string

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- cli_utils.py ready for import by Plan 02 (command migration to --project)
- All resolution utilities tested and committed

---
*Phase: 69-cli-workflow-cleanup*
*Completed: 2026-03-06*
