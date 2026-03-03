---
phase: 49-tuningorchestrator-and-aquapose-tune-cli
plan: 02
subsystem: cli
tags: [click, evaluation, tuning, cli]

# Dependency graph
requires:
  - phase: 49-01
    provides: TuningOrchestrator with sweep_association, sweep_reconstruction, and formatting helpers
provides:
  - aquapose tune CLI command with --stage, --config, --n-frames, --n-frames-validate, --top-n options
  - Deletion of legacy scripts/tune_association.py and scripts/tune_threshold.py
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "tune_cmd follows eval_cmd inline-import pattern: imports from aquapose.evaluation.tuning inside the function body"
    - "StaleCacheError and FileNotFoundError both convert to click.ClickException"

key-files:
  created: []
  modified:
    - src/aquapose/cli.py
  deleted:
    - scripts/tune_association.py
    - scripts/tune_threshold.py

key-decisions:
  - "tune_cmd registered as @cli.command('tune') matching eval_cmd naming convention"
  - "Inline imports used for TuningOrchestrator and formatting helpers, consistent with eval_cmd pattern"
  - "evaluation/__init__.py already exported TuningOrchestrator and TuningResult from Plan 01 — no changes needed"

patterns-established:
  - "tune_cmd: thin CLI wrapper with inline imports; all logic delegated to TuningOrchestrator"

requirements-completed: [CLEAN-01, CLEAN-02]

# Metrics
duration: 5min
completed: 2026-03-03
---

# Phase 49 Plan 02: TuningOrchestrator CLI Wiring Summary

**`aquapose tune` CLI command wired to TuningOrchestrator, legacy tune_association.py and tune_threshold.py scripts deleted**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-03T20:00:00Z
- **Completed:** 2026-03-03T20:03:42Z
- **Tasks:** 2
- **Files modified:** 1 modified, 2 deleted

## Accomplishments
- Added `aquapose tune --stage [association|reconstruction] -c <config>` command to cli.py
- Command prints 2D yield matrix (association sweeps), comparison table, and config diff
- Deleted scripts/tune_association.py and scripts/tune_threshold.py (superseded by CLI)
- All 823 unit tests pass, lint clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire tune CLI command and update exports** - `dd00bc3` (feat)
2. **Task 2: Delete legacy tuning scripts** - `c17312d` (chore)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `src/aquapose/cli.py` - Added tune_cmd with --stage, --config, --n-frames, --n-frames-validate, --top-n options
- `scripts/tune_association.py` - DELETED (superseded by aquapose tune --stage association)
- `scripts/tune_threshold.py` - DELETED (superseded by aquapose tune --stage reconstruction)

## Decisions Made
- `evaluation/__init__.py` required no changes — TuningOrchestrator and TuningResult were already exported by Plan 01
- Docstring/comment references to deleted scripts in src/ and tests/ left as-is (historical context, not functional imports)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 49 complete: TuningOrchestrator + `aquapose tune` CLI fully implemented
- `aquapose tune --stage association -c <config>` and `aquapose tune --stage reconstruction -c <config>` both operational
- Legacy tuning scripts removed

---
*Phase: 49-tuningorchestrator-and-aquapose-tune-cli*
*Completed: 2026-03-03*
