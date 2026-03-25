---
phase: 103-training-data-mining
plan: 02
subsystem: cli
tags: [click, cli, reid, mining]

requires:
  - phase: 103-training-data-mining
    provides: "TrainingDataMiner, MinerConfig from plan 01"
provides:
  - "mine-reid-crops CLI command for mining ReID training crops"
affects: [104-fine-tuning]

tech-stack:
  added: []
  patterns: ["CLI command following stitch_cmd pattern with lazy imports"]

key-files:
  created: []
  modified:
    - src/aquapose/cli.py

key-decisions:
  - "Placed command near stitch_cmd as both are post-pipeline commands"
  - "--overwrite flag uses shutil.rmtree for clean re-runs"

patterns-established:
  - "Post-pipeline command pattern: resolve_run + lazy import + config dataclass"

requirements-completed: [TRAIN-01]

duration: 5min
completed: 2026-03-25
---

# Plan 103-02: mine-reid-crops CLI Command Summary

**CLI command wiring TrainingDataMiner into aquapose with all MinerConfig options and --overwrite flag**

## Performance

- **Duration:** ~5 min
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Added `mine-reid-crops` command to aquapose CLI
- All MinerConfig parameters exposed as CLI options with defaults matching the dataclass
- `--overwrite/--no-overwrite` flag for clean re-runs
- Command appears in `aquapose --help` output

## Task Commits

1. **Task 1: Add mine-reid-crops CLI command** - `1c5d615` (feat)

## Files Created/Modified
- `src/aquapose/cli.py` - Added mine_reid_crops_cmd function with Click decorators

## Decisions Made
None - followed plan as specified.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 103 complete: mining infrastructure ready for Phase 104 fine-tuning
- CLI provides entry point: `aquapose -p YH mine-reid-crops [RUN]`

---
*Phase: 103-training-data-mining*
*Completed: 2026-03-25*
