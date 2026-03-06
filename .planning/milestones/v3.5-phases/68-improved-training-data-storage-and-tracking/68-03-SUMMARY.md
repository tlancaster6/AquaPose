---
phase: 68-improved-training-data-storage-and-tracking
plan: 03
subsystem: cli
tags: [click, cli, training-data, assembly, symlinks, yolo, sqlite]

# Dependency graph
requires:
  - phase: 68-01
    provides: SampleStore class with CRUD, query, exclude/include/remove
  - phase: 68-02
    provides: Data CLI import/convert commands, data_group Click group
provides:
  - "SampleStore.assemble() for symlink-based YOLO dataset creation with pseudo-excluded val split"
  - "SampleStore.save_dataset/get_dataset/list_datasets for manifest persistence"
  - "SampleStore.summary() for store-level statistics"
  - "aquapose data assemble CLI command with query filters and pseudo-in-val override"
  - "aquapose data status/list/exclude/include/remove CLI commands for lifecycle management"
affects: [68-04]

# Tech tracking
tech-stack:
  added: []
  patterns: [symlink-based-dataset-assembly, pseudo-excluded-val-split, relative-symlinks]

key-files:
  created: []
  modified:
    - src/aquapose/training/store.py
    - src/aquapose/training/data_cli.py
    - tests/unit/training/test_store.py
    - tests/unit/training/test_data_cli.py

key-decisions:
  - "Relative symlinks for dataset assembly (portable across machines)"
  - "Pseudo-labels excluded from val split by default (manual+corrected only in val)"
  - "Assembly removes existing dataset dir before recreating (clean reassembly)"

patterns-established:
  - "_resolve_store() helper for config-to-store-path resolution in CLI commands"
  - "Query dict passed through to store.query() for flexible filtering"

requirements-completed: [STORE-05, STORE-06]

# Metrics
duration: 7min
completed: 2026-03-06
---

# Phase 68 Plan 03: Dataset Assembly and Lifecycle CLI Summary

**Symlink-based YOLO dataset assembly with pseudo-excluded val split, plus status/list/exclude/include/remove lifecycle commands**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-06T17:45:32Z
- **Completed:** 2026-03-06T17:52:05Z
- **Tasks:** 2 (both TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- SampleStore gains 5 new methods: save_dataset, get_dataset, list_datasets, assemble, summary
- assemble() creates YOLO directory structure with relative symlinks, deterministic seeded splitting, pseudo-excluded val split (configurable), min_confidence filter, and dataset.yaml generation
- 6 new CLI commands: assemble, status, list, exclude, include, remove --purge
- 21 new tests (11 store + 10 CLI) all passing, total test suite at 1076

## Task Commits

Each task was committed atomically:

1. **RED: Failing tests for SampleStore assembly** - `8620559` (test)
2. **GREEN: Implement SampleStore assembly methods** - `fd2b99c` (feat)
3. **RED: Failing tests for CLI commands** - `855191c` (test)
4. **GREEN: Implement CLI commands** - `a66d812` (feat)

## Files Created/Modified
- `src/aquapose/training/store.py` - Added save_dataset, get_dataset, list_datasets, assemble, summary methods
- `src/aquapose/training/data_cli.py` - Added assemble, status, list, exclude, include, remove commands
- `tests/unit/training/test_store.py` - 11 new tests for assembly, persistence, summary
- `tests/unit/training/test_data_cli.py` - 10 new tests for CLI commands

## Decisions Made
- Relative symlinks via os.path.relpath for portability across machines
- Pseudo-labels excluded from validation split by default; --pseudo-in-val flag to override
- Assembly removes and recreates dataset directory for clean reassembly
- Source filter in assemble CLI supports single source only (query dict pattern)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test expectation for augmented child source inheritance**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Test expected by_source["manual"] == 1 but augmented children inherit parent source, so count is 2
- **Fix:** Updated test assertion to match actual behavior (augmented child inherits "manual" source)
- **Files modified:** tests/unit/training/test_store.py
- **Committed in:** fd2b99c (task commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test expectation)
**Impact on plan:** Minor test fix. No scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Full data lifecycle management complete (import, convert, assemble, status, list, exclude, include, remove)
- Ready for Plan 04 (model registry) to track trained models against assembled datasets

## Self-Check: PASSED

All 4 files verified. All 4 commit hashes confirmed.

---
*Phase: 68-improved-training-data-storage-and-tracking*
*Completed: 2026-03-06*
