---
phase: quick-19
plan: 19
subsystem: training
tags: [cli, frame-selection, pseudo-labels, dataset-assembly]

provides:
  - "selected_frames parameter on assemble_dataset with _filter_by_frames helper"
  - "CLI flags --temporal-step, --diversity-bins, --diversity-max-per-bin on assemble command"
  - "End-to-end wiring from CLI to frame_selection.py functions"
affects: [pseudo-label-assembly, training-data-curation]

tech-stack:
  added: []
  patterns:
    - "Dynamic import of evaluation.runner for diagnostic cache loading in CLI (boundary compliance)"

key-files:
  created: []
  modified:
    - "src/aquapose/training/dataset_assembly.py"
    - "src/aquapose/training/pseudo_label_cli.py"
    - "tests/unit/training/test_dataset_assembly.py"
    - "tests/unit/training/test_pseudo_label_cli.py"

key-decisions:
  - "Frame index parsed from first 6 chars of stem (int(stem[:6])), consistent with Phase 65 decision"
  - "Runs not in selected_frames dict are kept unfiltered (Phase 65 decision preserved)"
  - "Frame selection filter applied after confidence/gap filtering but before max_frames cap"

requirements-completed: [TODO-19]

duration: 4min
completed: 2026-03-05
---

# Quick Task 19: Wire Frame Selection into Pseudo-Label Assembly CLI

**Connected temporal_subsample and diversity_sample functions to assemble CLI via selected_frames filtering on assemble_dataset**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-05T22:08:10Z
- **Completed:** 2026-03-05T22:12:23Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Added `_filter_by_frames` helper and `selected_frames` parameter to `assemble_dataset()` with full test coverage
- Added `--temporal-step`, `--diversity-bins`, `--diversity-max-per-bin` CLI flags to `aquapose pseudo-label assemble`
- Wired diagnostic cache loading and frame selection functions end-to-end when flags are active
- Behavior unchanged when no frame selection flags are set (selected_frames=None)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add selected_frames parameter with filtering (TDD)** - `5226b64` (test)
2. **Task 2: Add CLI flags and wire frame selection** - `287a6a8` (feat)

## Files Created/Modified
- `src/aquapose/training/dataset_assembly.py` - Added `_filter_by_frames` helper and `selected_frames` parameter to `assemble_dataset`
- `src/aquapose/training/pseudo_label_cli.py` - Added 3 CLI flags, frame selection wiring with diagnostic cache loading
- `tests/unit/training/test_dataset_assembly.py` - Added `TestFilterByFrames` and `TestAssembleDatasetWithFrameSelection` test classes
- `tests/unit/training/test_pseudo_label_cli.py` - Updated help text test to expect new flags

## Decisions Made
- Frame index parsed from first 6 chars of stem, consistent with Phase 65 decision
- Runs not in selected_frames dict are kept unfiltered (Phase 65 decision)
- Filter applied after confidence/gap filtering but before max_frames cap (per plan spec)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated stale help text test assertions**
- **Found during:** Task 2 (CLI flags)
- **Issue:** `test_assemble_help_text` explicitly asserted `--temporal-step` etc. should NOT appear (written when flags were unimplemented)
- **Fix:** Flipped assertions to expect the flags now that they are wired
- **Files modified:** tests/unit/training/test_pseudo_label_cli.py
- **Verification:** All 1004 tests pass
- **Committed in:** 287a6a8 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Test was asserting old behavior; trivial fix, no scope creep.

## Issues Encountered
- Pre-existing lint error in `tests/unit/training/test_run_manager.py` (unsorted imports). Out of scope, not caused by this task.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Frame selection is now fully wired; `aquapose pseudo-label assemble` can subsample frames temporally and by curvature diversity
- Ready for real-world testing with pipeline runs

---
*Quick Task: 19-wire-frame-selection*
*Completed: 2026-03-05*
