---
phase: 65-frame-selection-and-dataset-assembly
plan: 03
subsystem: training
tags: [dataset-assembly, frame-selection, pseudo-labels, yolo]

requires:
  - phase: 65-frame-selection-and-dataset-assembly
    provides: "frame_selection module and dataset_assembly module from plans 01 and 02"
provides:
  - "selected_frames parameter on assemble_dataset for frame-level filtering"
  - "gap_reason field in pseudo_val_metadata.json sidecar entries"
  - "CLI wiring of frame selection results into assembly"
affects: [pseudo-label-pipeline, training-data-quality]

tech-stack:
  added: []
  patterns:
    - "_filter_by_frames: frame-index parsing from 6-digit stem prefix"
    - "_extract_dominant_gap_reason: Counter-based dominant reason extraction"

key-files:
  created: []
  modified:
    - src/aquapose/training/dataset_assembly.py
    - src/aquapose/training/pseudo_label_cli.py
    - tests/unit/training/test_dataset_assembly.py

key-decisions:
  - "Frame index parsed from first 6 chars of stem (int(stem[:6])), consistent with existing naming convention"
  - "Runs not in selected_frames dict are kept unfiltered (safe default for multi-run assembly)"
  - "Dominant gap_reason via collections.Counter most_common (deterministic on ties)"

patterns-established:
  - "_filter_by_frames pattern: run_id-aware frame filtering with safe passthrough for unlisted runs"

requirements-completed: [FRAME-01, FRAME-02, FRAME-03, DATA-01, DATA-02, DATA-03]

duration: 4min
completed: 2026-03-05
---

# Phase 65 Plan 03: Wire Frame Selection and Gap Reason Summary

**Frame selection filtering wired into assemble_dataset with gap_reason in pseudo-val metadata sidecar**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-05T20:38:21Z
- **Completed:** 2026-03-05T20:42:01Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- Added `selected_frames` parameter to `assemble_dataset` enabling CLI frame selection (temporal-step, diversity-max-per-bin) to actually filter the output dataset
- Added `gap_reason` field to pseudo_val_metadata.json sidecar for post-training gap-reason breakdown analysis
- Removed TODO comment and wired CLI frame selection loop to build and pass selected_frames dict
- All 961 existing tests continue to pass (backward compatible)

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Add failing tests** - `b917152` (test)
2. **Task 1 (GREEN): Implement frame filtering and gap_reason** - `52b26af` (feat)

## Files Created/Modified
- `src/aquapose/training/dataset_assembly.py` - Added `_filter_by_frames`, `_extract_dominant_gap_reason`, `selected_frames` param
- `src/aquapose/training/pseudo_label_cli.py` - Wire selected_frames dict from frame selection loop to assemble_dataset call
- `tests/unit/training/test_dataset_assembly.py` - 4 new/updated tests for frame filtering and gap_reason

## Decisions Made
- Frame index parsed from first 6 chars of stem via `int(lbl["stem"][:6])`, matching existing pseudo-label stem convention
- Runs not present in `selected_frames` dict pass through unfiltered (safe for multi-run assemblies where only some runs have frame selection)
- Dominant gap_reason extracted via `Counter.most_common(1)` from per-fish label metadata

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 65 gap closure complete; all verification gaps from 65-VERIFICATION.md resolved
- Frame selection flags (--temporal-step, --diversity-max-per-bin) now produce filtered datasets
- pseudo_val_metadata.json now includes gap_reason for DATA-03 post-training analysis

---
*Phase: 65-frame-selection-and-dataset-assembly*
*Completed: 2026-03-05*
