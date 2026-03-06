---
phase: 64-gap-detection-and-fill-source-b
plan: 02
subsystem: training
tags: [pseudo-labels, cli, gap-fill, directory-restructure, confidence-sidecar]

requires:
  - phase: 64-gap-detection-and-fill-source-b
    plan: 01
    provides: detect_gaps(), generate_gap_fish_labels() in pseudo_labels.py

provides:
  - Refactored CLI with --consensus/--gaps flags and --min-cameras
  - Restructured output directories pseudo_labels/{consensus,gap}/{obb,pose}/
  - Per-subset dataset.yaml and confidence.json
  - Gap sidecar with gap_reason and n_source_cameras

affects: [65-dataset-assembly]

tech-stack:
  added: []
  patterns: [flag-gated-generation-paths, shared-frame-io]

key-files:
  created: []
  modified:
    - src/aquapose/training/pseudo_label_cli.py
    - tests/unit/training/test_pseudo_label_cli.py

key-decisions:
  - "Duplicated _LutConfigFromDict locally (private class, not worth extracting to shared module)"
  - "Pre-build frame-to-tracklet index for O(1) gap classification lookup"
  - "Shared frame reads between consensus and gap when both flags active"

patterns-established:
  - "Flag-gated generation paths sharing frame I/O"

requirements-completed: [GAP-01, GAP-02, GAP-03, GAP-04]

duration: 8min
completed: 2026-03-05
---

# Plan 64-02 Summary

**Refactored pseudo-label CLI with --consensus/--gaps flags and restructured output directories**

## Performance

- **Duration:** 8 min
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- Added --consensus and --gaps flags with at-least-one validation
- Added --min-cameras flag (default 3) for gap detection camera floor
- Restructured output from flat pseudo_labels/{obb,pose}/ to pseudo_labels/{consensus,gap}/{obb,pose}/
- Each subset gets its own dataset.yaml and confidence.json
- Gap confidence sidecar entries include gap_reason and n_source_cameras
- Wired detect_gaps() and generate_gap_fish_labels() into gap generation loop
- Pre-built frame-to-tracklet index for O(1) lookup during gap classification
- Frame reads shared between consensus and gap when both flags active
- 8 CLI tests covering all flag combinations, output structure, sidecar format

## Task Commits

1. **Task 1: CLI refactor with --consensus/--gaps flags** - `4008248` (feat)

## Files Created/Modified
- `src/aquapose/training/pseudo_label_cli.py` - Refactored generate command with flags, directory restructure, gap generation path
- `tests/unit/training/test_pseudo_label_cli.py` - 8 tests for CLI flag validation, output structure, gap sidecar

## Decisions Made
- Duplicated _LutConfigFromDict locally rather than importing from prep.py (private class)
- Added assert for tracks_2d not None after explicit validation check (type checker satisfaction)
- Pre-initialized all output path variables to satisfy type checker even when flags not active

## Deviations from Plan
None - plan executed as specified.

## Issues Encountered
None.

## Next Phase Readiness
- Phase 64 complete: gap detection and fill (Source B) fully implemented
- Ready for Phase 65: dataset assembly

---
*Phase: 64-gap-detection-and-fill-source-b*
*Completed: 2026-03-05*
