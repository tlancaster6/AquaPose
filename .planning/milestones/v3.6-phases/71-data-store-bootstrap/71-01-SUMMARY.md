---
phase: 71-data-store-bootstrap
plan: 01
subsystem: training
tags: [sqlite, cli, temporal-split, yolo-pose, yolo-obb]

requires:
  - phase: 70-metrics-comparison-infrastructure
    provides: evaluation infrastructure for model comparison
provides:
  - temporal split for COCO-to-YOLO conversion
  - val tagging on import from val/ subdirectories
  - tagged assemble mode for deterministic val sets
  - exclusion reason tracking with audit trail
  - updated training defaults (pose mosaic=0.1, imgsz=128, rect=True; obb mosaic=0.3)
affects: [71-data-store-bootstrap plan 02, training workflows]

tech-stack:
  added: []
  patterns: [split_mode parameter pattern for extensible splitting strategies]

key-files:
  created: []
  modified:
    - src/aquapose/training/coco_convert.py
    - src/aquapose/training/store.py
    - src/aquapose/training/data_cli.py
    - src/aquapose/training/yolo_pose.py
    - src/aquapose/training/cli.py
    - src/aquapose/training/__init__.py
    - tests/unit/training/test_store.py
    - tests/unit/training/test_data_cli.py

key-decisions:
  - "Temporal split puts last N% of sorted frame indices into val (chronological, not random)"
  - "Reason tags persist after include() as audit trail (only 'excluded' is removed)"
  - "Reserved tags ('excluded', 'val', 'augmented') excluded from reason breakdown in status"

patterns-established:
  - "split_mode parameter: extensible split strategy pattern used in both convert and assemble"
  - "val_candidates_tag: tag-based filtering for val-eligible samples in random mode"

requirements-completed: [BOOT-01, BOOT-04, BOOT-05]

duration: 9min
completed: 2026-03-07
---

# Phase 71 Plan 01: Data Store Bootstrap Features Summary

**Temporal split, val tagging, tagged assemble, exclusion reasons, and training defaults for the data management CLI**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-07T14:16:07Z
- **Completed:** 2026-03-07T14:25:00Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Temporal split groups images by frame index for train/val splitting (no data leakage across cameras)
- Val tagging on import automatically tags samples from val/ subdirectories
- Tagged assemble mode uses val tags for deterministic, user-controlled val sets
- Exclusion reasons add audit trail tags alongside "excluded" (preserved after include)
- Training defaults updated: pose (mosaic=0.1, imgsz=128, rect=True), OBB (mosaic=0.3)
- All 1104 tests pass including 14 new tests; backward-compatible defaults everywhere

## Task Commits

Each task was committed atomically:

1. **Task 1: Temporal split + val tagging + tagged assemble** - `31d5376` (feat)
2. **Task 2: Exclusion reasons + training defaults** - `458beaf` (feat)

_Both tasks used TDD: tests written first (RED), then implementation (GREEN)._

## Files Created/Modified
- `src/aquapose/training/coco_convert.py` - Added parse_frame_index, temporal_split, split_mode param
- `src/aquapose/training/store.py` - Added split_mode, val_candidates_tag to assemble; reason to exclude
- `src/aquapose/training/data_cli.py` - Added --split-mode, --val-candidates, --reason options; val tagging; reason breakdown in status
- `src/aquapose/training/yolo_pose.py` - Added rect parameter
- `src/aquapose/training/cli.py` - Updated defaults (mosaic, imgsz), added --rect/--no-rect
- `src/aquapose/training/__init__.py` - Exported parse_frame_index, temporal_split
- `tests/unit/training/test_store.py` - Added 6 new tests for tagged split, val candidates, exclusion reasons
- `tests/unit/training/test_data_cli.py` - Added 8 new tests for CLI integration

## Decisions Made
- Temporal split uses chronological ordering (last N% frames as val) rather than random to prevent data leakage
- Reason tags are audit-only: include() removes "excluded" but keeps reason tags for traceability
- Reserved tags ("excluded", "val", "augmented") are excluded from reason breakdown in status output

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All code features for Plan 02 (end-to-end bootstrap workflow) are in place
- Backward-compatible defaults ensure existing workflows are unaffected

---
*Phase: 71-data-store-bootstrap*
*Completed: 2026-03-07*
