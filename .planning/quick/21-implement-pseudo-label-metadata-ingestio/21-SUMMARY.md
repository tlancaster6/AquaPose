---
phase: 21-implement-pseudo-label-metadata-ingestion
plan: 01
subsystem: training
tags: [pseudo-labels, curvature, metadata, sidecar, cli]

requires:
  - phase: 65-pseudo-label-generation
    provides: pseudo-label pipeline with confidence.json sidecar
provides:
  - automatic confidence.json sidecar ingestion at data import
  - 2D curvature computation at import time for pose store
  - 3D curvature in pseudo-label confidence.json entries
  - compute_curvature relocated to pseudo_labels.py
affects: [training, pseudo-labels, data-import]

tech-stack:
  added: []
  patterns:
    - sidecar auto-detection at import time
    - per-sample metadata merging (sidecar + curvature + CLI override)

key-files:
  created: []
  modified:
    - src/aquapose/training/pseudo_labels.py
    - src/aquapose/training/pseudo_label_cli.py
    - src/aquapose/training/data_cli.py
    - src/aquapose/training/__init__.py
    - src/aquapose/training/run_manager.py

key-decisions:
  - "compute_curvature docstring generalized to (N, D) for 2D and 3D use"
  - "Sidecar metadata flattened from labels[0] at import (confidence, gap_reason, n_source_cameras, raw_metrics, source)"
  - "--metadata-json overrides sidecar values (priority: CLI > sidecar > computed)"

requirements-completed: [QUICK-21]

duration: 3min
completed: 2026-03-06
---

# Quick Task 21: Pseudo-Label Metadata Ingestion Summary

**Auto-read confidence.json sidecar at import, compute 2D curvature for pose store, add 3D curvature to pseudo-label sidecar, and clean up dead frame_selection module**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-06T20:55:16Z
- **Completed:** 2026-03-06T20:58:23Z
- **Tasks:** 2
- **Files modified:** 5 (+ 2 deleted)

## Accomplishments
- Relocated compute_curvature from frame_selection.py to pseudo_labels.py with generalized (N, D) docstring
- Deleted dead frame_selection.py and test_frame_selection.py with no remaining references
- Data import auto-detects confidence.json sidecar and merges per-image metadata into samples
- Data import computes 2D curvature from visible keypoints for pose store samples
- Pseudo-label generate writes curvature_3d field into confidence.json for both consensus and gap labels
- Fixed run_manager next-steps to use --project instead of --config

## Task Commits

1. **Task 1: Move compute_curvature, delete frame_selection, fix run_manager** - `07750a0` (refactor)
2. **Task 2: Auto-read confidence.json sidecar and compute 2D curvature at import** - `24bd82f` (feat)

## Files Created/Modified
- `src/aquapose/training/pseudo_labels.py` - Added compute_curvature function
- `src/aquapose/training/data_cli.py` - Sidecar auto-detection and 2D curvature at import
- `src/aquapose/training/pseudo_label_cli.py` - Added curvature_3d to confidence entries, added compute_curvature import
- `src/aquapose/training/__init__.py` - Removed frame_selection imports, added compute_curvature from pseudo_labels
- `src/aquapose/training/run_manager.py` - Fixed --config to --project in next-steps output
- `src/aquapose/training/frame_selection.py` - Deleted
- `tests/unit/training/test_frame_selection.py` - Deleted

## Decisions Made
- compute_curvature docstring generalized from (7, 3) to (N, D) since it now serves both 2D keypoints and 3D control points
- Sidecar metadata flattened from labels[0] keys at import time
- --metadata-json CLI flag still works as override on top of sidecar data

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

---
*Quick Task: 21-implement-pseudo-label-metadata-ingestion*
*Completed: 2026-03-06*
