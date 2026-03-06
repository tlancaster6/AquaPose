---
phase: 39-migrate-legacy-domain-libraries-into-core-submodules
plan: "04"
subsystem: documentation
tags: [docs, docstrings, migration, cleanup]

requires:
  - phase: 39-02
    provides: all src/ consumer imports rewired to new core paths; shim files deleted
  - phase: 39-03
    provides: all test imports updated to core paths

provides:
  - GUIDEBOOK.md source layout reflects post-migration structure (core/types/ present, legacy reconstruction/segmentation/tracking removed)
  - CLAUDE.md Architecture section updated to match actual directory tree
  - No stale legacy path references in module docstrings across all moved files
  - Lint passes, 656 tests pass

affects: []

tech-stack:
  added: []
  patterns:
    - "~aquapose.core.types.detection.Detection used in docstring cross-references (not ~aquapose.segmentation.detector.Detection)"
    - "~aquapose.core.midline.crop.extract_affine_crop used in docstring cross-references (not ~aquapose.segmentation.crop.extract_affine_crop)"

key-files:
  created:
    - .planning/phases/39-migrate-legacy-domain-libraries-into-core-submodules/39-04-SUMMARY.md
  modified:
    - .planning/GUIDEBOOK.md
    - CLAUDE.md
    - src/aquapose/core/midline/stage.py
    - src/aquapose/core/midline/midline.py
    - src/aquapose/core/detection/backends/yolo_obb.py
    - src/aquapose/core/detection/backends/yolo.py
    - src/aquapose/engine/config.py
    - src/aquapose/io/midline_writer.py

key-decisions:
  - "GUIDEBOOK.md source layout section updated: legacy reconstruction/segmentation/tracking lines removed; core/types/ added; core/<stage>/ descriptions updated with actual filenames"
  - "CLAUDE.md Architecture section updated: core/ subtree now shows types/, detection/, midline/, reconstruction/, tracking/, association/ as expected post-migration structure"
  - "segmentation/ references in docstrings disambiguated: backend kind strings ('segmentation', 'pose_estimation') left as-is (they name runtime backends, not Python packages); cross-reference paths updated to canonical core locations"

patterns-established:
  - "Docstring cross-references use full canonical module path (~aquapose.core.types.detection.Detection), not legacy package path"

requirements-completed:
  - STAB-04

duration: 3min
completed: "2026-03-02"
---

# Phase 39 Plan 04: Documentation and Docstring Cleanup Summary

**GUIDEBOOK.md source layout and CLAUDE.md Architecture section updated to post-migration reality; U-Net and legacy module path references purged from all module docstrings**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-02T18:25:35Z
- **Completed:** 2026-03-02T18:29:11Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- GUIDEBOOK.md Section 4 (Source Layout) now reflects actual directory structure: `core/types/` added, `reconstruction/`, `segmentation/`, `tracking/` legacy lines removed, `core/<stage>/` descriptions updated with actual module names
- CLAUDE.md Architecture directory tree updated: `core/` now shows full subtree (`types/`, `detection/`, `midline/`, `reconstruction/`, `tracking/`, `association/`), `training/` added, legacy top-level `reconstruction/`, `segmentation/`, `tracking/` removed
- All U-Net references purged from module docstrings (`core/midline/stage.py`, `core/midline/midline.py`)
- All `~aquapose.segmentation.detector.Detection` cross-references updated to `~aquapose.core.types.detection.Detection`
- All `~aquapose.segmentation.crop.extract_affine_crop` cross-references updated to `~aquapose.core.midline.crop.extract_affine_crop`
- All `~aquapose.reconstruction.triangulation.Midline3D` cross-references updated to `~aquapose.core.types.reconstruction.Midline3D`
- Lint passes and 656 tests pass with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Update GUIDEBOOK.md and CLAUDE.md with post-migration structure** - `db53f44` (docs)
2. **Task 2: Update stale module docstrings in moved files** - `a73c66a` (docs)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `.planning/GUIDEBOOK.md` - Removed legacy reconstruction/segmentation/tracking lines; added core/types/; updated core/<stage>/ descriptions
- `CLAUDE.md` - Updated Architecture directory tree to post-migration structure
- `src/aquapose/core/midline/stage.py` - Removed U-Net reference from module docstring
- `src/aquapose/core/midline/midline.py` - Removed U-Net references from `_crop_to_frame` docstring
- `src/aquapose/core/detection/backends/yolo_obb.py` - Updated `segmentation.detector` cross-refs to `core.types.detection`
- `src/aquapose/core/detection/backends/yolo.py` - Updated module docstring and `segmentation.detector` cross-ref
- `src/aquapose/engine/config.py` - Updated `segmentation.crop` cross-ref to `core.midline.crop`
- `src/aquapose/io/midline_writer.py` - Updated `reconstruction.triangulation` cross-ref to `core.types.reconstruction`

## Decisions Made

- Backend kind string literals (`"segmentation"`, `"pose_estimation"`) in docstrings were deliberately NOT changed — they refer to runtime backend names, not Python package paths, and are correct
- `core/reconstruction/` sub-directory in CLAUDE.md was intentionally left (it's a sub-package of `core/`, not a top-level legacy directory)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Also updated engine/config.py and io/midline_writer.py stale docstring refs**
- **Found during:** Task 2 (broad sweep grep)
- **Issue:** The grep sweep found two additional files with stale `~aquapose.segmentation.crop` and `~aquapose.reconstruction.triangulation` cross-references not listed in the plan's explicit file list
- **Fix:** Updated both files to canonical core paths
- **Files modified:** `src/aquapose/engine/config.py`, `src/aquapose/io/midline_writer.py`
- **Verification:** Grep confirms 0 remaining `segmentation.detector`, `segmentation.crop`, `reconstruction.triangulation` docstring references
- **Committed in:** `a73c66a` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (expanded scope within Task 2 to cover all stale docstring refs found by grep)
**Impact on plan:** Minor scope expansion — two additional files caught by the broad grep sweep specified in the plan itself. No behavioral changes.

## Issues Encountered

None - plan executed cleanly. All changes were purely documentation/docstring updates with no behavioral impact.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 39 is now complete: all 4 plans executed
- The repository has been fully migrated: legacy `reconstruction/`, `segmentation/`, `tracking/` directories deleted; all consumers (src/, tests/) import from `core/types/` and `core/<stage>/`; documentation reflects the new structure
- STAB-04 requirement fulfilled: no stale references to legacy directories, U-Net, or no-op stubs remain

---
*Phase: 39-migrate-legacy-domain-libraries-into-core-submodules*
*Completed: 2026-03-02*
