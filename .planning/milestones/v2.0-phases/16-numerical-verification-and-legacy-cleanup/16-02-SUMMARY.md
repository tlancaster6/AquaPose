---
phase: 16-numerical-verification-and-legacy-cleanup
plan: 02
subsystem: testing
tags: [cleanup, legacy, pipeline, diagnostics]

# Dependency graph
requires:
  - phase: 16-01
    provides: Regression test suite and generate_golden_data.py updated to PosePipeline
provides:
  - Legacy diagnostic scripts archived to scripts/legacy/ with archive marker
  - Import boundary clean: no active code imports from archived scripts
  - Test suite updated to remove v1.0 execution path references
affects: [phase-17-observers, phase-18-cli]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "scripts/legacy/ holds archived v1.0 scripts with __init__.py archive marker"
    - "Importability check pattern: test file replaced with minimal import assertion when regression suite covers canonical path"

key-files:
  created:
    - scripts/legacy/__init__.py
    - scripts/legacy/diagnose_pipeline.py
    - scripts/legacy/diagnose_tracking.py
    - scripts/legacy/diagnose_triangulation.py
    - scripts/legacy/per_camera_spline_overlay.py
  modified:
    - tests/unit/pipeline/test_stages.py
    - tests/unit/tracking/test_diagnose_tracking.py

key-decisions:
  - "test_stages.py v1.0 functional tests replaced with importability check — regression suite in tests/regression/ covers the canonical PosePipeline execution path"
  - "test_diagnose_tracking.py path updated from scripts/ to scripts/legacy/ — test still valid, just references archived location"
  - "diagnose_triangulation.py was untracked (not git mv) — copied and removed manually since git mv requires tracked files"

patterns-established:
  - "Legacy archive pattern: scripts/legacy/ with __init__.py docstring explaining supersession"

requirements-completed: [VER-04]

# Metrics
duration: 15min
completed: 2026-02-26
---

# Phase 16 Plan 02: Legacy Script Archive and Import Cleanup Summary

**Four v1.0 diagnostic scripts archived to scripts/legacy/ with import boundary scrubbed clean across src/ and tests/**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-02-26T03:44:00Z
- **Completed:** 2026-02-26T03:59:13Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Moved `diagnose_pipeline.py`, `diagnose_tracking.py`, `diagnose_triangulation.py`, `per_camera_spline_overlay.py` from `scripts/` to `scripts/legacy/` (git mv preserves history for tracked files)
- Created `scripts/legacy/__init__.py` with archival docstring explaining these are preserved-for-reference v1.0 scripts superseded by PosePipeline
- Replaced `test_stages.py` v1.0 functional test suite with minimal importability check (3 tests testing `run_tracking`/`run_triangulation` directly are redundant with regression suite)
- Fixed `test_diagnose_tracking.py` broken path: `scripts/diagnose_tracking.py` -> `scripts/legacy/diagnose_tracking.py`
- All 504 fast tests pass after cleanup

## Task Commits

1. **Task 1: Move legacy scripts to scripts/legacy/ and create archive marker** - `2569ce8` (chore)
2. **Task 2: Import audit and test cleanup** - `a63301a` (feat)

**Plan metadata:** TBD (docs: complete plan)

## Files Created/Modified

- `scripts/legacy/__init__.py` - Archive marker with docstring explaining v1.0 supersession
- `scripts/legacy/diagnose_pipeline.py` - Archived (git mv from scripts/)
- `scripts/legacy/diagnose_tracking.py` - Archived (git mv from scripts/)
- `scripts/legacy/diagnose_triangulation.py` - Archived (untracked copy+delete from scripts/)
- `scripts/legacy/per_camera_spline_overlay.py` - Archived (git mv from scripts/)
- `tests/unit/pipeline/test_stages.py` - Replaced 3 v1.0 functional tests with 1 importability check
- `tests/unit/tracking/test_diagnose_tracking.py` - Updated script path to scripts/legacy/

## Decisions Made

- test_stages.py v1.0 functional tests replaced with importability check — regression suite in tests/regression/ covers the canonical PosePipeline execution path; the old `run_tracking`/`run_triangulation` tests were testing v1.0 library functions no longer on the critical path
- test_diagnose_tracking.py path updated from scripts/ to scripts/legacy/ — test still valid, just references archived location
- diagnose_triangulation.py was untracked (not git mv) — copied and removed manually since git mv requires tracked files

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed test_diagnose_tracking.py broken path after Task 1 script move**

- **Found during:** Task 2 (import audit)
- **Issue:** After moving `diagnose_tracking.py` to `scripts/legacy/` in Task 1, `test_diagnose_tracking.py` pointed to the old `scripts/diagnose_tracking.py` path — test collection error
- **Fix:** Updated `_SCRIPT_PATH` in `test_diagnose_tracking.py` to `scripts/legacy/diagnose_tracking.py`
- **Files modified:** `tests/unit/tracking/test_diagnose_tracking.py`
- **Verification:** `hatch run test` passes with 504 tests, no collection errors
- **Committed in:** a63301a (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking)
**Impact on plan:** Necessary fix to keep test suite runnable after script move. No scope creep.

## Issues Encountered

- `diagnose_triangulation.py` was listed as untracked in git status — it was a file at `scripts/diagnose_triangulation.py` that git didn't know about. Used copy+delete instead of `git mv` since the file had no git history to preserve.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Legacy script archive complete. Import boundary is clean.
- Phase 16 verification plan complete: regression suite (16-01) + legacy cleanup (16-02)
- Ready for Phase 17 (Observers) and Phase 18 (CLI) execution

---
*Phase: 16-numerical-verification-and-legacy-cleanup*
*Completed: 2026-02-26*

## Self-Check: PASSED

- scripts/legacy/__init__.py: FOUND
- scripts/legacy/diagnose_pipeline.py: FOUND
- scripts/legacy/diagnose_tracking.py: FOUND
- scripts/legacy/diagnose_triangulation.py: FOUND
- scripts/legacy/per_camera_spline_overlay.py: FOUND
- tests/unit/pipeline/test_stages.py: FOUND
- tests/unit/tracking/test_diagnose_tracking.py: FOUND
- Commit 2569ce8: FOUND
- Commit a63301a: FOUND
