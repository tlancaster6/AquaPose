---
phase: 94-dead-code-removal
plan: 01
subsystem: reconstruction
tags: [dlt, triangulation, dead-code-removal, refactor]

requires:
  - phase: 93-config-plumbing
    provides: n_sample_points config plumbing; DltBackend fully wired to vectorized path

provides:
  - Dead scalar _triangulate_body_point() and _tri_rays() methods removed from DltBackend
  - _MIN_RAY_ANGLE_DEG and _COS_MIN_RAY_ANGLE constants removed
  - TestVectorizedEquivalence class removed from test suite
  - dlt.py is now vectorized-only with no dead code paths

affects: [reconstruction, dlt-backend]

tech-stack:
  added: []
  patterns:
    - "Vectorized-only triangulation: DltBackend has a single code path via _triangulate_fish_vectorized"

key-files:
  created: []
  modified:
    - src/aquapose/core/reconstruction/backends/dlt.py
    - tests/unit/core/reconstruction/test_dlt_backend.py

key-decisions:
  - "Removed unused triangulate_rays import alongside _tri_rays deletion (it was only called there)"
  - "Removed math import alongside constants deletion (only used by _MIN_RAY_ANGLE_DEG/_COS_MIN_RAY_ANGLE)"
  - "Updated TestModuleConstants to only test the three remaining module constants"

patterns-established:
  - "DltBackend triangulation is vectorized-only: single call to _triangulate_fish_vectorized per fish"

requirements-completed: [CLEAN-01, CLEAN-02]

duration: 8min
completed: 2026-03-13
---

# Phase 94 Plan 01: Dead Scalar Path Removal Summary

**Deleted ~135 lines of dead scalar triangulation code (_triangulate_body_point, _tri_rays, two constants) from dlt.py and purged equivalence tests that tested the removed code**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-13T20:20:00Z
- **Completed:** 2026-03-13T20:28:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Removed `_triangulate_body_point()` (136 lines) and `_tri_rays()` (35 lines) from `dlt.py`
- Removed `_MIN_RAY_ANGLE_DEG` / `_COS_MIN_RAY_ANGLE` constants and their docstrings
- Removed also-dead `triangulate_rays` import and `math` import (both only used by deleted code)
- Updated `_triangulate_fish_vectorized` docstring to no longer reference the scalar fallback
- Deleted `TestVectorizedEquivalence` class (6 tests) from `test_dlt_backend.py`
- Updated `TestModuleConstants` to remove assertions on the two deleted constants
- All 1198 unit tests pass; typecheck reports 0 errors

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete scalar fallback code and constants from dlt.py** - `8cb85cb` (refactor)
2. **Task 2: Remove scalar-equivalence tests and verify test suite passes** - `1e57e1b` (refactor)

## Files Created/Modified
- `src/aquapose/core/reconstruction/backends/dlt.py` - Removed dead scalar path, constants, and unused imports
- `tests/unit/core/reconstruction/test_dlt_backend.py` - Removed equivalence test class and stale constant assertions

## Decisions Made
- Removed `triangulate_rays` import alongside `_tri_rays` deletion — it was the only call site, making it an unused import (Rule 1 auto-fix)
- Removed `math` import alongside constant deletion — it was only used by the now-deleted `_COS_MIN_RAY_ANGLE` expression
- Kept `TestModuleConstants` class but stripped the two assertions for deleted constants rather than deleting the whole class

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed unused triangulate_rays import**
- **Found during:** Task 1 (Delete scalar fallback code and constants from dlt.py)
- **Issue:** `triangulate_rays` was imported from `aquapose.calibration.projection` but only called by `_tri_rays`, which was deleted
- **Fix:** Removed the import line
- **Files modified:** `src/aquapose/core/reconstruction/backends/dlt.py`
- **Verification:** ruff lint passes; parse OK
- **Committed in:** `8cb85cb` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 unused import)
**Impact on plan:** Necessary cleanup — leftover import would have caused ruff lint failure in pre-commit. No scope creep.

## Issues Encountered
- Pre-commit ruff hooks auto-fixed formatting on both commits (extra blank line and trailing whitespace). Re-staged and recommitted in both cases.

## Next Phase Readiness
- Dead code removal complete; dlt.py is clean with a single vectorized triangulation path
- Ready for remaining 94-dead-code-removal plans (if any)

---
*Phase: 94-dead-code-removal*
*Completed: 2026-03-13*
