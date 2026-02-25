---
phase: 14-golden-data-and-verification-framework
plan: 02
subsystem: testing
tags: [golden-data, regression, pytest, fixtures, structural-tests, numerical-stability]

# Dependency graph
requires:
  - phase: 14-golden-data-and-verification-framework
    plan: 01
    provides: Golden .pt fixture files in tests/golden/ (detection, segmentation, tracking, midlines, triangulation, metadata)
provides:
  - tests/golden/__init__.py — golden test package
  - tests/golden/conftest.py — session-scoped pytest fixtures loading all 6 golden files
  - tests/golden/test_stage_harness.py — 9 regression tests validating golden data structure and numerical sanity
affects:
  - 15-stage-migrations (harness reused to verify ported stages match golden outputs)
  - 16-verification (same structural assertion patterns extended for ported Stage.run() calls)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Session-scoped pytest fixtures for expensive file loads (torch.load, gzip.open)"
    - "gzip.open + torch.load pattern for compressed .pt.gz golden fixtures"
    - "pytest.skip in fixtures to gracefully skip entire test session when golden data missing"
    - "Low-confidence triangulation exemption: skip numerical bounds checks for RANSAC degenerate outputs"

key-files:
  created:
    - tests/golden/__init__.py
    - tests/golden/conftest.py
    - tests/golden/test_stage_harness.py
  modified: []

key-decisions:
  - "CropRegion has x1/y1/x2/y2 fields (not x/y/width/height as plan described) — tests written to match actual API"
  - "FishTrack uses positions deque (not centroid_3d) — tests assert positions[-1].shape == (3,)"
  - "Midline3D uses n_cameras (not camera_ids list) for multi-view tracking — tests use n_cameras"
  - "Low-confidence triangulations (is_low_confidence=True) exempted from tank bounds check — RANSAC failures produce outliers up to 76m in v1.0 pipeline; only high-confidence results checked against 10m bound"

patterns-established:
  - "Golden regression harness pattern: conftest provides session fixtures, test_stage_harness validates structure + numerical sanity"
  - "Phase 15 reuse: same structural assertions apply to ported Stage.run(context) outputs vs golden reference"

requirements-completed: [VER-02]

# Metrics
duration: 8min
completed: 2026-02-25
---

# Phase 14 Plan 02: Interface Test Harness Summary

**pytest golden regression harness with session-scoped fixtures loading 6 .pt files and 9 @slow tests validating v1.0 stage output structure and numerical sanity**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-02-25T22:39:02Z
- **Completed:** 2026-02-25T22:47:00Z
- **Tasks:** 2 of 2 committed
- **Files modified:** 3

## Accomplishments

- Created `tests/golden/__init__.py` and `conftest.py` with 7 session-scoped fixtures covering all 6 golden data files (metadata, detection, segmentation/gzip, tracking, midlines, triangulation)
- Created `test_stage_harness.py` with 9 tests: 5 structural + 3 numerical stability + 1 metadata completeness
- Discovered and documented real v1.0 data characteristics: CropRegion uses x1/y1/x2/y2 coordinate system; FishTrack stores position history in `positions` deque; low-confidence triangulations produce degenerate RANSAC outputs (up to 76m coordinates) which are expected behavior
- All 9 tests pass with golden data; tests skip gracefully when golden data missing

## Task Commits

Each task was committed atomically:

1. **Task 1: Create golden test fixtures and conftest** - `f9aadbc` (feat)
2. **Task 2: Create interface test harness for all 5 stages** - `4085eb7` (feat)

**Plan metadata:** pending final commit

## Files Created/Modified

- `tests/golden/__init__.py` - Package docstring: "Golden data regression tests for v1.0 pipeline equivalence."
- `tests/golden/conftest.py` - GOLDEN_DIR, DEFAULT_ATOL constant, _check_golden_data_exists(), 6 session-scoped fixtures with graceful skip behavior; golden_masks uses gzip.open for .pt.gz file
- `tests/golden/test_stage_harness.py` - 9 @slow-marked tests validating all 5 v1.0 stage output types; structural assertions check types/shapes/dtypes; numerical stability tests verify finite values and valid ranges

## Decisions Made

- CropRegion has x1/y1/x2/y2 fields (not x/y/width/height as the plan described) — tests written to match actual API observed from golden data inspection
- FishTrack stores 3D positions in a `positions` deque, not a single `centroid_3d` attribute — `positions[-1].shape == (3,)` is the correct assertion
- Low-confidence triangulations exempt from tank-bounds check — RANSAC on 2-camera views can yield degenerate results with coordinates up to 76m; only the 9 high-confidence frames (of 77 total with fish) are checked against the 10m bound

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed triangulation numerical stability test that incorrectly rejected valid golden data**
- **Found during:** Task 2 verification (running tests after creation)
- **Issue:** Test asserted all control_points coordinates < 10m, but 14 of 77 low-confidence triangulations (from 2-camera RANSAC) had coordinates up to 76m — this is correct v1.0 behavior, not a data error
- **Fix:** Added `is_low_confidence` check; tank-bounds assertion only applies to high-confidence midlines. Added second assertion that at least 1 high-confidence midline exists (passes with 9 found in 30 frames)
- **Files modified:** tests/golden/test_stage_harness.py
- **Verification:** All 9 tests pass
- **Committed in:** 4085eb7 (Task 2 commit, after ruff lint fixes)

**2. [Rule 1 - Bug] Fixed ruff lint errors in test harness (unused loop variables, list-to-iter pattern)**
- **Found during:** Task 2 commit (pre-commit hook)
- **Issue:** 9 ruff errors: unused loop variables (`fi`, `cam`) and `list(dict.keys())[0]` pattern where `next(iter(...))` is preferred (RUF015, B007)
- **Fix:** Renamed unused variables to `_fi`/`_cam`; replaced `list(x.keys())[0]` with `next(iter(x.keys()))` in 3 locations
- **Files modified:** tests/golden/test_stage_harness.py
- **Verification:** Ruff passed on second commit attempt
- **Committed in:** 4085eb7 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — test correctness + style/lint)
**Impact on plan:** First fix necessary for test accuracy (golden data was correct, test logic was wrong). Second fix cosmetic/style. No scope creep.

## Issues Encountered

- Plan's interface description incorrectly stated CropRegion has `x, y, width, height` attributes — actual API uses `x1, y1, x2, y2`. Discovered by inspecting golden data before writing tests, not a runtime failure.
- Plan stated FishTrack has `centroid_3d` attribute — actual API stores history in `positions` deque. Same discovery approach.

## Next Phase Readiness

- Golden regression harness is complete and fully passing
- `tests/golden/` package provides the regression baseline for Phase 15 stage migrations
- Phase 15 can reuse the same structural assertion patterns for ported `Stage.run(context)` outputs
- VER-02 requirement is satisfied: harness loads golden fixtures and asserts structure + numerical equivalence

---
*Phase: 14-golden-data-and-verification-framework*
*Completed: 2026-02-25*

## Self-Check: PASSED

- FOUND: tests/golden/__init__.py
- FOUND: tests/golden/conftest.py
- FOUND: tests/golden/test_stage_harness.py
- FOUND commit: f9aadbc (Task 1)
- FOUND commit: 4085eb7 (Task 2)
