---
phase: 40-diagnostic-capture
plan: "02"
subsystem: io
tags: [numpy, npz, midline, fixture, deserialization, round-trip]

requires:
  - phase: 40-diagnostic-capture/40-01
    provides: MidlineFixture dataclass, NPZ key convention, DiagnosticObserver.export_midline_fixtures

provides:
  - load_midline_fixture() function that deserializes midline_fixtures.npz into MidlineFixture
  - Version validation with clear error messages on mismatch or missing version
  - Round-trip test coverage for single/multi fish, empty fixtures, and non-uniform confidence arrays

affects:
  - phase: 41-evaluation-harness (loads fixtures without running pipeline)

tech-stack:
  added: []
  patterns:
    - "NPZ deserialization: flat slash-key iteration, grouping by (frame_idx, fish_id, camera_id), ordered frame assembly"
    - "Validation-first loading: check meta/version before parsing any midline keys"

key-files:
  created:
    - tests/unit/io/test_midline_fixture.py
  modified:
    - src/aquapose/io/midline_fixture.py
    - src/aquapose/io/__init__.py

key-decisions:
  - "Frame indices are derived from parsed midline keys (not from meta/frame_indices array) — avoids redundancy and works for empty fixtures"
  - "camera_ids in loaded MidlineFixture comes from meta/camera_ids (preserves original capture ordering)"

patterns-established:
  - "TDD round-trip: write test helpers that fire DiagnosticObserver events, export NPZ, then assert Midline2D fields match within float32 tolerance"

requirements-completed: [DIAG-02]

duration: 15min
completed: 2026-03-02
---

# Phase 40 Plan 02: Midline Fixture Loader Summary

**NPZ-to-MidlineFixture deserialization via load_midline_fixture with version validation and six round-trip tests covering single fish, multi-fish/camera, empty fixtures, and non-uniform confidence arrays**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-02T19:50:41Z
- **Completed:** 2026-03-02T20:05:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments

- Implemented `load_midline_fixture(path)` in `src/aquapose/io/midline_fixture.py`
- Version validation raises `ValueError` with clear messages for missing or wrong version
- Round-trip confirms export -> load produces identical `Midline2D` arrays within float32 tolerance
- `load_midline_fixture` and `MidlineFixture` re-exported from `aquapose.io`

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing round-trip and validation tests** - `ce66995` (test)
2. **Task 1 GREEN: load_midline_fixture implementation** - `7ec4f96` (feat)

## Files Created/Modified

- `src/aquapose/io/midline_fixture.py` - Added `load_midline_fixture()` and updated `__all__`
- `src/aquapose/io/__init__.py` - Added `load_midline_fixture` to imports and `__all__`
- `tests/unit/io/test_midline_fixture.py` - Six tests: round-trip (1 fish, multi fish/camera), error cases (missing version, wrong version), empty fixture, non-uniform confidence

## Decisions Made

- Frame indices in the returned `MidlineFixture` are derived from the midline keys themselves (not `meta/frame_indices`) — this avoids redundancy and correctly handles the empty-fixture case where no midline keys exist
- `camera_ids` are loaded from `meta/camera_ids` to preserve the original capture ordering

## Deviations from Plan

None - plan executed exactly as written. One lint fix during GREEN phase (unused `frame_indices_arr` variable removed; frame_indices derived from midline keys instead).

## Issues Encountered

- Pre-commit ruff formatter reformatted the test file and implementation file (long tuple literal) on commit — re-staged and committed successfully on second attempt. No logic changes.

## Next Phase Readiness

- `load_midline_fixture` is importable from `aquapose.io` and round-trip tested
- Phase 41 (evaluation harness) can call `load_midline_fixture` to load fixture files produced by the capture pipeline without re-running inference
- No blockers

---
*Phase: 40-diagnostic-capture*
*Completed: 2026-03-02*
