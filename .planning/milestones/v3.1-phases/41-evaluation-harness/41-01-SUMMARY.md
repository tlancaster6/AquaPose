---
phase: 41-evaluation-harness
plan: "01"
subsystem: io
tags: [npz, calibration, fixture, dataclass, numpy, torch]

# Dependency graph
requires:
  - phase: 40-diagnostic-capture
    provides: "MidlineFixture dataclass, NPZ v1.0 format, export_midline_fixtures, load_midline_fixture"
provides:
  - "CalibBundle frozen dataclass with per-camera K_new/R/t and shared water_z/interface_normal/n_air/n_water"
  - "NPZ v2.0 format with calib/ keys for self-contained offline evaluation"
  - "Backward-compatible v1.0 loading (calib_bundle=None)"
  - "export_midline_fixtures models parameter for v2.0 export"
affects: [41-02-evaluation-harness, 42-pipeline-integration, triangulation, reconstruction]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Versioned NPZ format with backward compat: v1.0 omits calib/, v2.0 includes calib/ keys"
    - "CalibBundle as immutable data-transfer object between fixture IO and projection models"
    - "CUDA safety pattern: .cpu().numpy() on all torch tensors before NPZ serialization"

key-files:
  created: []
  modified:
    - src/aquapose/io/midline_fixture.py
    - src/aquapose/engine/diagnostic_observer.py
    - src/aquapose/io/__init__.py
    - tests/unit/io/test_midline_fixture.py
    - tests/unit/engine/test_diagnostic_observer.py

key-decisions:
  - "NPZ_VERSION updated to 2.0; loader supports both 1.0 and 2.0 via _SUPPORTED_VERSIONS frozenset"
  - "export_midline_fixtures writes 1.0 (no calib/) when models=None, 2.0 (with calib/) when models provided"
  - "CalibBundle is a frozen dataclass — matches MidlineFixture immutability convention"
  - "Shared calibration params (water_z, n_air, n_water, interface_normal) taken from first model in dict"
  - "Per-camera calib/ keys discovered dynamically by scanning NPZ key names during load"

patterns-established:
  - "Versioned NPZ with forward-slash key convention for structured data"
  - "Optional calib_bundle field on MidlineFixture (None = v1.0 backward compat)"

requirements-completed: [EVAL-01]

# Metrics
duration: 8min
completed: 2026-03-02
---

# Phase 41 Plan 01: Calibration Bundle for Self-Contained Fixtures Summary

**CalibBundle dataclass and NPZ v2.0 format bundling per-camera K/R/t plus shared refractive params for offline evaluation without a separate calibration JSON**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-02T20:08:41Z
- **Completed:** 2026-03-02T20:17:18Z
- **Tasks:** 2 (TDD — 4 commits total: 2 RED + 2 GREEN)
- **Files modified:** 5

## Accomplishments
- Added `CalibBundle` frozen dataclass to `midline_fixture.py` with per-camera `K_new`, `R`, `t` dicts and shared `water_z`, `interface_normal`, `n_air`, `n_water` fields
- Extended `MidlineFixture` with optional `calib_bundle` field (None for v1.0, populated for v2.0)
- Updated `load_midline_fixture` to support both v1.0 (backward compat) and v2.0 (parses calib/ keys into CalibBundle)
- Extended `export_midline_fixtures` with optional `models` parameter — writes v2.0 with calib/ keys when models dict provided, v1.0 without calib/ when models is None
- All Phase 40 tests continue to pass (677 total, 0 failures)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for CalibBundle and v2.0 loading** - `2c99a75` (test)
2. **Task 1 GREEN: CalibBundle dataclass and v2.0 NPZ loader** - `e2aa926` (feat)
3. **Task 2 RED: Failing tests for export with models** - `4a90ff7` (test)
4. **Task 2 GREEN: export_midline_fixtures v2.0 with models** - `e6b3369` (feat)

**Plan metadata:** (docs commit follows)

_Note: TDD tasks have multiple commits (test RED → feat GREEN)_

## Files Created/Modified
- `src/aquapose/io/midline_fixture.py` - Added CalibBundle dataclass, NPZ v2.0 support, _parse_calib_bundle(), updated __all__ and NPZ_VERSION
- `src/aquapose/io/__init__.py` - Added CalibBundle to imports and __all__
- `src/aquapose/engine/diagnostic_observer.py` - Added models parameter to export_midline_fixtures, added _write_calib_arrays() helper
- `tests/unit/io/test_midline_fixture.py` - Added backward-compat and v2.0 CalibBundle tests
- `tests/unit/engine/test_diagnostic_observer.py` - Added v2.0 export tests with mock models, updated NPZ_VERSION assertion

## Decisions Made
- `NPZ_VERSION` updated to "2.0" as the current export default; both "1.0" and "2.0" accepted during load
- When `models=None`, exporter writes literal "1.0" (not NPZ_VERSION constant) to preserve backward compat
- `CalibBundle.camera_ids` is sorted tuple (alphabetical) derived from discovered calib/ keys
- Shared calibration params extracted from first model in dict — all rig cameras share same water surface geometry
- `_write_calib_arrays` is module-level function (not method) for testability and clarity

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated hardcoded NPZ_VERSION assertion in test_diagnostic_observer.py**
- **Found during:** Task 1 GREEN (implementing CalibBundle and updating NPZ_VERSION to "2.0")
- **Issue:** `test_midline_fixture_importable` asserted `NPZ_VERSION == "1.0"` — failed after version bump
- **Fix:** Updated assertion to `NPZ_VERSION == "2.0"`
- **Files modified:** tests/unit/engine/test_diagnostic_observer.py
- **Verification:** All tests pass
- **Committed in:** e2aa926 (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Necessary to keep existing test accurate. No scope creep.

## Issues Encountered
- After bumping `NPZ_VERSION` to "2.0", the exporter wrote "2.0" but lacked calib/ keys, causing the loader to fail on round-trip tests. Resolved by making the exporter write the literal "1.0" string when `models=None`, decoupling the export version from the `NPZ_VERSION` module constant.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CalibBundle available at `aquapose.io.CalibBundle` for Plan 02 evaluation harness
- `load_midline_fixture` returns `fixture.calib_bundle` populated from v2.0 NPZ files
- Plan 02 can reconstruct `RefractiveProjectionModel` from `CalibBundle` fields without a separate calibration JSON
- No blockers for Plan 02 evaluation harness implementation

## Self-Check: PASSED

- FOUND: src/aquapose/io/midline_fixture.py
- FOUND: src/aquapose/engine/diagnostic_observer.py
- FOUND: .planning/phases/41-evaluation-harness/41-01-SUMMARY.md
- FOUND: 2c99a75 (test RED Task 1)
- FOUND: e2aa926 (feat GREEN Task 1)
- FOUND: 4a90ff7 (test RED Task 2)
- FOUND: e6b3369 (feat GREEN Task 2)

---
*Phase: 41-evaluation-harness*
*Completed: 2026-03-02*
