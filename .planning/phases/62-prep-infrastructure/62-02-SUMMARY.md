---
phase: 62-prep-infrastructure
plan: 02
subsystem: training, association, engine
tags: [cli, luts, fail-fast, pipeline-validation]

requires:
  - phase: 23
    provides: LUT generation and load/save functions
  - phase: 25
    provides: AssociationStage with lazy LUT generation
provides:
  - generate-luts CLI command with --config and --force flags
  - AssociationStage fail-fast on missing LUTs (no lazy generation)
  - Early LUT existence check in build_stages()
  - Autouse conftest fixtures for LUT mocking in engine and reconstruction tests
affects: [63, 64]

tech-stack:
  added: []
  patterns: [_LutConfigFromDict to satisfy LutConfigLike protocol without engine import, nested _check_luts_if_needed in build_stages]

key-files:
  created:
    - tests/unit/test_generate_luts_cli.py
    - tests/unit/engine/conftest.py
    - tests/unit/core/reconstruction/conftest.py
  modified:
    - src/aquapose/training/prep.py
    - src/aquapose/core/association/stage.py
    - src/aquapose/engine/pipeline.py

key-decisions:
  - "_LutConfigFromDict class avoids training->engine import boundary violation"
  - "LUT check runs after _truncate() so stop_after before association skips the check"
  - "Autouse conftest fixtures mock LUT loading for all engine and reconstruction tests"

requirements-completed: [PREP-03, PREP-04]

duration: ~15min
completed: 2026-03-05
---

# Plan 62-02: Add generate-luts CLI and Remove Lazy Generation

**New `aquapose prep generate-luts` CLI command replaces lazy LUT generation; pipeline fails fast at construction time if LUTs missing**

## Performance

- **Duration:** ~15 min
- **Tasks:** 1 (multi-part)
- **Files modified:** 3 production, 3 test

## Accomplishments
- Added `aquapose prep generate-luts --config <yaml> [--force]` CLI command
- _LutConfigFromDict satisfies LutConfigLike protocol without importing engine config (import boundary enforced)
- Removed ~30 lines of lazy LUT generation from AssociationStage.run(), replaced with fail-fast FileNotFoundError
- Added _check_luts_if_needed() in build_stages() that conditionally checks LUT existence only when AssociationStage is in the truncated stage list
- Created autouse conftest fixtures in tests/unit/engine/ and tests/unit/core/reconstruction/ to mock LUT loading

## Task Commits

1. **Task 1: Add generate-luts CLI and remove lazy generation** - `aa72a54` (feat)

## Decisions Made
- Plain yaml.safe_load + _LutConfigFromDict instead of load_config() to avoid import boundary violation
- LUT check after _truncate() respects stop_after (detection/tracking-only runs skip LUT check)
- Module-level import of load_forward_luts/load_inverse_luts in pipeline.py (not lazy import)

## Deviations from Plan
- Plan suggested per-camera progress output during generation; implemented batch-level progress instead (simpler, uses existing generate_forward_luts batch function)
- Plan suggested conditional import in pipeline.py; used module-level import instead (cleaner)

## Issues Encountered
- Import boundary AST check caught even deferred imports of engine.config; resolved with _LutConfigFromDict
- Existing engine/reconstruction tests failed due to new LUT check in build_stages; resolved with autouse conftest fixtures
- Ruff format auto-fixed files on first commit attempt

## User Setup Required
None

## Next Phase Readiness
- Phase 63 and 64 can assume LUTs are pre-generated and available
- Phase 64 (gap detection) can use generate-luts CLI independently of the pipeline

---
*Phase: 62-prep-infrastructure*
*Completed: 2026-03-05*
