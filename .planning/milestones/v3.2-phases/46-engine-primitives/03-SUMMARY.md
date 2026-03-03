---
phase: 46-engine-primitives
plan: "03"
subsystem: engine
tags: [cli, pickle, resume, cache, click, pipeline]

# Dependency graph
requires:
  - phase: 46-engine-primitives
    plan: "01"
    provides: "StaleCacheError, load_stage_cache, context_fingerprint in core/context.py"
  - phase: 46-engine-primitives
    plan: "02"
    provides: "DiagnosticObserver cache writing, PosePipeline.run(initial_context=...)"
provides:
  - "--resume-from CLI flag on aquapose run that loads a stage cache and passes it as initial_context"
  - "StaleCacheError and FileNotFoundError converted to click.ClickException for user-friendly error messages"
  - "Integration tests covering cache load, error paths, and end-to-end DiagnosticObserver round-trip"
affects: [phase-47, phase-48, phase-49, cli, evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Plan 02 prerequisite work completed as blocking deviation (Rule 3) before Plan 03"
    - "CLI --resume-from uses click.Path(exists=True) for automatic missing-file validation"
    - "load_stage_cache() imported inline inside CLI to avoid top-level engine/core import coupling"
    - "StaleCacheError and FileNotFoundError both raise click.ClickException for consistent UX"

key-files:
  created:
    - tests/unit/engine/test_resume_cli.py
    - tests/unit/engine/test_stage_cache_write.py
    - tests/unit/engine/test_stage_skip.py
  modified:
    - src/aquapose/cli.py
    - src/aquapose/engine/pipeline.py
    - src/aquapose/engine/diagnostic_observer.py

key-decisions:
  - "Inline import of load_stage_cache/StaleCacheError inside CLI run() avoids top-level import coupling"
  - "FileNotFoundError raised with 'from None' to avoid chaining in B904 lint rule"
  - "Test for corrupt-file error path uses mock_pipeline fixture to bypass config validation"
  - "Plan 02 blocking dependency auto-executed: initial_context+stage-skip added to pipeline.run()"

patterns-established:
  - "CLI option ordering: --resume-from placed after --verbose, before positional args"
  - "Stage cache test pattern: DiagnosticObserver fires PipelineStart then StageComplete to write cache"

requirements-completed: [INFRA-01, INFRA-02, INFRA-03, INFRA-04]

# Metrics
duration: 8min
completed: 2026-03-03
---

# Phase 46 Plan 03: CLI --resume-from Flag and Integration Test Summary

**--resume-from CLI flag added to aquapose run, with StaleCacheError-to-ClickException conversion and 6 integration tests covering load, error paths, and DiagnosticObserver end-to-end round-trip**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-03T18:25:18Z
- **Completed:** 2026-03-03T18:33:00Z
- **Tasks:** 2 (plan 03) + 3 (plan 02 blocking dependency)
- **Files modified:** 6

## Accomplishments
- Added `--resume-from PATH` option to `aquapose run` using `click.Path(exists=True)`
- CLI loads stage cache via `load_stage_cache()` and passes result as `initial_context` to `pipeline.run()`
- Both `StaleCacheError` and `FileNotFoundError` converted to `click.ClickException` for user-friendly messages
- Each resumed run gets its own `run_id` — not inherited from cache
- Also completed missing Plan 02 work (blocking dependency): `PosePipeline.run(initial_context=...)`, stage-skip logic with `_STAGE_OUTPUT_FIELDS` lookup, `CarryForward` extraction/injection, and DiagnosticObserver cache writing with envelope format

## Task Commits

1. **Plan 02 (blocking dependency): initial_context, stage-skip, cache write tests** - `3d732b6` (feat)
2. **Task 46.3.1: --resume-from flag on aquapose run CLI** - `a9e3e1b` (feat)
3. **Task 46.3.2: Integration tests for --resume-from round-trip** - `a9e3e1b` (feat, same commit)

## Files Created/Modified
- `src/aquapose/cli.py` - Added --resume-from option, load_stage_cache call, StaleCacheError/FileNotFoundError handling
- `src/aquapose/engine/pipeline.py` - Added initial_context parameter, stage-skip logic, carry extraction/injection
- `src/aquapose/engine/diagnostic_observer.py` - Added per-stage pickle cache writing with envelope format
- `tests/unit/engine/test_resume_cli.py` - 6 tests: load round-trip, corrupt-file error, nonexistent-file error, e2e DiagnosticObserver round-trip, importability, invalid-envelope error
- `tests/unit/engine/test_stage_cache_write.py` - 7 tests for DiagnosticObserver cache writing behavior
- `tests/unit/engine/test_stage_skip.py` - 5 tests for PosePipeline stage-skip logic

## Decisions Made
- Inline import of `load_stage_cache`/`StaleCacheError` inside `run()` to avoid coupling CLI module to core at import time
- `FileNotFoundError` raised `from None` to satisfy B904 lint rule (suppress chaining from `except` clause)
- CLI test for corrupt-file error path uses mock patches on `load_config`, `build_stages`, and `PosePipeline` to bypass config validation and reach the `--resume-from` code path
- Stub stages in `test_stage_skip.py` use class names matching `_STAGE_OUTPUT_FIELDS` keys (`DetectionStage`) so skip logic triggers correctly

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Executed Plan 02 as prerequisite before Plan 03**
- **Found during:** Pre-execution analysis (STATE.md showed only Plan 01 complete)
- **Issue:** `PosePipeline.run()` lacked `initial_context` parameter; `test_stage_cache_write.py` and `test_stage_skip.py` were missing; Plan 03 requires both
- **Fix:** Implemented Plan 02 tasks: added `initial_context` + stage-skip logic to `pipeline.run()`, fixed pre-existing failing tests in `test_stage_skip.py` (stub class names must match `_STAGE_OUTPUT_FIELDS`), confirmed `DiagnosticObserver` already had cache writing
- **Files modified:** `src/aquapose/engine/pipeline.py`, `tests/unit/engine/test_stage_cache_write.py`, `tests/unit/engine/test_stage_skip.py`
- **Verification:** 709 tests pass after fix; 2 previously failing skip tests now pass
- **Committed in:** `3d732b6`

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking dependency)
**Impact on plan:** Blocking prerequisite work was necessary. Plan 03 is now fully operational.

## Issues Encountered
- `test_stage_skip.py` had `StubDetection` class that didn't match `_STAGE_OUTPUT_FIELDS` keys — fixed by renaming to `DetectionStage` so pipeline skip logic recognizes it
- `test_resume_from_stale_cache_gives_click_exception` initially failed because config validation (`n_animals required`) blocked execution before reaching the `--resume-from` code path — fixed by mocking `load_config`/`build_stages`/`PosePipeline`

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 46 complete: all 3 plans executed — StaleCacheError/load_stage_cache (01), stage-skip/cache-write (02), CLI --resume-from (03)
- Phase 47 (Evaluation Primitives) can now proceed — per-stage cache files are written in diagnostic mode, loadable via CLI --resume-from or programmatically via `load_stage_cache()`

## Self-Check: PASSED

All created files verified to exist:
- `src/aquapose/cli.py` - FOUND
- `tests/unit/engine/test_resume_cli.py` - FOUND
- `tests/unit/engine/test_stage_cache_write.py` - FOUND
- `tests/unit/engine/test_stage_skip.py` - FOUND
- `.planning/phases/46-engine-primitives/03-SUMMARY.md` - FOUND

Commits verified to exist:
- `3d732b6` (feat(46-02): add initial_context, stage-skip logic, and cache write tests) - FOUND
- `a9e3e1b` (feat(46-03): add --resume-from CLI flag with integration tests) - FOUND

---
*Phase: 46-engine-primitives*
*Completed: 2026-03-03*
