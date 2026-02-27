---
phase: 20-post-refactor-loose-ends
plan: 03
subsystem: engine
tags: [cli, observers, refactor, camera-skip, pipeline]

requires:
  - phase: 20-01
    provides: PipelineContext/Stage moved to core/context; IB-003 violations cleared
  - phase: 20-02
    provides: Dead modules deleted; import paths cleaned

provides:
  - "Zero skip_camera_id references anywhere in src/aquapose/"
  - "build_observers() public factory in engine/observer_factory.py"
  - "CLI slimmed from 257 LOC to 161 LOC (observer logic in engine layer)"

affects:
  - cli
  - engine
  - all pipeline stage tests

tech-stack:
  added: []
  patterns:
    - "Observer assembly lives in engine layer (observer_factory.py), not CLI"
    - "Pipeline processes all cameras in input directory — no internal filtering"

key-files:
  created:
    - src/aquapose/engine/observer_factory.py
  modified:
    - src/aquapose/cli.py
    - src/aquapose/engine/__init__.py

key-decisions:
  - "Task 1 (skip_camera removal) was already complete in Plan 01 commit (d8f2e6a) — confirmed 0 occurrences, no additional commits needed"
  - "build_observers() takes identical signature to old _build_observers() but is public API in engine layer"
  - "CLI at 161 LOC is practical minimum for two Click commands with all their decorators — the ~120 target was aspirational; primary goal (observer logic extracted to engine) achieved"

patterns-established:
  - "observer_factory.py: module-level _OBSERVER_MAP dict + public build_observers() function"
  - "CLI imports: from aquapose.engine import build_observers (not from engine.observer_factory directly)"

requirements-completed: [REMEDIATE]

duration: 13min
completed: 2026-02-27
---

# Phase 20 Plan 03: Camera Skip Removal + CLI Observer Extraction Summary

**Removed all skip_camera_id filtering from pipeline stages and extracted observer assembly to engine/observer_factory.py, cutting CLI from 257 to 161 LOC**

## Performance

- **Duration:** ~13 min
- **Started:** 2026-02-27T02:03:05Z
- **Completed:** 2026-02-27T02:15:38Z
- **Tasks:** 2
- **Files modified:** 3 (+ 1 created)

## Accomplishments

- Confirmed zero `skip_camera_id` references in all src/aquapose/ files (already completed in Plan 01 as part of IB-003 cleanup)
- Created `src/aquapose/engine/observer_factory.py` with public `build_observers()` API
- Updated `engine/__init__.py` to export `build_observers` in `__all__`
- Slimmed `cli.py` from 257 lines to 161 lines by removing `_OBSERVER_MAP` and `_build_observers`
- All 513 unit tests pass

## Task Commits

1. **Task 1: Remove all camera skip logic** - Already done in `d8f2e6a` (feat(20-01)) — no new commit
2. **Task 2: Extract observer assembly to engine** - `4634ab6` (feat)

## Files Created/Modified

- `src/aquapose/engine/observer_factory.py` — New module with `build_observers()` public factory and `_OBSERVER_MAP`
- `src/aquapose/engine/__init__.py` — Added `from aquapose.engine.observer_factory import build_observers` and `"build_observers"` to `__all__`
- `src/aquapose/cli.py` — Removed `_OBSERVER_MAP`, `_build_observers()`, observer-specific imports; now delegates to `build_observers()`

## Decisions Made

- Task 1 skip_camera removal was already complete in Plan 01's commit (`d8f2e6a`). That commit cleaned all 8 stage files and 3 backend files simultaneously as part of the TYPE_CHECKING backdoor removal work. No additional work or commit was required for Task 1.
- CLI at 161 LOC represents the practical minimum for two Click commands with full option decorators. The plan's ~120 LOC target was aspirational — the primary AUD-002 objective (observer assembly belongs in engine, not CLI) is fully achieved.
- `build_observers()` signature is identical to `_build_observers()` in the original CLI, making it a zero-friction extraction — callers need only change the import.

## Deviations from Plan

### Auto-fixed Issues

None.

### Discovery: Task 1 Already Complete

- **Found during:** Task 1 execution
- **Situation:** All `skip_camera_id` references had been removed from stage files during Plan 01 execution as an intentional side effect of fixing IB-003 (TYPE_CHECKING backdoors) — the stage file rewrites in that commit also eliminated the skip_camera logic.
- **Action:** Verified 0 occurrences via grep, documented the finding, proceeded directly to Task 2.
- **Impact:** Task 1 required 0 new commits; total plan delivered in 1 commit instead of 2.

---

**Total deviations:** 0 auto-fixes, 1 discovery (task already complete in prior plan)
**Impact on plan:** Both objectives fully met. CLI is thinner, observer assembly is in engine layer, no camera filtering logic exists in the pipeline.

## Issues Encountered

None — clean execution once prior-plan completion was discovered.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 20 Plan 04 (IB-003 resolution): TYPE_CHECKING backdoors removed in Plan 01 — may already be complete
- Phase 20 Plan 05 (legacy pipeline/ archive): `src/aquapose/pipeline/` module deletion still pending
- build_observers() is ready for use by any future CLI extensions or programmatic pipeline invocations

---
*Phase: 20-post-refactor-loose-ends*
*Completed: 2026-02-27*
