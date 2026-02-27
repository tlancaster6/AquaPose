---
phase: 20-post-refactor-loose-ends
plan: 01
subsystem: core
tags: [import-boundary, refactor, pipeline, architecture]

# Dependency graph
requires:
  - phase: 19-alpha-refactor-audit
    provides: IB-003 audit findings identifying 7 TYPE_CHECKING backdoors in core/ stage files
provides:
  - PipelineContext and Stage Protocol relocated to core/context.py
  - Zero IB-003 violations in import boundary checker
  - Direct (non-TYPE_CHECKING) imports of PipelineContext in all 5 core stage files
  - engine/__init__.py re-exports PipelineContext/Stage for backward compat
affects: [21-retrospective-prospective, future-phases-using-pipeline-context]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PipelineContext and Stage Protocol belong in core/ not engine/ — pure data contracts"
    - "IB-003 allowlist in import_boundary_checker.py for documented downward config dependencies"
    - "SR-002 rule updated to allow aquapose.core.context imports from engine/ files"

key-files:
  created:
    - src/aquapose/core/context.py
  modified:
    - src/aquapose/core/__init__.py
    - src/aquapose/engine/__init__.py
    - src/aquapose/engine/pipeline.py
    - src/aquapose/core/detection/stage.py
    - src/aquapose/core/midline/stage.py
    - src/aquapose/core/association/stage.py
    - src/aquapose/core/tracking/stage.py
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/core/synthetic.py
    - tools/import_boundary_checker.py
    - tests/unit/engine/test_stages.py
    - tests/unit/engine/test_pipeline.py
    - tests/unit/engine/test_diagnostic_observer.py
    - tests/unit/core/tracking/test_tracking_stage.py
    - tests/unit/core/test_synthetic.py
    - tests/unit/core/reconstruction/test_reconstruction_stage.py
    - tests/unit/core/midline/test_midline_stage.py
    - tests/unit/core/association/test_association_stage.py
    - tests/unit/core/detection/test_detection_stage.py

key-decisions:
  - "PipelineContext and Stage moved to core/context.py — pure data contracts belong in core, not engine"
  - "engine/stages.py deleted with no backward-compat shim — clean break, all consumers updated"
  - "SyntheticConfig TYPE_CHECKING import in core/synthetic.py is permitted (config flows downward, engine->core)"
  - "IB-003 allowlist added to import_boundary_checker.py for the SyntheticConfig exception"
  - "SR-002 rule updated to exclude aquapose.core.context — engine files importing it is correct design"

patterns-established:
  - "Pattern: Core data contracts (Stage Protocol, PipelineContext) live in core/ not engine/"
  - "Pattern: Import boundary allowlists document intentional exceptions with rationale"

requirements-completed: [REMEDIATE]

# Metrics
duration: 25min
completed: 2026-02-27
---

# Phase 20 Plan 01: Fix IB-003 Import Boundary Violations Summary

**PipelineContext and Stage Protocol relocated from engine/stages.py to core/context.py, eliminating all 7 IB-003 TYPE_CHECKING backdoors in core/ stage files**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-02-27T01:45:00Z
- **Completed:** 2026-02-27T02:12:08Z
- **Tasks:** 2
- **Files modified:** 24 (source + tests + checker)

## Accomplishments
- Created `core/context.py` housing Stage Protocol and PipelineContext dataclass
- Deleted `engine/stages.py` entirely — no backward-compat shim needed
- Removed all 7 TYPE_CHECKING backdoors from 5 core stage files and core/synthetic.py
- Updated import_boundary_checker.py: IB-003 allowlist for SyntheticConfig, SR-002 exclusion for core.context
- Updated all 9 affected test files to import from aquapose.core.context
- Import boundary checker: 0 violations (was 7 IB-003 errors before this plan)
- Unit test suite: 513 tests pass, 0 failures

## Task Commits

Each task was committed atomically:

1. **Task 1: Create core/context.py and update all source imports** - `d8f2e6a` (feat)
2. **Task 1: Ruff format auto-fixes** - `e29b9cc` (style)
3. **Task 2: Update all test imports** - included in `d8f2e6a` (both tasks committed together)

Note: Task 2 (test updates) was committed together with Task 1 by the prior agent session.

## Files Created/Modified
- `src/aquapose/core/context.py` - New home of Stage Protocol and PipelineContext (moved from engine/stages.py)
- `src/aquapose/engine/stages.py` - DELETED
- `src/aquapose/core/__init__.py` - Added PipelineContext and Stage to exports
- `src/aquapose/engine/__init__.py` - Re-export from core.context instead of engine.stages
- `src/aquapose/engine/pipeline.py` - Direct import from core.context
- `src/aquapose/core/detection/stage.py` - Removed TYPE_CHECKING block, direct import
- `src/aquapose/core/midline/stage.py` - Removed TYPE_CHECKING block, direct import
- `src/aquapose/core/association/stage.py` - Removed TYPE_CHECKING block, direct import
- `src/aquapose/core/tracking/stage.py` - Removed TYPE_CHECKING block, direct import
- `src/aquapose/core/reconstruction/stage.py` - Removed TYPE_CHECKING block, direct import
- `src/aquapose/core/synthetic.py` - PipelineContext moved to direct import; SyntheticConfig stays TYPE_CHECKING
- `tools/import_boundary_checker.py` - IB-003 allowlist + SR-002 core.context exclusion
- `tests/unit/engine/test_stages.py` - Import from core.context; updated boundary check
- 8 additional test files - Updated from engine.stages to core.context imports

## Decisions Made
- PipelineContext and Stage are pure data contracts — no engine logic — so they belong in core/
- engine/stages.py deleted cleanly: no re-export shim, all consumers updated directly
- SyntheticConfig TYPE_CHECKING import in core/synthetic.py is an acceptable exception: config flows downward from engine to core, annotation-only dependency
- import_boundary_checker.py updated with documented IB-003 allowlist for the SyntheticConfig exception
- SR-002 rule updated to exclude aquapose.core.context since engine files importing from it is correct design

## Deviations from Plan

None — plan executed exactly as written. The import_boundary_checker.py updates (IB-003 allowlist, SR-002 exclusion) were part of the planned Task 1 step 8.

## Issues Encountered
- Pre-commit hook (ruff format) applied formatting changes to staged files during the commit process, requiring a separate style commit (`e29b9cc`) to capture the auto-formatted files cleanly.

## Next Phase Readiness
- DoD criterion 7 IB-003 violations: RESOLVED (0 remaining)
- Import boundary checker passes clean
- 513 unit tests pass
- Ready for Phase 20 remaining plans (20-02 through 20-05)

---
*Phase: 20-post-refactor-loose-ends*
*Completed: 2026-02-27*
