---
phase: 20-post-refactor-loose-ends
plan: 02
subsystem: cleanup
tags: [dead-code, refactoring, audit, pipeline, mesh, initialization]

# Dependency graph
requires:
  - phase: 19-alpha-refactor-audit
    provides: "AUD-008, AUD-010, AUD-011, AUD-019, AUD-020: dead module catalog"
  - phase: 20-post-refactor-loose-ends-01
    provides: "PipelineContext move from engine/stages to core/context; engine/stages.py deleted"
provides:
  - "Clean source tree: only active modules remain in src/aquapose/"
  - "Zero collection errors from dead modules (pytorch3d no longer needed)"
  - "No dangling imports to deleted modules in src/ or tests/"
affects:
  - "20-03, 20-04, 20-05: subsequent Phase 20 plans will have clean import baseline"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dead code identified by audit report deleted in corresponding remediation plan"
    - "Orphaned tests co-deleted with their source modules"
    - "Legacy scripts depending on deleted modules removed together"

key-files:
  created: []
  modified:
    - CLAUDE.md
    - pyproject.toml
  deleted:
    - src/aquapose/pipeline/ (orchestrator.py, stages.py, report.py, __init__.py)
    - src/aquapose/initialization/ (keypoints.py, triangulator.py, __init__.py)
    - src/aquapose/mesh/ (builder.py, cross_section.py, profiles.py, spine.py, state.py, __init__.py)
    - src/aquapose/utils/__init__.py
    - tests/unit/pipeline/, tests/unit/initialization/, tests/unit/mesh/
    - scripts/legacy/diagnose_pipeline.py, diagnose_tracking.py, diagnose_triangulation.py, per_camera_spline_overlay.py
    - tests/unit/tracking/test_diagnose_tracking.py

key-decisions:
  - "Delete dead modules entirely rather than stub/deprecate — all 5 had zero pipeline consumers"
  - "test_diagnose_tracking.py co-deleted with its source (diagnose_tracking.py) — orphaned test has no value without the script"
  - "pytorch3d comment block removed from pyproject.toml — mesh/ (the only pytorch3d consumer) is deleted"
  - "CLAUDE.md architecture diagram updated to reflect current module structure (engine, reconstruction, tracking, visualization)"
  - "Plan 01 work (PipelineContext move) was in working tree uncommitted — completed Plan 01 Task 2 (test import updates) and committed Plan 01 before Plan 02"

patterns-established:
  - "Audit-driven deletion: Phase 19 audit produced finding IDs; Phase 20 plans target those IDs directly"

requirements-completed: [REMEDIATE]

# Metrics
duration: 30min
completed: 2026-02-26
---

# Phase 20 Plan 02: Dead Module Deletion Summary

**Deleted 5 dead modules (pipeline/, initialization/, mesh/, utils/, optimization/), 3 orphaned test directories, 4 legacy scripts, and 1 orphaned test — codebase now has zero dead-code modules and unit tests pass cleanly (513 tests).**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-02-26
- **Completed:** 2026-02-26
- **Tasks:** 1 (+ Plan 01 catch-up: test import updates)
- **Files modified/deleted:** 37 files deleted, 2 modified (CLAUDE.md, pyproject.toml)

## Accomplishments

- All 5 dead modules identified by the Phase 19 audit (AUD-008, AUD-010, AUD-011, AUD-019, AUD-020) deleted from src/aquapose/
- All 3 orphaned test directories (pipeline/, initialization/, mesh/) deleted from tests/unit/
- All 4 legacy scripts importing from deleted pipeline/ deleted from scripts/legacy/
- pytorch3d comment block removed from pyproject.toml (mesh/ was the sole consumer)
- CLAUDE.md architecture diagram updated to reflect current active modules
- Unit test suite: 513 tests pass, 0 failures, 0 collection errors

## Task Commits

1. **Plan 01 Task 2 (catch-up): Update all test imports from engine.stages to core.context** - `d8f2e6a` (feat)
2. **Style fixup: ruff formatting after import boundary fixes** - `e29b9cc` (style)
3. **Task 1: Delete dead modules, orphaned tests, legacy scripts** - `fa21b8b` (chore)

## Files Deleted

- `src/aquapose/pipeline/` — v1.0 orchestrator API (orchestrator.py, stages.py, report.py, __init__.py)
- `src/aquapose/initialization/` — v1.0 cold-start init (keypoints.py, triangulator.py, __init__.py)
- `src/aquapose/mesh/` — v1.0 parametric mesh (builder.py, cross_section.py, profiles.py, spine.py, state.py, __init__.py)
- `src/aquapose/utils/__init__.py` — empty stub with `__all__ = []` and zero imports
- `tests/unit/pipeline/`, `tests/unit/initialization/`, `tests/unit/mesh/` — orphaned test directories
- `scripts/legacy/diagnose_pipeline.py`, `diagnose_tracking.py`, `diagnose_triangulation.py`, `per_camera_spline_overlay.py` — all imported aquapose.pipeline.stages
- `tests/unit/tracking/test_diagnose_tracking.py` — orphaned test for deleted diagnose_tracking.py

## Files Modified

- `CLAUDE.md` — architecture diagram updated from v1.0 (mesh/initialization/utils) to current (engine/reconstruction/tracking/visualization)
- `pyproject.toml` — pytorch3d install instructions comment block removed (mesh/ deleted)

## Decisions Made

- Delete entirely rather than stub/deprecate: all 5 dead modules had zero pipeline consumers per Phase 19 audit
- test_diagnose_tracking.py co-deleted: orphaned test has no value without its source script
- Plan 01's Task 2 (test import updates from engine.stages to core.context) was found uncommitted in working tree — completed it before Plan 02 to restore clean test suite baseline

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Completed Plan 01 Task 2: test import updates from engine.stages to core.context**
- **Found during:** Task 1 (initial test run after deletions)
- **Issue:** Plan 01's Task 2 (updating 9 test files from `aquapose.engine.stages` to `aquapose.core.context`) was in the working tree uncommitted but incomplete — 5 source stage files still had TYPE_CHECKING backdoors to `aquapose.engine.stages`, causing 10 collection errors
- **Fix:** Updated all 9 test files + 5 source stage files + 1 test boundary check to import from `aquapose.core.context`; deleted orphaned `test_diagnose_tracking.py` which hard-referenced the deleted diagnose_tracking.py script
- **Files modified:** tests/unit/core/*/test_*.py (7 files), tests/unit/engine/test_stages.py, test_pipeline.py, test_diagnostic_observer.py, src/aquapose/core/*/stage.py (5 files)
- **Verification:** 513 tests pass, 0 collection errors
- **Committed in:** d8f2e6a (feat(20-01)) + e29b9cc (style)

---

**Total deviations:** 1 auto-fixed (Rule 1 — completing uncommitted Plan 01 work found in working tree)
**Impact on plan:** Essential — test suite was broken without this fix. No scope creep beyond what Plan 01 specified.

## Issues Encountered

- Plan 01 changes were in the working tree uncommitted (engine/stages.py deleted, core/context.py created, but test imports not updated). Handled by completing Plan 01's Task 2 and committing both plans' work in proper order before Plan 02 commit.

## Next Phase Readiness

- Source tree is clean: only active modules remain
- Import boundary (IB-003) fully resolved: no TYPE_CHECKING backdoors from core/ to engine/
- Test suite passes cleanly: 513 tests, 0 collection errors
- Ready for Phase 20 Plan 03 and beyond

---
*Phase: 20-post-refactor-loose-ends*
*Completed: 2026-02-26*
