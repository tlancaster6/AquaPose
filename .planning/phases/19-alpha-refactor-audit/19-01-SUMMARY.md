---
phase: 19-alpha-refactor-audit
plan: 01
subsystem: tooling
tags: [ast, pre-commit, import-boundary, linting, audit]

requires:
  - phase: 13-engine-core
    provides: engine/ and core/ layer structure that the boundary rules apply to

provides:
  - AST-based import boundary checker covering all 6 rule categories (IB-001 through SR-002)
  - Pre-commit hook enforcing one-way import layering on every commit touching src/aquapose/
  - Audit catalog: 7 existing IB-003 violations in core/ stage files documented (not fixed)

affects: [19-02, 19-03, 19-04, future refactor plans that touch core/ or engine/]

tech-stack:
  added: []
  patterns:
    - "Import boundary enforcement: IB-001/IB-003 (core), IB-002 (engine), IB-004 (legacy), SR-001 (stage run I/O), SR-002 (observer core imports)"
    - "Pre-commit hook pattern: shell wrapper in .hooks/ delegates to Python script in tools/"

key-files:
  created:
    - tools/import_boundary_checker.py
    - .hooks/check-import-boundary.sh
  modified:
    - .pre-commit-config.yaml

key-decisions:
  - "IB-003 violations in core/ stage files are cataloged but NOT fixed in this plan (audit phase only) — Phase 19-02+ will address the refactoring"
  - "Legacy computation dirs (calibration, segmentation, tracking, reconstruction, initialization, mesh, optimization) classified as Layer 1 and checked under IB-004"
  - "SR-002 (observer core imports) is a WARNING not an error — some legitimate core type imports may be needed by observers"
  - "Checker accepts file paths as CLI args for pre-commit per-file invocation, or scans all src/aquapose/ when run standalone"

patterns-established:
  - "Violation format: {filepath}:{line}: {rule_id} [error|warning] — {description}"
  - "Exit code 0 = clean, exit code 1 = hard violations found (warnings alone return 0)"

requirements-completed: [AUDIT]

duration: 12min
completed: 2026-02-26
---

# Phase 19 Plan 01: Import Boundary Checker Summary

**AST-based import boundary checker with pre-commit hook enforcing core/engine/cli layering — found 7 existing IB-003 (TYPE_CHECKING backdoor) violations in core/ stage files**

## Performance

- **Duration:** 12 min
- **Started:** 2026-02-26T22:58:12Z
- **Completed:** 2026-02-26T23:10:00Z
- **Tasks:** 2
- **Files modified:** 3 created, 1 modified

## Accomplishments
- Built `tools/import_boundary_checker.py` — pure stdlib (ast module) script with all 6 rule categories
- Wired checker as a pre-commit hook via `.hooks/check-import-boundary.sh` and `.pre-commit-config.yaml`
- Audited full codebase: found 7 IB-003 violations (TYPE_CHECKING backdoors in all 5 core stage files + core/synthetic.py)
- Verified no false positives on legitimate engine->core imports

## Task Commits

Each task was committed atomically:

1. **Task 1: Build AST-based import boundary and structural rule checker** - `6b0a2e5` (feat)
2. **Task 2: Wire checker as pre-commit hook** - `9bc8dd8` (feat)

## Files Created/Modified
- `tools/import_boundary_checker.py` - Standalone AST-based checker, 6 rule categories, --verbose flag, exit codes
- `.hooks/check-import-boundary.sh` - Shell wrapper for pre-commit hook invocation
- `.pre-commit-config.yaml` - Added import-boundary hook to local repo section

## Decisions Made
- IB-003 violations in core/ stage files are cataloged only — not fixed in this plan (audit phase)
- Legacy computation dirs follow core/ rules under IB-004
- SR-002 is a warning (not error) since some legitimate core imports may be needed in observers
- Checker accepts file-level args for pre-commit compatibility, falls back to full scan

## Violations Cataloged

The following **7 IB-003** violations exist in the current codebase and must be addressed in future refactor plans:

| File | Line | Violation |
|------|------|-----------|
| `src/aquapose/core/association/stage.py` | 21 | imports `aquapose.engine.stages` under TYPE_CHECKING |
| `src/aquapose/core/detection/stage.py` | 22 | imports `aquapose.engine.stages` under TYPE_CHECKING |
| `src/aquapose/core/midline/stage.py` | 21 | imports `aquapose.engine.stages` under TYPE_CHECKING |
| `src/aquapose/core/reconstruction/stage.py` | 24 | imports `aquapose.engine.stages` under TYPE_CHECKING |
| `src/aquapose/core/synthetic.py` | 24 | imports `aquapose.engine.config` under TYPE_CHECKING |
| `src/aquapose/core/synthetic.py` | 25 | imports `aquapose.engine.stages` under TYPE_CHECKING |
| `src/aquapose/core/tracking/stage.py` | 29 | imports `aquapose.engine.stages` under TYPE_CHECKING |

**Root cause:** Core stage files use `PipelineContext` as a type annotation for their `run()` method parameter, but `PipelineContext` is defined in `engine/stages.py`. The fix is to move `PipelineContext` to `core/` or use a Protocol/string annotation instead.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed nested `if` SIM102 ruff violations**
- **Found during:** Task 1 (during pre-commit hook on commit)
- **Issue:** Ruff SIM102 rule flagged 3 nested `if` statements that should be combined with `and`
- **Fix:** Merged nested `if` into single conditions using `and`
- **Files modified:** `tools/import_boundary_checker.py`
- **Verification:** `python -m ruff check tools/import_boundary_checker.py` passes
- **Committed in:** `9bc8dd8` (Task 2 commit, after fixing and re-staging)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Minor linting fix with no behavioral impact. No scope creep.

## Issues Encountered
- Pre-commit hook ran ruff on first commit attempt and found SIM102 violations — fixed inline and re-committed cleanly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Import boundary checker is active and will catch any new violations from this point forward
- 7 IB-003 violations documented for Phase 19-02+ to address
- The checker provides the automated gate that makes refactoring safe

## Self-Check: PASSED

- FOUND: tools/import_boundary_checker.py
- FOUND: .hooks/check-import-boundary.sh
- FOUND: .planning/phases/19-alpha-refactor-audit/19-01-SUMMARY.md
- FOUND: commit 6b0a2e5 (Task 1)
- FOUND: commit 9bc8dd8 (Task 2)
- FOUND: commit 87cf8ff (metadata)

---
*Phase: 19-alpha-refactor-audit*
*Completed: 2026-02-26*
