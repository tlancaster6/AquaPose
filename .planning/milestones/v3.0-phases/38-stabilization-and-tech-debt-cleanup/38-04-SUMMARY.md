---
phase: 38-stabilization-and-tech-debt-cleanup
plan: "04"
subsystem: visualization
tags: [dead-code, ast, import-graph, cleanup, visualization]

# Dependency graph
requires:
  - phase: 38-01
    provides: Config field consolidation and weights_path rename
  - phase: 38-02
    provides: NDJSON to txt+yaml training format migration
provides:
  - Import analysis report classifying all legacy directory files as canonical, thin wrapper, or dead code
  - visualization/diagnostics.py deleted (confirmed zero importers)
affects: [39-migrate-legacy-libraries, future-dead-code-audits]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Import graph analysis via ast.parse() to classify files before any deletion — no code deleted without evidence"
    - "Backward-compat shims deleted once confirmed no importers remain"

key-files:
  created:
    - .planning/phases/38-stabilization-and-tech-debt-cleanup/38-DEAD-CODE-REPORT.md
  modified:
    - src/aquapose/visualization/diagnostics.py (deleted)

key-decisions:
  - "visualization/diagnostics.py deleted — confirmed zero importers across src/, tests/, scripts/ via ast import graph"
  - "All other legacy modules (reconstruction/, segmentation/, tracking/) classified as canonical and load-bearing — kept in place"
  - "Broader organization concern (misleading directory names, cross-package private imports) deferred to new Phase 39"

patterns-established:
  - "Evidence-first deletion: build import graph report before any removal; no code deleted without explicit user approval"

requirements-completed: [STAB-04]

# Metrics
duration: ~20min
completed: 2026-03-02
---

# Phase 38 Plan 04: Dead Code Analysis and Cleanup Summary

**Import graph analysis of 86 legacy files confirmed visualization/diagnostics.py as the only dead code; file deleted after user approval; all other legacy directories confirmed load-bearing and retained**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-03-02T16:55:00Z
- **Completed:** 2026-03-02T17:16:11Z
- **Tasks:** 2 (Task 1: auto, Task 2: checkpoint-approved cleanup)
- **Files modified:** 2 (38-DEAD-CODE-REPORT.md created, diagnostics.py deleted)

## Accomplishments

- Produced a complete AST-based import graph analysis of 86 Python files across 4 legacy directories, classifying each file as canonical, thin wrapper, or dead code with evidence
- Confirmed `visualization/diagnostics.py` as the only dead file (zero importers in src/, tests/, scripts/) and deleted it per user approval
- Established that reconstruction/, segmentation/, and tracking/ modules are canonical, load-bearing implementations actively imported by core/ — not legacy in the "obsolete" sense
- Deferred organizational cleanup (misleading directory names, cross-package private imports) to a dedicated Phase 39

## Task Commits

Each task was committed atomically:

1. **Task 1: Produce import analysis report for legacy directories** - `d36d1b2` (feat)
2. **Task 2: User reviews dead code report and approves actions** - `846b157` (chore)

**Plan metadata:** (recorded in final docs commit)

## Files Created/Modified

- `.planning/phases/38-stabilization-and-tech-debt-cleanup/38-DEAD-CODE-REPORT.md` - Complete import graph analysis report classifying all 16 legacy files across 4 directories
- `src/aquapose/visualization/diagnostics.py` - Deleted (backward-compat shim, zero importers)

## Decisions Made

- `visualization/diagnostics.py` deleted — AST analysis confirmed zero importers across all of src/, tests/, and scripts/. The shim was created when the original 2200-LOC diagnostics.py was split; the migration was already complete and the shim was unreferenced.
- All other legacy modules retained — reconstruction/, segmentation/, and tracking/ files are canonical implementations with 20+ active import sites in core/. Not legacy in the "obsolete" sense; they are the implementation layer beneath the pipeline abstraction.
- Broader organization concerns deferred — the misleading "legacy" label for directories and cross-package private imports (e.g., segmentation.crop private helpers imported by core/) are scoped to a new Phase 39, not this plan.
- `visualization/__init__.py` required no changes — it already imported directly from midline_viz and triangulation_viz, not from diagnostics.py.

## Deviations from Plan

None - plan executed exactly as written. The checkpoint was approved as specified, and only the single user-approved deletion was executed.

## Issues Encountered

None. All 656 unit tests passed after deletion of diagnostics.py.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 38 stabilization plans 01-04 are all complete
- Phase 39 is ready to begin: migrate legacy domain libraries (reconstruction/, segmentation/, tracking/) into proper core/ submodules, fix misleading directory naming, resolve cross-package private imports
- The dead code report (38-DEAD-CODE-REPORT.md) provides the migration map for Phase 39 planning

---
*Phase: 38-stabilization-and-tech-debt-cleanup*
*Completed: 2026-03-02*
