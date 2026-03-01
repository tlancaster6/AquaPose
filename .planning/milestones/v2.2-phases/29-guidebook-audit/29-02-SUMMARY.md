---
phase: 29-guidebook-audit
plan: "02"
subsystem: documentation
tags: [guidebook, v2.2, backends, keypoint-midline, yolo-obb, documentation]

# Dependency graph
requires:
  - phase: 29-guidebook-audit
    provides: 29-01 audited and corrected stale GUIDEBOOK.md content (v2.1 accurate)
provides:
  - Backend registry subsection in Section 8 with YOLO-OBB concrete example
  - Keypoint midline approach documented with 6 anatomical keypoints and partial-skeleton policy
  - All v2.2 planned features marked inline with (v2.2) tags across relevant sections
  - Confidence-weighted reconstruction documented for both backends
  - Training CLI (aquapose train) mentioned as v2.2 planned
affects: [30-config-contracts, 31-training-infra, 32-yolo-obb, 33-keypoint-midline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "(v2.2) inline tag convention for planned feature markers — no banner, no new sections"
    - "backends/ sub-package under core/<stage>/ for swappable backends"
    - "Factory function pattern: resolves backend from config key, imports dynamically"
    - "Configurable model vs new backend distinction: OBB uses same data flow, gets optional fields"

key-files:
  created: []
  modified:
    - .planning/GUIDEBOOK.md

key-decisions:
  - "YOLO-OBB documented as a configurable model (not new backend) — optional angle/obb_points fields on Detection, no pipeline changes needed"
  - "Keypoint midline NaN policy locked: evaluate spline only within [t_min_observed, t_max_observed], everything outside NaN + confidence=0"
  - "Confidence-weighted triangulation: per-point confidence from keypoint backend flows to Stage 5; uniform weights when confidence is None"
  - "aquapose train CLI: mention existence only, full details deferred to Phase 31"

patterns-established:
  - "Backend Registration pattern: backends/ sub-package, factory reads config key, dynamic import, types.py output contract"
  - "v2.2 feature markers: inline (v2.2) after feature name, batch-removed at milestone end"

requirements-completed: [DOCS-02]

# Metrics
duration: 3min
completed: 2026-02-28
---

# Phase 29 Plan 02: Add v2.2 Planned Features Summary

**Backend registry subsection with YOLO-OBB walkthrough and keypoint midline backend fully documented with 6-keypoint anatomical model, fixed-t NaN policy, and confidence-weighted reconstruction across GUIDEBOOK.md**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-28T17:55:56Z
- **Completed:** 2026-02-28T17:58:08Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added Backend Registration subsection to Section 8 with concrete YOLO-OBB walkthrough enabling readers to identify where to add new backends
- Documented keypoint midline backend in full detail: 6 anatomical keypoints, fixed t values, spline-within-observed-range only, NaN + confidence=0 for extrapolation
- Added (v2.2) inline tags across 5 sections: Stage 1 (YOLO-OBB, affine crops), Stage 4 (keypoint backend), Stage 5 (confidence-weighted reconstruction), Section 11 (config additions), Section 14 (training CLI)
- Verified v2.2 milestone history entry in Section 15 already covers all required scope items

## Task Commits

Each task was committed atomically:

1. **Task 29-02.1: Add backend registry subsection to Section 8** - `a82c8a7` (docs)
2. **Task 29-02.2: Document v2.2 planned features inline with (v2.2) tags** - `7f4948b` (docs)

**Plan metadata:** (pending final commit)

## Files Created/Modified

- `.planning/GUIDEBOOK.md` - Added backend registry subsection (Section 8), all (v2.2) inline tags across Stage 1, 4, 5, Section 11, Section 14

## Decisions Made

- YOLO-OBB documented as a configurable model with optional Detection fields — not a new backend — consistent with STATE.md decision
- Keypoint midline partial-skeleton NaN policy locked as architectural contract for Phase 33 implementers
- Training CLI mention stays brief — details fully deferred to Phase 31 as specified in CONTEXT.md

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- GUIDEBOOK.md is now fully updated for v2.1 accuracy and v2.2 planned features
- Phase 30 (Config/Contracts) can proceed — Section 11 v2.2 config changes are documented as the target state
- Phase 33 (Keypoint Midline) implementers have a locked contract for the NaN policy and keypoint architecture

---
*Phase: 29-guidebook-audit*
*Completed: 2026-02-28*
