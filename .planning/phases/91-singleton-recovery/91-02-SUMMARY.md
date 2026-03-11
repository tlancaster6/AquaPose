---
phase: 91-singleton-recovery
plan: "02"
subsystem: association
tags: [singleton-recovery, pipeline-wiring, config, stage-integration]

# Dependency graph
requires:
  - phase: 91-01
    provides: recovery.py with recover_singletons() and RecoveryConfigLike protocol

provides:
  - recovery_enabled guard in AssociationStage.run() step 5
  - recovery_* fields confirmed present on AssociationConfig
  - RecoveryConfigLike and recover_singletons confirmed exported from association __init__.py
affects:
  - phase 92 (tuning singleton recovery thresholds against real run data)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Lazy import pattern for optional pipeline step (import inside if-guard for zero overhead when disabled)
    - Config toggle guard at call site plus early return inside function for dual-layer disable

key-files:
  created: []
  modified:
    - src/aquapose/core/association/stage.py
    - src/aquapose/engine/config.py (already had fields from 91-01 wave)
    - src/aquapose/core/association/__init__.py (already had exports from 91-01 wave)

key-decisions:
  - "recovery_enabled checked at call site in stage.py (not just inside recover_singletons) to avoid lazy import overhead when disabled"
  - "config.py recovery fields and __init__.py exports were already present — only the call-site guard was missing"

patterns-established:
  - "Pipeline step guards: check both resource availability (forward_luts is not None) and config toggle before lazy import"

requirements-completed:
  - RECOV-02
  - RECOV-04

# Metrics
duration: 5min
completed: 2026-03-11
---

# Phase 91 Plan 02: Singleton Recovery Pipeline Wiring Summary

**recover_singletons() wired into AssociationStage step 5 with recovery_enabled config guard; AssociationConfig satisfies RecoveryConfigLike protocol structurally**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-11T20:32:42Z
- **Completed:** 2026-03-11T20:37:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Confirmed recovery_* config fields present on AssociationConfig (recovery_enabled, recovery_residual_threshold, recovery_min_shared_frames, recovery_min_segment_length)
- Confirmed RecoveryConfigLike and recover_singletons exported from association __init__.py
- Added missing `recovery_enabled` guard to stage.py Step 5 so the lazy import is skipped entirely when recovery is disabled
- All 1183 unit tests pass with no regressions

## Task Commits

1. **Task 1: Add config fields and wire stage integration** - `9c27f56` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/aquapose/core/association/stage.py` — Added `and self._config.association.recovery_enabled` to Step 5 if-guard

## Decisions Made

- The `recovery_enabled` check belongs at the call site (stage.py) to avoid even the lazy import overhead, in addition to the early-return inside `recover_singletons()` itself. This dual-layer guard is the established pattern for optional pipeline steps.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing recovery_enabled guard in stage.py call site**
- **Found during:** Task 1 (initial file inspection)
- **Issue:** Stage.py already had Step 5 wired but used only `if forward_luts is not None:` — missing the `and self._config.association.recovery_enabled` guard specified in the plan
- **Fix:** Added the config toggle check to the if-condition so lazy import is skipped when recovery is disabled
- **Files modified:** src/aquapose/core/association/stage.py
- **Verification:** `hatch run check` (lint + typecheck clean) and `hatch run test` (1183 passed)
- **Committed in:** 9c27f56

---

**Total deviations:** 1 auto-fixed (Rule 1 - incomplete call-site guard)
**Impact on plan:** Essential correctness fix — without the call-site guard, `recovery_enabled=False` would still trigger the lazy import (though recover_singletons early-returns immediately). Minor overhead issue but violates the plan specification.

## Issues Encountered

None — config fields and exports were already present from Phase 91 Plan 01. Only the call-site guard was missing.

## Next Phase Readiness

- Singleton recovery is fully wired and active in the pipeline
- Phase 92 can tune recovery thresholds (recovery_residual_threshold, recovery_min_shared_frames, recovery_min_segment_length) against real run data
- RECOV-02 and RECOV-04 requirements satisfied

---
*Phase: 91-singleton-recovery*
*Completed: 2026-03-11*
