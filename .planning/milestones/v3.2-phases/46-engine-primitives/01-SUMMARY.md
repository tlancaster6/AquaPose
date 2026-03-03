---
phase: 46-engine-primitives
plan: "01"
subsystem: core
tags: [pickle, cache, context, fingerprint, dataclass]

# Dependency graph
requires: []
provides:
  - StaleCacheError exception class in core/context.py
  - load_stage_cache() function for reading envelope-wrapped pickle caches
  - context_fingerprint() for stable field-name-based version hashing
  - carry_forward: CarryForward | None = None field on PipelineContext
affects:
  - 46-02 (DiagnosticObserver writes envelopes that load_stage_cache reads)
  - 46-03 (ContextLoader uses load_stage_cache to seed evaluation runs)
  - 47 (evaluation harness uses carry_forward state)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Envelope-wrapped pickle format: dict with run_id, timestamp, stage_name, version_fingerprint, context keys
    - context_fingerprint() provides structural version fingerprinting via SHA-256 of sorted field names
    - StaleCacheError catches AttributeError/ModuleNotFoundError/pickle.UnpicklingError on deserialization

key-files:
  created:
    - tests/unit/core/test_stage_cache.py
  modified:
    - src/aquapose/core/context.py
    - src/aquapose/core/__init__.py

key-decisions:
  - "StaleCacheError defined in core/context.py alongside PipelineContext (not a separate errors.py)"
  - "context_fingerprint() is public (not _context_fingerprint) so DiagnosticObserver can import it"
  - "Shape validation uses combined if-condition (SIM102 compliance) rather than nested ifs"
  - "carry_forward field placed after stage_timing as cross-batch state, not a stage output"

patterns-established:
  - "Cache envelope format: dict with run_id, timestamp, stage_name, version_fingerprint, context keys"
  - "Deserialization errors wrapped in StaleCacheError with re-run suggestion message"

requirements-completed: [INFRA-03, INFRA-04]

# Metrics
duration: 4min
completed: 2026-03-03
---

# Phase 46 Plan 01: StaleCacheError, load_stage_cache, and PipelineContext.carry_forward Summary

**StaleCacheError, load_stage_cache with envelope/shape validation, context_fingerprint hash, and carry_forward field on PipelineContext — foundational primitives for per-stage pickle cache evaluation in v3.2**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-03T18:18:32Z
- **Completed:** 2026-03-03T18:22:47Z
- **Tasks:** 3
- **Files modified:** 3 (1 created)

## Accomplishments

- `StaleCacheError` defined in `core/context.py` with descriptive messages including cache path and re-run suggestion
- `load_stage_cache()` implements full validation: pickle deserialization, envelope format check, shape validation (frame_count vs len(detections))
- `context_fingerprint()` produces stable 12-character hex hash of PipelineContext field names for version detection
- `carry_forward: CarryForward | None = None` field added to PipelineContext after stage_timing
- All three symbols exported from `aquapose.core.__init__` and `__all__`
- 7 unit tests written and passing (round-trip, stale cache, invalid envelope, shape mismatch, file-not-found, fingerprint stability, carry_forward field)

## Task Commits

Each task was committed atomically:

1. **Task 46.1.1 + 46.1.2: Add carry_forward, StaleCacheError, context_fingerprint, load_stage_cache** - `9216797` (feat)
2. **Task 46.1.3: Update __init__.py exports and write unit tests** - `339e769` (feat)

## Files Created/Modified

- `src/aquapose/core/context.py` - Added StaleCacheError, context_fingerprint(), load_stage_cache(), carry_forward field on PipelineContext; added pickle/hashlib/dataclasses/Path imports
- `src/aquapose/core/__init__.py` - Added StaleCacheError, load_stage_cache, context_fingerprint to imports and __all__
- `tests/unit/core/test_stage_cache.py` - 7 unit tests covering all plan requirements

## Decisions Made

- Tasks 46.1.1 and 46.1.2 were implemented together in a single commit since they both touch `context.py` and form an indivisible unit (StaleCacheError is required by load_stage_cache).
- Nested `if` for shape validation was combined into a single `if` with `and` to satisfy ruff SIM102 lint rule — this is a style-only change with identical semantics.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ruff SIM102 lint failure (nested if statements)**
- **Found during:** Task 46.1.1 commit (pre-commit hook)
- **Issue:** Shape validation used nested `if` statements; ruff SIM102 requires combining into single `if` with `and`
- **Fix:** Combined the two `if` statements into a single condition with `and`
- **Files modified:** src/aquapose/core/context.py
- **Verification:** Pre-commit hook passed on retry commit
- **Committed in:** 9216797 (fixed before successful commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - lint compliance)
**Impact on plan:** Minor style-only fix; no semantic change. No scope creep.

## Issues Encountered

- Pre-existing typecheck errors (40 errors in basedpyright) exist in unrelated files (association/stage.py, detection backends, midline backends, engine observers). None are in files modified by this plan. These are out of scope per deviation rules.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `load_stage_cache`, `StaleCacheError`, and `context_fingerprint` are ready for use by Plan 46.2 (DiagnosticObserver) which writes the envelopes these functions read.
- `carry_forward` field is ready for use by any stage needing cross-batch persistence.
- No blockers.

---
*Phase: 46-engine-primitives*
*Completed: 2026-03-03*
