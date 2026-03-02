---
phase: 35-codebase-cleanup
plan: 02
subsystem: core
tags: [midline, backends, stubs, config-validation, cleanup]

requires:
  - phase: 35-01
    provides: Deleted UNetSegmentor, _PoseModel, SAM2, MOG2 — backends referencing those symbols were broken

provides:
  - SegmentThenExtractBackend and DirectPoseBackend are no-op stubs that instantiate cleanly and return midline=None
  - MidlineConfig.__post_init__ validates backend names at construction time
  - test_direct_pose_backend.py with 5 stub tests
  - REQUIREMENTS.md CLEAN-03 description corrected
  - ROADMAP.md Phase 37 depends_on updated

affects:
  - 35-03 (if any further cleanup needed)
  - 37-pipeline-integration (wires YOLO models into these stubs)

tech-stack:
  added: []
  patterns:
    - No-op stub pattern: accept **kwargs and log a warning rather than loading a model

key-files:
  created:
    - tests/unit/core/midline/test_direct_pose_backend.py
  modified:
    - src/aquapose/core/midline/backends/segment_then_extract.py
    - src/aquapose/core/midline/backends/direct_pose.py
    - src/aquapose/core/midline/backends/__init__.py
    - src/aquapose/engine/config.py
    - tests/unit/core/midline/test_midline_stage.py
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Backends accept **kwargs and silently ignore all arguments — preserves API compatibility with get_backend() kwarg forwarding"
  - "MidlineConfig.__post_init__ validates backend against hardcoded set — rejects typos at config construction time"
  - "test_midline_stage.py no longer needs sys.modules injection for aquapose.segmentation.model — stubs have no model imports"

patterns-established:
  - "No-op stub pattern: constructor accepts **kwargs, logs a warning, does no model loading"
  - "Config validation in __post_init__ for backend kind strings, mirroring DetectionConfig pattern"

requirements-completed:
  - CLEAN-03

duration: 35min
completed: 2026-03-01
---

# Phase 35 Plan 02: Codebase Cleanup (Stub Midline Backends) Summary

**Both midline backends rewritten as no-op stubs returning midline=None, MidlineConfig validates backend names, and planning documents corrected to match CONTEXT.md semantics**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-03-01T20:05:00Z
- **Completed:** 2026-03-01T20:42:09Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Rewrote `SegmentThenExtractBackend` and `DirectPoseBackend` as no-op stubs — no model loading, no deleted symbol imports, `midline=None` for every detection
- Added `MidlineConfig.__post_init__` that rejects unknown backend names with `ValueError`, mirroring the existing `DetectionConfig` validation pattern
- Removed the `sys.modules` injection hack from `test_midline_stage.py` (no longer needed since stubs have no model imports)
- Created `test_direct_pose_backend.py` with 5 tests covering instantiation, kwarg compatibility, empty cameras, and AnnotatedDetection field correctness
- Corrected REQUIREMENTS.md CLEAN-03 and ROADMAP.md Phase 37 to reflect that backends survive as stubs (not removed)
- Full test suite (614 tests) passes; lint passes; zero grep hits for any deleted symbols

## Task Commits

1. **Task 1: Stub midline backends and add config validation** - `2451c2a` (feat)
2. **Task 2: Correct planning documents and run final verification** - `682b680` (feat)

## Files Created/Modified

- `src/aquapose/core/midline/backends/segment_then_extract.py` - Complete rewrite as no-op stub (~80 lines vs 290)
- `src/aquapose/core/midline/backends/direct_pose.py` - Complete rewrite as no-op stub (~80 lines vs 416)
- `src/aquapose/core/midline/backends/__init__.py` - Updated module docstring and get_backend() docstring to reflect stub status
- `src/aquapose/engine/config.py` - Added MidlineConfig.__post_init__ with backend validation; updated docstring
- `tests/unit/core/midline/test_midline_stage.py` - Removed sys.modules injection; added stub behavior test
- `tests/unit/core/midline/test_direct_pose_backend.py` - New file: 5 stub backend tests
- `.planning/ROADMAP.md` - Phase 37 depends_on clarified
- `.planning/REQUIREMENTS.md` - CLEAN-03 description corrected

## Decisions Made

- Backends accept `**kwargs` and silently ignore all arguments — preserves API compatibility with `get_backend()` kwarg forwarding so callers don't need to change
- `MidlineConfig.__post_init__` validates `backend` against a hardcoded set `{"segment_then_extract", "direct_pose"}` — rejects typos at config construction time, same pattern as `DetectionConfig`
- `test_midline_stage.py` no longer needs `sys.modules` injection for `aquapose.segmentation.model` because the stubs have zero model imports

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Ruff auto-fixed one import style issue in `test_midline_stage.py` during pre-commit (multi-line import formatting for `SegmentThenExtractBackend`). Re-staged and committed cleanly on second attempt.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 35 is complete: both plans executed, all 5 CLEAN requirements satisfied (CLEAN-01 through CLEAN-05)
- Phase 36 (Training Wrappers) can begin: codebase is clean, no legacy model code, no conflicting training CLI commands
- Phase 37 (Pipeline Integration) will wire YOLO-seg and YOLO-pose into `segment_then_extract` and `direct_pose` backends respectively — the stubs are the insertion points

---
*Phase: 35-codebase-cleanup*
*Completed: 2026-03-01*
