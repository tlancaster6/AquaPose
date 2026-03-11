---
phase: 85-code-quality-audit-cli-smoke-test
plan: 01
subsystem: infra
tags: [basedpyright, ruff, vulture, boxmot, type-safety]

requires:
  - phase: 84-integration-evaluation
    provides: keypoint_bidi tracker as sole production tracker
provides:
  - BoxMot dependency removed from pyproject.toml and all source code
  - Zero basedpyright type errors across the codebase
  - Dead code at 80% vulture confidence cleaned
affects: [85-02, pipeline, tracking]

tech-stack:
  added: []
  patterns:
    - "LutConfigLike Protocol uses @property for frozen dataclass compat"
    - "type: ignore[attr-defined] for ultralytics/cv2 stub gaps"

key-files:
  created: []
  modified:
    - pyproject.toml
    - src/aquapose/engine/config.py
    - src/aquapose/core/tracking/stage.py
    - src/aquapose/calibration/luts.py

key-decisions:
  - "Clean break on BoxMot — no aliases, no fallback, ocsort_wrapper.py deleted"
  - "TrackingConfig drops iou_threshold field (ocsort-only) and valid_kinds is {keypoint_bidi}"
  - "Test stubs updated with keypoints for KeypointTracker compatibility"

patterns-established:
  - "Protocol with @property: use @property members in Protocol classes when implementors are frozen dataclasses"

requirements-completed: [INTEG-03, INTEG-04]

duration: 12min
completed: 2026-03-11
---

# Plan 85-01 Summary

**Removed BoxMot/OC-SORT dependency entirely and fixed all 25 basedpyright type errors to zero**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-11
- **Completed:** 2026-03-11
- **Tasks:** 2
- **Files modified:** 27 (15 + 15, with overlap)

## Accomplishments
- BoxMot dependency completely removed — deleted ocsort_wrapper.py (1087 lines), its test file, config options, and all docstring references across the codebase
- All 25 basedpyright type errors fixed to zero using casts, type: ignore for third-party stubs, protocol @property pattern, and isinstance guards
- 4 dead code items at 80% vulture confidence cleaned (show_fish_id, exc_val, args, augment_count)
- Test stubs in test_tracking_stage.py updated with keypoints for KeypointTracker compatibility — all 1152 tests pass

## Task Commits

1. **Task 1: Remove BoxMot dependency and all OC-SORT references** - `0773394` (refactor)
2. **Task 2: Fix all type errors and clean dead code** - `39d9c37` (fix)

## Files Created/Modified
- `pyproject.toml` - Removed boxmot>=11.0 dependency
- `src/aquapose/engine/config.py` - TrackingConfig: removed iou_threshold, ocsort from valid_kinds
- `src/aquapose/core/tracking/stage.py` - Removed OC-SORT branch, now only KeypointTracker
- `src/aquapose/core/tracking/ocsort_wrapper.py` - DELETED
- `tests/unit/tracking/test_ocsort_wrapper.py` - DELETED
- `src/aquapose/calibration/luts.py` - LutConfigLike protocol uses @property members
- `src/aquapose/cli.py` - isinstance guards for h5py Group/Dataset
- `src/aquapose/core/association/stage.py` - cast tracks_2d from context.get()
- `tests/unit/core/tracking/test_tracking_stage.py` - Added keypoints to _FakeDet stub

## Decisions Made
- Clean break on BoxMot: no backward-compatible aliases, no fallback path
- Used @property in Protocol for frozen dataclass compatibility (not plain attributes)
- Used type: ignore for third-party stub gaps (ultralytics YOLO, cv2.VideoWriter_fourcc) — these are correct at runtime

## Deviations from Plan

### Auto-fixed Issues

**1. Test stubs needed keypoints for KeypointTracker**
- **Found during:** Task 1 (BoxMot removal)
- **Issue:** test_tracking_stage.py _FakeDet lacked keypoints/keypoint_conf; KeypointTracker skips keypoint-less detections producing 0 tracklets
- **Fix:** Added keypoints (6x2 ndarray) and keypoint_conf to _FakeDet and _make_det
- **Files modified:** tests/unit/core/tracking/test_tracking_stage.py
- **Verification:** All 9 previously-failing tests now pass
- **Committed in:** 0773394 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (test stub compatibility)
**Impact on plan:** Necessary for test correctness after OC-SORT removal. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Codebase is clean: zero type errors, zero BoxMot references, dead code cleaned
- Ready for Plan 85-02: CLI smoke test and audit report

---
*Phase: 85-code-quality-audit-cli-smoke-test*
*Completed: 2026-03-11*
