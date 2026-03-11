---
phase: 83-custom-tracker-implementation
plan: 01
subsystem: tracking
tags: [kalman-filter, oks, ocm, hungarian-assignment, numpy, scipy, tdd]

# Dependency graph
requires:
  - phase: 82-association-upgrade-keypoint-centroid
    provides: spine1 keypoint centroid in Tracklet2D, Detection.keypoints populated by PoseStage
  - phase: 81-pose-stage
    provides: Detection.keypoints and Detection.keypoint_conf enriched before tracking
provides:
  - _KalmanFilter: 24-dim constant-velocity KF with confidence-scaled noise
  - compute_oks_matrix: vectorized (N,M) OKS similarity via NumPy broadcasting
  - compute_ocm_matrix: direction-consistency via cosine similarity of spine heading
  - build_cost_matrix: combined OKS+OCM cost for Hungarian assignment
  - _KFTrack: per-track state with ORU/OCR mechanisms
  - _SinglePassTracker: full predict/match/update/birth/death loop producing Tracklet2D
  - keypoint_sigmas.py: DEFAULT_SIGMAS and compute_keypoint_sigmas
affects: [83-02, 84-custom-tracker-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD: tests written first (RED), then implementation (GREEN)"
    - "Vectorized OKS via NumPy broadcasting — no Python loops over track-detection pairs"
    - "Confidence-scaled R matrix: R_k = base_R / max(conf_k, epsilon)"
    - "ORU: save pre-coast x/P, restore on re-match before KF update"
    - "OCR: scan obs_history deque (maxlen=5) for OKS > 0.5 during coast"
    - "_KptTrackletBuilder independent from ocsort_wrapper._TrackletBuilder — stores keypoints+confs for Plan 02 merge"

key-files:
  created:
    - src/aquapose/core/tracking/keypoint_sigmas.py
    - src/aquapose/core/tracking/keypoint_tracker.py
    - tests/unit/core/tracking/test_keypoint_sigmas.py
    - tests/unit/core/tracking/test_keypoint_tracker.py
  modified:
    - src/aquapose/core/tracking/__init__.py

key-decisions:
  - "KF state dimension is 24 (6 kpts x 2D x 2), not 60 — CONTEXT.md '60-dim' was conceptual shorthand"
  - "OKS denominator uses det_confs.sum(axis=-1) directly (shape (M,)), not sum of the broadcast tensor, to avoid spurious extra dimension"
  - "ORU implementation: snapshot x/P at time_since_update=0→1 transition, restore on re-match before KF update"
  - "OCR implementation: reverse-scan obs_history deque, apply first entry with OKS > 0.5 against predicted positions"
  - "_KptTrackletBuilder is independent from ocsort_wrapper._TrackletBuilder — avoids coupling and stores keypoints+confs for Plan 02 bidirectional merge"
  - "match_update() calls predict() implicitly via the tracker's predict-then-update cycle, not independently"

patterns-established:
  - "Vectorized cost matrix: always use NumPy broadcasting (N,1,K,2) vs (1,M,K,2) for OKS"
  - "State serialization: get_state() returns JSON-safe nested lists; from_state() reconstructs via __new__"
  - "ORU/OCR state stored on _KFTrack, not on tracker — each track owns its recovery logic"

requirements-completed: [TRACK-02, TRACK-03, TRACK-04, TRACK-06]

# Metrics
duration: 9min
completed: 2026-03-11
---

# Phase 83 Plan 01: Keypoint Tracker Core Engine Summary

**24-dim constant-velocity Kalman filter with confidence-scaled R, vectorized OKS+OCM cost matrix, ORU/OCR recovery mechanisms, and single-pass predict/match/update/birth/death tracker loop**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-03-11T02:58:04Z
- **Completed:** 2026-03-11T03:07:17Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 5

## Accomplishments

- Built `_KalmanFilter` with 24-dim state (6 keypoints x 2D x position+velocity), confidence-scaled measurement noise (R_k = base_R / max(conf_k, 1e-6)), and JSON-safe state serialization for chunk handoff
- Implemented vectorized OKS cost matrix via NumPy broadcasting — no Python loops over track-detection pairs; shape `(N, 1, K, 2)` vs `(1, M, K, 2)` pattern with correct denominator derivation
- Built `_KFTrack` with ORU (pre-coast state snapshot restored on re-match) and OCR (obs_history ring buffer, scan for OKS > 0.5 during coast)
- `_SinglePassTracker.update()` runs full predict/match/update/birth/death cycle; `get_tracklets()` returns `Tracklet2D` for confirmed tracks only
- `keypoint_sigmas.py` with `DEFAULT_SIGMAS=[0.08, 0.06, 0.04, 0.04, 0.05, 0.07]` (endpoints larger) and COCO-methodology `compute_keypoint_sigmas()`
- 38 new tests added covering KF math, OKS, OCM, tracker lifecycle; all 1151 tests passing

## Task Commits

Each task was committed atomically (TDD pattern):

1. **Task 1 RED: failing tests** - `1f63426` (test)
2. **Task 1 GREEN: implementation** - `0f9f25b` (feat)

## Files Created/Modified

- `src/aquapose/core/tracking/keypoint_sigmas.py` — DEFAULT_SIGMAS + compute_keypoint_sigmas
- `src/aquapose/core/tracking/keypoint_tracker.py` — _KalmanFilter, OKS/OCM functions, _KFTrack, _KptTrackletBuilder, _SinglePassTracker
- `src/aquapose/core/tracking/__init__.py` — exports for new public symbols
- `tests/unit/core/tracking/test_keypoint_sigmas.py` — sigma defaults and compute tests
- `tests/unit/core/tracking/test_keypoint_tracker.py` — KF, OKS, OCM, tracker lifecycle tests

## Decisions Made

- KF state dimension is 24 (6 kpts x 2D x 2), not 60 — CONTEXT.md "60-dim" was conceptual shorthand; the mathematical dimension is 24.
- OKS denominator bug fixed during implementation: `np.sum(confs, axis=-1)[np.newaxis, :]` where `confs=(1,M,K)` was giving shape `(1,1,M)` (3D) instead of `(1,M)` (2D). Fixed to `det_confs.sum(axis=-1)[np.newaxis, :]` using the original `(M,K)` array.
- `_KptTrackletBuilder` kept independent from `ocsort_wrapper._TrackletBuilder` to avoid coupling and to store `keypoints`/`keypoint_conf` lists needed by Plan 02's bidirectional merge.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] OKS denominator shape was 3D instead of 2D**
- **Found during:** Task 1 GREEN (implementation verification)
- **Issue:** `np.sum(confs, axis=-1)[np.newaxis, :]` where `confs` had shape `(1, M, K)` produced a `(1, 1, M)` tensor. Combined with `(N, M)` numerator, division yielded `(1, N, M)` output.
- **Fix:** Changed denominator to `det_confs.sum(axis=-1)[np.newaxis, :]` using the original `(M, K)` array, giving correct `(1, M)` shape.
- **Files modified:** src/aquapose/core/tracking/keypoint_tracker.py
- **Verification:** `compute_oks_matrix(pred(2,6,2), det(3,6,2), ...)` now correctly returns shape `(2, 3)`
- **Committed in:** `0f9f25b` (feat commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential correctness fix. No scope creep.

## Issues Encountered

- Pre-commit ruff hooks flagged import ordering and unused variables across two commit attempts; all resolved with ruff --fix and manual edits before final commit.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 83-02 (bidirectional architecture + merge) can now consume `_SinglePassTracker` directly — the forward/backward pass engine is complete and tested
- `_KptTrackletBuilder` stores `keypoints` and `keypoint_conf` per frame for the merge step
- All existing 1113 tests remain green; 38 new tests added
- `hatch run check` clean on new files (pre-existing typecheck errors in pipeline.py, overlay.py are unrelated)

---
*Phase: 83-custom-tracker-implementation*
*Completed: 2026-03-11*
