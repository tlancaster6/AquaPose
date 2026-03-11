---
phase: 83-custom-tracker-implementation
verified: 2026-03-10T00:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
human_verification:
  - test: "Run pipeline end-to-end with tracker_kind='keypoint_bidi' on YH clip"
    expected: "Tracks produced per camera, fewer fragments than OC-SORT baseline (< 27 tracks on benchmark clip)"
    why_human: "Requires GPU, real video data, and qualitative assessment of fragmentation improvement"
---

# Phase 83: Custom Tracker Implementation Verification Report

**Phase Goal:** A bidirectional batched keypoint tracker replaces OC-SORT, using OKS cost, OCM direction consistency, Kalman filter over keypoint positions, asymmetric birth/death, ORU/OCR mechanisms, bidirectional merge, chunk handoff, and gap interpolation
**Verified:** 2026-03-10
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1 | 24-dim constant-velocity Kalman filter tracks 6 keypoints x 2D positions + velocities with confidence-scaled measurement noise | VERIFIED | `_KalmanFilter` at line 74 of `keypoint_tracker.py`; `_DIM_X=24`; R_k = base_r / max(conf_k, eps); predict/update/serialize methods all present; 1580-line file |
| 2 | OKS cost matrix is computed vectorized over N tracks x M detections using per-keypoint sigmas and confidence weighting | VERIFIED | `compute_oks_matrix` uses NumPy broadcasting `(N,1,K,2)` vs `(1,M,K,2)`; no Python loops over pairs; `DEFAULT_SIGMAS` in `keypoint_sigmas.py` with 6 entries |
| 3 | OCM direction consistency uses cosine similarity of spine1-to-spine3 heading vectors as an additive cost term | VERIFIED | `compute_ocm_matrix`, `compute_heading` (spine1=idx 2, spine3=idx 4); `build_cost_matrix` = (1-OKS) + lambda*(1-OCM) |
| 4 | ORU re-updates KF state after coasting recovery; OCR checks observation buffer for track recovery | VERIFIED | `_KFTrack` stores `obs_history` deque (maxlen=5); `_pre_coast_state` snapshot at time_since_update 0→1 transition; OCR scans history for OKS > 0.5 |
| 5 | A single-pass tracker runs predict/match/update/birth/death cycle per frame | VERIFIED | `_SinglePassTracker.update()` at line 658; uses `linear_sum_assignment`; tentative/confirmed promotion; culls by max_age |
| 6 | KeypointTracker runs independent forward and backward passes over a chunk, then merges tracklets via Hungarian assignment on overlap-OKS | VERIFIED | `_fwd_tracker` + `_bwd_tracker` at lines 1368-1372; `get_tracklets()` replays stored frames in reverse for backward pass; `_collect_merged_builders()` applies Hungarian on OKS overlap cost |
| 7 | Merged tracklets have a unified ID space with no duplicate track_ids | VERIFIED | `get_tracklets()` re-assigns fresh monotonic IDs `(0, 1, 2, ...)` at lines 1432-1444 |
| 8 | Chunk boundary handoff serializes KF mean, covariance, and observation history as JSON-safe lists | VERIFIED | `get_state()` at line 1453 returns `kind`, `camera_id`, config params, and forward-pass track states with `.tolist()`; `from_state()` at line 1481 reconstructs; all lists/dicts/ints/floats |
| 9 | Small tracklet gaps (up to max_gap_frames) are filled via cubic spline interpolation | VERIFIED | `interpolate_gaps()` uses `scipy.interpolate.CubicSpline`; skips gaps > max_gap_frames; inserts "coasted" frames |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/core/tracking/keypoint_tracker.py` | `_KalmanFilter, compute_oks_matrix, build_cost_matrix, _KFTrack, _SinglePassTracker, KeypointTracker, merge_forward_backward, interpolate_gaps` | VERIFIED | 1580 lines; all named symbols present; substantive implementations confirmed |
| `src/aquapose/core/tracking/keypoint_sigmas.py` | `compute_keypoint_sigmas, DEFAULT_SIGMAS` | VERIFIED | 74 lines; `DEFAULT_SIGMAS = [0.08, 0.06, 0.04, 0.04, 0.05, 0.07]` (6 entries, endpoints larger); COCO-methodology compute function present |
| `tests/unit/core/tracking/test_keypoint_tracker.py` | Unit tests for KF, OKS, OCM, single-pass tracker, merge, gap fill, KeypointTracker wrapper | VERIFIED | 992 lines; covers `TestKalmanFilter`, `TestOks`, `TestOcm`, `TestSinglePassTracker`, `TestMergeForwardBackward`, `TestInterpolateGaps`, `TestKeypointTracker` |
| `src/aquapose/engine/config.py` | Extended `TrackingConfig` with `base_r`, `lambda_ocm`, `max_gap_frames` and `__post_init__` validation | VERIFIED | All three fields present with defaults; `__post_init__` validates `tracker_kind` against `{"ocsort", "keypoint_bidi"}` |
| `src/aquapose/core/tracking/stage.py` | `tracker_kind='keypoint_bidi'` dispatch in `TrackingStage.run()` | VERIFIED | Conditional at line 106; constructs `KeypointTracker` with all config fields; `from_state` path also dispatches correctly |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_SinglePassTracker.update()` | `_KalmanFilter.predict() + .update()` | Per-frame predict-match-update cycle | WIRED | `kf.predict()` and `kf.update()` called inside the update loop; `linear_sum_assignment` for Hungarian matching |
| `_SinglePassTracker.update()` | `compute_oks_matrix + build_cost_matrix` | Cost matrix construction before Hungarian assignment | WIRED | `compute_oks_matrix` → `build_cost_matrix` → `linear_sum_assignment` chain confirmed in `_SinglePassTracker.update()` |
| `KeypointTracker.update()` | `_SinglePassTracker` (forward pass) | Forward pass accumulates frames 0..N-1 | WIRED | `self._fwd_tracker.update(frame_idx=frame_idx, detections=detections)` at line 1391; stores frames for backward |
| `KeypointTracker.get_tracklets()` | `merge_forward_backward()` / `_collect_merged_builders()` | Hungarian assignment on overlap-OKS cost matrix | WIRED | `_collect_merged_builders()` called at line 1425; backward pass replay at line 1406; `linear_sum_assignment` inside merge |
| `TrackingStage.run()` | `KeypointTracker` | `tracker_kind` conditional dispatch | WIRED | `if tracker_kind == "keypoint_bidi": from aquapose.core.tracking.keypoint_tracker import KeypointTracker` at lines 106-107 |
| `KeypointTracker.get_state()` | `ChunkHandoff.tracks_2d_state` | Dict serialization with `.tolist()` numpy arrays | WIRED | `get_state()` returns JSON-safe dict; `TrackingStage.run()` stores in `new_tracks_2d_state`; `from_state` reconstructs on next chunk |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| TRACK-01 | 83-02 | Custom keypoint tracker runs forward and backward passes over each chunk | SATISFIED | `KeypointTracker` with `_fwd_tracker` + `_bwd_tracker` implemented |
| TRACK-02 | 83-01 | OKS-based association cost replaces IoU | SATISFIED | `compute_oks_matrix` + `build_cost_matrix` implemented and used in `_SinglePassTracker` |
| TRACK-03 | 83-01 | OCM direction consistency term | SATISFIED | `compute_ocm_matrix` + `compute_heading` (spine1-spine3 vector) in cost matrix |
| TRACK-04 | 83-01 | Kalman filter tracks keypoint positions and velocities | SATISFIED | `_KalmanFilter` 24-dim state (documented as equivalent to "60-dim" conceptual description) |
| TRACK-05 | 83-02 | Asymmetric track birth/death | SATISFIED BY DESIGN | Module docstring documents: "forward pass catches track entries, backward pass catches exits"; `KeypointTracker` docstring explicitly confirms; per CONTEXT.md locked decision |
| TRACK-06 | 83-01 | ORU and OCR mechanisms | SATISFIED | `_KFTrack` with `_pre_coast_state` for ORU, `obs_history` deque for OCR; both exercised in tests |
| TRACK-07 | 83-02 | Bidirectional merge with overlap-based matching | SATISFIED | `merge_forward_backward()` + `_collect_merged_builders()` with Hungarian OKS assignment |
| TRACK-08 | 83-02 | Chunk boundary handoff via serialized KF state | SATISFIED | `get_state()` / `from_state()` with JSON-safe serialization; wired through `TrackingStage` |
| TRACK-09 | 83-02 | Gap interpolation via spline | SATISFIED | `interpolate_gaps()` with `CubicSpline`; max_gap_frames configurable |
| TRACK-10 | 83-02 | BYTE-style secondary pass (conditional) | SATISFIED (DEFERRED) | Explicitly deferred to Phase 84 per CONTEXT.md decision and user agreement. `KeypointTracker` docstring documents deferral. REQUIREMENTS.md conditional qualifier ("implement if INV-04 reveals...") applies — Phase 84 will evaluate need. REQUIREMENTS.md marks it [x] Complete, consistent with the deferred-by-design decision. |

**Note on TRACK-10:** REQUIREMENTS.md marks TRACK-10 as `[x]` Complete and maps it to Phase 83. The actual outcome is documented deferral with a code comment. This is intentional: CONTEXT.md explicitly decided "Deferred to Phase 84 evaluation" and the plan's `must_haves` listed the truth as "TRACK-10 BYTE-style pass is explicitly deferred with documentation (not implemented)". The deferred status is the intended outcome, not a gap. The conditional nature of the requirement ("implement if INV-04 reveals...") further supports this treatment.

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|---------|--------|
| None | — | — | No TODOs, stubs, empty returns, or placeholder patterns detected |

### Human Verification Required

#### 1. End-to-End Pipeline Run with `tracker_kind='keypoint_bidi'`

**Test:** Set `tracking.tracker_kind: keypoint_bidi` in the YH project config and run the pipeline on the benchmark clip.
**Expected:** Tracks produced per camera with fewer fragments than OC-SORT baseline. Phase 80 baseline showed 27 tracks vs 9 expected fish; the new tracker should reduce this. `tracks_2d` output contains `Tracklet2D` objects with valid frame sequences.
**Why human:** Requires GPU, real video data (`~/aquapose/projects/YH/`), and qualitative/quantitative comparison of fragmentation metrics against baseline.

#### 2. Chunk Handoff Continuity Across Real Chunks

**Test:** Run a multi-chunk pipeline invocation and verify track IDs are stable across chunk boundaries.
**Expected:** Tracks that are active at end of chunk N continue in chunk N+1 with the same global identity.
**Why human:** The `from_state()` path reconstructs KF states from JSON; real data may expose edge cases (empty chunks, all tracks coasted, etc.) not covered by unit tests.

### Gaps Summary

No gaps found. All 9 observable truths are verified, all 5 required artifacts exist at sufficient line counts with substantive implementations, all 6 key links are wired, and all 10 TRACK requirements are addressed (9 implemented, 1 explicitly deferred with documentation per the pre-decided project decision in CONTEXT.md).

Tests pass: 1179 passed, 3 skipped, 14 deselected (no regressions).

---

_Verified: 2026-03-10_
_Verifier: Claude (gsd-verifier)_
