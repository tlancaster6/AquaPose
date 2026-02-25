---
phase: quick-8
verified: 2026-02-24T23:00:00Z
status: gaps_found
score: 4/5 must-haves verified
gaps:
  - truth: "Velocity prediction accuracy improves for tracks with noisy per-frame positions"
    status: partial
    reason: "A 6th test added in commit 0cf5295 (test_near_claim_penalty_suppresses_ghost) is currently failing. The test asserts 2 confirmed tracks but gets 3. This test was bundled with the windowed velocity test commit but tests ghost suppression behaviour, not velocity smoothing. The SUMMARY claimed all pre-existing tests pass, which is inaccurate — this new test fails."
    artifacts:
      - path: "tests/unit/tracking/test_tracker.py"
        issue: "test_near_claim_penalty_suppresses_ghost fails: AssertionError: Expected 2 confirmed tracks, got 3 (line 815)"
    missing:
      - "Fix test_near_claim_penalty_suppresses_ghost to either: (a) fix the underlying ghost suppression so only 2 confirmed tracks emerge, or (b) move the test to the correct task scope and mark it as a known limitation"
---

# Quick Task 8: Windowed Velocity Smoothing Verification Report

**Task Goal:** Windowed velocity smoothing for tracking priors — FishTrack.velocity reflects a smoothed mean of recent frame-to-frame position deltas (default window=5) rather than a single noisy delta.
**Verified:** 2026-02-24T23:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | FishTrack.velocity reflects a smoothed average of recent frame-to-frame deltas, not just the last single-frame delta | VERIFIED | `update_from_claim` appends raw delta to `velocity_history` deque then sets `self.velocity = np.mean(list(self.velocity_history), axis=0)` (tracker.py lines 223-229) |
| 2 | Velocity prediction accuracy improves for tracks with noisy per-frame positions | PARTIAL | The ring buffer averaging logic is correct and 5 windowed velocity tests pass, but `test_near_claim_penalty_suppresses_ghost` (added in the same commit) fails with AssertionError — 3 confirmed tracks instead of 2 |
| 3 | Coasting prediction still works correctly with smoothed velocity | VERIFIED | `mark_missed()` is unchanged and consumes `self.velocity` directly. `test_windowed_velocity_coasting_uses_smoothed` passes with damping=1.0, confirming coasting advances by the smoothed velocity |
| 4 | Single-view update_position_only still freezes velocity (no smoothing applied) | VERIFIED | `update_position_only` does not touch `velocity_history` or `self.velocity`. `test_windowed_velocity_single_view_does_not_update_history` passes: velocity frozen and `len(velocity_history)==1` after single-view call |
| 5 | Existing tracker behavior is unchanged when velocity_window=1 (backward compatible) | VERIFIED | `test_windowed_velocity_window_1_matches_raw_delta` passes: window=1 deque holds only the last delta, producing identical output to the pre-change single-delta behaviour |

**Score:** 4/5 truths verified (Truth 2 is partial due to failing test)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/tracking/tracker.py` | Windowed velocity smoothing in FishTrack | VERIFIED | File exists, substantive (729 lines), contains `velocity_history`, `DEFAULT_VELOCITY_WINDOW`, `velocity_window`, `__post_init__` initialization, and mean-averaging in `update_from_claim`. Wired: `velocity_window` passed from `FishTracker.__init__` through `_create_track` to `FishTrack` constructor |
| `tests/unit/tracking/test_tracker.py` | Tests for windowed velocity smoothing | PARTIAL | File exists, substantive (847 lines), contains all 5 required windowed tests (`test_windowed_velocity_*`). All 5 pass. However, a 6th test committed alongside them (`test_near_claim_penalty_suppresses_ghost`) currently fails |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/aquapose/tracking/tracker.py` | `FishTrack.update_from_claim` | `velocity_history` ring buffer averaging | WIRED | Lines 223-229: `delta` computed, appended to `self.velocity_history`, `self.velocity` set to `np.mean(list(self.velocity_history), axis=0)` |
| `src/aquapose/tracking/tracker.py` | `FishTrack.predict` | `self.velocity` (now smoothed) | WIRED | `predict()` returns `last_pos + self.velocity` (line 200). `self.velocity` is the smoothed value. `mark_missed()` also uses `self.velocity * self.velocity_damping` (lines 319, 327). No changes needed to consumers |

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| QUICK-8 | Windowed velocity smoothing for tracking priors | SATISFIED (partial) | Core smoothing is fully implemented and tested. One incidentally-bundled ghost-suppression test fails |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/unit/tracking/test_tracker.py` | 785-846 | `test_near_claim_penalty_suppresses_ghost` — fails with AssertionError | Blocker | Test suite reports 1 failed test. This was committed in the task's test commit but tests behaviour outside the scope of windowed velocity smoothing |

### Human Verification Required

None — all relevant behaviour is mechanically verifiable.

### Gaps Summary

The windowed velocity smoothing implementation is complete and correct. All five specified windowed velocity tests pass. The ring buffer, `__post_init__` initialization, `velocity_window` parameter threading, and mean-averaging are all properly implemented and wired.

The single gap is a failing test (`test_near_claim_penalty_suppresses_ghost`) that was bundled into the task's test commit (`0cf5295`) but is not related to windowed velocity smoothing — it tests ghost birth suppression. The test asserts that after establishing 2 confirmed tracks and running one more frame, exactly 2 confirmed tracks remain. The tracker currently produces 3, meaning either the ghost suppression logic is not working as intended or the test expectation is wrong. This failure was not disclosed in the SUMMARY (which claimed "All 17 pre-existing tracker tests: PASS / All 5 new windowed velocity tests: PASS"), and the commit message for `0cf5295` does not mention this test.

Resolution options:
1. Fix the ghost birth suppression so the tracker holds at 2 confirmed tracks in that scenario.
2. Remove or rewrite the test to match actual expected tracker behaviour.

---

_Verified: 2026-02-24T23:00:00Z_
_Verifier: Claude (gsd-verifier)_
