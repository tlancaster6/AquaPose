---
phase: 84-integration-evaluation
verified: 2026-03-11T10:30:00Z
status: human_needed
score: 5/5 must-haves verified
human_verification:
  - test: "Review 84-EVALUATION.md metrics table and interpretation paragraph"
    expected: "Side-by-side OC-SORT vs keypoint_bidi columns, both with valid numbers, plus an interpretation that makes sense given the results"
    why_human: "Document exists and has content, but correctness and clarity of the interpretation requires human judgment"
  - test: "Watch eval_tracker_output/keypoint_bidi_tracking.mp4 annotated video"
    expected: "Track IDs are stable (no two tracks share the same ID in the same frame), colors are consistent across frames, tracks are plausible"
    why_human: "Visual quality and absence of duplicate IDs in the rendered video cannot be verified programmatically"
---

# Phase 84: Integration Evaluation Verification Report

**Phase Goal:** The new tracker is wired into the reordered pipeline and evaluated against the Phase 80 baselines, with iteration on parameters if needed
**Verified:** 2026-03-11T10:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                              | Status     | Evidence                                                                                                           |
| --- | -------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------ |
| 1   | TrackingStage dispatches to KeypointTracker when tracker_kind='keypoint_bidi' and to OcSortTracker when tracker_kind='ocsort' | ✓ VERIFIED | `stage.py` lines 107–146: explicit `if tracker_kind == "keypoint_bidi"` / `else` with correct construction in both branches |
| 2   | TrackingConfig accepts keypoint_bidi fields (base_r, lambda_ocm, max_gap_frames) with backward-compatible defaults | ✓ VERIFIED | `config.py` lines 205–213: all three fields present with defaults 10.0 / 0.2 / 5; `__post_init__` validates tracker_kind |
| 3   | A standalone evaluation script runs the custom tracker on the same clip/frames as Phase 80 baseline (e3v83eb, frames 3300-4500) | ✓ VERIFIED | `scripts/evaluate_custom_tracker.py` (906 lines); `comparison_metrics.json` present with config block showing camera=e3v83eb, start_frame=3300, end_frame=4500 |
| 4   | Side-by-side comparison of OC-SORT vs keypoint_bidi metrics exists in a markdown evaluation document | ✓ VERIFIED | `84-EVALUATION.md` contains a 4-column table (Ph80 baseline, OC-SORT this run, keypoint_bidi bidi, keypoint_bidi single-pass) with 10 metric rows |
| 5   | tracker_kind default is 'keypoint_bidi' in TrackingConfig                                         | ✓ VERIFIED | `config.py` line 205: `tracker_kind: str = "keypoint_bidi"`                                                       |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact                                                              | Expected                                             | Status     | Details                                                                                        |
| --------------------------------------------------------------------- | ---------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------- |
| `src/aquapose/core/tracking/stage.py`                                 | TrackingStage with keypoint_bidi dispatch            | ✓ VERIFIED | 181 lines; imports `KeypointTracker` and `OcSortTracker` under conditional; both branches fully implemented |
| `src/aquapose/engine/config.py`                                       | Extended TrackingConfig with keypoint_bidi fields    | ✓ VERIFIED | `TrackingConfig` has `base_r`, `lambda_ocm`, `max_gap_frames`; `__post_init__` validates kind  |
| `scripts/evaluate_custom_tracker.py`                                  | Standalone evaluation script (min 100 lines)         | ✓ VERIFIED | 906 lines; runs both trackers on same cached detections; uses `evaluate_tracking` and `evaluate_fragmentation_2d` |
| `.planning/phases/84-integration-evaluation/84-EVALUATION.md`        | Side-by-side metrics comparison with interpretation  | ✓ VERIFIED | Present; contains full comparison table, interpretation section, BYTE trigger assessment, and parameter tuning analysis |

---

### Key Link Verification

| From                        | To                                   | Via                                                  | Status     | Details                                                                                                  |
| --------------------------- | ------------------------------------ | ---------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------- |
| `TrackingStage.run()`       | `KeypointTracker`                    | `tracker_kind='keypoint_bidi'` conditional dispatch  | ✓ WIRED    | `stage.py` line 108: `from aquapose.core.tracking.keypoint_tracker import KeypointTracker`; lines 113–127 construct/restore it |
| `TrackingConfig`            | `TrackingStage`                      | Config fields threaded to KeypointTracker constructor | ✓ WIRED    | `stage.py` lines 119–124: `base_r=self._config.base_r`, `lambda_ocm=self._config.lambda_ocm`, `max_gap_frames=self._config.max_gap_frames` all threaded correctly |
| `evaluate_custom_tracker.py` | `evaluate_tracking + evaluate_fragmentation_2d` | Same metrics functions as Phase 80 baseline script  | ✓ WIRED    | Lines 440–441: both functions imported; lines 537–538 and 557–558: called for both trackers; results in comparison_metrics.json |

---

### Requirements Coverage

| Requirement | Source Plans | Description                                                               | Status       | Evidence                                                                                                     |
| ----------- | ------------ | ------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------ |
| INTEG-01    | 84-01, 84-02 | New tracker evaluated against INV-03 baseline metrics on perfect-tracking target | ✓ SATISFIED  | `84-EVALUATION.md` has Phase 80 baseline numbers; `comparison_metrics.json` has actual run results; side-by-side table present |
| INTEG-02    | 84-01        | Full pipeline runs end-to-end from CLI with new stage ordering            | ✓ SATISFIED  | TrackingStage with keypoint_bidi default wired end-to-end; evaluation script ran 1200 frames without errors per SUMMARY; both tracker backends operational |

No orphaned requirements: REQUIREMENTS.md maps both INTEG-01 and INTEG-02 to Phase 84 and marks them complete. INTEG-03 and INTEG-04 are assigned to Phase 85 (pending).

---

### Anti-Patterns Found

| File                                          | Lines   | Pattern                                     | Severity | Impact                                                                                                  |
| --------------------------------------------- | ------- | ------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `src/aquapose/core/tracking/stage.py`         | 8–13    | Module docstring says "Custom bidirectional keypoint tracker ... Runs forward+backward passes and merges via Hungarian OKS assignment" | ⚠️ Warning | Stale docstring — bidi merge was removed after Phase 84-01. Does not affect runtime behavior but misleads future readers. |
| `src/aquapose/core/tracking/stage.py`         | 42      | Class docstring says "an OcSortTracker" (still singular, doesn't mention keypoint_bidi) | ℹ️ Info    | Minor; the `run()` docstring at lines 80–83 correctly describes dual dispatch                            |
| `src/aquapose/core/tracking/stage.py`         | 83      | "`KeypointTracker (custom bidirectional KF)`" in `run()` docstring | ⚠️ Warning | Stale — the tracker is no longer bidirectional. Same category as the module docstring.                   |

No blockers found. No TODO/FIXME/placeholder comments in any modified file. No empty implementations or return-null stubs.

---

### Human Verification Required

#### 1. Review Evaluation Document

**Test:** Open `.planning/phases/84-integration-evaluation/84-EVALUATION.md` and read the comparison table and interpretation section.
**Expected:** The table shows four columns (Ph80 OC-SORT baseline, current OC-SORT, keypoint_bidi bidi, keypoint_bidi single-pass), all metric rows are populated with the numbers from `eval_tracker_output/comparison_metrics.json`, and the interpretation paragraph correctly identifies the root cause of over-fragmentation (occlusion reacquisition) and the BYTE trigger status.
**Why human:** Document exists and has the right structure, but whether the interpretation is accurate and actionable requires human judgment on the domain context.

#### 2. Review Annotated Video

**Test:** Watch `eval_tracker_output/keypoint_bidi_tracking.mp4`.
**Expected:** Track IDs are stable across frames, no two fish share the same track ID at the same time, track colors are consistent (same fish = same color), tracks roughly follow fish through the clip.
**Why human:** The exclusive-assignment renderer fix (Phase 84-02) is supposed to eliminate duplicate IDs, but visual verification of track quality and absence of ID conflicts cannot be confirmed programmatically.

---

### Gaps Summary

No automated gaps found. All five truths verified, all artifacts substantive and wired, both requirement IDs satisfied, tests pass (1172 passed). Two stale docstring references in `stage.py` are warnings (not blockers) and do not prevent goal achievement.

The phase goal is achieved: KeypointTracker is wired into TrackingStage as the default backend (`tracker_kind="keypoint_bidi"`), the evaluation script ran both trackers on the Phase 80 clip and produced `comparison_metrics.json`, and `84-EVALUATION.md` documents the side-by-side results with interpretation. Verification of the annotated video and evaluation document quality is delegated to the human checkpoint (Task 3 in 84-01 / Task 4 in 84-02).

---

_Verified: 2026-03-11T10:30:00Z_
_Verifier: Claude (gsd-verifier)_
