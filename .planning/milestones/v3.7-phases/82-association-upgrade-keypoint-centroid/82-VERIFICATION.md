---
phase: 82-association-upgrade-keypoint-centroid
verified: 2026-03-11T00:00:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
---

# Phase 82: Association Upgrade — Keypoint Centroid Verification Report

**Phase Goal:** Cross-view association uses the mid-body keypoint position instead of the OBB centroid, making ray-based matching more stable under partial occlusion and frame-edge clipping

**Verified:** 2026-03-11
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md success criteria)

| #   | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1 | `Tracklet2D.centroids` is populated from the selected mid-body keypoint (spine1, index 2) rather than OBB center | VERIFIED | `_TrackletBuilder.add_frame()` in `ocsort_wrapper.py` lines 93-101: when `status == "detected"` and detection has a keypoint at `centroid_keypoint_index` with `conf >= centroid_confidence_floor`, reads `kpts[idx, 0/1]` and assigns to `cx`, `cy`, overriding the OBB center. Fallback to OBB center preserved in all other cases. |
| 2 | The association stage runs end-to-end without modification to the downstream LUT/ray-ray scoring/Leiden clustering machinery | VERIFIED | Commit `05997c8` changed only `config.py`, `ocsort_wrapper.py`, `stage.py`, and `pipeline.py`. No association stage files touched. Git diff of association directory against phase commits shows no changes. |
| 3 | A brief note documents which keypoint index was selected and why (confidence statistics) | VERIFIED | `82-NOTES.md` exists, documents spine1 (index 2) selection, frame-edge clipping stability rationale, confidence statistics from Phase 78.1 production model (mAP50-95=0.974), fallback behavior, and YAML configuration examples. |
| 4 | Both centroid parameters are YAML-tunable via AssociationConfig | VERIFIED | `config.py` lines 168-169: `centroid_keypoint_index: int = 2` and `centroid_confidence_floor: float = 0.3` added to `AssociationConfig` dataclass with full docstring coverage. `load_config()` passes these through `_filter_fields(AssociationConfig, assoc_kwargs)` so they are YAML-tunable under `association:` key. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `src/aquapose/engine/config.py` | `AssociationConfig` with `centroid_keypoint_index` and `centroid_confidence_floor` fields | VERIFIED | Both fields present at lines 168-169 with defaults 2 and 0.3. Full docstring attributes section documents both fields. |
| `src/aquapose/core/tracking/ocsort_wrapper.py` | Keypoint centroid extraction in `_TrackletBuilder.add_frame()` | VERIFIED | `add_frame()` signature extended with `detection`, `centroid_keypoint_index`, `centroid_confidence_floor` params. Keypoint extraction logic present at lines 93-101. `keypoint_centroid_count` field added to dataclass. Debug logging in `to_tracklet2d()`. |
| `src/aquapose/core/tracking/stage.py` | Config threading from `AssociationConfig` to `OcSortTracker` | VERIFIED | `TrackingStage.__init__()` accepts `centroid_keypoint_index=2` and `centroid_confidence_floor=0.3` params (lines 53-54). Both stored as `self._centroid_keypoint_index` and `self._centroid_confidence_floor`. Passed to `OcSortTracker()` at lines 111-112. |
| `src/aquapose/engine/pipeline.py` | `TrackingStage` constructed with association config centroid fields | VERIFIED | Lines 331-335: `TrackingStage(config=config.tracking, centroid_keypoint_index=config.association.centroid_keypoint_index, centroid_confidence_floor=config.association.centroid_confidence_floor)` |
| `tests/unit/tracking/test_ocsort_wrapper.py` | `TestKeypointCentroid` class with 6 new tests | VERIFIED | 6 tests present: high-confidence override, low-confidence fallback, `None` keypoints fallback, coasted frame, custom index, constructor params. `_FakeDet` extended with `keypoints` and `keypoint_conf` optional fields. |
| `tests/unit/core/tracking/test_tracking_stage.py` | `TestCentroidConfigThreading` class with 2 new tests | VERIFIED | 2 tests present: passes custom centroid params, verifies defaults (index=2, floor=0.3). |
| `.planning/phases/82-association-upgrade-keypoint-centroid/82-NOTES.md` | Keypoint selection rationale document | VERIFIED | File exists with spine1 rationale, confidence statistics, fallback behavior, and YAML configuration examples. |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `config.py` (AssociationConfig) | `stage.py` (TrackingStage) | `pipeline.py` reads `config.association.centroid_keypoint_index` and passes to `TrackingStage` | WIRED | `pipeline.py` lines 333-334 read `config.association.centroid_keypoint_index` and `centroid_confidence_floor` directly. |
| `stage.py` (TrackingStage) | `ocsort_wrapper.py` (OcSortTracker) | `OcSortTracker` constructor receives centroid config | WIRED | `stage.py` lines 111-112 pass `centroid_keypoint_index=self._centroid_keypoint_index` and `centroid_confidence_floor=self._centroid_confidence_floor` to `OcSortTracker()`. |
| `ocsort_wrapper.py` (OcSortTracker) | `Detection.keypoints` | `_TrackletBuilder.add_frame` reads `detection.keypoints[idx]` | WIRED | `ocsort_wrapper.py` lines 94-101: `getattr(detection, "keypoints", None)` and `getattr(detection, "keypoint_conf", None)`, index access via `kpts[idx, 0]` and `kpts[idx, 1]`. |
| `ocsort_wrapper.py` (OcSortTracker.update) | `_TrackletBuilder.add_frame` | Detection recovered via boxmot column 7 index | WIRED | Lines 267-278: `det_idx = int(row[7])`, `source_det = detections[det_idx] if 0 <= det_idx < len(detections) else None`, passed as `detection=source_det`. |
| `ocsort_wrapper.py` (get_state/from_state) | centroid config preservation | `get_state()` serializes, `from_state()` restores with `.get()` defaults | WIRED | `get_state()` lines 353-354 include both centroid fields. `from_state()` lines 375-378 use `.get()` with defaults for backward compat. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| ASSOC-01 | 82-01-PLAN.md | Cross-view association uses mid-body keypoint centroid instead of OBB centroid for ray-based matching | SATISFIED | Spine1 (index 2) keypoint position replaces OBB centroid in `Tracklet2D.centroids` via `_TrackletBuilder.add_frame()`. Config threaded from `AssociationConfig` through pipeline to tracker. REQUIREMENTS.md marks ASSOC-01 as Complete / Phase 82. |

### Anti-Patterns Found

None. Scanned `config.py`, `ocsort_wrapper.py`, `stage.py`, and `pipeline.py` for TODO/FIXME/placeholder patterns — no issues found.

### Human Verification Required

None. All goal conditions are verifiable through code inspection.

The one behavioral aspect that cannot be verified statically — whether keypoint centroids actually improve cross-view association quality under real occlusion — is an integration/runtime question outside the scope of this phase. The phase goal is the implementation of the mechanism, not empirical validation of the quality improvement.

### Commits Verified

All three commits from SUMMARY.md confirmed present in git history:

- `cf1e0e3` — test(82-01): add failing tests for keypoint centroid extraction
- `05997c8` — feat(82-01): implement keypoint centroid extraction in tracker
- `4f9eea3` — docs(82-01): add keypoint selection rationale notes

### Summary

Phase 82 goal is fully achieved. The keypoint centroid mechanism is implemented end-to-end:

1. `AssociationConfig` exposes `centroid_keypoint_index=2` (spine1) and `centroid_confidence_floor=0.3` as YAML-tunable fields.
2. The config is threaded through `pipeline.py` → `TrackingStage` → `OcSortTracker` → `_TrackletBuilder.add_frame()` with no gaps.
3. `_TrackletBuilder.add_frame()` reads the keypoint position from `detection.keypoints[idx]` when detection has a confident keypoint, silently falling back to OBB centroid for coasted frames, missing keypoints, and low-confidence detections.
4. `get_state()`/`from_state()` preserve centroid config across chunk boundaries with backward-compatible defaults.
5. 6 new unit tests cover all keypoint centroid cases; 2 new tests cover config threading through `TrackingStage`.
6. The association stage, LUT generation, ray-ray scoring, and Leiden clustering are completely untouched.
7. `82-NOTES.md` documents spine1 selection rationale with confidence statistics and YAML configuration examples.

---

_Verified: 2026-03-11_
_Verifier: Claude (gsd-verifier)_
