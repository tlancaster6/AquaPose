---
phase: 10-store-3d-consensus-centroids-on-tracklet
verified: 2026-02-28T21:30:00Z
status: passed
score: 3/3 must-haves verified
gaps: []
human_verification: []
---

# Quick Task 10: Store 3D Consensus Centroids on TrackletGroup — Verification Report

**Task Goal:** Store 3D consensus centroids on TrackletGroup and write centroid correspondences to disk via DiagnosticObserver. Purpose: provide high-fidelity 2D-to-3D correspondences for iterative calibration fine-tuning.
**Verified:** 2026-02-28T21:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | TrackletGroup carries per-frame 3D consensus centroids after refinement | VERIFIED | `TrackletGroup.consensus_centroids: tuple \| None = None` at line 59 of `types.py`; populated at lines 172-175 of `refinement.py` using `cleaned_consensus` dict from re-triangulation |
| 2 | Groups that skip refinement (< min_cameras or disabled) have consensus_centroids=None | VERIFIED | Disabled path returns groups unchanged (line 79-80 `refinement.py`); below-threshold path appends group unchanged (line 91-94); evicted singletons explicitly set `consensus_centroids=None` (line 139) |
| 3 | DiagnosticObserver can serialize 2D-to-3D centroid correspondences to disk as NPZ | VERIFIED | `export_centroid_correspondences(output_path)` method at lines 144-236 of `diagnostic_observer.py`; writes 5 arrays (fish_ids, frame_indices, points_3d, camera_ids, centroids_2d) via `np.savez_compressed` |

**Score:** 3/3 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/core/association/types.py` | TrackletGroup.consensus_centroids field | VERIFIED | Field declared at line 59; full docstring at lines 45-53 describing tuple of (frame_idx, ndarray\|None) pairs |
| `src/aquapose/core/association/refinement.py` | Consensus centroids populated on refined TrackletGroups | VERIFIED | 3 construction sites handled: evicted singletons (line 139: None), refined groups (lines 172-175: populated), singleton ID reassignment (line 186: carried through) |
| `src/aquapose/engine/diagnostic_observer.py` | NPZ export of 2D-3D correspondences | VERIFIED | Method `export_centroid_correspondences` at lines 144-236; substantive implementation with iteration logic, array construction, and `np.savez_compressed` call |
| `tests/unit/core/association/test_refinement.py` | Tests for consensus_centroids on TrackletGroup | VERIFIED | `TestConsensusCentroids` class with 4 tests at lines 381-466 covering: populated after refinement, None for skipped groups, None for evicted singletons, None when disabled |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `refinement.py` | `types.py` | TrackletGroup constructor with consensus_centroids kwarg | VERIFIED | Pattern `consensus_centroids=` found at lines 139, 172-175, 186 — all three construction sites wired |
| `diagnostic_observer.py` | `types.py` | reads TrackletGroup.consensus_centroids + tracklets for export | VERIFIED | Pattern `consensus_centroids` found at lines 182, 198; `group.tracklets` iterated at line 191 to build per-camera 2D centroids; fully connected |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CENTROID-01 | 10-PLAN.md | TrackletGroup stores per-frame 3D consensus centroids from refinement | SATISFIED | Field added, populated in all 3 refinement construction sites, default None for skip/eviction paths |
| CENTROID-02 | 10-PLAN.md | DiagnosticObserver.export_centroid_correspondences() writes structured NPZ with 2D-3D pairs | SATISFIED | Method implemented; raises ValueError on missing snapshot; skips None groups; writes 5 arrays in correct shapes |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

No TODO/FIXME/placeholder comments, empty implementations, or stub returns found in modified files.

---

### Human Verification Required

None. All goal behaviors are verifiable through static code analysis and test execution.

---

### Test Results

589 tests passed, 0 failed across the full unit test suite (including the 4 new `TestConsensusCentroids` tests and 3 new `test_export_centroid_correspondences_*` tests).

---

### Gaps Summary

No gaps. All three observable truths are verified:

1. **Truth 1 (field and population):** `consensus_centroids` is a proper frozen-dataclass field with None default, and `refine_clusters()` populates it on refined groups using the `cleaned_consensus` dict that already existed from the re-triangulation step. All three TrackletGroup construction sites are handled correctly.

2. **Truth 2 (None for skipped paths):** Disabled refinement returns groups unchanged (field stays None). Groups below `min_cameras_refine` are appended unchanged. Evicted singletons are constructed with explicit `consensus_centroids=None`. The ID-reassignment path carries through `singleton.consensus_centroids` (which is None from the prior step). All four skip/eviction paths produce None as expected.

3. **Truth 3 (NPZ export):** `export_centroid_correspondences()` is a substantive implementation — not a stub. It iterates groups, builds per-tracklet frame-to-centroid lookups, collects one row per (frame, camera) observation where both a valid 3D point and a 2D centroid exist, assembles 5 typed numpy arrays, and saves via `np.savez_compressed`. Raises `ValueError` on missing/empty AssociationStage snapshot. Returns the resolved absolute path.

---

_Verified: 2026-02-28T21:30:00Z_
_Verifier: Claude (gsd-verifier)_
