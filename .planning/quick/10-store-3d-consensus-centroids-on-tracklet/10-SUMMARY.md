---
phase: 10-store-3d-consensus-centroids-on-tracklet
plan: "01"
subsystem: core/association, engine/diagnostic_observer
tags: [tracklet, consensus, calibration, npz-export, diagnostics]
dependency_graph:
  requires: []
  provides: [TrackletGroup.consensus_centroids, DiagnosticObserver.export_centroid_correspondences]
  affects: [refinement pipeline, calibration fine-tuning workflow]
tech_stack:
  added: []
  patterns: [frozen-dataclass-field-extension, type-ignore-for-import-boundary, pytest-tmp_path]
key_files:
  created: []
  modified:
    - src/aquapose/core/association/types.py
    - src/aquapose/core/association/refinement.py
    - src/aquapose/engine/diagnostic_observer.py
    - tests/unit/core/association/test_refinement.py
    - tests/unit/engine/test_diagnostic_observer.py
decisions:
  - "consensus_centroids stored as tuple of (frame_idx, ndarray|None) pairs, matching existing tuple-field style in TrackletGroup"
  - "Used type: ignore comments in diagnostic_observer.py for Tracklet2D attribute access, preserving core/ import boundary without TYPE_CHECKING gymnastics"
  - "Used pytest tmp_path fixture (not tempfile.TemporaryDirectory) in tests to avoid Windows file-locking on NPZ files"
metrics:
  duration: "~12 min"
  completed: "2026-02-28T21:02:43Z"
  tasks_completed: 2
  files_modified: 5
---

# Quick Task 10: Store 3D Consensus Centroids on TrackletGroup — Summary

**One-liner:** Per-frame 3D consensus centroids from RANSAC triangulation are now stored on TrackletGroup and exportable as structured NPZ files for calibration fine-tuning.

## What Was Built

### Task 1: `consensus_centroids` field on TrackletGroup

Added `consensus_centroids: tuple | None = None` to the frozen `TrackletGroup` dataclass. The field stores per-frame 3D consensus points as `tuple[tuple[int, np.ndarray | None], ...]` — one `(frame_idx, point_3d)` pair per frame in the union frame range.

**Population logic in `refinement.py`:**
- Refined groups: `consensus_centroids=tuple((f, cleaned_consensus.get(f)) for f in frame_list)` — uses the already-computed `cleaned_consensus` dict from the re-triangulation step
- Evicted singletons (initial creation): `consensus_centroids=None`
- Evicted singletons (ID reassignment): carries through `singleton.consensus_centroids` (which is None)
- Groups below `min_cameras_refine`: returned unchanged (field stays None from construction)
- `refinement_enabled=False`: groups returned unchanged (field stays None)

**Files modified:**
- `src/aquapose/core/association/types.py` — new field with full docstring
- `src/aquapose/core/association/refinement.py` — populate at all 3 construction sites

### Task 2: `export_centroid_correspondences` on DiagnosticObserver

Added `export_centroid_correspondences(output_path: Path | str) -> Path` to `DiagnosticObserver`. The method:

1. Looks up the `AssociationStage` snapshot; raises `ValueError` if missing
2. Iterates all `TrackletGroup` objects with non-None `consensus_centroids`
3. For each `(frame_idx, point_3d)` with a valid 3D point, finds contributing tracklets that have the frame and records one row per camera observation
4. Saves a compressed NPZ with arrays: `fish_ids` (int64, N), `frame_indices` (int64, N), `points_3d` (float64, N×3), `camera_ids` (object, N), `centroids_2d` (float64, N×2)
5. Returns the resolved absolute path

**Files modified:**
- `src/aquapose/engine/diagnostic_observer.py` — new method + `Path` and `numpy` imports

## Tests Added

**`tests/unit/core/association/test_refinement.py`** — 4 new tests in `TestConsensusCentroids`:
- `test_consensus_centroids_populated_after_refinement` — verifies tuple of (int, ndarray) pairs, shape (3,)
- `test_consensus_centroids_none_for_skipped_groups` — groups below min_cameras stay None
- `test_consensus_centroids_none_for_evicted_singletons` — evicted singletons have None
- `test_consensus_centroids_none_when_disabled` — refinement_enabled=False leaves None

**`tests/unit/engine/test_diagnostic_observer.py`** — 3 new tests:
- `test_export_centroid_correspondences_writes_npz` — NPZ exists, correct arrays, shapes, values
- `test_export_centroid_correspondences_raises_without_association` — ValueError on missing snapshot
- `test_export_centroid_correspondences_skips_none_consensus` — zero rows, not error

**Test results:** 589 passed, 0 failed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Used pytest `tmp_path` fixture instead of `tempfile.TemporaryDirectory`**
- **Found during:** Task 2 test execution
- **Issue:** `np.savez_compressed` holds an OS file handle on Windows; `TemporaryDirectory.__exit__` fails with `PermissionError (WinError 32)` when trying to delete the NPZ file during cleanup
- **Fix:** Changed all 3 new test functions to accept `tmp_path: Path` pytest fixture, which defers temp directory deletion to after the test session when file handles are released
- **Files modified:** `tests/unit/engine/test_diagnostic_observer.py`

**2. [Rule 2 - Missing type safety] Added `type: ignore` comments for Tracklet2D attribute access in DiagnosticObserver**
- **Found during:** Task 2 typecheck
- **Issue:** `group.tracklets` is a generic `tuple` (import boundary constraint); basedpyright inferred element type as `object`, causing `Cannot access attribute "camera_id"` error at line 204
- **Fix:** Added `# type: ignore[union-attr]` to tracklet loop and attribute accesses, consistent with the same pattern used in `refinement.py` (explicit cast via type annotation on line 86)
- **Files modified:** `src/aquapose/engine/diagnostic_observer.py`

### Out-of-Scope Issues (Deferred)

Pre-existing issues in files NOT modified by this task:
- `src/aquapose/engine/overlay_observer.py`: ruff I001 import-sort error (pre-existing from previous session work — visible in git diff before this task)
- Multiple basedpyright errors across `stage.py`, `observer_factory.py`, `reconstruction/midline.py`, etc. — all pre-existing

## Self-Check

Files created:
- `.planning/quick/10-store-3d-consensus-centroids-on-tracklet/10-SUMMARY.md` — this file

Files modified (all exist):
- `src/aquapose/core/association/types.py` — FOUND
- `src/aquapose/core/association/refinement.py` — FOUND
- `src/aquapose/engine/diagnostic_observer.py` — FOUND
- `tests/unit/core/association/test_refinement.py` — FOUND
- `tests/unit/engine/test_diagnostic_observer.py` — FOUND

Commits:
- `454c9db` feat(10-01): add consensus_centroids field to TrackletGroup and populate in refinement — FOUND
- `21f2827` feat(10-01): add export_centroid_correspondences to DiagnosticObserver — FOUND

## Self-Check: PASSED
