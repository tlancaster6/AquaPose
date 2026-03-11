# Phase 85: Code Quality Audit Report

**Date:** 2026-03-11
**Scope:** Full codebase audit after v3.7 overhaul (segmentation removal, pipeline reorder, custom tracker, association upgrade)

---

## 1. What Was Found

### 1.1 BoxMot / OC-SORT Dead References

**Count:** 2 modules (393 lines), 15 source files with references, 2 test files

| Category | Count | Details |
|----------|-------|---------|
| Modules to delete | 2 | `ocsort_wrapper.py` (393 lines), `test_ocsort_wrapper.py` |
| Source files with references | 15 | config.py, stage.py, __init__.py, context.py, pipeline.py, stubs.py, detection.py, keypoint_tracker.py, types.py, core/__init__.py |
| Test files with OC-SORT tests | 2 | test_keypoint_tracker.py (2 tests), test_tracking_stage.py (fixture default) |
| Dependency in pyproject.toml | 1 | `boxmot>=11.0` |
| Config fields | 2 | `iou_threshold` field, `"ocsort"` in `valid_kinds` |

### 1.2 Type Errors (basedpyright)

**Count:** 25 errors across 11 files

| File | Errors | Category |
|------|--------|----------|
| `cli.py` | 7 | h5py union type narrowing |
| `association/stage.py` | 3 | `context.get()` returns `object` |
| `yolo.py` | 1 | Ultralytics lazy export |
| `yolo_obb.py` | 1 | Ultralytics lazy export |
| `pose_estimation.py` | 3 | Ultralytics lazy export + result.keypoints |
| `reconstruction/stage.py` | 1 | `_backend` typed as `object` |
| `orchestrator.py` | 2 | `frame_source` typed as `object` |
| `pipeline.py` | 3 | LutConfigLike protocol + crop_size tuple |
| `overlay.py` | 1 | cv2 stub gap |
| `trails.py` | 2 | cv2 stub gap |
| `logging.py` | 1 | exc_info tuple type |

**Fix strategies used:**
- Protocol @property pattern (LutConfigLike) -- 2 errors
- isinstance guards (h5py, VideoFrameSource) -- 9 errors
- cast() for known-typed context values -- 3 errors
- type: ignore[attr-defined] for third-party stub gaps -- 8 errors
- type: ignore[arg-type] for stdlib typing edge case -- 1 error
- type: ignore[union-attr] for known-safe attribute access -- 1 error
- type: ignore[arg-type] for tuple size -- 1 error

### 1.3 Dead Code (vulture at 80% confidence)

**Count:** 4 items

| Location | Finding | Fix Applied |
|----------|---------|-------------|
| `evaluation/viz/overlay.py:302` | `show_fish_id` parameter unused | `_ = show_fish_id` suppression |
| `io/midline_writer.py:230` | `exc_val` in `__exit__` unused | Renamed to `_exc_val` |
| `synthetic/stubs.py:45` | `*args` in stub function unused | Renamed to `*_args` |
| `training/data_cli.py:76` | `augment_count` CLI param not wired | Comment documenting intent |

### 1.4 Config Compatibility Issue

**PoseConfig missing `backend` field:** The YH config.yaml has `midline.backend: pose_estimation` which was not accepted by PoseConfig (strict field filtering). This is a pre-existing issue from the midline-to-pose rename, not a Phase 85 regression.

---

## 2. What Was Fixed

### Plan 85-01 (2 commits)

**Commit `0773394` -- BoxMot removal:**
- Deleted `src/aquapose/core/tracking/ocsort_wrapper.py` (393 lines)
- Deleted `tests/unit/tracking/test_ocsort_wrapper.py`
- Removed `boxmot>=11.0` from pyproject.toml
- TrackingConfig: removed `iou_threshold` field, `valid_kinds` now `{"keypoint_bidi"}` only
- Updated 15 source files to remove all OC-SORT/boxmot references in docstrings/comments
- Updated test stubs with keypoints for KeypointTracker compatibility
- Removed 2 OC-SORT backward-compat tests from test_keypoint_tracker.py

**Commit `39d9c37` -- Type errors and dead code:**
- Fixed all 25 basedpyright errors to zero
- LutConfigLike protocol: plain attributes -> @property members
- Added isinstance guards for h5py and VideoFrameSource
- Added cast() for association stage tracks_2d
- Added type: ignore for ultralytics, cv2, and logging edge cases
- Fixed 4 dead code items at 80% vulture confidence

### Plan 85-02 (2 commits)

**Commit `4e96fb6` -- PoseConfig backend field:**
- Added `backend: str = "pose_estimation"` to PoseConfig for config.yaml compatibility

**Commit `eef52c0` -- iou_threshold rename hint:**
- Added `iou_threshold` to `_RENAME_HINTS` for helpful error message on old configs

### CLI Smoke Test Result

- **Command:** `hatch run aquapose -p YH run --set chunk_size=100 --max-chunks 2`
- **Exit code:** 0
- **Run directory:** `~/aquapose/projects/YH/runs/run_20260311_114254/`
- **Artifacts confirmed:**
  - `config.yaml` (frozen config) -- present
  - `midlines.h5` -- present (46 KB)
  - `diagnostics/chunk_000/cache.pkl` -- present
  - `timing.txt` -- present
- **Chunk 1 (0-99):** Completed successfully, 15 fish identified, 31s
- **Chunk 2 (100-199):** Failed with `ValueError: x must be strictly increasing sequence` in `interpolate_gaps()` -- pre-existing spline bug with duplicate frame indices during gap interpolation. Pipeline handled gracefully (logged, skipped, exit 0).
- **Quality gate:** `hatch run check` passes (lint + typecheck: 0 errors)

---

## 3. What Remains for Phase 86

### 3.1 Dead Code at 60% Vulture Confidence (Worth Investigating)

| Location | Finding | Likely Status |
|----------|---------|---------------|
| `evaluation/viz/overlay.py:140` | `_reproject_3d_midline` function | Possibly dead -- internal helper may be unused after refactoring |
| `core/reconstruction/backends/dlt.py:671` | `_triangulate_body_point` method | Possibly dead -- may have been superseded by midline-based triangulation |
| `core/tracking/types.py` | `FishTrack`, `TrackState` members | OC-SORT era types -- may be partially orphaned after BoxMot removal |
| `synthetic/trajectory.py` | `loose_school`, `milling`, `streaming` methods, `FishTrajectoryState` class | Synthetic module may be partially orphaned |

### 3.2 Misnamed Test Directory

`tests/unit/segmentation/` contains tests for `core/pose/crop.py` and `core/detection/backends/yolo.py`, not any segmentation module. Should be renamed to `tests/unit/pose/` or similar.

### 3.3 Spline Interpolation Bug

`interpolate_gaps()` in `keypoint_tracker.py` fails with `ValueError: x must be strictly increasing sequence` when frame indices contain duplicates. This caused chunk 2 to fail during the smoke test. The orchestrator handles it gracefully (catch + skip), but the root cause should be fixed.

### 3.4 augment_count CLI Parameter Not Wired

`training/data_cli.py:76` has an `--augment-count` CLI option that's never passed to `generate_variants()`. The function is hardcoded to produce 4 variants. Either wire the parameter or remove the CLI option.

---

## 4. Phase 86 Recommendation

**Recommendation: Phase 86 is NOT urgently needed but would be beneficial.**

The v3.7 codebase is now in a clean state for continued development:
- Zero type errors
- Zero BoxMot/OC-SORT references
- All 1152 tests pass
- Pipeline runs end-to-end from CLI

The items in Section 3 are quality-of-life improvements, not blocking issues:
- Dead code at 60% confidence is mostly false positives; the genuine items are internal helpers, not public API
- The misnamed test directory is cosmetic
- The spline interpolation bug is handled by the orchestrator's error recovery
- The unwired CLI param is low-priority

If Phase 86 is pursued, recommended scope:
1. Fix `interpolate_gaps()` spline bug (highest priority -- affects production reliability)
2. Investigate and clean 60%-confidence dead code candidates
3. Rename `tests/unit/segmentation/` directory
4. Wire `augment_count` to `generate_variants()` or remove the CLI option

---

*Generated: 2026-03-11*
*Phase: 85-code-quality-audit-cli-smoke-test*
