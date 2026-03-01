---
phase: quick-12
plan: 01
subsystem: scripts
tags: [yolo, training-data, obb, pose, coco, geometry]
dependency_graph:
  requires: []
  provides: [scripts/build_yolo_training_data.py]
  affects: []
tech_stack:
  added: []
  patterns: [PCA-OBB orientation, affine warp crop, COCO keypoint parsing, Ultralytics label format]
key_files:
  created:
    - scripts/build_yolo_training_data.py
    - tests/unit/test_build_yolo_training_data.py
  modified: []
decisions:
  - "Standalone script (no aquapose imports) — operates on annotation files not live video"
  - "PCA via SVD for OBB orientation — handles curved fish correctly"
  - "Edge extrapolation overwrites first/last visible keypoint position (extend in-chain direction)"
  - "Pose dataset: one crop per annotation, bbox is full crop (0.5 0.5 1.0 1.0)"
  - "Used np.array(dtype=np.float32) over np.float32([...]) for cv2 type compatibility"
metrics:
  duration: "~20 minutes"
  completed: "2026-03-01"
  tasks_completed: 2
  files_created: 2
  files_modified: 0
---

# Quick Task 12: YOLO Training Data Builder Summary

Standalone CLI script that converts COCO-format multi-animal keypoint annotations into two Ultralytics-ready training datasets: YOLO-OBB with PCA-derived oriented bounding boxes, and YOLO-Pose with affine-warped axis-aligned crops at 64x128.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Implement core geometry utilities and COCO parser | 4a82111 | scripts/build_yolo_training_data.py |
| 1 (fix) | Fix cv2 type error in affine_warp_crop | e3ae68d | scripts/build_yolo_training_data.py |
| 2 | Implement dataset generation pipeline and tests | afe9a50 | tests/unit/test_build_yolo_training_data.py |

## What Was Built

### `scripts/build_yolo_training_data.py`

Standalone CLI tool (no aquapose imports) with:

**COCO parsing:**
- `load_coco(path)` — loads JSON, builds `image_id -> info` and `image_id -> annotations` lookup dicts
- `parse_keypoints(ann, n_keypoints=6)` — extracts `(N,2)` coords and `(N,)` bool visibility from `[x, y, v, ...]` COCO format

**Geometry:**
- `compute_arc_length(coords, visible)` — sums consecutive visible keypoint distances in chain order
- `compute_median_arc_length(annotations)` — median arc over fully-visible fish; used for lateral pad
- `pca_obb(coords, visible, lateral_pad)` — SVD-based PCA to find fish long axis; returns 4 OBB corners (TL, TR, BR, BL); handles 0/1 visible degenerate case with 20x20 fallback box
- `extrapolate_edge_keypoints(...)` — extends first/last visible keypoint toward nearest image edge when within `lateral_pad * edge_factor` pixels of boundary
- `affine_warp_crop(image, obb_corners, crop_w, crop_h)` — `cv2.getAffineTransform` mapping OBB to axis-aligned `(crop_w, crop_h)` rectangle
- `transform_keypoints(coords, visible, affine_matrix, crop_w, crop_h)` — applies 2x3 affine to keypoints; marks OOB as invisible

**Output formatters:**
- `format_obb_label(corners, img_w, img_h, class_id=0)` — Ultralytics OBB format: `class x1 y1 x2 y2 x3 y3 x4 y4` normalized to [0,1]
- `format_pose_label(cx, cy, w, h, keypoints, visible, crop_w, crop_h, class_id=0)` — Ultralytics pose format: `class cx cy w h x1 y1 v1 ...` with invisible keypoints as `0 0 0`

**Dataset generators:**
- `generate_obb_dataset(...)` — all annotations (including partial), train/val split by image, writes `obb/{images,labels}/{train,val}/` + `data.yaml`
- `generate_pose_dataset(...)` — annotations with `>= min_visible` keypoints, one crop per fish per image, writes `pose/{images,labels}/{train,val}/` + `data.yaml` with `kpt_shape: [6, 3]`

**CLI arguments:** `--annotations`, `--images-dir`, `--output-dir`, `--crop-height=64`, `--crop-width=128`, `--lateral-ratio=0.18`, `--min-visible=4`, `--val-split=0.2`, `--seed=42`, `--edge-threshold-factor=2.0`

### `tests/unit/test_build_yolo_training_data.py`

28 unit tests across 7 test classes:
- `TestParseKeypoints` (4 tests): all visible, mixed visibility, empty, partial
- `TestComputeArcLength` (4 tests): collinear, diagonal, <2 visible, none visible
- `TestPcaObb` (5 tests): 4-corner output, orientation for horizontal fish, centroid enclosure, degenerate single/zero visible
- `TestExtrapolateEdgeKeypoints` (3 tests): nose near left edge extended, mid-frame unchanged, returns copies
- `TestFormatObbLabel` (3 tests): format, normalized values, custom class
- `TestFormatPoseLabel` (3 tests): all visible, invisible as zeros, mixed
- `TestTransformKeypoints` (3 tests): identity, translation, OOB marking
- `TestIntegrationPipeline` (3 tests): OBB directory structure, Pose directory + crop sizes + label format, data.yaml content

## Verification Results

```
1. python -c "import ast; ast.parse(...)"  -> syntax OK
2. hatch run test tests/unit/test_build_yolo_training_data.py  -> 28 passed
3. python scripts/build_yolo_training_data.py --help  -> shows all 9 documented arguments
4. hatch run lint scripts/build_yolo_training_data.py tests/unit/test_build_yolo_training_data.py  -> All checks passed
5. hatch run typecheck scripts/build_yolo_training_data.py  -> 0 errors in new files
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Ruff SIM108 / F841 in initial implementation**
- **Found during:** Task 1 commit (pre-commit hook)
- **Issue:** `if/else` block should be ternary; unused `proj_perp` variable
- **Fix:** Converted to ternary `main_axis = np.array([1.0, 0.0]) if norm < 1e-9 else main_axis / norm`; removed `proj_perp` assignment
- **Files modified:** `scripts/build_yolo_training_data.py`
- **Commit:** 4a82111

**2. [Rule 1 - Bug] Ruff I001/PT018/RUF059/E741 in test file**
- **Found during:** Task 2 commit (pre-commit hook)
- **Issue:** Import block unsorted; combined assert; unused variable names; ambiguous `l` variable
- **Fix:** Reorganized imports; split asserts; prefixed unused vars with `_`; renamed `l` to `ln`
- **Files modified:** `tests/unit/test_build_yolo_training_data.py`
- **Commit:** afe9a50

**3. [Rule 1 - Bug] basedpyright type error in affine_warp_crop**
- **Found during:** Final verification (typecheck)
- **Issue:** `np.float32([...])` doesn't satisfy cv2 stub `MatLike` type — pyright saw `floating[_32Bit]` not `ndarray`
- **Fix:** Changed to `np.array([...], dtype=np.float32)` which produces proper `ndarray`
- **Files modified:** `scripts/build_yolo_training_data.py`
- **Commit:** e3ae68d

## Self-Check: PASSED

- FOUND: scripts/build_yolo_training_data.py
- FOUND: tests/unit/test_build_yolo_training_data.py
- FOUND commit: 4a82111 (feat: core implementation)
- FOUND commit: afe9a50 (test: unit tests)
- FOUND commit: e3ae68d (fix: type error)
