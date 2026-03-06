---
phase: quick-12
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/build_yolo_training_data.py
  - tests/unit/test_build_yolo_training_data.py
autonomous: true
requirements: [QUICK-12]

must_haves:
  truths:
    - "Script reads COCO keypoint JSON and source images, produces YOLO-OBB dataset"
    - "Script reads COCO keypoint JSON and source images, produces YOLO-Pose crop dataset"
    - "OBB labels use PCA-derived oriented bounding boxes with lateral padding from median arc length"
    - "Pose crops are affine-warped to axis-aligned landscape orientation at 64x128"
    - "Partial skeletons included in OBB (all), filtered to >=4/6 visible for Pose crops"
    - "Edge keypoints near image boundary are extrapolated toward the edge"
    - "Train/val split applied with configurable fraction and seed"
  artifacts:
    - path: "scripts/build_yolo_training_data.py"
      provides: "CLI tool for YOLO-OBB + YOLO-Pose training data generation"
    - path: "tests/unit/test_build_yolo_training_data.py"
      provides: "Unit tests for core geometry and format conversion functions"
  key_links:
    - from: "scripts/build_yolo_training_data.py"
      to: "COCO keypoint JSON"
      via: "json.load + annotation parsing"
      pattern: "keypoints.*x.*y.*v"
    - from: "scripts/build_yolo_training_data.py"
      to: "Ultralytics OBB format"
      via: "per-image txt with x1 y1 x2 y2 x3 y3 x4 y4 class"
    - from: "scripts/build_yolo_training_data.py"
      to: "Ultralytics Pose format"
      via: "per-image txt with cx cy w h + x y visible per keypoint"
---

<objective>
Create a standalone tool script that converts COCO-format multi-animal keypoint annotations into two Ultralytics-ready training datasets: (1) YOLO-OBB with PCA-derived oriented bounding boxes, and (2) YOLO-Pose with affine-warped axis-aligned crops.

Purpose: Enable training YOLO-OBB detection and YOLO-Pose keypoint models from existing COCO keypoint annotations without manual relabeling.
Output: `scripts/build_yolo_training_data.py` with CLI interface and unit tests.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@scripts/build_training_data.py (reference for CLI patterns, argparse structure)
@src/aquapose/training/pose.py (COCO keypoint format: [x, y, v, x, y, v, ...] where v=0 invisible, v=1 occluded, v=2 visible)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement core geometry utilities and COCO parser</name>
  <files>scripts/build_yolo_training_data.py</files>
  <action>
Create `scripts/build_yolo_training_data.py` with the following structure. The script is standalone (no aquapose imports) since it operates on annotation files, not live video.

**Constants and types:**
- `KEYPOINT_NAMES = ["nose", "head", "spine1", "spine2", "spine3", "tail"]` (6 keypoints, spine-only chain)
- `N_KEYPOINTS = 6`

**COCO parsing functions:**
- `load_coco(path) -> dict`: Load and validate COCO JSON. Build `image_id -> image_info` and `image_id -> list[annotation]` lookup dicts.
- `parse_keypoints(ann, n_keypoints=6) -> tuple[np.ndarray, np.ndarray]`: Extract from annotation's `keypoints` field (COCO format: [x,y,v,...]) into `(coords: (N,2), visible: (N,) bool)`. `v > 0` means visible.

**Geometry functions:**
- `compute_arc_length(coords, visible) -> float | None`: Sum Euclidean distances between consecutive VISIBLE keypoints in chain order (nose->head->spine1->spine2->spine3->tail). Return None if fewer than 2 visible in sequence.
- `compute_median_arc_length(annotations, n_keypoints=6) -> float`: Pre-pass over all annotations. For annotations with all 6/6 visible, compute arc length. Return median. Raise ValueError if no complete skeletons found.
- `pca_obb(coords, visible, lateral_pad) -> np.ndarray`: Given visible keypoint coords, compute PCA on them to get the long axis direction. Project all visible points onto the PCA axes. The OBB center is the centroid of projections. Half-lengths: along long axis = max projection extent + small epsilon (2px), across = `lateral_pad`. Return 4 corner points as `(4, 2)` array (clockwise from top-left in OBB frame). Handle degenerate case (1 visible keypoint) by using a default 20x20px box.
- `extrapolate_edge_keypoints(coords, visible, img_w, img_h, lateral_pad, edge_factor=2.0) -> tuple[np.ndarray, np.ndarray]`: Check if the first or last visible keypoint in the chain is within `lateral_pad * edge_factor` pixels of the image boundary. If so, extend the polyline from that keypoint toward the nearest edge along the chain direction. Return updated coords and visible arrays (copies, not in-place).
- `affine_warp_crop(image, obb_corners, crop_w, crop_h) -> tuple[np.ndarray, np.ndarray]`: Compute affine transform mapping OBB to axis-aligned rectangle of size `(crop_w, crop_h)`. Use `cv2.getAffineTransform` with 3 of the 4 OBB corners mapped to destination corners. Return `(warped_image, affine_matrix_2x3)`. Ensure fish is in landscape orientation (long axis = horizontal).
- `transform_keypoints(coords, visible, affine_matrix, crop_w, crop_h) -> tuple[np.ndarray, np.ndarray]`: Apply 2x3 affine matrix to keypoint coords. Clamp to [0, crop_w) x [0, crop_h). Return transformed coords and updated visibility (mark OOB as invisible).

**Output format functions:**
- `format_obb_label(obb_corners, img_w, img_h, class_id=0) -> str`: Format one OBB annotation as Ultralytics OBB format: `class x1 y1 x2 y2 x3 y3 x4 y4` with coordinates normalized to [0,1] by image dimensions.
- `format_pose_label(cx, cy, w, h, keypoints, visible, crop_w, crop_h, class_id=0) -> str`: Format one pose annotation as Ultralytics keypoint format: `class cx cy w h x1 y1 v1 x2 y2 v2 ...` with bbox and keypoints normalized to [0,1]. For invisible keypoints, output `0 0 0`. For visible, output `x y 2` (COCO visible=2 convention).

**CLI argument parser:**
- Arguments as specified in the planning context description (--annotations, --images-dir, --output-dir, --crop-height=64, --crop-width=128, --lateral-ratio=0.18, --min-visible=4, --val-split=0.2, --seed=42, --edge-threshold-factor=2.0)
- Use argparse with Path types, following the pattern in `scripts/build_training_data.py`
  </action>
  <verify>
    `python -c "import ast; ast.parse(open('scripts/build_yolo_training_data.py').read()); print('syntax OK')"` passes
  </verify>
  <done>Script file exists with all geometry functions, COCO parser, output formatters, and CLI argument parser implemented. No runtime dependencies beyond numpy, opencv-python, and stdlib.</done>
</task>

<task type="auto">
  <name>Task 2: Implement dataset generation pipeline and train/val split</name>
  <files>scripts/build_yolo_training_data.py, tests/unit/test_build_yolo_training_data.py</files>
  <action>
Add the main generation pipeline to `scripts/build_yolo_training_data.py`:

**`generate_obb_dataset(coco, images_dir, output_dir, median_arc, lateral_ratio, edge_factor, val_split, seed)`:**
1. Create `output_dir/obb/images/train/`, `output_dir/obb/images/val/`, `output_dir/obb/labels/train/`, `output_dir/obb/labels/val/`
2. Collect all (image_id, image_info) pairs. Shuffle with seed, split into train/val by `val_split` fraction.
3. For each image: load image from `images_dir`, get all annotations for that image.
4. For each annotation: parse keypoints, run `extrapolate_edge_keypoints`, compute `pca_obb` with `lateral_pad = median_arc * lateral_ratio`. Include ALL annotations (even partial — detector must learn partial fish).
5. Collect all OBB labels for the image into a single txt file (one line per annotation).
6. Copy source image to `images/train/` or `images/val/`, write label txt to `labels/train/` or `labels/val/`. Use same filename stem for image and label (Ultralytics convention).
7. Write `output_dir/obb/data.yaml` with: `path: .` (relative), `train: images/train`, `val: images/val`, `names: {0: fish}`, `nc: 1`.

**`generate_pose_dataset(coco, images_dir, output_dir, median_arc, lateral_ratio, edge_factor, crop_w, crop_h, min_visible, val_split, seed)`:**
1. Create `output_dir/pose/images/train/`, `output_dir/pose/images/val/`, `output_dir/pose/labels/train/`, `output_dir/pose/labels/val/`
2. For each image, for each annotation: parse keypoints, filter to `>= min_visible` visible keypoints.
3. Run `extrapolate_edge_keypoints`, compute `pca_obb`, `affine_warp_crop` to get axis-aligned crop.
4. Transform keypoints into crop space via `transform_keypoints`.
5. If two fish share the same OBB (overlapping crops), annotate both keypoint sets in that crop's label file. Detect overlap by checking if another annotation's centroid falls inside this OBB.
6. Format label: normalized bbox is the full crop `0.5 0.5 1.0 1.0` (since each crop IS the bounding box), plus keypoints. For overlapping fish, compute their relative bbox within the crop.
7. Save warped crop image and label txt. Naming: `{image_stem}_{ann_idx:03d}.jpg`.
8. Shuffle all crop filenames with seed, split into train/val, move to respective directories.
9. Write `output_dir/pose/data.yaml` with: `path: .`, `train: images/train`, `val: images/val`, `names: {0: fish}`, `nc: 1`, `kpt_shape: [6, 3]`.

**`main()` function:**
1. Parse CLI args
2. Load COCO JSON
3. Compute `median_arc = compute_median_arc_length(all_annotations)`
4. Print summary: total images, total annotations, median arc length, lateral pad
5. Call `generate_obb_dataset(...)` and `generate_pose_dataset(...)`
6. Print final summary: counts per split for each dataset

**Unit tests in `tests/unit/test_build_yolo_training_data.py`:**
- Test `parse_keypoints` with valid COCO annotation (6 keypoints, mix of visible/invisible)
- Test `compute_arc_length` with known coordinates (3 visible points in a line = predictable length)
- Test `pca_obb` produces 4 corners, roughly enclosing the points, correct orientation for horizontal fish
- Test `pca_obb` degenerate case (1 visible keypoint) returns a valid box
- Test `extrapolate_edge_keypoints` extends nose near left edge, does NOT extend mid-frame keypoints
- Test `format_obb_label` produces correct normalized string format
- Test `format_pose_label` produces correct format with invisible keypoints as `0 0 0`
- Test `transform_keypoints` correctly applies a known affine (e.g., 90-degree rotation)
- Tests use synthetic numpy arrays, no file I/O needed (except for integration-style tests of the full pipeline which can use `tmp_path`)
- Add one integration test: create a minimal COCO JSON (2 images, 3 annotations) in tmp_path, create dummy images, run `generate_obb_dataset` and `generate_pose_dataset`, verify output directory structure and file counts.
  </action>
  <verify>
    <automated>cd C:/Users/tucke/PycharmProjects/AquaPose && hatch run test tests/unit/test_build_yolo_training_data.py -x -v</automated>
  </verify>
  <done>Full pipeline generates both YOLO-OBB and YOLO-Pose datasets with correct Ultralytics directory structure, data.yaml files, and normalized label formats. All unit tests pass covering geometry, formatting, and end-to-end directory structure.</done>
</task>

</tasks>

<verification>
1. `python -c "import ast; ast.parse(open('scripts/build_yolo_training_data.py').read())"` -- syntax valid
2. `hatch run test tests/unit/test_build_yolo_training_data.py -x -v` -- all tests pass
3. `python scripts/build_yolo_training_data.py --help` -- shows CLI usage with all documented arguments
4. `hatch run lint` -- no ruff errors in new files
5. `hatch run typecheck` -- no type errors in new files
</verification>

<success_criteria>
- Script accepts COCO keypoint JSON and outputs two complete Ultralytics training sets
- OBB set has images/ + labels/ + data.yaml with oriented bounding box annotations
- Pose set has affine-warped crops at 64x128 with transformed keypoint labels
- PCA-based OBB orientation handles curved fish correctly
- Edge extrapolation extends polylines near boundaries
- Partial skeletons handled correctly (all for OBB, >=4/6 for Pose)
- Train/val split is reproducible with seed
- All unit tests pass
</success_criteria>

<output>
After completion, create `.planning/quick/12-add-tool-script-for-yolo-obb-and-yolo-po/12-SUMMARY.md`
</output>
