---
phase: 18-fix-pseudo-label-pose-output
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/training/geometry.py
  - src/aquapose/training/__init__.py
  - src/aquapose/training/pseudo_label_cli.py
  - tests/unit/training/test_geometry.py
  - tests/unit/training/test_pseudo_label_cli.py
autonomous: true
requirements: [QUICK-18]

must_haves:
  truths:
    - "Pose pseudo-labels use OBB-cropped stretch-fitted images, not full-frame images"
    - "Pose keypoints are in crop space, normalized to crop dimensions"
    - "Each fish gets its own crop image; each crop includes labels for ALL fish visible in that crop"
    - "OBB output remains unchanged (full-frame images + full-frame OBB labels)"
    - "CLI accepts --crop-width and --crop-height flags (defaults 128x64)"
  artifacts:
    - path: "src/aquapose/training/geometry.py"
      provides: "affine_warp_crop and transform_keypoints functions"
      exports: ["affine_warp_crop", "transform_keypoints"]
    - path: "src/aquapose/training/pseudo_label_cli.py"
      provides: "Pose output with OBB-cropped images and crop-space keypoints"
      contains: "affine_warp_crop"
  key_links:
    - from: "src/aquapose/training/pseudo_label_cli.py"
      to: "src/aquapose/training/geometry.py"
      via: "import affine_warp_crop, transform_keypoints"
      pattern: "from aquapose.training.geometry import.*affine_warp_crop"
    - from: "src/aquapose/training/pseudo_label_cli.py"
      to: "src/aquapose/training/geometry.py"
      via: "pca_obb on crop-space keypoints for pose bbox"
      pattern: "pca_obb.*crop"
---

<objective>
Fix the pseudo-label CLI so pose output uses OBB-cropped, stretch-fitted images with crop-space keypoints, matching the format produced by scripts/build_yolo_training_data.py. Currently the CLI writes full-frame images for pose, but YOLO-pose training expects cropped images.

Purpose: Pseudo-label pose data must match the training pipeline's expected input format (OBB crops with crop-space annotations) so YOLO-pose models can be trained on pseudo-labels.
Output: Updated CLI that writes crop images to pose/images/train/ with crop-space normalized keypoints + AABB in pose/labels/train/.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/training/geometry.py
@src/aquapose/training/pseudo_label_cli.py
@src/aquapose/training/pseudo_labels.py
@src/aquapose/training/__init__.py
@tests/unit/training/test_pseudo_label_cli.py
@tests/unit/training/test_geometry.py

<interfaces>
<!-- Reference implementation in scripts/build_yolo_training_data.py (lines 173-252, 500-548) -->

From scripts/build_yolo_training_data.py (to be extracted into geometry.py):
```python
def affine_warp_crop(
    image: np.ndarray,
    obb_corners: np.ndarray,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp OBB region to axis-aligned rectangle. Returns (warped_image, affine_matrix)."""
    src = np.array([obb_corners[0], obb_corners[1], obb_corners[3]], dtype=np.float32)
    dst = np.array([[0, 0], [crop_w - 1, 0], [0, crop_h - 1]], dtype=np.float32)
    affine = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(image, affine, (crop_w, crop_h))
    return warped, affine.astype(np.float64)

def transform_keypoints(
    coords: np.ndarray,
    visible: np.ndarray,
    affine_matrix: np.ndarray,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply affine to keypoints, mark OOB as invisible. Returns (coords_out, visible_out)."""
```

From scripts/build_yolo_training_data.py (multi-fish-per-crop logic, lines 512-540):
```python
# For each crop, iterate all other fish and check if their keypoints land in the crop
for other_ann in annotations:
    kp_crop, vis_crop = transform_keypoints(other_coords, other_vis, affine_mat, crop_w, crop_h)
    if int(vis_crop.sum()) < min_visible:
        continue
    crop_obb = pca_obb(kp_crop, vis_crop, lateral_pad)
    crop_obb[:, 0] = np.clip(crop_obb[:, 0], 0, crop_w - 1)
    crop_obb[:, 1] = np.clip(crop_obb[:, 1], 0, crop_h - 1)
    x_min, y_min = crop_obb.min(axis=0)
    x_max, y_max = crop_obb.max(axis=0)
    cx = (x_min + x_max) / 2.0 / crop_w
    cy = (y_min + y_max) / 2.0 / crop_h
    bw = (x_max - x_min) / crop_w
    bh = (y_max - y_min) / crop_h
    pose_rows.append(format_pose_annotation(cx, cy, bw, bh, kp_crop, vis_crop, crop_w, crop_h))
```

From aquapose/training/geometry.py (existing exports):
```python
def pca_obb(coords, visible, lateral_pad) -> np.ndarray:  # (4,2) TL,TR,BR,BL
def extrapolate_edge_keypoints(coords, visible, img_w, img_h, lateral_pad, edge_factor) -> tuple
def format_obb_annotation(obb_corners, img_w, img_h, class_id=0) -> list[float]
def format_pose_annotation(cx, cy, w, h, keypoints, visible, crop_w, crop_h, class_id=0) -> list[float]
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Extract affine_warp_crop and transform_keypoints into training/geometry.py</name>
  <files>src/aquapose/training/geometry.py, src/aquapose/training/__init__.py, tests/unit/training/test_geometry.py</files>
  <action>
1. Add `import cv2` to `src/aquapose/training/geometry.py` (needed for affine_warp_crop).

2. Copy `affine_warp_crop` and `transform_keypoints` from `scripts/build_yolo_training_data.py` (lines 173-252) into `src/aquapose/training/geometry.py`. Preserve the exact function signatures, docstrings, and implementations. These functions are already well-documented and tested in the script.

3. Update `src/aquapose/training/__init__.py`:
   - Add `affine_warp_crop` and `transform_keypoints` to the imports from `.geometry`
   - Add both names to `__all__`

4. Add unit tests for both functions in `tests/unit/training/test_geometry.py`:
   - `test_affine_warp_crop_output_shape`: Given a 100x200 image and 4 OBB corners, output crop has shape `(crop_h, crop_w, 3)` and affine matrix has shape `(2, 3)`.
   - `test_affine_warp_crop_identity_like`: When OBB corners form an axis-aligned rectangle matching crop dims, the affine should be near-identity.
   - `test_transform_keypoints_in_bounds`: Keypoints inside the crop remain visible; coords are within [0, crop_w) x [0, crop_h).
   - `test_transform_keypoints_oob_marked_invisible`: Keypoints that land outside crop bounds after transform get visible=False.
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test tests/unit/training/test_geometry.py -x</automated>
  </verify>
  <done>affine_warp_crop and transform_keypoints are importable from aquapose.training.geometry, with passing unit tests.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor pseudo_label_cli.py to write OBB-cropped pose output with multi-fish-per-crop logic</name>
  <files>src/aquapose/training/pseudo_label_cli.py, tests/unit/training/test_pseudo_label_cli.py</files>
  <action>
**CLI changes in `pseudo_label_cli.py`:**

1. Add `--crop-width` (default 128) and `--crop-height` (default 64) CLI options to the `generate` command, matching `scripts/build_yolo_training_data.py` defaults.

2. Add imports at the top (alongside existing geometry imports via pseudo_labels):
   ```python
   from aquapose.training.geometry import (
       affine_warp_crop,
       format_pose_annotation,
       pca_obb,
       transform_keypoints,
   )
   ```

3. Pass `crop_width` and `crop_height` through to the label generation and image writing sections.

4. **Refactor the consensus pose image/label writing block** (around lines 390-410). Currently it writes full-frame images. Change to:
   - For each `(frame_idx, cam_id)` that has consensus labels, iterate each fish (each entry in `cons_image_labels[cam_id]`).
   - Each fish defines a "primary crop": use its `keypoints_2d` and `visibility` (returned in the result dict from `generate_fish_labels`) to compute `pca_obb` -> `affine_warp_crop(frame, obb_corners, crop_w, crop_h)` to get the crop image and affine matrix.
   - For each crop, collect pose annotations for ALL fish in that camera view whose keypoints are visible in the crop. For each fish (including the primary fish), call `transform_keypoints(fish_kp, fish_vis, affine_mat, crop_w, crop_h)`. Skip fish with fewer than 2 visible keypoints in crop. Compute crop-space PCA OBB for the bbox: `crop_obb = pca_obb(kp_crop, vis_crop, lateral_pad)`, clip to crop bounds, take AABB, normalize by crop dims, then `format_pose_annotation(cx, cy, bw, bh, kp_crop, vis_crop, crop_w, crop_h)`.
   - Write crop image as `{frame_idx:06d}_{cam_id}_{fish_idx:03d}.jpg` to `cons_pose_images`.
   - Write all pose annotation lines (one per visible fish in the crop) to `{frame_idx:06d}_{cam_id}_{fish_idx:03d}.txt` in `cons_pose_labels`.
   - OBB images/labels remain unchanged (full-frame).

5. **Refactor the gap pose image/label writing block** (around lines 412-434) with the same crop logic as consensus.

6. To make the multi-fish-per-crop logic work, the `cons_image_labels[cam_id]` and `gap_image_labels[cam_id]` dicts need to store per-fish `keypoints_2d` and `visibility` arrays (already present in the result dicts from `generate_fish_labels` / `generate_gap_fish_labels`). Add these to the accumulated label data:
   ```python
   cons_image_labels[cam_id]["fish_data"].append({
       "keypoints_2d": result["keypoints_2d"],
       "visibility": result["visibility"],
       "fish_id": int(fish_id),
   })
   ```

7. Note: The `generate_fish_labels` and `generate_gap_fish_labels` functions in `pseudo_labels.py` already return `keypoints_2d` and `visibility` in their result dicts, so NO changes to `pseudo_labels.py` are needed.

**Test updates in `test_pseudo_label_cli.py`:**

8. Update `test_consensus_produces_output_structure`:
   - Pose images should now be crop-sized, not full-frame. The filenames include a fish index suffix (`000000_cam0_000.jpg` instead of `000000_cam0.jpg`).
   - Add assertion that pose image files have fish-index suffix pattern.
   - Add assertion that pose label content uses crop-normalized coordinates (values should be within [0,1] but computed relative to crop_w/crop_h, not img_w/img_h).

9. Update `test_help_text` to check for `--crop-width` and `--crop-height` in help output.

10. Update `test_both_flags_together` if needed to account for new filename pattern.

11. Note: The existing test mocks `generate_fish_labels` results at a high level. The mock projection returns pixels around (450-550, 300), which are well within the 1920x1080 image. The mock results include `keypoints_2d` and `visibility`. The test needs the mock frame source to return actual numpy images (already does - `np.zeros((1080, 1920, 3), dtype=np.uint8)`) so `affine_warp_crop` can operate on them. The key change is that pose output is now per-crop, not per-frame-camera. You should NOT mock `affine_warp_crop` or `transform_keypoints` -- let them run on the real (zero) images and mock projection coordinates so the end-to-end crop flow is tested.
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test tests/unit/training/test_pseudo_label_cli.py -x</automated>
  </verify>
  <done>
- `pseudo-label generate --consensus` writes OBB-cropped images (crop_w x crop_h) to pose/images/train/ with fish-index suffixes
- Pose label files contain crop-space normalized keypoints + crop-space AABB
- Each crop includes annotations for all fish visible in that crop (multi-fish logic)
- OBB output is unchanged (full-frame images, full-frame labels)
- `--crop-width` and `--crop-height` CLI flags are accepted with defaults 128 and 64
- All existing and updated tests pass
  </done>
</task>

</tasks>

<verification>
```bash
cd /home/tlancaster6/Projects/AquaPose && hatch run test tests/unit/training/test_geometry.py tests/unit/training/test_pseudo_label_cli.py -x
cd /home/tlancaster6/Projects/AquaPose && hatch run check
```
</verification>

<success_criteria>
- `hatch run test tests/unit/training/` passes all tests
- `hatch run check` (lint + typecheck) passes
- Pose output uses cropped images with crop-space keypoints
- OBB output unchanged
- CLI accepts --crop-width and --crop-height
</success_criteria>

<output>
After completion, create `.planning/quick/18-fix-pseudo-label-pose-output-to-use-obb-/18-SUMMARY.md`
</output>
