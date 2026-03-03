# Pitfalls Research

**Domain:** Adding YOLO-OBB, keypoint midline backend, training infrastructure, and config cleanup to an existing multi-view 3D fish pose estimation pipeline (v2.2 Backends milestone); then replacing custom U-Net and keypoint regression models with Ultralytics-native YOLOv8-seg and YOLOv8-pose (v3.0 Ultralytics Unification)
**Researched:** 2026-02-28 (v2.2) / 2026-03-01 (v3.0 additions)
**Confidence:** HIGH — codebase directly inspected for all integration points; OBB angle convention verified against ultralytics GitHub issues #13003, #16235 and official docs; v3.0 pitfalls cross-verified against Ultralytics GitHub issues #4796, #15380, #17116, #1970 and official task documentation.

> **Scope note:** This file covers three tiers of pitfalls. The first section ("v3.0 Ultralytics Unification Pitfalls") is new research specific to replacing custom segmentation/pose models with Ultralytics-native equivalents. The second section ("v2.2 Integration Pitfalls") covers adding OBB detection, keypoint regression, and training infrastructure. The third section ("Foundation Pitfalls") preserves the original project-wide pitfalls from v1.0/v2.0/v2.1 research.

---

## v3.0 Ultralytics Unification Pitfalls

These pitfalls are specific to replacing custom U-Net segmentation and keypoint regression models with Ultralytics YOLOv8-seg and YOLOv8-pose. The primary risk sources are: annotation format conversion from SAM2 binary masks to YOLO polygon format, small dataset training on ~150 frames, coordinate space handling between Ultralytics inference output and the existing pipeline's crop-space / frame-space conventions, and ensuring the working YOLO detection model is not broken.

---

### Pitfall B1: SAM2 Multi-Region Masks Produce Invalid YOLO Segmentation Annotations

**What goes wrong:**
SAM2 frequently generates masks with multiple disconnected regions for a single fish instance — especially for low-contrast females where the mask may fragment across the body. YOLO segmentation format requires one polygon per object per label line. When a conversion script naively calls `cv2.findContours` on a SAM2 mask with multiple connected components, it produces multiple contours. If each contour is written as a separate polygon on separate lines, Ultralytics treats them as separate object instances. If only the largest contour is kept, small but anatomically important regions (fin tips, tail) are silently lost. If multiple contours are concatenated into one polygon line, the polygon self-intersects, which causes training errors or silently produces bad loss.

**Why it happens:**
The SAM2 crop-and-box-only pseudo-label pipeline (already validated on this project) is optimized for segmentation quality, not polygon suitability. Disconnected SAM2 masks are common at occlusion boundaries and on low-contrast fish bodies. The conversion step from binary mask to YOLO polygon is typically a single `findContours` call that developers assume produces one contour.

**How to avoid:**
- After `cv2.findContours`, check `len(contours) > 1`. If multiple regions exist, apply morphological closing (dilation then erosion) to merge nearby regions before polygon extraction. Kernel size should be ~5–10% of the fish width in pixels.
- If closing fails to merge, keep only the largest contour by area. Log a warning with the frame ID and discarded area fraction. Discarded area > 10% of the fish area warrants manual review.
- Validate every converted label file: load with `ultralytics.data.utils.check_det_dataset()` or a custom check that counts polygon points per line and asserts all coordinates are in [0, 1].
- Run Ultralytics' dataset checker (`YOLO(...).val(data=yaml)`) on the converted dataset before any training to surface corrupt annotations early.

**Warning signs:**
- Training logs show "ignoring corrupt image/label" warnings with specific frame paths.
- mAP stays near zero after 10+ epochs (indicates corrupted annotations that produce no valid predictions).
- Visual inspection of label overlay images shows multiple disconnected colored regions per fish.

**Phase to address:**
Training data preparation phase (annotation conversion tooling). The mask-to-polygon conversion must be verified visually on a sample of 20+ frames before training begins.

---

### Pitfall B2: YOLO Pose Keypoints Must Be Normalized Relative to the FULL IMAGE, Not the Crop

**What goes wrong:**
The existing pseudo-label and training data pipeline operates on crops: SAM2 is run on a crop, the U-Net is trained on 128x128 crops, and keypoints from the custom regression model are in crop-space coordinates. When preparing YOLOv8-pose training labels, developers may naturally write keypoint coordinates relative to the crop image (since that's what SAM2 and U-Net used). YOLO pose label format requires ALL coordinates — bounding box center, bounding box size, AND all keypoint coordinates — to be normalized relative to the full frame image dimensions. Keypoints in crop space will fall near (0, 0) in normalized full-frame space, causing the model to learn that all fish have keypoints in the top-left corner.

**Why it happens:**
The Ultralytics pose format documentation states "normalize between 0 and 1" but does not explicitly say "relative to the full image, not a sub-region." The official issue #1970 explicitly warns that "keypoint coordinates should be relative to the global image and not the cropped image." The existing codebase's crop-centric workflow makes this the default mental model for anyone working in it.

**How to avoid:**
- Training label generation must convert keypoint coordinates from crop space to frame space BEFORE normalization: `frame_x = crop_x1 + keypoint_x_in_crop; frame_y = crop_y1 + keypoint_y_in_crop`. Then normalize: `norm_x = frame_x / frame_width; norm_y = frame_y / frame_height`.
- The bounding box in YOLO pose format is also in full-frame normalized coordinates. Use the detection bbox (already in frame space) as the bounding box.
- Add a sanity check: for every label line, verify `0 <= px_i <= 1` and `0 <= py_i <= 1` for all keypoints. Any value outside [0, 1] indicates crop-space coordinates were written without frame-space conversion.
- Run training for 2–3 epochs and visualize predictions on training images: if keypoints cluster near the top-left of the full frame, the coordinate space is wrong.

**Warning signs:**
- Training warnings: "ignoring corrupt image/label: non-normalized or out of bounds coordinate" from Ultralytics (this fires when coords are > 1).
- After training, predicted keypoints on full-frame images cluster near the image origin.
- Bounding boxes look correct but keypoints are systematically offset from the fish body.

**Phase to address:**
Training data preparation phase (annotation format conversion). Verify with a single label file overlay before building the full dataset.

---

### Pitfall B3: `kpt_shape` Missing from dataset.yaml Causes Silent Training Failure

**What goes wrong:**
YOLOv8 pose training requires a `kpt_shape: [N, 3]` entry in `dataset.yaml` specifying the number of keypoints and dimensionality (2 for xy-only, 3 for xy+visibility). If this field is missing, Ultralytics raises a `KeyError` during training initialization — but the error message may be obscure enough that developers assume it is a data path issue. Additionally, if `kpt_shape: [6, 2]` is specified but label files contain `[6, 3]` columns (with visibility), Ultralytics silently misparses columns, shifting coordinates by one position for all keypoints past the first.

**Why it happens:**
`kpt_shape` is a pose-task-specific field not present in detection or segmentation dataset YAML files. Developers copying a detection YAML as a starting point for pose training will not have this field. The mismatch between kpt_shape dimensionality (2 vs 3) and actual label file columns is a silent data corruption.

**How to avoid:**
- Always specify `kpt_shape: [6, 3]` for 6 keypoints with visibility (recommended). Use visibility=2 for visible keypoints, visibility=1 for annotated-but-occluded, visibility=0 for missing (coordinates should be (0, 0) with visibility=0).
- Use `kpt_shape: [6, 3]` even if not all fish have all 6 keypoints — mark missing keypoints with `(0.0, 0.0, 0)` rather than omitting them or changing the count.
- Validate the dataset YAML before training using `from ultralytics.data.utils import check_det_dataset; check_det_dataset(yaml_path)`.
- Use the COCO8-pose dataset as a reference for the expected YAML structure.

**Warning signs:**
- `KeyError: 'kpt_shape'` during training initialization.
- Predicted keypoints are shifted by one index relative to ground truth (head predicted at tail position, etc.).
- Training begins and immediately terminates in the first few iterations with loss = NaN.

**Phase to address:**
Training data preparation phase (dataset YAML construction). Add a YAML validation step as a prerequisite gate before training.

---

### Pitfall B4: YOLOv8-seg Inference on Crops Returns Masks in Letterboxed/Padded Space, Not Original Crop Dimensions

**What goes wrong:**
When YOLOv8-seg is run on a crop image (non-square, e.g., 256x128 for OBB-aligned crops), Ultralytics letterboxes the crop to 640x640 before inference. The `result.masks.data` tensor has shape `(N, 640, 640)` — the padded square dimensions, not the original crop dimensions. The black letterbox padding is included in the mask. If this mask is naively resized to `(crop_h, crop_w)` using `cv2.resize`, the fish mask is compressed into the original crop region but the padding is incorrectly scaled. The resulting mask covers the wrong region.

**Why it happens:**
Ultralytics automatically rescales bounding boxes back to original image coordinates, but `result.masks.data` is in the padded inference space. Users who follow the official docs' simple examples (which use square images) never encounter this. The Ultralytics GitHub issue #4796 (2023) documented this and Glenn Jocher acknowledged it as a "common enough use case" but it was not automatically handled in the library at that time. The correct path is to use `result.masks.xy` (polygon format, already scaled to original image coords) rather than `result.masks.data` when coordinates are needed for downstream use.

**How to avoid:**
- Use `result.masks.xy` for polygon coordinates — these ARE already in original image pixel space (Ultralytics applies scale restoration to xy outputs).
- Use `result.masks.data` ONLY if you need the binary mask matrix, and apply the scale/crop restoration: compute the letterbox padding from `result.orig_shape` and the inference input size, crop out the padding, then resize to original dimensions. Alternatively, use `result.plot()` to obtain the correctly scaled overlay, but this is only for visualization.
- The safest approach for this project: convert `result.masks.xy[i]` (polygon in frame pixels) to a binary mask using `cv2.fillPoly` on a canvas of `(crop_h, crop_w)`.
- Test explicitly: run YOLOv8-seg on a 256x128 crop, verify that the returned mask region matches the visible fish area, not a compressed version of a 640x640 letterboxed mask.

**Warning signs:**
- Segmentation masks on non-square crops appear squished or offset — the fish is in one region but the mask covers a different region.
- Mask IoU against ground truth is poor for rectangular crops but good for square crops.
- `result.masks.data.shape` is `(N, 640, 640)` instead of `(N, crop_h, crop_w)`.

**Phase to address:**
Segmentation backend integration phase (pipeline integration of YOLOv8-seg). Write a test that runs inference on a non-square crop and verifies mask pixel coverage against known ground truth.

---

### Pitfall B5: YOLOv8-pose Keypoint Output Is in Original Frame Space, Not Crop Space — But the Pipeline Expects Crop Space

**What goes wrong:**
When YOLOv8-pose is run on a crop image, `result.keypoints.xy` returns keypoints in the crop's coordinate space (pixel coordinates within the crop image). The existing pipeline's `Midline2D` contract requires frame-space coordinates. If these crop-space keypoints are stored in `Midline2D.points` without back-projection, all downstream triangulation receives wrong coordinates — exactly the same failure mode as Pitfall A2 but now specifically for Ultralytics inference output rather than custom model output.

The complementary failure: if YOLOv8-pose is run on the FULL FRAME instead of a crop (to avoid the coordinate conversion), all fish in the frame are detected simultaneously, which conflicts with the per-fish, per-crop inference model the pipeline uses. Running full-frame pose inference also degrades per-fish keypoint accuracy because the model must handle up to 9 fish simultaneously at 1600x1200 resolution.

**Why it happens:**
`result.keypoints.xy` is always in the coordinate space of the image passed to `predict()`. Running on a crop gives crop-space coordinates; running on the full frame gives frame-space coordinates. The pipeline's crop-based inference architecture (detect → crop → segment/pose → paste back) requires an explicit coordinate translation step that is not automatic.

**How to avoid:**
- Run YOLOv8-pose on individual crops (one fish per crop). After inference, back-project: `frame_kp_x = crop_region.x1 + kp_x; frame_kp_y = crop_region.y1 + kp_y`.
- For OBB-aligned affine crops (Phase 32), additionally apply `invert_affine_points(kp_xy, affine_crop.M)` before adding the crop offset.
- Add a coordinate plausibility assertion: every keypoint in frame space should fall within `[detection.bbox.x1 - margin, detection.bbox.x2 + margin]` and similarly for Y. Log and skip any keypoint outside this range.
- Unit test the back-projection with a synthetic crop at a known position: assert back-projected keypoints match their expected frame-space positions to within 1 pixel.

**Warning signs:**
- Midline visualization overlays show keypoints near the top-left of the full frame when fish are in the center.
- Triangulation produces 3D midlines clustered near the tank center regardless of fish position.
- `midline.points[:, 0].mean()` is much smaller than `detection.bbox[0]` (x1 of the detection).

**Phase to address:**
Segmentation/pose backend integration phase (pipeline wiring). The back-projection step must be written and tested before any reconstruction testing.

---

### Pitfall B6: Small Dataset Overfitting — YOLOv8 Mosaic Augmentation Uses the Same 150 Images Repeatedly

**What goes wrong:**
YOLOv8's default mosaic augmentation combines 4 randomly selected training images into one. With only ~150 annotated frames, the same image pairs appear together repeatedly across epochs, and the model memorizes specific frame configurations rather than learning general fish appearance. Val loss matches train loss from epoch 5 onward (temporal leakage from the small pool), giving false confidence. On held-out video frames the model fails on females (never generalized beyond training appearances) and on unusual body orientations.

Additionally, YOLOv8 defaults to closing mosaic augmentation in the last 10 epochs (`close_mosaic=10`). With only 50–100 training epochs on a small dataset, turning off mosaic for the last 10 epochs removes the primary augmentation source just as the model is nearing convergence, often causing a sharp validation metric drop and instability.

**Why it happens:**
Ultralytics default hyperparameters are tuned for COCO-scale datasets (tens of thousands of images). A 150-frame dataset with 12-camera correlations behaves very differently from a diverse object detection dataset. The default configuration is inappropriate at this data scale.

**How to avoid:**
- Set `close_mosaic=0` to keep mosaic active throughout training (no sudden augmentation removal).
- Increase other augmentation intensities: `degrees=15` (rotation), `scale=0.5` (scale jitter), `fliplr=0.5`, `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4` (color jitter to help with low-contrast females).
- Start from a pretrained model (`yolov8n-seg.pt` or `yolov8n-pose.pt`) — transfer learning from COCO significantly reduces the data volume needed for the backbone.
- Freeze backbone layers for the first 10–20 epochs (`freeze=10`) to prevent overfitting the pretrained features to the small dataset, then unfreeze for fine-tuning. This is especially important for seg/pose models where the backbone is significantly larger than the head.
- Use a lower learning rate than the default: `lr0=0.001` instead of `0.01` for small dataset fine-tuning.
- Split training data by temporal segment (see Pitfall A7 for the temporal leakage concern — applies here too).

**Warning signs:**
- Val mAP50 exceeds 0.90 within the first 5–10 epochs on a 150-frame dataset (memorization, not generalization).
- Sharp metric drop in the last 10 epochs (mosaic closure effect).
- Model fails on held-out frames from different recording sessions despite excellent val metrics.
- Loss curve for females is consistently higher than for males throughout training (insufficient female appearance coverage).

**Phase to address:**
Training data preparation and model training phases. Establish a held-out validation clip from a different recording session (or a non-overlapping temporal window) BEFORE any training begins.

---

### Pitfall B7: Breaking the Existing Working YOLO Detection Model When Adding Seg/Pose Training

**What goes wrong:**
The existing YOLO detection model (`yolo_fish/train_v1/weights/best.pt`) is the pipeline's entry point and works correctly. When developers add training infrastructure for seg/pose, they may accidentally:
1. Overwrite the existing weights path by using the same `project=` and `name=` arguments in `YOLO.train()`.
2. Load the detection weights as the starting point for seg model training — detection and seg models have different head architectures, so this silently trains with incompatible weight initialization on the detection head.
3. Change the Ultralytics version (e.g., `pip install --upgrade ultralytics`) to get a newer feature, which changes the inference output format (e.g., `results[0].boxes` attribute changes) and breaks existing detection parsing code.

**Why it happens:**
Ultralytics uses `YOLO("yolov8n.pt")` for detect and `YOLO("yolov8n-seg.pt")` for seg — the file suffix determines the task. Developers unfamiliar with this may load a detect `.pt` file when intending to train a seg model. The `project/name` collision is easy to miss in the training script.

**How to avoid:**
- Write seg and pose training to explicitly load from task-specific starting weights: `YOLO("yolov8n-seg.pt")` for seg, `YOLO("yolov8n-pose.pt")` for pose. Never use the existing detection weights file as the starting point for seg/pose training.
- Use distinct `project` and `name` values: `project="runs/seg", name="fish_seg_v1"` and `project="runs/pose", name="fish_pose_v1"`. Never use `runs/detect`.
- Pin the Ultralytics version in `pyproject.toml` before adding new training infrastructure. Do not upgrade during active development.
- After adding any new Ultralytics training code, run the existing detection pipeline smoke test: `aquapose run --config tests/fixtures/minimal_config.yaml` and verify detection count matches baseline.

**Warning signs:**
- `runs/detect/fish_seg_v1` directory exists (project/name mismatch with detect task).
- Existing detection pipeline starts failing with `AttributeError: 'Results' object has no attribute 'boxes'` after an Ultralytics version upgrade.
- Seg model `.pt` file size is the same as the detect model `.pt` file (indicates the detect weights were loaded as seg, which may fail silently during transfer).

**Phase to address:**
Training infrastructure phase and model integration phase. Add a regression test for the detection pipeline before any new Ultralytics model training is attempted.

---

### Pitfall B8: YOLO Segmentation Annotation Polygon Winding Order and Minimum Point Count

**What goes wrong:**
Ultralytics YOLO segmentation format requires polygon coordinates as `class_id x1 y1 x2 y2 ... xn yn` where coordinates are normalized [0, 1] and the polygon must have at least 3 points. Two specific failures:
1. Degenerate polygons: a fish seen nearly head-on in an edge camera produces a very narrow mask. The bounding contour from `cv2.findContours` may degenerate to 1–2 points after simplification, which Ultralytics silently ignores (the label line is skipped, reducing training data without warning).
2. Excessive polygon complexity: an unprocessed SAM2 mask contour may have 500–2000 points. This slows dataset loading and may cause memory issues with large batches.

**Why it happens:**
`cv2.findContours` returns raw pixel-boundary contours without simplification. SAM2 masks have smooth but detailed boundaries that produce many contour points. Developers converting masks to polygons rarely think about point count as a training concern.

**How to avoid:**
- Simplify contours with `cv2.approxPolyDP(contour, epsilon=2.0, closed=True)` before writing to label files. Epsilon of 1.5–3.0 pixels balances boundary fidelity against polygon complexity.
- After simplification, assert `len(simplified_contour) >= 3`. If fewer than 3 points remain, log and skip the annotation (the mask is degenerate and not usable as training data).
- Cap polygon point count at 100–150 points by increasing epsilon until below the cap. This is acceptable for fish body shapes which are smooth ellipsoids.
- Verify the final dataset has no label files with fewer than 3 polygon points: `for f in labels/; check len(line.split()) >= 7 (class + 3 xy pairs)`.

**Warning signs:**
- Ultralytics dataset check reports "no labels found in image" for some frames despite label files existing.
- Training data count is lower than expected (degenerate polygons silently skipped).
- Dataset loading is slow (thousands of points per polygon causing parsing overhead).

**Phase to address:**
Training data preparation phase (polygon extraction and validation step).

---

### Pitfall B9: Missing Keypoints Must Be Represented as (0.0, 0.0, 0) — Not Omitted or Padded With Other Values

**What goes wrong:**
Fish viewed at extreme angles or with occlusion will not have all 6 anatomical keypoints visible. Three wrong approaches:
1. **Omit the keypoint entirely** — changes the column count per label line. Ultralytics expects exactly `kpt_shape[0]` keypoints per object, every time. Variable column counts cause a parsing crash.
2. **Use (-1, -1, 0)** — negative normalized coordinates are out-of-range and trigger "non-normalized or out of bounds" warnings. Some values will be clipped or the label skipped.
3. **Use (0.5, 0.5, 0)** — marking missing keypoints at image center teaches the model to predict center for occluded points, which is wrong.

The correct representation is `(0.0, 0.0, 0)` — coordinates at image origin with visibility=0 — which signals to the loss function that this keypoint should not contribute to the gradient.

**Why it happens:**
The Ultralytics issue #17116 explicitly documents this. The COCO keypoint convention (v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible) is not obvious to developers unfamiliar with the COCO spec. Custom datasets often have simpler annotation schemes that do not include visibility.

**How to avoid:**
- In the annotation pipeline, explicitly mark each of the 6 keypoints with: if annotated → `(norm_x, norm_y, 2)` if visible, `(norm_x, norm_y, 1)` if occluded but inferred; if not annotatable → `(0.0, 0.0, 0)`.
- In the data conversion script, never use negative coordinates or coordinates outside [0, 1].
- During inference post-processing, filter out keypoints where the visibility score < threshold OR the confidence from the model is low — do not use keypoints predicted at (0, 0) as real body positions.

**Warning signs:**
- Training warnings: "non-normalized or out of bounds coordinate" for specific frames.
- Inference shows predicted keypoints at (0, 0) in frame coordinates on some fish (model has learned to predict image origin for missing points using wrong training convention).
- Column count in label files is inconsistent across fish within the same frame.

**Phase to address:**
Training data preparation phase (keypoint annotation pipeline). Validate by loading a sample label file and asserting exactly `5 + kpt_shape[0] * kpt_shape[1]` values per line.

---

### Pitfall B10: Running YOLOv8-seg on Crops vs Full Frame — Data Format Must Match Inference Mode

**What goes wrong:**
If YOLOv8-seg is trained on full-frame images (normalized polygon coordinates relative to the full 1600x1200 frame) but then run at inference time on crops (256x128 regions), the model receives images where the fish fills the entire frame. The input distribution is completely different from training. The model has only ever seen a fish as a small polygon in the center of a 1600x1200 frame; it now sees the same fish filling 90% of the image. Detection will fail.

The reverse failure: if trained on crops but inference runs on full frames, the model receives a full 1600x1200 frame with many small fish and must detect and segment all of them simultaneously — which it has no training experience for.

**Why it happens:**
The existing pipeline has two inference modes that look similar: full-frame YOLO detection (already working), and crop-based U-Net segmentation. Developers adding YOLOv8-seg must decide which mode to use for BOTH training AND inference. The existing detect + crop workflow must inform the seg training mode. Mixing modes invalidates the training distribution.

**How to avoid:**
- Decide explicitly at the start: train on crops (one fish per image) and run inference on crops, OR train on full frames and run inference on full frames. For this pipeline, training on crops is recommended because:
  - It matches the existing YOLO-detect → crop → segment architecture.
  - It keeps images small (faster training on 150 frames).
  - It avoids multi-instance seg at inference time (one fish per crop, simpler).
- Encode the inference mode in the dataset YAML description comment and in the model documentation.
- Do NOT mix: if the training dataset contains cropped images, the inference code must crop before calling `model.predict()`.

**Warning signs:**
- Model trained on crops fails to detect fish when run on full frames (or vice versa).
- Segmentation mAP is high during training but near zero in pipeline integration testing.
- `result.boxes` is empty when running inference on a full frame with a model trained on crops.

**Phase to address:**
Training data preparation phase (design of training image format). This must be decided before any annotation preparation begins, as it affects all label coordinate normalization.

---

## Technical Debt Patterns (v3.0 Specific)

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Train on raw SAM2 masks without visual validation of polygon conversion | Faster pipeline setup | Silent corrupt annotations; model trains on garbage | Never — always validate 20+ samples visually before full training |
| Use `result.masks.data` directly without letterbox correction | Simpler code | Masks in wrong coordinate space for non-square crops | Never for non-square inputs — use `result.masks.xy` and `cv2.fillPoly` |
| Skip `kpt_shape` validation in dataset YAML | No extra code | Silent column offset; model predicts wrong keypoints | Never — add YAML validation as a pipeline step |
| Train seg/pose from scratch instead of pretrained weights | No dependency on COCO weights | 10x more data needed; poor generalization on 150 frames | Never for this dataset size |
| Use full-frame inference for seg/pose after training on crops | Simpler inference code | Distribution mismatch; model fails on real frames | Never — inference mode must match training mode |
| Keep old U-Net code during transition | Fallback if Ultralytics fails | Maintenance burden; import conflicts; mixed model interfaces | Acceptable only as a temporary stage gate during phased replacement |

---

## Integration Gotchas (v3.0 Specific)

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| SAM2 mask → YOLO seg polygon | Write each `findContours` contour as a separate label line | Merge multi-region masks with morphological closing; keep only largest contour if merging fails |
| Keypoint labels | Normalize coords relative to crop image | Normalize relative to full frame image; convert crop coords first |
| Missing keypoints | Omit or use (-1,-1) | Always write `(0.0, 0.0, 0)` for missing keypoints; never change column count |
| YOLOv8-seg inference masks | Use `result.masks.data` directly | Use `result.masks.xy` (already in original image coords) then `cv2.fillPoly` |
| YOLOv8-pose keypoints at inference | Use `result.keypoints.xy` as frame coords | `result.keypoints.xy` is in crop coords when crop is passed; add `crop_region.x1/y1` offset |
| Existing YOLO detect model | Load detect weights for seg training | Use task-specific weights: `YOLO("yolov8n-seg.pt")` for seg |
| Ultralytics version pinning | Upgrade for new features mid-project | Pin version in `pyproject.toml`; verify detect pipeline still works after any upgrade |

---

## "Looks Done But Isn't" Checklist (v3.0 Specific)

- [ ] **Polygon conversion validated:** 20+ label files visually inspected with overlaid polygons on source images — no disconnected regions, no degenerate (< 3 point) polygons.
- [ ] **YOLO seg training on crops:** Confirm dataset images are crops (not full frames); label coordinates normalized to crop dimensions, not frame dimensions.
- [ ] **YOLO pose keypoints in frame space:** Verify label file keypoint coordinates for a known fish are in expected full-frame normalized range (not near 0 which would indicate crop space).
- [ ] **kpt_shape in dataset.yaml:** `kpt_shape: [6, 3]` present; `kpt_shape[1] == 3` (not 2) to support visibility flags.
- [ ] **Missing keypoints use (0.0, 0.0, 0):** Search label files for negative values or values > 1.0 — both indicate wrong convention.
- [ ] **Inference mask coordinate space correct:** Test inference on a 256x128 crop; verify `result.masks.xy[0]` points are within (0, 256) for x and (0, 128) for y.
- [ ] **Keypoint back-projection verified:** After pose model inference on a crop, verify back-projected frame keypoints fall within the detection bounding box.
- [ ] **Existing detect model unaffected:** Run existing YOLO detect pipeline after any Ultralytics version change or training run; verify detection count on reference frame matches baseline.
- [ ] **No project/name collision:** seg and pose training use distinct `project/name` paths; `runs/detect/` is unchanged.
- [ ] **Temporal split validated:** Val set is from a different recording session or non-overlapping temporal window; val loss is NOT matching train loss from epoch 1.

---

## Recovery Strategies (v3.0 Specific)

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| SAM2 multi-region polygon corruption discovered after training | HIGH | Fix conversion pipeline, regenerate all labels, retrain from scratch |
| Keypoint labels in crop space discovered after training | HIGH | Recompute all keypoint labels with frame-space normalization, retrain |
| Missing kpt_shape in YAML or wrong dimensionality | LOW | Add/fix YAML field, no label changes needed, re-run training |
| Masks in letterboxed space used incorrectly | MEDIUM | Switch from `result.masks.data` to `result.masks.xy` + fillPoly; no retraining needed |
| Existing detect model broken by Ultralytics upgrade | MEDIUM | Pin previous version, check release notes for breaking changes, update parsing code |
| Model trained on crops, inference on full frames | HIGH | Retrain with correct dataset format; or change inference to crop-first (preferred) |
| Small dataset overfitting discovered | MEDIUM | Add stronger augmentation (especially color jitter for low-contrast females), freeze backbone, retrain |

---

## Pitfall-to-Phase Mapping (v3.0 Specific)

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| B1: SAM2 multi-region masks | Training data prep (annotation conversion) | Visual overlay check on 20+ frames; zero corrupt-label warnings at dataset check |
| B2: Keypoint coords in crop space | Training data prep (pose label generation) | Assert normalized keypoint values in expected frame-space range; no > 1.0 or < 0 |
| B3: Missing kpt_shape in YAML | Training data prep (YAML construction) | `check_det_dataset(yaml_path)` passes without KeyError |
| B4: Masks in letterboxed space | Seg backend integration | Test: non-square crop inference → verify mask pixel region matches fish location |
| B5: Pose keypoints in crop space at inference | Pose backend integration | Coordinate plausibility assertion: keypoints within detection bbox |
| B6: Small dataset overfitting | Model training (hyperparameter setup) | Hold-out val clip from different session; val loss diverges from train after ~20 epochs |
| B7: Breaking existing detect model | Any training phase | Detect pipeline regression test after each Ultralytics model training run |
| B8: Degenerate polygons | Training data prep (polygon extraction) | Assert len >= 3 for all polygons; count check matches expected fish count per frame |
| B9: Wrong missing keypoint convention | Training data prep (annotation pipeline) | Assert all values in [0, 1]; no negative coords; constant column count per label line |
| B10: Train/inference mode mismatch | Training data prep (design decision) | Explicitly document training image format; verify inference code matches it |

---

---

## v2.2 Integration Pitfalls

These pitfalls are specific to adding the v2.2 feature set to the existing pipeline. They involve coordinate system mismatches, contract changes, and config system fragility when integrating new components with existing consumers.

---

### Pitfall A1: OBB Angle Convention Mismatch Between Extraction and Affine Crop

**What goes wrong:**
Ultralytics YOLO-OBB outputs angles in **radians** in the range `[-pi/4, 3pi/4)`, using a **clockwise** convention (angle=0 means no rotation, positive angle rotates clockwise). Code that assumes degrees, counter-clockwise, or the `[0, pi/2)` range produced by OpenCV `minAreaRect` will generate affine crops rotated by the wrong amount — often 90 degrees off — silently producing valid-looking but misoriented crops. The fish body appears horizontal when it should be diagonal. The keypoint model trained on correctly oriented crops receives garbage input at inference time.

**Why it happens:**
Three separate angle representations coexist in this ecosystem. OpenCV `minAreaRect()` returns angles in degrees in `(-90, 0]`. Ultralytics OBB model outputs radians in `[-pi/4, 3pi/4)` (clockwise). Label format (`xyxyxyxy` corners) is angle-free; conversion via `minAreaRect` introduces the OpenCV degree convention. When training labels are prepared with OpenCV and the model is queried via ultralytics, developers assume both use the same convention. There is a documented inconsistency between the `[0, pi/2)` range used in label conversion and the `[-pi/4, 3pi/4)` range of model predictions (ultralytics issues #13003 and #16235).

**How to avoid:**
- Always extract angle from `result.obb.xywhr` (not re-derived from `xyxyxyxy`) and confirm units are radians before passing to `cv2.getRotationMatrix2D`.
- Convert: `angle_deg = float(box.xywhr[0, 4]) * 180 / math.pi` — this is clockwise from horizontal.
- Add an explicit smoke test before any keypoint model training: given a synthetic fish at a known angle, verify the affine crop has the fish axis aligned with the crop horizontal.
- Never mix `minAreaRect` angles with ultralytics OBB angles in the same pipeline step without an explicit conversion guard.

**Warning signs:**
- Affine-cropped images consistently look sideways or upside-down relative to the OBB overlay drawn from the same detection.
- Keypoint model outputs confidences near 0.5 uniformly across all body points (random-looking, indicating random orientation input).
- OBB overlay on the full frame looks correct but the crop-based visualization looks wrong.

**Phase to address:**
YOLO-OBB detection backend phase. Add a crop-orientation smoke test before any keypoint model training begins.

---

### Pitfall A2: Keypoint Coordinates Returned in Crop-Local Space, Consumed as Frame-Global

**What goes wrong:**
A keypoint regression head operating on a 128x128 crop returns points in crop coordinates `(0..crop_w, 0..crop_h)`. If those coordinates are stored directly into `Midline2D.points` without applying the `CropRegion` inverse transform (scale + translate), all downstream consumers receive midline points clustered near the image origin rather than at the actual fish position. Reconstruction will fail silently: triangulation attempts to triangulate near `(0, 0)` in each view, producing 3D points near the camera, not the fish. The `AnnotatedDetection` and HDF5 writer accept any `Midline2D` without checking coordinate plausibility.

**Why it happens:**
The existing `segment_then_extract` backend explicitly calls `_crop_to_frame()` before returning a `Midline2D`. A keypoint backend author implementing inference-then-return may return points immediately after argmax or soft-argmax without noticing the coordinate system obligation. `Midline2D` has no field that records whether points are in frame or crop space — the contract is implicit in the docstring ("Full-frame pixel coordinates") and is never enforced at runtime.

**How to avoid:**
- The keypoint backend **must** call `_crop_to_frame()` (or an equivalent) before constructing `Midline2D`. This function in `reconstruction/midline.py` handles the resize scale from model input size (128x128) to actual crop dimensions, then translates by `crop_region.x1, crop_region.y1`.
- Add an integration test: `midline.points[:, 0].min() > crop_region.x1 - 10` and `midline.points[:, 1].min() > crop_region.y1 - 10` for every non-None midline.
- Consider a runtime assertion in `MidlineStage.run()` after each backend call: verify points fall within detection bbox expanded by some margin.

**Warning signs:**
- Midline visualization overlays show all points clustered at `(0,0)` or top-left of the frame.
- Triangulation produces 3D points with X and Y near zero regardless of fish position.
- `AnnotatedDetection.midline.points.mean(axis=0)` is far from `detection.bbox` centroid.

**Phase to address:**
Keypoint midline backend phase, at the coordinate transform step. Verify with a frame-space coordinate assertion before integration testing.

---

### Pitfall A3: Changing Midline2D Point Count Breaks HDF5 Writer, Curve Optimizer, and Visualization

**What goes wrong:**
`N_SAMPLE_POINTS = 15` is hardcoded in `reconstruction/triangulation.py` and imported by `io/midline_writer.py`, which pre-allocates HDF5 datasets with shape `(N, max_fish, 15)`. The curve optimizer (`reconstruction/curve_optimizer.py`) also imports `N_SAMPLE_POINTS` directly. Visualization code in `visualization/triangulation_viz.py:514` and `visualization/midline_viz.py:615` has bare `if n_skel < 15:` guards. If a keypoint backend produces a different point count, the HDF5 writer silently truncates with `n_hw = min(len(hw), N_SAMPLE_POINTS)`. If a user sets `midline.n_points = 20` via YAML, only `MidlineStage` and its backends respect it — the HDF5 writer, curve optimizer, and visualization stay at 15.

**Why it happens:**
`n_points=15` is a config parameter in `MidlineConfig` that flows through `MidlineStage.__init__()` → `get_backend()` → backends. But the HDF5 writer, curve optimizer, and visualization modules import the constant directly rather than receiving it from config. They do not observe `MidlineConfig.n_points`. This is a pre-existing partial wiring that becomes a landmine when v2.2 adds a keypoint backend.

**How to avoid:**
- Do not change `n_points` from 15 in v2.2 unless all consumers are audited first.
- If configurable point count is needed, add `n_sample_points` to `ReconstructionConfig` and thread it through `Midline3DWriter` and the curve optimizer constructor.
- Remove all bare `15` literals from visualization code — use the `N_SAMPLE_POINTS` constant.
- Add a CI check: `grep -rn "< 15\b\|== 15\b" src/aquapose/visualization/` should return zero matches after the config cleanup phase.
- The keypoint backend should use the same `n_points` parameter passed to it by `MidlineStage` — no independent hardcoding inside the backend.

**Warning signs:**
- HDF5 `half_widths` dataset has trailing NaN columns when n_points differs from 15.
- Visualization skeleton-length check fires at a threshold that disagrees with the backend minimum.
- `Midline3D.half_widths.shape[0]` is not 15 but downstream analysis indexes `[:15]` silently.

**Phase to address:**
Config cleanup phase (scatter audit). Keypoint backend phase (ensure backend receives and uses the same n_points). The HDF5 writer fix should precede reconstruction integration.

---

### Pitfall A4: OBB NMS Suppresses Overlapping Fish Differently from AABB NMS

**What goes wrong:**
OBB NMS uses rotated IoU rather than axis-aligned IoU. For fish that are nearly parallel and close (common schooling behavior), rotated IoU is significantly higher than AABB IoU even when the fish are distinct individuals — because elongated boxes sharing the same orientation have large overlap. OBB NMS with the same `iou_threshold=0.45` used for AABB will suppress one of two nearby parallel fish, silently reducing detection count. Downstream tracking loses a fish, Association stage gets fewer tracklets, and reconstruction drops a fish from the 3D output without error.

**Why it happens:**
The existing `YOLODetector` uses `iou_threshold=0.45` tuned for axis-aligned boxes. Developers test with solo or spaced fish and see correct results, but schooling cases only appear in full recordings. The AABB threshold is not the right starting point for OBB.

**How to avoid:**
- Use a higher OBB `iou_threshold` (0.60–0.70) as the starting point for elongated fish; 0.45 is too aggressive for aligned orientations.
- Test with frames that have at least 2 parallel fish within 100px of each other — this is the stress case.
- After deploying OBB detection, compare per-frame detection counts against the existing YOLO AABB baseline on the same video segment.

**Warning signs:**
- Per-frame detection count is lower than expected, especially in schooling frames.
- Fish pairs that are visually distinct in the frame are missing one member.
- OC-SORT shows coasting tracks in frames where fish are grouped.

**Phase to address:**
YOLO-OBB detection backend phase. Include a detection count regression test against the existing AABB backend on a reference frame set.

---

### Pitfall A5: Affine Crop Produces Black Border Artifacts That Confuse Segmentor and Keypoint Model

**What goes wrong:**
An affine rotation crop (used to de-rotate the fish to horizontal) fills areas outside the original image with `borderValue=0` (black) when using `cv2.warpAffine`. If the fish is near a frame edge, up to 30–40% of the crop may be black border. The U-Net segmentor, trained on natural crops without large black regions, may predict foreground probability in the black area (treating it as dark water or a fish body). The keypoint model may regress keypoints into the black padding region. `_check_skip_mask` may also incorrectly fire "boundary-clipped" on artificial black borders.

**Why it happens:**
`extract_crop()` in `segmentation/crop.py` is a simple rectangle slice with no border artifacts. Affine-rotated crops using `cv2.warpAffine` introduce hard black edges at rotation boundaries. Developers testing on fish in the tank center (far from edges) never observe the artifact.

**How to avoid:**
- Use `cv2.BORDER_REPLICATE` or `cv2.BORDER_REFLECT` instead of `cv2.BORDER_CONSTANT` in all `warpAffine` calls. Replicated borders are semantically neutral (background-like) rather than black.
- Test with a detection whose bbox is within 50px of the frame edge. Verify the affine crop has no all-zero rows or columns inside the fish region.

**Warning signs:**
- Masks for fish near frame edges include large rectangular black regions.
- `_check_skip_mask` reports "boundary-clipped" on affine crops that aren't near actual frame boundaries.
- Keypoint confidence is systematically lower for fish observed by cameras that see the tank wall.

**Phase to address:**
YOLO-OBB affine crop implementation within the midline backend phase.

---

### Pitfall A6: Training Augmentation Breaks Spatial Consistency Between Image and Keypoint Labels

**What goes wrong:**
Standard image augmentation (horizontal flip, random crop, perspective warp) applied to the image without applying the exact same transform to keypoint coordinates produces mismatched labels. The keypoint model learns from image patches where the fish is flipped left-right but the label says "head is at the left end." Training loss decreases normally (the model memorizes a random mapping) but inference orientation is systematically wrong. This is particularly insidious because the loss looks healthy.

**Why it happens:**
Image augmentation libraries have two modes: image-only and image+annotations. Albumentations, torchvision, and imgaug all support keypoint-aware transforms but require explicitly registering keypoints as `KeypointParams` or similar. If a developer extends the existing `BinaryMaskDataset` pattern by adding augmentation at the image level without registering keypoints, the geometric transforms are applied to images only.

**How to avoid:**
- Use Albumentations with `KeypointParams(format="xy", remove_invisible=False)` so all geometric transforms (flip, rotate, crop, warp) apply to both image and keypoint labels simultaneously.
- Non-geometric augmentations (brightness, contrast, blur, noise) are safe to apply to the image only.
- Add a visual validation step: augment a batch, overlay keypoints on the augmented image, and confirm alignment before training for more than a few epochs.

**Warning signs:**
- Training loss decreases to a low value but validation keypoint metrics are poor.
- Head-end prediction accuracy is near 50% (random) despite good total loss.
- Flipped fish in augmented frames have keypoints that do not match the flip.

**Phase to address:**
Training infrastructure phase. Validate augmentation pipeline visually before any model training begins.

---

### Pitfall A7: Train/Val Split Leakage With Temporally Correlated Pseudo-Label Frames

**What goes wrong:**
Pseudo-labels are generated from consecutive video frames. If the train/val split is done by randomly shuffling individual frames, consecutive frames from the same sequence appear in both train and val. The model memorizes fish trajectories rather than generalizing. Validation loss looks excellent from early epochs, the model appears well-trained, but it fails on held-out clips because it has memorized specific fish positions and orientations rather than learned general appearance.

**Why it happens:**
The existing `BinaryMaskDataset` draws from a list of `(image, mask)` pairs without temporal structure awareness. A random 80/20 frame-level split leaks: frames t=100 and t=101 are nearly identical, so the model sees t=100 in train and t=101 in val, which is functionally train data.

**How to avoid:**
- Split by contiguous temporal segment, not by individual frame. Hold out a full contiguous clip (different recording session or a non-overlapping temporal window) as the val set.
- If only one recording is available, split by non-overlapping temporal windows with a gap: e.g., frames 0–200 train, frames 200–250 val (with a 50-frame buffer, not random selection from the pool).

**Warning signs:**
- Val loss matches train loss almost exactly from early epochs onward.
- Model accuracy degrades significantly when tested on a different recording.
- Keypoint metrics on val are much better than on a held-out clip.

**Phase to address:**
Training infrastructure phase, before pseudo-label dataset construction.

---

### Pitfall A8: Device Propagation Failure When Multiple New Backends Each Default to "cuda"

**What goes wrong:**
`MidlineStage` defaults `device="cuda"`. `DetectionConfig` defaults `device="cuda"`. A new OBB backend will add another device parameter. If each component has its own device default rather than inheriting from a single source, two failure modes arise: (1) CPU-only machines fail at construction with "CUDA is not available" unless the user knows to set `device=cpu` in YAML; (2) if the OBB detector (ultralytics auto-selects device) ends up on CPU while the U-Net stays on GPU, tensors from different stages end up on different devices and inference fails mid-frame with a confusing error.

**Why it happens:**
The current `PipelineConfig` has no single top-level `device` field — each sub-config (`DetectionConfig.device`, `MidlineConfig` via `MidlineStage`) has its own default. A `device` set at `detection.device=cpu` does not propagate to `midline` device.

**How to avoid:**
- Add a single `device: str = "cuda"` field at the top level of `PipelineConfig`.
- In `load_config()`, propagate `top_kwargs["device"]` to `det_kwargs` and `mid_kwargs` if those sub-configs do not explicitly override device.
- In the training CLI, default to `"cuda" if torch.cuda.is_available() else "cpu"` rather than hardcoding "cuda".
- Add a test: construct the full pipeline with `device="cpu"` and verify no model tensor ends up on CUDA.

**Warning signs:**
- `RuntimeError: Tensors are on different devices` appearing in the midline backend after adding an OBB crop step.
- CI failures with "CUDA is not available" even though tests are expected to be CPU-only.
- The OBB detector works (ultralytics auto-selects CPU) but the U-Net fails (explicit device="cuda").

**Phase to address:**
Config cleanup phase. Top-level device propagation should precede any new backend addition so new backends inherit it correctly.

---

### Pitfall A9: Adding Fields to Midline2D Without Defaults Breaks All Construction Sites

**What goes wrong:**
`Midline2D` currently has no `confidence` or `per_point_confidence` field. Adding a field to this dataclass without a default value causes `TypeError` at all construction sites. `Midline2D` is constructed in at least 4 locations: `SegmentThenExtractBackend._extract_midline_from_mask()`, `MidlineExtractor.extract_midlines()` (legacy), `core/synthetic.py`, and any new keypoint backend. The HDF5 writer already expects `is_low_confidence: bool` on `Midline3D` (`midline_writer.py:168`) — if the equivalent is not added to `Midline2D` and threaded through `AnnotatedDetection`, the writer always writes `False`.

**Why it happens:**
`Midline2D` is defined in `reconstruction/midline.py` and re-exported from two `core/` type modules. Its construction is scattered. The field addition looks trivial but has wide blast radius.

**How to avoid:**
- Add `per_point_confidence: np.ndarray | None = None` with a default (keyword-only, not positional) so existing construction sites work without modification.
- Audit all construction sites before the field addition: `grep -rn "Midline2D(" src/aquapose/`.
- Update the reconstruction backend to check `per_point_confidence is not None` and apply confidence weighting only when present (backward compatible).

**Warning signs:**
- `TypeError: __init__() missing 1 required positional argument` at any `Midline2D()` call site after adding a field without a default.
- The HDF5 `is_low_confidence` dataset is all-False even when the keypoint backend flags low confidence.
- Triangulation uses uniform weights even when per-point confidence data is available.

**Phase to address:**
Keypoint midline backend phase, as a prerequisite to confidence-weighted reconstruction.

---

### Pitfall A10: Config Backward Compatibility — New Fields Without Defaults Break Existing YAML Files

**What goes wrong:**
When new fields are added to `DetectionConfig`, `MidlineConfig`, or `ReconstructionConfig` without default values, existing YAML config files that do not include those fields cause `TypeError` at load time. The existing `load_config()` has a `_filter_fields()` helper that strips unknown keys for `AssociationConfig` and `TrackingConfig`, but it is **not** applied to `DetectionConfig`, `MidlineConfig`, or `ReconstructionConfig`. Adding new required fields to those dataclasses — or removing old fields — silently breaks all existing YAML configs.

**Why it happens:**
The `_filter_fields()` safety net is partial by accident — it was added to fix an issue with `AssociationConfig` and `TrackingConfig` during v2.1 refactoring but was never applied universally. `DetectionConfig(**det_kwargs)` and `MidlineConfig(**mid_kwargs)` receive raw dicts from YAML without filtering.

**How to avoid:**
- Apply `_filter_fields()` to ALL stage config dataclasses in `load_config()`, not just association and tracking.
- All new config fields must have defaults — never add a required field to an existing production dataclass.
- If a field is renamed, keep the old name as a deprecated alias in the YAML loading layer for at least one milestone.
- After any config schema change, test `load_config(yaml_path=pinned_v21_yaml)` in CI against a representative saved YAML file.

**Warning signs:**
- `TypeError: __init__() missing 1 required positional argument` when loading an existing YAML after a dataclass change.
- Researchers who saved working YAML configs from v2.1 find they no longer work after upgrading.
- CI tests pass with hardcoded test configs but user-generated configs fail.

**Phase to address:**
Config cleanup phase. Apply `_filter_fields()` universally and add a pinned YAML regression test before any other config schema changes.

---

### Pitfall A11: Reconstruction Assumes Midlines Have Consistent Point Count — Keypoint Backend Must Not Vary It

**What goes wrong:**
The triangulation backend uses arc-length position `t[i]` as the correspondence key across cameras: body point at arc-length index `i` in camera A is matched to body point at index `i` in camera B. This works only when both cameras produce midlines with the same number of points sampled at the same arc-length positions (`t = linspace(0, 1, n_points)`). A keypoint backend that returns a variable number of points (e.g., 13 high-confidence points for an occluded fish) produces a mismatched arc-length mapping. Point `t[i]` on the 13-point midline corresponds to a different body position than `t[i]` on the 15-point midline, corrupting triangulation silently.

**Why it happens:**
The segment-then-extract backend always produces exactly `n_points` points (or `None`). A keypoint backend author may decide to omit low-confidence points to "improve quality," not realizing that the correspondence constraint requires a fixed point count.

**How to avoid:**
- The keypoint backend must always produce exactly `n_points` output points, regardless of per-point confidence. Store confidence per point but never omit points.
- If partial midlines are desired (e.g., only head-side points visible), pad to `n_points` with the last known position and mark the padded points as low-confidence in `per_point_confidence`.
- Add a contract assertion in `MidlineStage.run()`: `assert midline.points.shape == (n_points, 2)` for every non-None midline.

**Warning signs:**
- Triangulation produces 3D midlines with incorrect curvature (S-shaped when the fish is straight).
- `midline.points.shape[0]` varies frame-to-frame for the same fish.
- Reconstruction succeeds with 2 cameras but fails with 4+ cameras (more cameras expose the mismatch).

**Phase to address:**
Keypoint midline backend phase, during output format definition. The n_points contract must be locked before reconstruction integration.

---

### Pitfall A12: Import Boundary Violation — New training/ Module Must Not Import from engine/

**What goes wrong:**
The AST-based import boundary checker enforces that `core/` never imports from `engine/`. A new `src/aquapose/training/` module that imports training utilities from `engine/config.py` (which is in engine/) will violate the import boundary and fail the pre-commit hook. If developers work around this by adding `training/` to the boundary checker's allowlist without thought, they may inadvertently permit circular imports.

**Why it happens:**
Training infrastructure naturally wants access to config (for hyperparameters) and possibly to pipeline stages (for data generation). The easy path is to import from wherever the class lives. The correct path is to either have `training/` depend only on `core/` and stdlib, or to explicitly declare that `training/` is allowed to depend on `engine/` as a top-level module.

**How to avoid:**
- Decide before implementation: is `training/` a peer of `engine/` (allowed to import from engine/) or a consumer of `core/` only? Document this in the import boundary checker config.
- If `training/` needs `PipelineConfig`, import it from `engine/config.py` explicitly and declare the allowance in the boundary checker.
- Run `hatch run pre-commit run --all-files` after adding any new import in `training/` — do not wait until the module is complete.

**Warning signs:**
- Pre-commit fails with "import boundary violation: training imports from engine".
- Workarounds like `TYPE_CHECKING` guards proliferating in `training/` to avoid import errors.
- Circular import at runtime when `training/` is imported.

**Phase to address:**
Training infrastructure phase, before writing any training module code.

---

## Technical Debt Patterns (v2.2 Specific)

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Leaving bare `15` literals in visualization code | No refactor needed now | Any n_points change silently breaks visualization | Never — replace with `N_SAMPLE_POINTS` constant in v2.2 cleanup |
| OBB backend config in `DetectionConfig.extra` dict | No new config fields | OBB params undocumented, not type-checked, not validated | Only as a temporary shim during development; add proper fields before shipping |
| Confidence as `None` in segment-then-extract backend after adding `per_point_confidence` field | No change to existing backend | Reconstruction cannot confidence-weight existing backend output | Acceptable for v2.2 if reconstruction checks for None |
| No `_filter_fields()` on `DetectionConfig`/`MidlineConfig` | Less code in load_config | Any YAML from v2.1 breaks on new field additions | Never — apply filter universally in v2.2 config cleanup |
| Separate `device` fields per sub-config | Each stage independently configurable | Device mismatch errors mid-pipeline; no single override point | Acceptable only if top-level propagation is implemented |
| Training CLI invents its own config loading | Faster initial implementation | Two config systems diverge; YAML files not interchangeable | Never — reuse `load_config()` from engine/config.py |

---

## Integration Gotchas (v2.2 Specific)

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OBB → CropRegion | Deriving CropRegion from OBB corners using their axis-aligned bounding box | Use OBB `xywhr` to compute affine rotation matrix; CropRegion describes the axis-aligned bounding box of the rotated crop in frame space |
| OBB angle → affine crop | Using `cv2.minAreaRect` on OBB corners to re-derive angle (introduces degree/OpenCV convention) | Use `result.obb.xywhr[..., 4]` (radians, clockwise) directly |
| Keypoint → Midline2D | Storing crop-space coordinates directly in `Midline2D.points` | Always call `_crop_to_frame()` from `reconstruction/midline.py` before constructing `Midline2D` |
| Training CLI → config system | Adding a new Click command that re-implements config loading from scratch | Reuse `load_config()` from `engine/config.py`; add training-specific fields to an extension of `PipelineConfig` if needed |
| New midline backend → MidlineStage | Implementing OBB crop logic directly in the stage | Follow the `SegmentThenExtractBackend` pattern: backend class implements `process_frame()`, registered in `core/midline/backends/__init__.py:get_backend()`, stage stays thin |
| Per-point confidence → HDF5 | Adding a new HDF5 dataset without updating `Midline3DWriter` | The writer pre-allocates datasets at `open()` — new fields require updating `_make()` calls in the constructor; existing files cannot be appended to with new schema |

---

## "Looks Done But Isn't" Checklist (v2.2 Specific)

- [ ] **OBB backend registered:** `make_detector("yolo-obb", ...)` in `segmentation/detector.py` returns a `YOLOOBBBackend` — verify the factory handles the new kind string.
- [ ] **Affine crop orientation tested:** Given a known-angle detection, verify the affine-cropped image has the fish axis aligned to horizontal — visual check, not just "no exception raised."
- [ ] **Keypoint coordinate transform verified:** `midline.points[:, 0].min() > crop_region.x1 - 20` for every non-None midline produced by the keypoint backend.
- [ ] **Config filter universal:** `_filter_fields()` applied to `DetectionConfig`, `MidlineConfig`, `ReconstructionConfig` — not just `AssociationConfig` and `TrackingConfig`.
- [ ] **n_points contract enforced:** `assert midline.points.shape == (config.midline.n_points, 2)` in `MidlineStage.run()` after each backend call.
- [ ] **HDF5 schema versioned if changed:** If `Midline3DWriter` gains new datasets, the HDF5 file has a schema version attribute so downstream analysis tools can detect format changes.
- [ ] **Training CLI uses shared config:** `aquapose train` loads via `load_config()` — not a separate config class invented for training.
- [ ] **Import boundary clean:** After adding `src/aquapose/training/`, `hatch run pre-commit run --all-files` reports 0 import boundary violations.
- [ ] **OBB NMS threshold tested on parallel fish:** Run on at least one frame with 2 adjacent parallel fish and verify both are detected with the chosen `iou_threshold`.
- [ ] **Pinned YAML regression test:** Loading a saved v2.1 YAML with the new config schema raises no `TypeError`.

---

## Recovery Strategies (v2.2 Specific)

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| OBB angle convention mismatch discovered after training keypoint model | HIGH | Re-generate all affine crops with correct angle convention, retrain from scratch — no post-hoc correction exists |
| Keypoint coordinates in crop space discovered in production | MEDIUM | Add `_crop_to_frame()` call in backend and re-run inference — no retraining needed |
| n_points mismatch in HDF5 file | MEDIUM | Write migration script that reads old HDF5, pads/truncates half_widths to 15, writes new file |
| Config backward compat broken by new required field | LOW | Add default value to the new field and release a patch — existing YAMLs work again without user changes |
| Train/val temporal leakage discovered after training | HIGH | Re-split by temporal segment and retrain — metrics from leaky split are unreliable |
| Device mismatch error mid-pipeline | LOW | Add device propagation to `load_config()`, no model changes needed |

---

## Pitfall-to-Phase Mapping (v2.2 Specific)

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| A1: OBB angle convention mismatch | YOLO-OBB detection backend | Smoke test: known-angle detection → affine crop → axis-aligned fish |
| A2: Keypoint coordinates in crop space | Keypoint midline backend | Integration test: assert points within detection bbox |
| A3: N_SAMPLE_POINTS scatter | Config cleanup | Grep for bare `15` literals in visualization; assert zero |
| A4: OBB NMS suppressing parallel fish | YOLO-OBB detection backend | Detection count regression test vs AABB baseline on schooling frames |
| A5: Affine crop border artifacts | YOLO-OBB affine crop step | Test: detection within 50px of frame edge produces valid crop |
| A6: Augmentation spatial inconsistency | Training infrastructure | Visual overlay check: augmented image + transformed keypoints aligned |
| A7: Train/val temporal leakage | Training infrastructure | Hold out a full contiguous clip as val; val loss diverges from train |
| A8: Device propagation failure | Config cleanup | Test: construct pipeline with `device="cpu"`, no CUDA tensors anywhere |
| A9: Midline2D contract change | Keypoint midline backend | All construction sites updated; no positional-arg `TypeError` |
| A10: Config backward compat breakage | Config cleanup | Load pinned v2.1 YAML after any schema change — no `TypeError` |
| A11: Arc-length mismatch (variable n_points) | Keypoint midline backend | Assert `midline.points.shape == (n_points, 2)` in `MidlineStage.run()` |
| A12: Import boundary violation | Training infrastructure | `hatch run pre-commit run --all-files` = 0 violations after each new import |

---

---

## Foundation Pitfalls (Retained from v1.0–v2.1 Research)

*These pitfalls from prior research remain relevant to the overall system.*

---

### Pitfall 1: Treating Refractive Distortion as Depth-Independent

**What goes wrong:** Refractive projection through a flat air-water port is depth-dependent. Systems that model refraction as a fixed pixel-wise correction produce systematic 3D errors that grow with distance from calibration depth. **Status:** Addressed in v1.0 via `RefractiveProjectionModel` with per-ray Snell's law tracing.

**Phase to address:** Camera model and calibration phase (resolved).

---

### Pitfall 2: All-Top-Down Camera Configuration Creates Weak Z-Reconstruction

**What goes wrong:** 13 cameras all looking straight down share nearly parallel optical axes. Z-reconstruction uncertainty is 132x larger than XY. **Status:** Quantified in v1.0; XY-only tracking cost matrix applied in early versions, superseded by OC-SORT per-camera in v2.1.

**Phase to address:** Geometry validation phase (resolved; 132x Z/XY anisotropy documented).

---

### Pitfall 3: MOG2 Background Subtraction Fails on Low-Contrast Female Fish

**What goes wrong:** Female cichlids have lower visual contrast against tank substrate. MOG2 absorbs slow/stationary fish into the background model. **Status:** YOLO added as primary detector; MOG2 retained as fallback. Detection recall for females remains a known limitation.

**Phase to address:** Detection module phase (mitigated via YOLO).

---

### Pitfall 4: Arc-Length Correspondence Errors on Curved Fish

**What goes wrong:** Arc-length normalization assumes the midline projection preserves parameterization across views. For significantly curved fish viewed from different angles, foreshortening compresses the arc-length mapping unevenly, creating triangulation errors at body points away from the head/tail endpoints.

**How to avoid:** RANSAC per body point during triangulation; view-angle weighting. **Status:** Partially mitigated by RANSAC + view-angle weighting in `triangulate_midlines()`.

**Phase to address:** Triangulation (active concern for reconstruction quality).

---

### Pitfall 5: Medial Axis Instability on Noisy Masks (IoU ~0.62)

**What goes wrong:** `skeletonize` on masks with boundary noise produces unstable, branchy skeletons that wobble frame-to-frame. **Status:** Mitigated by `_adaptive_smooth()` with morphological closing/opening and adaptive kernel radius. The keypoint backend, if implemented correctly, bypasses this entirely.

**Phase to address:** Midline extraction (partially mitigated; keypoint backend is the intended long-term fix).

---

### Pitfall 6: Head-Tail Ambiguity in Arc-Length Parameterization

**What goes wrong:** Without orientation information, the skeleton may be ordered tail-to-head in some cameras and head-to-tail in others, corrupting arc-length correspondence. **Status:** Addressed in v2.1 via `resolve_orientation()` using cross-camera geometry, velocity, and temporal prior signals.

**Phase to address:** Midline extraction (resolved in v2.1).

---

## Sources

### v3.0 Sources

- Ultralytics YOLOv8-seg task documentation: [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/)
- Ultralytics YOLOv8-pose task documentation: [Pose Estimation](https://docs.ultralytics.com/tasks/pose/)
- Ultralytics pose dataset format: [Pose Estimation Datasets Overview](https://docs.ultralytics.com/datasets/pose/)
- Ultralytics issue #4796 "Inference on rectangular image returns padded mask": [GitHub](https://github.com/ultralytics/ultralytics/issues/4796)
- Ultralytics issue #15380 "Handling Multiple Connected Regions from SAM2 for YOLO Segmentation Training": [GitHub](https://github.com/ultralytics/ultralytics/issues/15380)
- Ultralytics issue #17116 "Question about kpt_shape parameter and handling missing keypoints": [GitHub](https://github.com/ultralytics/ultralytics/issues/17116)
- Ultralytics issue #1970 "YOLOv8-Pose annotations format": [GitHub](https://github.com/ultralytics/ultralytics/issues/1970)
- Ultralytics GitHub discussion #6421 "SAM segmentation masks to YOLO format": [GitHub](https://github.com/orgs/ultralytics/discussions/6421)
- Ultralytics small dataset training discussion #6201: [GitHub](https://github.com/ultralytics/ultralytics/issues/6201)
- Ultralytics layer freezing discussion #3862: [GitHub](https://github.com/orgs/ultralytics/discussions/3862)
- Roboflow discussion on kpt_shape configuration: [What format to use for YOLOV8 Pose model training](https://discuss.roboflow.com/t/what-format-to-use-for-yolov8-pose-model-training/10568)

### v2.2 Sources

- Ultralytics YOLO OBB documentation: [Oriented Bounding Boxes Object Detection](https://docs.ultralytics.com/tasks/obb/)
- Ultralytics issue #13003 "Is the angle value given by OBB correct?": [GitHub](https://github.com/ultralytics/ultralytics/issues/13003)
- Ultralytics issue #16235 "YOLOv8-OBB angle conversion": [GitHub](https://github.com/ultralytics/ultralytics/issues/16235)
- PyTorch mixed precision training: [What Every User Should Know About Mixed Precision Training in PyTorch](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- Direct codebase inspection: `src/aquapose/reconstruction/midline.py`, `core/midline/types.py`, `core/midline/stage.py`, `core/midline/backends/segment_then_extract.py`, `engine/config.py`, `io/midline_writer.py`, `reconstruction/triangulation.py`, `segmentation/crop.py`, `segmentation/detector.py`, `core/context.py`
- Prior project pitfalls research: 2026-02-19 / 2026-02-21 (v1.0–v2.1)

---
*Pitfalls research for: v3.0 Ultralytics Unification (replacing U-Net segmentation and keypoint regression with YOLOv8-seg and YOLOv8-pose); v2.2 Backends; v1.0–v2.1 foundation*
*Researched: 2026-02-28 (v2.2) / 2026-03-01 (v3.0 additions)*

---

## v3.2 Evaluation Ecosystem Pitfalls

These pitfalls are specific to adding a unified evaluation and parameter tuning system (`aquapose eval`, `aquapose tune`) to the existing AquaPose pipeline. The primary risk sources are: pickle fragility when caching complex pipeline objects, proxy metrics that mislead rather than guide optimization, import boundary violations when adding a new orchestration layer, cache invalidation bugs in the sweep workflow, combinatorial explosion in grid search, frame selection bias, and NPZ migration hazards when splitting the monolithic diagnostic file.

---

### Pitfall C1: Pickle Cache Invalidation After Any Code Change

**What goes wrong:**
The tuning orchestrator caches upstream stage outputs (Detection, Tracking, Association, Midline) as pickle files so that downstream stage sweeps don't re-run the full pipeline per parameter combo. If the pickle cache is loaded after any change to the pickled classes — including renaming a field, adding a field with no default, changing a dataclass from mutable to frozen, or even importing from a different path — Python raises `AttributeError`, `TypeError`, or silent data corruption. The sweep machinery fails at load time with an obscure error rather than a clear "cache is stale" message.

**Why it happens:**
Pickle serializes the full Python object graph including class references and attribute names at the time of writing. The seed document explicitly acknowledges pickle's "no round-trip fidelity concerns since the pipeline code won't change mid-tuning session" — but this assumption fails the moment a developer iterates on core types during the same milestone. `Detection`, `Tracklet2D`, `TrackletGroup`, and `Midline2D` are actively developed dataclasses; any field addition or rename invalidates all cached pickles silently or noisily depending on the change.

**How to avoid:**
- Write a `cache_version` string into every pickle file (hash of the class definitions or a manually bumped version constant). On load, compare versions and raise a `StaleCacheError` with a clear message: "Tuning cache is stale — re-run baseline pipeline to refresh."
- The cache loader must catch `(AttributeError, TypeError, ModuleNotFoundError)` on unpickle and convert them to `StaleCacheError` with actionable guidance, not raw tracebacks.
- Store caches under a run-timestamped directory (not a fixed path), so re-running the baseline naturally creates a new cache without overwriting the old one.
- Document clearly in `aquapose tune` help text: pickle caches are valid only for the session that created them and are discarded after tuning completes.

**Warning signs:**
- `AttributeError: 'Detection' object has no attribute 'X'` during sweep load (field added post-cache).
- `TypeError: __init__() missing required argument` during sweep load (field added without default).
- Sweep produces metrics inconsistent with what a fresh run produces (silent data corruption).

**Phase to address:**
Cache infrastructure phase — before any sweep logic is built. The `StaleCacheError` path must be tested explicitly with a unit test that writes a cache, modifies a dataclass field, and asserts the loader raises the right error.

---

### Pitfall C2: Import Boundary Violation When Adding Orchestrator Layer

**What goes wrong:**
The new `EvalRunner` and `TuningOrchestrator` classes live above `PosePipeline` in the call stack. Developers naturally reach into `core/` types (Detection, Tracklet2D, TrackletGroup, Midline2D, MidlineSet) from the orchestrator to interpret cached stage outputs. If the orchestrator lives in a new `aquapose.evaluation` package that is technically not `engine/`, it is easy to miss that importing `core/` types from `evaluation/` is fine, but importing `engine/` types (PipelineConfig, PosePipeline, observers) from `core/` would be a boundary violation. The more subtle risk is the reverse: placing orchestration logic that instantiates `PosePipeline` inside `core/` modules, which would cause core/ to import engine/.

The AST-based import boundary checker currently enforces `core/ -> nothing from engine/ or cli/`. The evaluation package must be treated as a Layer 2 or Layer 3 module (above core/, alongside or above engine/), never as core/.

**Why it happens:**
The evaluation harness (`evaluation/harness.py`) already imports both `core/` and `engine/` modules — it is effectively a Layer 3 orchestrator. When expanding it to a full `EvalRunner`/`TuningOrchestrator`, the temptation is to colocate metric functions and orchestration logic in the same `evaluation/` package without thinking about where that package sits in the layer hierarchy. Metric functions (pure computation on numpy arrays) belong in Layer 1 (alongside core/), while orchestration (instantiating PosePipeline, loading config) belongs in Layer 3.

**How to avoid:**
- Treat `evaluation/metrics.py` (pure metric computation, no pipeline imports) as Layer 1 — it may NOT import `engine/`.
- Treat `evaluation/harness.py`, `EvalRunner`, `TuningOrchestrator` as Layer 3 — they may import both `core/` and `engine/`.
- Update the import boundary checker rule set to cover the evaluation package explicitly: `evaluation/metrics.py` and similar pure-computation files must not import `engine/`.
- Run the boundary checker in CI on every commit, not just pre-push.

**Warning signs:**
- `import_boundary_checker.py` reports violations in `evaluation/metrics.py` or `evaluation/output.py` involving `engine/` imports.
- Circular import error at runtime: `engine/pipeline.py` -> `evaluation/` -> `engine/config.py`.

**Phase to address:**
Evaluation package setup phase — before metric functions are written. Define the package's layer position in a module docstring and add a boundary checker rule before any imports are written.

---

### Pitfall C3: Proxy Metrics That Don't Correlate With Real Quality

**What goes wrong:**
Every stage-specific metric in this system is a proxy — there is no ground truth. Optimizing a proxy metric can improve the metric while degrading actual reconstruction quality. Specific failure modes:

- **Association fish yield**: Increasing `ray_distance_threshold` allows looser cross-camera matching, increasing apparent yield while incorrectly merging distinct fish into one cluster. The yield metric rises, reconstruction becomes wrong.
- **Midline smoothness**: A midline that is temporally smooth can be consistently wrong (e.g., always extracting the background instead of the fish). Smoothness measures consistency, not accuracy.
- **Detection yield stability**: Stable detection counts (low frame-to-frame variance) can be achieved by a model that always detects the same background patches, not fish.
- **Reconstruction reprojection error**: Can be minimized by rejecting most camera views as outliers (high inlier threshold), leaving only views that agree — but this reduces the effective triangulation baseline and increases 3D uncertainty.

**Why it happens:**
Proxy metrics are necessary in the absence of annotations, but they measure a single facet of quality. Developers run a sweep, see the metric improve, declare success, and move on without checking whether the downstream reconstruction improved commensurately. The modular architecture makes it easy to not check cross-stage effects.

**How to avoid:**
- Every stage sweep must include E2E validation (full pipeline + reconstruction metrics) for the top-N winners, not just stage-specific metrics. This is already in the design; it must be enforced in the code — the validation step must not be optional for production use.
- For association sweeps: after selecting a winner by fish yield, manually inspect cluster assignments for a sample of frames. A 2-camera "cluster" where both cameras are at adjacent angles is a sign of over-grouping.
- For reconstruction sweeps: track inlier count alongside reprojection error. If inlier count drops as error drops, the optimizer is excluding valid views, not actually improving.
- Document metric limitations in the eval report output. Each metric should be annotated with its known failure mode.

**Warning signs:**
- Stage metric improves, E2E reprojection error does not improve or worsens.
- Winning parameter expands a threshold (distance, angle) to an extreme value near the edge of the search grid.
- Inlier ratio drops while reprojection error drops.

**Phase to address:**
Metric design phase and sweep validation phase. The E2E validation step must be mandatory (not a flag) in the `tune` CLI, and the final report must always include both stage-specific and E2E metrics side by side.

---

### Pitfall C4: Frame Selection Bias in Sweep Evaluation

**What goes wrong:**
The sweep evaluates each parameter combination on a fixed subset of frames selected from the available fixture data. If the selection is biased (e.g., always selecting the first N frames, or selecting frames that happen to have unusually high fish visibility), the winning parameters are tuned to that subset and may generalize poorly. The project already encountered this: early sweeps with `n_frames=15` sampled too few frames to detect that 2 of 9 fish were being systematically missed (documented in MEMORY.md).

A subtler form occurs in cascade tuning: if the frame selection for the association sweep and the reconstruction sweep sample different frames (because the fixture changes between D0 and D1), the comparison between stages is confounded.

**Why it happens:**
`select_frames` currently samples uniformly from available frame indices. With 15 frames from a 1500-frame run, each sample represents 1% of the data — a single unusual batch of frames can dominate the metrics. The fix (bumping to n_frames=100) is documented but not yet enforced as a minimum.

**How to avoid:**
- Enforce a minimum `n_frames` for production sweeps (suggest 50 frames, recommend 100). Add a CLI warning if `--n-frames < 50`.
- Use stratified sampling if temporal structure matters (e.g., sample 10 frames from each decile of the recording).
- In cascade tuning, use the same frame indices across all stages within a cascade run. Pass the selected indices explicitly rather than re-sampling at each stage.
- Document frame count impact in the final report: include the number of frames evaluated and flag when it is below the recommended minimum.

**Warning signs:**
- Sweep winner differs substantially from the baseline even though the parameter change seems small.
- Running the same sweep twice with different random seeds produces different winners.
- Fish count in the sweep results varies more than expected across parameter values (frame sampling noise dominating parameter effect).

**Phase to address:**
Frame selection design phase. The `select_frames` function should be upgraded to accept a `seed` parameter for reproducibility and a `min_frames` validation. This should happen before any sweep is built on top of it.

---

### Pitfall C5: Combinatorial Grid Search Explosion

**What goes wrong:**
Grid search over N parameters with K values each requires K^N pipeline runs (or K^N stage runs for stage-specific sweeps). Association has 5 tunable parameters; reconstruction has 4. A 5-parameter grid with 7 values each = 16,807 runs. Even a 2D joint grid (7x8 = 56 runs) at 3+ minutes per full pipeline run = 3+ hours. The existing `tune_association.py` already uses a 2D joint grid for the two most interactive parameters and sequential search for the rest — this is the right approach, but it must be carried forward and not regressed.

The failure mode is adding a 3rd parameter to the joint grid (making it a 3D grid) or widening the value ranges without thinking about runtime consequences. Developers optimizing for "thoroughness" add more values, the sweep takes 8+ hours, and nobody runs it.

**Why it happens:**
Grid search is intuitive and easy to implement. The combinatorial cost is easy to underestimate when each individual run takes seconds to conceive but minutes to execute. The project currently runs on real data with a GPU; each full pipeline run is 3-5 minutes minimum.

**How to avoid:**
- Hard limit: joint grids cap at 2 parameters. Sequential (coordinate descent) search for remaining parameters.
- Default sweep ranges must be small (3-7 values per parameter). Wider ranges require explicit `--range` CLI override.
- Print estimated runtime before starting: `Estimated runtime: N combos x ~3 min/run = X hours. Proceed? [y/N]`.
- The two-tier frame count design (fewer frames during sweep, more during validation) is essential — enforce that sweep-phase `n_frames` defaults to something fast (e.g., 30 frames).
- For reconstruction sweeps specifically: reconstruction is fast (milliseconds per frame on GPU); the bottleneck is fixture generation. Since reconstruction sweeps re-use cached upstream data and only re-run Stage 5, they can use many more values safely.

**Warning signs:**
- A sweep has 3+ parameters in a joint grid.
- Default value ranges have 10+ values.
- Runtime estimate exceeds 4 hours on default settings.

**Phase to address:**
Sweep engine design phase. The sweep engine must compute and display runtime estimates before running. Default grids must be reviewed for runtime cost, not just parameter coverage.

---

### Pitfall C6: Pre-Populating PipelineContext With Wrong Types

**What goes wrong:**
The "resume from stage N" pattern requires pre-populating a `PipelineContext` with cached outputs from stages 1..N-1. `PipelineContext` fields are typed as generic stdlib types (`list`, `dict`) at the dataclass level, but downstream stages expect specific element types:
- `context.detections`: `list[dict[str, list[Detection]]]`
- `context.tracks_2d`: `dict[str, list[Tracklet2D]]`
- `context.tracklet_groups`: `list[TrackletGroup]`
- `context.annotated_detections`: `list[dict[str, list[AnnotatedDetection]]]`

If the context loader deserializes from pickle and the types match, everything works. But if a developer constructs a context manually for testing (e.g., putting a `list[dict]` instead of `list[dict[str, list[Detection]]]`), the stage that consumes that field will silently get wrong data. Type errors surface at attribute access inside the stage (e.g., `detection.bbox`) rather than at context assignment, making the source of the error hard to trace.

**Why it happens:**
`PipelineContext` uses `list | None` and `dict | None` annotations (not generic parameterized types) to avoid importing core types into `context.py` (which lives in core/ and must not import specific stage types). This means the type checker cannot catch mismatched element types at the context assignment site.

**How to avoid:**
- The context loader (the function that reconstructs a PipelineContext from pickle) must be the only code path that populates context fields from cache. Never construct a pre-populated context manually in production code.
- For the test suite: create a `build_test_context(stage: str)` fixture factory that returns correctly-typed synthetic data. This factory is the single source of truth for test context construction.
- Add runtime validation in the context loader: after unpickling, check that `isinstance(context.detections[0], dict)` and that the dict values are lists with the expected element type. Raise `ContextTypeError` with field name and actual type if violated.

**Warning signs:**
- `AttributeError: 'dict' object has no attribute 'bbox'` inside a stage run (dict where Detection expected).
- Stage completes without error but produces empty or wrong outputs (silent type mismatch).
- Test that passes on correct fixture fails with confusing errors when context is hand-constructed.

**Phase to address:**
Context loader design phase. The loader's type validation must be tested with both correct and incorrect inputs before any sweep logic uses it.

---

### Pitfall C7: Monolithic NPZ Migration Breaks Existing Consumers

**What goes wrong:**
The existing diagnostic observer exports a single `pipeline_diagnostics.npz` file. The v3.2 design replaces this with per-stage files. Any code that reads `pipeline_diagnostics.npz` — including the existing `aquapose eval` harness, Jupyter notebooks for exploration, and the fixture loader — will break silently or noisily when the file disappears.

The migration has two hazards:
1. **Hard break**: Old run directories contain `pipeline_diagnostics.npz`; new code looks for per-stage files and finds nothing.
2. **Silent partial read**: New code reads per-stage files; some stages are missing because the pipeline was stopped early (`stop_after`). Code that assumes all 5 stage files exist will fail or silently use stale data.

**Why it happens:**
Replacing a monolithic format with a structured one is a breaking change to the serialized output format. The existing codebase treats `pipeline_diagnostics.npz` as a stable artifact. The seed document acknowledges this but does not specify a migration strategy.

**How to avoid:**
- Implement a compatibility shim: if `pipeline_diagnostics.npz` exists in the run directory but per-stage files do not, the context loader reads from the monolithic file. Log a deprecation warning.
- The per-stage file names must be well-defined constants (not constructed from strings inline). Define them in a single location: `evaluation/fixtures.py` or similar.
- The context loader must handle missing stage files gracefully — `stop_after` runs legitimately lack downstream stage files.
- Add an `aquapose migrate-run <run-dir>` subcommand that converts old monolithic NPZ to per-stage files, so existing run directories can be upgraded without re-running the pipeline.

**Warning signs:**
- `FileNotFoundError` for per-stage files in old run directories after upgrade.
- `KeyError` when reading NPZ keys that have been renamed or restructured.
- Silent empty metrics because per-stage file for an early-stopped run is missing and code defaults to empty arrays.

**Phase to address:**
Diagnostic file restructuring phase. The shim and the per-stage filename constants must be in place before any consumer code is migrated. Migrate consumers one at a time, testing each before moving to the next.

---

### Pitfall C8: Over-Tuning on the Sweep Dataset (Overfitting to a Recording)

**What goes wrong:**
All tuning is performed on a single recording (the YH project data). Parameters tuned on this recording — lighting conditions, fish behavior, tank geometry, water turbidity, specific camera positions — may not transfer to other recordings. This is the evaluation equivalent of overfitting. It is particularly acute for association parameters (which depend on ray geometry and camera arrangement) and reconstruction parameters (which depend on image quality and fish contrast).

The current codebase has already experienced this: association defaults were kept (not tuned) because sweeps showed only ~1% yield improvement — a correct decision that could easily be rationalized away as "the sweep didn't converge" if the metric were noisier.

**Why it happens:**
The tuning system is designed for a single-recording, single-rig workflow. There is no cross-recording validation set. Developers trust the sweep results because they are quantitative.

**How to avoid:**
- Treat tuned parameter changes as hypotheses, not conclusions. For any parameter that deviates significantly from the default, manually inspect reconstructions on 10-20 frames to confirm the improvement is real.
- Document the recording used for tuning in the config diff output ("tuned on YH/run_20260303..."). When parameters are used on a new recording, re-run the sweep on that recording's data.
- Keep default grid ranges narrow (centered around the current default). Wide ranges suggest the developer is searching, not validating.
- The final report must state explicitly: "These parameters were tuned on [recording name]. Re-tune for different rigs or recordings."

**Warning signs:**
- Winning parameters are at the extreme edge of the search grid (suggests the real optimum is outside the searched range, or the metric is noisy).
- Winning parameters differ substantially from the manufacturer/domain defaults with no intuitive explanation.
- Manual inspection shows reconstruction looks no better (or worse) despite metric improvement.

**Phase to address:**
Reporting design phase. The output report must include the recording identifier and a caveat about generalization. This is a documentation and UX concern, not a code correctness concern.

---

### Pitfall C9: TrackingStage's Non-Standard run() Signature Breaks Resume-From-Stage Logic

**What goes wrong:**
`TrackingStage` uses a different `run()` signature from all other stages:
```python
# Standard Stage protocol:
def run(self, context: PipelineContext) -> PipelineContext: ...

# TrackingStage:
def run(self, context: PipelineContext, carry: CarryForward | None) -> tuple[PipelineContext, CarryForward]: ...
```
`PosePipeline.run()` has a special `isinstance(stage, TrackingStage)` branch to handle this. The "resume from stage N" pattern loads cached context and runs a truncated stage list starting from stage N. If the resume point is after tracking (stages 4 or 5), the carry forward state from the cached run must also be loaded and passed — it is not stored in PipelineContext, it is a separate `CarryForward` object.

If a context loader loads only `PipelineContext` from the cache and ignores `CarryForward`, then a resume that happens to include TrackingStage (e.g., for a stage 2 sweep) will instantiate a fresh TrackingStage with no prior state — silently producing incorrect tracklets that don't match the cached upstream association data.

**Why it happens:**
`CarryForward` is an architectural exception to the "everything lives in PipelineContext" rule. It exists to persist OC-SORT state across batches, but tuning operates on fixed-length frame windows (not batches), so `CarryForward` is less relevant in the tuning context. Developers may assume `CarryForward` can be ignored for single-batch sweeps and discover this is wrong only when results are wrong.

**How to avoid:**
- The cache format for a full baseline run must include both `PipelineContext` and `CarryForward`. The context loader must restore both.
- When the resume point is after TrackingStage (i.e., stages 3/4/5 sweeps), the `CarryForward` from the cached run must be passed to the truncated pipeline's `TrackingStage` if tracking is included in the truncated stage list.
- Add an integration test: cache a full run, resume from stage 3 with the cached stage 1+2 outputs, verify that association outputs match the original full run.

**Warning signs:**
- Sweep yields different results than the cached baseline despite using identical parameters.
- Association metrics show high singleton rates even for the baseline parameter combo when running from cache.

**Phase to address:**
Context loader design phase. The loader must explicitly handle CarryForward alongside PipelineContext. This must be verified with an integration test before any sweep relies on cached tracking output.

---

### Pitfall C10: Evaluating Only "evaluate only" Stages Produces False Confidence

**What goes wrong:**
The design marks Detection, Tracking, and Midline as "evaluate only" (no sweep capability). Evaluation metrics for these stages (detection yield, track length, midline completeness) give a numeric score but no lever to act on. A developer who runs `aquapose eval` and sees "detection yield: 0.72" knows something is wrong but cannot improve it without retraining or changing the camera setup. The eval report may create false confidence that the system is being managed when the actionable decisions (retrain, recalibrate) are outside the scope of this system.

Additionally, "evaluate only" stages must still have their metrics tested — if the metric computation has a bug, the number is wrong but nothing fails. A metric that always returns a plausible value regardless of actual stage quality is worse than no metric.

**Why it happens:**
It is easy to write a metric function that returns a reasonable-looking number. Without ground truth, there is no oracle to check against. Unit tests for proxy metrics typically only check that the function runs without error and returns a number in the expected range, not that the number reflects reality.

**How to avoid:**
- For each "evaluate only" metric, write a test with a synthetic context where the correct answer is known: e.g., inject a context where 8 of 13 cameras detect a fish and verify detection yield = 8/13.
- Distinguish clearly in the eval report between "actionable metrics" (association, reconstruction — can be improved by tuning) and "diagnostic metrics" (detection, tracking, midline — inform decisions outside this system).
- Include in the report: "Detection yield is below 0.85. Possible causes: lighting change, detector needs retraining. Run `aquapose train yolo-obb` to retrain."
- Do not surface "evaluate only" metrics with the same prominence as tunable metrics in the report.

**Warning signs:**
- Metric computation passes unit tests with no edge cases tested.
- Metric reports a value that does not change across different configurations (metric is always returning a constant).
- Eval report is read and filed away without triggering any action (metrics are not actionable).

**Phase to address:**
Metric implementation phase. Every metric function must have a unit test with a synthetic context where the expected value is analytically known. This is a testability requirement, not just a quality bar.

---

## v3.2 Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip StaleCacheError, let AttributeError surface naturally | No error handling code to write | Confusing errors; developers re-run sweeps from scratch unnecessarily | Never — cost is low, benefit is high |
| Use fixed frame indices (first N frames) instead of stratified sampling | Simple implementation | Systematic bias; parameters tuned to recording opening may not generalize | Only for quick exploratory runs with `--n-frames < 10`, explicitly flagged |
| Report only stage-specific metric in sweep, skip E2E validation | Faster sweeps | Parameters that win stage metric but hurt reconstruction go undetected | Never for production tune runs; acceptable for exploratory debugging |
| Hardcode sweep ranges inline in CLI commands | Avoids designing a DEFAULT_GRIDS structure | Ranges scattered across files; changing them requires touching CLI code | Never — DEFAULT_GRIDS in evaluator modules is the right design |
| Keep monolithic NPZ and add per-stage reading | No migration needed | Monolithic file is loaded entirely even when only one stage is needed; migration never happens | Acceptable temporarily during transition if a shim is implemented |
| Use `eval()` or string formatting to construct parameter override dicts | Quick for prototyping | Security risk in CLI context; brittle for parameter names with dots | Never — use dataclasses.replace() directly |

---

## v3.2 Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PosePipeline with initial context | Passing a context pre-populated with data from a different pipeline version (different camera set, different frame range) | Context loader must validate camera_ids and frame_count against the new config before resuming |
| DiagnosticObserver + per-stage files | DiagnosticObserver writes a snapshot after each StageComplete event — if a stage fails, later stages are not captured but the partial NPZ is still written | Context loader must check for a `pipeline_complete` sentinel key in per-stage files; partial files must be treated as invalid |
| Frozen dataclass config + sweep overrides | Using `dataclasses.replace()` on a frozen config creates a new frozen object; passing the modified config to `build_stages()` works, but the run_id and output_dir will still be the same as the original config, causing all sweep runs to write to the same output directory | Each sweep combo must get a unique run_id (e.g., `run_id + "_sweep_" + param_hash`) and output_dir |
| Click CLI + numeric parameter ranges | `--range min:max:step` parsed as a string then split; edge cases include negative values, scientific notation, and integer-vs-float distinction | Parse with explicit `np.arange(min, max+step, step)` after splitting; validate result is non-empty |
| Evaluation harness + stop_after runs | `aquapose eval <run-dir>` on a run that used `stop_after=association` will have no midline or reconstruction data; eval code that unconditionally reads all stage files will fail | Check which per-stage files exist before evaluating; report metrics only for available stages |

---

## v3.2 Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Full pipeline run per association combo | Sweep takes hours; GPU is utilized for all 5 stages when only stage 3 parameters change | Cache stages 1+2 outputs; run only stage 3 per combo | Immediately at >20 combos with 3-minute pipeline runtime |
| Tier 2 leave-one-out during sweep (not just validation) | Sweep takes 13x longer (13 camera dropout iterations per frame per combo) | Always skip_tier2=True during sweep phase; only run Tier 2 for top-N validation | At >10 combos |
| Unpickling large context objects for every sweep combo | Memory spikes; GC pressure; slow sweeps | Unpickle once at sweep start, keep in memory as the "cache object", copy or slice as needed | At >50 sweep combos with large frame counts |
| Loading all per-stage files when eval targets one stage | I/O overhead for unused files | Lazy loading — open per-stage file only when the corresponding stage metric is requested | At runs with large per-stage files (association fixture can be hundreds of MB) |

---

## v3.2 "Looks Done But Isn't" Checklist

- [ ] **Pickle cache**: verify StaleCacheError is raised (not AttributeError) when a dataclass field changes between cache write and load.
- [ ] **Sweep E2E validation**: verify the validation step runs for top-N winners even when the stage-specific metric has already converged — it must not be short-circuited.
- [ ] **Import boundaries**: run `tools/import_boundary_checker.py` on the new `evaluation/` package after adding orchestrator imports; verify no IB-001 violations.
- [ ] **Frame selection seed**: verify that two runs of `aquapose tune` with the same arguments produce the same frame selection (reproducibility).
- [ ] **Per-stage file migration shim**: verify that `aquapose eval` on an old run directory (monolithic NPZ, no per-stage files) produces a deprecation warning and correct metrics, not a FileNotFoundError.
- [ ] **CarryForward in cache**: verify that a cached run's CarryForward is restored correctly and that resuming from stage 3 produces the same tracklet_groups as the original full run.
- [ ] **Metric unit tests**: verify each of the 5 stage metric functions has at least one test with a synthetic context where the expected value is analytically known (not just "returns a number in range").
- [ ] **Config diff in report**: verify the final report includes the parameter diff between baseline config and winning config in copy-pasteable YAML format.

---

## v3.2 Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Stale pickle cache | LOW | Delete the tuning work directory; re-run baseline pipeline to regenerate cache |
| Import boundary violation | MEDIUM | Move the violating import to a wrapper in the correct layer; re-run boundary checker |
| Proxy metric misleads tuning | MEDIUM | Manually inspect reconstructions for the "winning" parameters; revert config if inspection fails |
| Frame selection bias | LOW | Re-run sweep with larger n_frames (100+) and different seed; compare winners |
| Grid explosion (sweep takes 8+ hours) | MEDIUM | Kill sweep; reduce grid dimensions (remove 1 parameter from joint grid); restart |
| Monolithic NPZ migration breaks consumers | MEDIUM | Implement compatibility shim reading old format; do not merge migration until shim is tested |
| CarryForward not cached | HIGH | All swept association results are invalid (wrong tracklets); must re-run baseline and all sweep combos |

---

## v3.2 Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| C1: Pickle cache invalidation | Cache infrastructure design | Unit test: write cache, modify dataclass field, assert StaleCacheError on load |
| C2: Import boundary violation | Evaluation package setup | `tools/import_boundary_checker.py` reports 0 violations on evaluation/ package |
| C3: Proxy metric misleads | Metric design + sweep validation | Final report shows both stage-specific and E2E metrics; E2E step is not skippable |
| C4: Frame selection bias | Frame selection design | `select_frames` accepts seed; min_frames validation; same indices used across cascade stages |
| C5: Combinatorial explosion | Sweep engine design | Runtime estimate printed before sweep starts; default grids have ≤7 values per parameter |
| C6: PipelineContext type mismatch | Context loader design | Loader runtime-validates element types; `build_test_context` fixture used in all tests |
| C7: Monolithic NPZ migration | Diagnostic file restructuring | Compatibility shim tested on old run directory before any consumer is migrated |
| C8: Over-tuning on one recording | Reporting design | Report states recording identifier; parameter diff is human-reviewable before applying |
| C9: CarryForward ignored | Context loader design | Integration test: cached resume produces same tracklet_groups as full run |
| C10: Evaluate-only metric bugs | Metric implementation | Every metric has synthetic test with analytically known expected value |

---

### v3.2 Sources

- Direct codebase inspection: `src/aquapose/engine/pipeline.py`, `src/aquapose/core/context.py`, `src/aquapose/engine/diagnostic_observer.py`, `src/aquapose/evaluation/harness.py`, `scripts/tune_association.py`, `tools/import_boundary_checker.py`
- Python pickle documentation (module docs, warning on class evolution): https://docs.python.org/3/library/pickle.html
- Scikit-learn parameter tuning anti-patterns (proxy metric overfitting): https://scikit-learn.org/stable/common_pitfalls.html
- MEMORY.md: n_frames=15 sampling artifact hid 2/9 fish — documented real-world frame selection bias
- PROJECT.md: association sweep showed ~1% yield improvement — correct decision to not over-tune

---
*Pitfalls research for: v3.2 Evaluation Ecosystem (unified eval/tune CLI, per-stage proxy metrics, cascade tuning, partial pipeline execution)*
*Researched: 2026-03-03*
