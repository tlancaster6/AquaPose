# Detection & Segmentation: Implementation Brief

## System Parameters

- **Image resolution**: 1600×1200, 13 cameras, 30 fps
- **Subjects**: 9 fish (~80-100px long in nearest cameras), 3 male, 6 female
- **Environment**: Clear water, controlled diffuse lighting, static cameras, some shadow artifacts
- **Challenge**: Females are low-contrast against background
- **Processing mode**: Batch (not real-time)

## Stage 1: Detection

**Objective**: Produce a bounding box for every fish in every camera frame. Optimize for recall over precision — false positives are acceptable, missed fish are not.

**Primary method**: OpenCV MOG2 background subtraction

- Color space: HSV or LAB (chromaticity-based thresholding to reject shadows)
- Enable MOG2 shadow detection flag
- Background adaptation rate: slow (fish that pause briefly should not be absorbed)
- Post-processing: connected components, filter by area (reject blobs outside plausible fish-size range at known depth)
- Output: bounding boxes with ~30% padding

**Fallback method**: YOLOv8-det (detection only, no mask head), fine-tuned on project annotations. Deploy only if MOG2 per-camera detection rate falls below acceptance threshold.

**Safety net** (always active): 3D tracker injects predicted bounding boxes for any tracked fish not detected in a given camera. This runs regardless of which detection method is active.

**Acceptance criteria**:

- Per-camera per-frame detection rate ≥95% (measured against manual annotation on 500+ frames spanning all cameras)
- Evaluate separately for males and females — female recall must meet the same ≥95% bar
- False positive rate is unconstrained but should not exceed ~3× the true detection count (to keep Stage 2 compute reasonable)

## Stage 2: Segmentation

**Objective**: Produce a binary body mask (excluding fins) for each detected fish. Mask boundary accuracy directly determines Phase III reconstruction quality.

**Input**: Cropped and resized patches (256×256) from Stage 1 bounding boxes.

**Primary method**: Mask R-CNN (ResNet-50-FPN backbone) via Detectron2, fine-tuned on project annotations, running on crops.

**Fallback if boundary quality insufficient**: Add PointRend head (~25% slower).

**Alternative if Mask R-CNN is overkill on crops**: Binary U-Net encoder-decoder. Lighter, potentially faster, but requires custom training loop.

**Training data**: SAM single-frame pseudo-labels, human-corrected. Target: 300+ corrected frames per camera viewpoint, with deliberate oversampling of low-contrast females and overlapping-fish cases.

**Acceptance criteria**:

- Mean mask IoU ≥0.90 against human-corrected ground truth, measured on a held-out validation set
- IoU for low-contrast females specifically ≥0.85
- Mean boundary error ≤3 pixels (measured as mean distance from predicted boundary to GT boundary)
- Visual QA: projected 3D mesh overlay (from Phase III) should align with visible fish body in all camera views — this is the downstream acceptance test that ultimately matters

## Decision Sequence

1. Implement MOG2 background subtraction. Evaluate detection rate on 500+ annotated frames.
2. If ≥95% recall (including females): proceed. If not: train and deploy YOLOv8-det.
3. Implement Mask R-CNN on crops. Evaluate mask IoU on held-out validation set.
4. If IoU ≥0.90 (≥0.85 for females): proceed. If not: add PointRend and re-evaluate.
5. If still insufficient: investigate U-Net alternative or revisit annotation quality.
