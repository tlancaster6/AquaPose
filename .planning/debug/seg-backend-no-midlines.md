---
status: awaiting_human_verify
trigger: "Seg backend produces almost no midlines despite excellent training metrics (mAP50=0.995). Pipeline run run_20260301_190623 had near-zero reconstruction (0.8s)."
created: 2026-03-01T00:00:00Z
updated: 2026-03-01T03:00:00Z
---

## Current Focus

hypothesis: CONFIRMED — training-inference crop mismatch caused near-zero mask output.
test: Fixed _extract_crop in SegmentationBackend to replicate training's cv2.getAffineTransform stretch-fill. All 14 unit tests pass.
expecting: Pipeline re-run should produce midlines for most detections.
next_action: User verifies fix by re-running pipeline with seg backend.

## Symptoms

expected: Seg backend should produce fish midlines for most detections across 12 cameras, leading to successful 3D reconstruction. Training mAP50=0.995 for both box and mask.
actual: Very few midlines made it to 2D visualizations. Reconstruction took only 0.8s (near-zero work), meaning most midlines failed or were empty.
errors: No crash errors reported — the pipeline completes, it just produces near-empty results.
reproduction: Run pipeline with seg backend using --mode diagnostic. Run output at C:\Users\tucke\aquapose\projects\YH\runs\run_20260301_190623.
timeline: First pipeline run with v3.0 YOLO-seg model. Just trained and integrated in phase 37-02.

## Eliminated

(none)

## Evidence

- timestamp: 2026-03-01T02:00:00Z
  checked: segmentation.py _extract_crop method (lines 245-254, original)
  found: called extract_affine_crop(..., fit_obb=True, mask_background=True, padding_fraction=0.15 default)
  implication: inference crops are LETTERBOXED — fish is scaled to fit inside 128x64 while preserving aspect ratio, with black padding around it

- timestamp: 2026-03-01T02:00:00Z
  checked: tmp/convert_all_annotations.py build_seg_dataset (line 432)
  found: training data uses affine_warp_crop(crop_img, obb_corners, CROP_W=128, CROP_H=64) — cv2.getAffineTransform mapping TL/TR/BL OBB corners to canvas corners
  implication: training crops STRETCH the fish to fill the entire 128x64 canvas, no letterboxing, no black padding

- timestamp: 2026-03-01T02:00:00Z
  checked: scripts/build_yolo_training_data.py affine_warp_crop function (lines 346-379)
  found: src=[TL, TR, BL] -> dst=[(0,0), (crop_w-1,0), (0,crop_h-1)] via cv2.getAffineTransform
  implication: OBB is stretched to fill the rectangle. Fish occupies the entire canvas at training time.

- timestamp: 2026-03-01T02:00:00Z
  checked: segmentation/crop.py extract_affine_crop with fit_obb=True
  found: scale = min(crop_w / obb_w, crop_h / obb_h) — letterbox scale. For a fish at say 200x40px OBB, scale = 0.64, fish occupies 128x25.6 inside the 128x64 canvas (40% fill)
  implication: Large portions of inference crop are black padding. Model trained on full-canvas fish never learned to find fish in a letterboxed canvas with black borders.

- timestamp: 2026-03-01T02:00:00Z
  checked: stage.py lines 172-180 (segmentation backend instantiation)
  found: crop_size parameter NOT passed to get_backend("segmentation", ...) — only pose_estimation branch passes crop_size
  implication: latent bug fixed as part of this work

- timestamp: 2026-03-01T02:00:00Z
  checked: run config.yaml midline section
  found: confidence_threshold=0.5, min_area=300
  implication: model outputs low-confidence masks due to distribution mismatch, all filtered at conf=0.5. Zero midlines out.

## Resolution

root_cause: Training-inference crop preparation mismatch. Training uses cv2.getAffineTransform (stretch-to-fill, 3-point affine from OBB corners to canvas corners) producing fish that fill the entire 128x64 canvas. Inference used extract_affine_crop with fit_obb=True which letterboxes the fish (scale-preserving fit) with black borders. The YOLO-seg model sees a completely different input distribution at inference — fish are small and centered with black padding — causing poor/no mask predictions, all filtered by confidence_threshold=0.5.

fix: |
  1. src/aquapose/core/midline/backends/segmentation.py:
     Rewrote _extract_crop to use cv2.getAffineTransform with 3 OBB corner points
     mapped to canvas corners (TL→(0,0), TR→(W-1,0), BL→(0,H-1)) — exactly matching
     training data preparation. Falls back to extract_affine_crop only when obb_points
     is absent (non-OBB detectors).
  2. src/aquapose/core/midline/stage.py:
     Added crop_size parameter to segmentation backend instantiation (was missing,
     latent bug — would matter if crop_size ever differed from default (128,64)).

verification: 14 unit tests (test_segmentation_backend.py + test_midline_stage.py) all pass. Lint clean.
files_changed:
  - src/aquapose/core/midline/backends/segmentation.py
  - src/aquapose/core/midline/stage.py
