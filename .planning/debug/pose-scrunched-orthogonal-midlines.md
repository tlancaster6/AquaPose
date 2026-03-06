---
status: awaiting_human_verify
trigger: "pose-backend-scrunched-orthogonal-midlines"
created: 2026-03-01T00:00:00Z
updated: 2026-03-01T00:05:00Z
---

## Current Focus

hypothesis: CONFIRMED — Ultralytics OBB does not guarantee w>=h. When h>w the inference affine crop was mapping fish short-axis to crop width (128px) and long-axis to crop height (64px), 90 degrees opposite training.
test: Code fix applied, 32 relevant unit tests pass.
expecting: With fix, all crops will have long axis along width regardless of w/h ordering
next_action: Human verification — run pipeline and check crop diagnostics

## Symptoms

expected: All midlines should follow the fish body axis (long axis of OBB) with reasonable length
actual: Binary outcome — either decent midline OR small scrunched midline perpendicular to fish body
errors: No crashes. Pipeline completes normally.
reproduction: Run pipeline with pose backend. Output at C:\Users\tucke\aquapose\projects\YH\runs\run_20260301_204119
started: After fixing OBB corner ordering (pts[2] as TL) and stretch-fill crop

## Eliminated

(none yet)

## Evidence

- timestamp: 2026-03-01T00:00:00Z
  checked: symptom description
  found: binary good/bad outcome with scrunched+orthogonal pattern
  implication: strongly suggests OBB orientation ambiguity — sometimes w>h, sometimes h>w, changing which axis maps to long crop dimension

- timestamp: 2026-03-01T00:01:00Z
  checked: scripts/build_yolo_training_data.py pca_obb + affine_warp_crop
  found: pca_obb always builds corners [TL,TR,BR,BL] where long axis is horizontal (PCA main axis = first component). affine_warp_crop maps src=[TL,TR,BL] → dst=[(0,0),(W-1,0),(0,H-1)]. TL→TR is ALWAYS the long axis.
  implication: training guarantees long axis maps to crop width (128px)

- timestamp: 2026-03-01T00:02:00Z
  checked: _extract_crop in pose_estimation.py and segmentation.py
  found: maps src=[pts[2],pts[1],pts[3]] = [LT,RT,LB] → [(0,0),(W-1,0),(0,H-1)]. LT→RT = Ultralytics "w" direction (along angle). LT→LB = Ultralytics "h" direction (perp to angle). NO check of which side is longer.
  implication: when Ultralytics OBB has w<h (fish long axis along h), the short side maps to width 128px, long side maps to height 64px — fish is rotated 90 degrees AND compressed

- timestamp: 2026-03-01T00:03:00Z
  checked: yolo_obb.py backend
  found: stores raw corners from r.obb.xyxyxyxy without any axis normalization. xywhr w and h are stored but not passed to Detection — only angle is stored.
  implication: Detection.obb_points has no guarantee that "long axis" = w direction

- timestamp: 2026-03-01T00:04:00Z
  checked: root cause confirmed
  found: Ultralytics OBB doesn't guarantee w>=h for elongated objects. When h>w (fish oriented perpendicular to angle), inference crop maps short→128px and long→64px, exactly opposite of training. Model sees a 90-degree rotated, aspect-ratio-wrong crop → keypoints predicted in wrong positions → midline scrunched and orthogonal.
  implication: fix = before building affine, measure side lengths and if LT→RT < LT→LB, rotate the corner assignment 90 degrees so long axis maps to crop width

## Resolution

root_cause: Ultralytics OBB does not guarantee w>=h. The YOLO-OBB model can output detections where w<h (fish long axis in h direction). The inference _extract_crop in both pose and seg backends naively mapped LT→RT (= w direction) to crop width (128px) without checking which side is longer. When h>w, this maps the short axis to 128px and long axis to 64px — a 90-degree rotated, aspect-ratio-inverted crop. The model was trained on crops where long axis always = horizontal (guaranteed by pca_obb), so it produces wrong keypoints on these flipped crops, resulting in scrunched orthogonal midlines.
fix: In _extract_crop for both backends, measure side lengths before building affine src. If LT→LB (h direction) > LT→RT (w direction), rotate the corner assignment: use [LB, LT, RB] as src (so LB→LT = long axis maps to crop width) instead of [LT, RT, LB].
verification: 32 relevant unit tests pass. Pending human verification with a pipeline run.
files_changed:
  - src/aquapose/core/midline/backends/pose_estimation.py
  - src/aquapose/core/midline/backends/segmentation.py
