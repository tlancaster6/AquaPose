---
status: awaiting_human_verify
trigger: "keypoint-midline-ribbon-loops"
created: 2026-02-28T00:00:00Z
updated: 2026-02-28T00:01:00Z
---

## Current Focus

hypothesis: CONFIRMED AND FIXED
test: ran model on 5 training images; observed ribbon fix via x-sort; all 687 tests pass
expecting: user to confirm smooth midlines in video output
next_action: await user verification on real video

## Symptoms

expected: Smooth head-to-tail arcs following the fish spine, no loops or crossings. Midlines should be gentle curves.
actual: Midlines consistently show an unusual loop shape like an "awareness ribbon" — very common across video output.
errors: None — model runs fine, produces bad geometry.
reproduction: Run full pipeline with trained keypoint model on video data.
started: First full deployment of keypoint midline backend (Phase 33). Model validated well (val error 0.0470 ≈ 6px) during training but fails at inference on real video.

## Eliminated

- hypothesis: coordinate space mismatch (training uses axis-aligned crops, inference uses affine crops)
  evidence: direct model testing shows predictions match GT for VISIBLE keypoints — the crop format is close enough
  timestamp: 2026-02-28

- hypothesis: keypoint ordering mismatch in annotation vs model output
  evidence: visible keypoints (2 annotated) are predicted accurately (kp0 and kp1 match GT within ~2px)
  timestamp: 2026-02-28

- hypothesis: sigmoid denormalization bug
  evidence: kp0 at x=7.0 vs GT 5.8 is correct; denormalization (*128) is working correctly
  timestamp: 2026-02-28

## Evidence

- timestamp: 2026-02-28
  checked: training annotation COCO JSON
  found: most annotations have only 2-3 visible keypoints (v=2) out of 6; remaining have v=0 (invisible, coords set to 0,0)
  implication: model can't learn full 6-keypoint spine from 2-3 visible points; invisible keypoints produce nonsense predictions

- timestamp: 2026-02-28
  checked: ran v2 model on 5 training images
  found: for image with only 2 visible keypoints (Nose + Head), model predicts x=[7.0, 94.6, 99.4, 77.9, 95.2, 104.7] — kp3 at x=77.9 drops BEHIND kp2 and kp4, creating non-monotonic sequence
  implication: non-monotonic x-values passed to CubicSpline with t=[0,0.2,0.4,0.6,0.8,1.0] forces the spline to backtrack, creating the ribbon loop shape

- timestamp: 2026-02-28
  checked: ran v2 model on image with 5 visible keypoints
  found: predictions [5.9, 27.0, 45.2, 69.5, 96.6, 121.2] are monotone and accurate — no loops would occur
  implication: the bug only triggers when many keypoints are invisible in training; dense annotations fix it

- timestamp: 2026-02-28
  checked: config.yaml keypoint_weights_path
  found: config points to output/best_model.pth (earlier model), not output_v2/best_model.pth (latest training)
  implication: additional issue — wrong model file; v2 was trained on letterboxed data with better preprocessing

## Resolution

root_cause: Training annotations are sparse (2-3 visible keypoints out of 6 per image). The masked loss correctly ignores invisible keypoints during training, but this means the model never learns consistent positions for them. At inference, all 6 keypoints pass the confidence floor (center-of-crop heuristic) but their x-values are non-monotone (e.g., x=[7.0, 94.6, 99.4, 77.9, 95.2, 104.7] — kp3 drops back to 77.9). CubicSpline through non-monotone points parameterized by t=[0,0.2,...,1.0] must backtrack, creating the ribbon/loop shape.

fix: In DirectPoseBackend._process_single_detection (step 6a), after filtering by confidence floor, sort visible keypoints by crop-space x-coordinate and re-assign uniform t-values via linspace(0,1,V). This enforces the left-to-right ordering consistent with training data, preventing non-monotone inputs to CubicSpline. Also updated config.yaml to use output_v2/best_model.pth (the model trained on letterboxed data). Also renamed test_partial_visibility_nan_padding to test_partial_visibility_no_loops with updated assertions reflecting the new behavior (full span always observed, no NaN-padding when visible kps cover [0,1]).

verification: ran model on 5 training images — loop-causing non-monotone x-values eliminated; sorted x-values are strictly increasing; all 687 unit tests pass
files_changed:
  - src/aquapose/core/midline/backends/direct_pose.py
  - tests/unit/core/midline/test_direct_pose_backend.py
  - C:/Users/tucke/aquapose/projects/YH/config.yaml
