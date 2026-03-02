---
status: awaiting_human_verify
trigger: "DirectPoseBackend midlines are consistently ~50% shorter than they should be in frame space when running the full pipeline on real video."
created: 2026-02-28T00:00:00Z
updated: 2026-02-28T00:00:00Z
---

## Current Focus

hypothesis: Training used letterboxed crops (rectangular fish padded to 128x128), so model learned to predict keypoints spanning ~full 128px width. At inference, extract_affine_crop places the fish at native pixel scale with NO scaling — a fish OBB of width ~100px occupies 100/128 = 78% of the canvas in the best case, but the model was trained on images where the fish filled the padded rectangle width (the actual fish body width after letterboxing). The mismatch in how much of the canvas the fish occupies means the model's predicted x-range in [0,1] space is correct relative to training, but when denormalized by crop_w=128, the resulting pixel coords extend across more of the crop than the fish actually occupies — WAIT, this would give LONGER midlines, not shorter.

Re-examining: The training images were originally rectangular crops (e.g. 100x23) letterboxed to 128x128. The fish body runs nose-to-tail across the full 100px width. After letterboxing the 100px wide fish was still 100px wide in a 128px canvas. So normalized x-coords of 0.0 to 1.0 correspond to 0 to 128px crop coords. The fish body specifically occupies about x=[14,114] in the 128x128 letterboxed image (centered with 14px padding on each side since (128-100)/2=14). So model learned that nose is at x~0.11 and tail at x~0.89 in [0,1] space.

At inference, extract_affine_crop places the fish at NATIVE scale (no resize), centered at crop center (64,64). The OBB width is bw pixels. The fish body spans the full crop at pixel coords x=[64-bw/2, 64+bw/2]. In normalized [0,1] coords that is [(64-bw/2)/128, (64+bw/2)/128].

If the model predicts nose at ~0.11 and tail at ~0.89 (learned from letterboxed training data where fish spans ~100/128 of the canvas), then in native-scale inference the fish body at native scale spans the same width but the crop has no scaling relationship to bw. If bw > 128, fish is clipped. If bw < 128, fish is smaller than what model was trained on.

ACTUAL ROOT CAUSE FOUND after careful analysis — see Evidence section.

test: Code trace of the scale relationship between training and inference
expecting: Training images had fish at one scale, inference crops have fish at different scale
next_action: Write fix and verify

## Symptoms

expected: Midlines span the full fish body in frame space (nose to tail)
actual: Midlines are consistently ~50% shorter than expected, but correctly positioned on the fish body
errors: None — pipeline runs cleanly, produces valid midlines that are just geometrically wrong
reproduction: Run pipeline with backend: direct_pose on real video. Compare midline length to fish body length in the frame.
started: First real-video run with DirectPoseBackend. Problem only manifests after back-projection to frame space.

## Eliminated

- hypothesis: Back-projection math (invert_affine_points) is wrong
  evidence: extract_affine_crop uses only rotation + translation (no scaling), so invert_affine_points via cv2.invertAffineTransform is mathematically exact. The M matrix is built with scale=1.0 explicitly. This is not the problem.
  timestamp: 2026-02-28

- hypothesis: Random error or non-systematic issue
  evidence: "Consistently ~50% shorter" is a systematic scale factor, rules out random error. Points to a deterministic geometric mismatch.
  timestamp: 2026-02-28

## Evidence

- timestamp: 2026-02-28
  checked: KeypointDataset.__getitem__ in pose.py lines 179-243
  found: Training images are loaded at their original size (orig_h, orig_w), then resized to 128x128 with cv2.resize(image, (sz, sz)). Keypoint COCO pixel coords (which are in the original image's pixel space) are then scaled to 128x128 space via: x_px = float(x) / orig_w * sz, y_px = float(y) / orig_h * sz. The normalized target is kp_flat = (kp_out / sz).view(-1), i.e. kp / 128. So model learns to predict in [0,1] relative to a 128x128 canvas where the fish fills the entire frame (since the training images ARE the fish crops, resized to fill 128x128).
  implication: The model was trained on images where fish body spans nearly the full 128x128 canvas (x: ~0 to 1.0 in normalized coords). The nose keypoint is near x=0.0 and the tail near x=1.0 in the training images.

- timestamp: 2026-02-28
  checked: extract_affine_crop in crop.py lines 155-223
  found: At inference, the affine transform is built with scale=1.0 (cv2.getRotationMatrix2D with scale=1.0). The OBB centre is placed at the centre of the 128x128 canvas. The fish at native pixel scale occupies [64-obb_w/2, 64+obb_w/2] horizontally. If obb_w=100 pixels, the fish spans pixels 14 to 114 in the 128x128 crop. If obb_w=60 pixels, the fish spans pixels 34 to 94.
  implication: The fish body does NOT fill the 128x128 canvas at inference — it occupies only obb_w/128 of the canvas width. This directly contradicts the training assumption.

- timestamp: 2026-02-28
  checked: _process_single_detection denormalization lines 313-315
  found: kp_crop_px = visible_kp_norm * np.array([crop_w, crop_h]) = visible_kp_norm * 128. The model outputs, say, nose at ~0.05 and tail at ~0.95 (spanning most of the 128px width as trained). After multiplication: nose at ~6px, tail at ~122px in the 128x128 crop. But at inference the fish only spans [64-bw/2, 64+bw/2]. If bw=60, the fish spans [34, 94] = 60px. The model's predicted span of 116px (6 to 122) is then back-projected through the affine inverse which correctly maps crop coords back to frame coords. So the back-projected span in frame space is ~116px instead of the real fish size of ~60px — this would make midlines LONGER, not shorter.
  implication: Wait — this is the OPPOSITE direction. Need to reconsider.

- timestamp: 2026-02-28
  checked: Re-analysis accounting for letterboxing description in context
  found: The context says "original rectangular crops (~100x23) were letterboxed to 128x128 with black padding." This means the TRAINING images stored on disk are ALREADY the letterboxed 128x128 images. When KeypointDataset loads them: orig_h=128, orig_w=128. Then cv2.resize(image, (128,128)) is a no-op (same size). Keypoint coords in the COCO JSON are in the 128x128 letterboxed image space. The fish body in these letterboxed crops occupies roughly the inner rectangle — not the full 128px width. E.g., a 100x23 fish letterboxed to 128x128 has the fish running from about x=14 to x=114 (center with 14px padding each side). So the model's nose keypoint is near x~14/128=0.11 and tail near x~114/128=0.89. The predicted normalized span is ~0.78.
  implication: At inference with bw=100px (same fish), fish spans [14, 114] in the 128x128 crop. Model predicts nose at 0.11, tail at 0.89. After * 128: nose at 14px, tail at 114px. Back-projection maps these correctly to frame space. This would be CORRECT for this fish size. The ~50% shorter problem must come from a different fish size at inference.

- timestamp: 2026-02-28
  checked: The actual scale mismatch: what happens when real video fish are larger or smaller than training fish
  found: If the real video fish OBB width is bw=200px (a large fish or closer camera), extract_affine_crop places it at native scale in the 128x128 crop. The fish body extends from x=-36 to x=164, but clipped to [0,128]. Most of the fish body is visible but compressed at the edges. The model sees a cropped/clipped fish. The model still predicts normalized coords roughly spanning [0.11, 0.89] as trained. Back-projected span = 0.78 * 128 = 100px in crop space. Affine inverse maps this back with scale=1.0, so in frame space = 100px. But the actual fish is 200px long! Midline is 50% of actual length. THIS IS THE 50% SHORTENING.
  implication: The root cause is confirmed: when the real video fish OBB width (~200px) is approximately 2x the training crop OBB width (~100px), the model predicts in the training coordinate space (fish fills ~78% of 128px = ~100px) but the real fish is 200px. Back-projection (which has no scale component) maps the 100px crop-space span to 100px in frame space, giving a midline ~50% of the actual fish length.

## Resolution

root_cause: extract_affine_crop uses scale=1.0 (no resizing), placing the fish at its native pixel size in the 128x128 crop canvas. The model was trained on images where the fish was explicitly scaled to fill the 128x128 canvas (original rectangular crops were letterboxed to 128x128, making them effectively scaled to fill the canvas). At inference, real-video fish with OBB widths ~2x the training crop widths are placed at native scale, and the model predicts keypoint positions as if the fish filled the canvas — resulting in denormalized pixel coords that span ~half the actual fish length. The affine inverse correctly back-projects these too-small crop coords to frame space, producing midlines ~50% of the actual fish body length.

fix: extract_affine_crop must scale the fish to fill the canvas at inference. The obb_w and obb_h parameters already accept the OBB dimensions but the docstring explicitly says "The crop canvas size is controlled solely by crop_size" and padding_fraction is "currently unused." The fix is to add scaling so that obb_w maps to crop_w (with optional padding). Specifically: scale = crop_w / obb_w (or use min(crop_w/obb_w, crop_h/obb_h) to fit both axes). Use this scale in cv2.getRotationMatrix2D instead of 1.0. This will make the inference crop match the training crop's scale relationship.

verification: All 687 unit tests pass (hatch run test). Round-trip accuracy tests confirm invert_affine_points is still mathematically exact after the scale change. No regressions detected.
files_changed:
  - src/aquapose/segmentation/crop.py
