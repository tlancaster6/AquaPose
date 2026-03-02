---
status: awaiting_human_verify
trigger: "pose-backend-v-shape-midlines"
created: 2026-03-01T00:00:00Z
updated: 2026-03-01T00:35:00Z
---

## Current Focus

hypothesis: CONFIRMED AND FIXED — OBB corner ordering mismatch between training (pca_obb: [TL,TR,BR,BL]) and Ultralytics inference output ([right-bottom, right-top, left-top, left-bottom]). Fix applied: change pts[0] to pts[2] in src corner selection in both backends.
test: unit tests pass (637 pass, 20 pre-existing failures unrelated to changed files)
expecting: V-shapes eliminated after rerunning pipeline. Orientation may still be ~50% reversed (secondary PCA sign ambiguity issue), but midlines should be smooth curves.
next_action: await human verification by re-running pipeline in diagnostic mode

## Symptoms

expected: Pose backend should produce smooth, anatomically correct midlines along the fish body axis with correct head-to-tail orientation for most detections.
actual: Two failure modes: (1) Many midlines are bent into V-shapes — the midline folds back on itself instead of following the fish spine. (2) Even "straight" midlines frequently have reversed head-to-tail orientation. Both visible in diagnostic overlay videos.
errors: No crash errors — pipeline completes, but visual output is poor.
reproduction: Run pipeline with pose backend in diagnostic mode. Run output at `C:\Users\tucke\aquapose\projects\YH\runs\run_20260301_200825`.
started: First pipeline runs with v3.0 YOLO-pose model, phase 37-02 implementation.

## Eliminated

- hypothesis: Seg vs Pose training crop preparation mismatch (different methods)
  evidence: Both use affine_warp_crop with src=[TL, TR, BL] -> dst=[(0,0),(W-1,0),(0,H-1)].
  timestamp: 2026-03-01T00:10:00Z

- hypothesis: Affine inverse transform error
  evidence: cv2.invertAffineTransform is numerically correct for any invertible affine.
  timestamp: 2026-03-01T00:10:00Z

## Evidence

- timestamp: 2026-03-01T00:05:00Z
  checked: scripts/build_yolo_training_data.py affine_warp_crop + pca_obb
  found: Training uses src=[obb_corners[0], obb_corners[1], obb_corners[3]] (TL, TR, BL) -> dst=[(0,0),(W-1,0),(0,H-1)]. pca_obb returns corners as:
    [0] = [-half_main, -half_perp] → LEFT-TOP (mapped to crop top-left (0,0))
    [1] = [+half_main, -half_perp] → RIGHT-TOP (mapped to crop top-right (W-1,0))
    [2] = [+half_main, +half_perp] → RIGHT-BOTTOM
    [3] = [-half_main, +half_perp] → LEFT-BOTTOM (mapped to crop bottom-left (0,H-1))
  implication: Training maps the "left" PCA end to the left of the crop consistently.

- timestamp: 2026-03-01T00:15:00Z
  checked: ultralytics xywhr2xyxyxyxy source code (inspected at runtime via python -c)
  found: Ultralytics obb_points = [pt1, pt2, pt3, pt4] where:
    pt1 = center + vec1 + vec2  → right-bottom
    pt2 = center + vec1 - vec2  → right-top
    pt3 = center - vec1 - vec2  → left-top  ← TRUE TL
    pt4 = center - vec1 + vec2  → left-bottom
  (vec1 = half long axis along fish, vec2 = half short axis perpendicular)
  implication: Ultralytics pts[2] is the true top-left, NOT pts[0].

- timestamp: 2026-03-01T00:20:00Z
  checked: pose_estimation.py and segmentation.py _extract_crop
  found: Both used `src = [pts[0], pts[1], pts[3]]` which maps:
    pts[0]=right-bottom → (0,0)   ← WRONG (should be left-top)
    pts[1]=right-top    → (W-1,0) ← incidentally correct axis (right end)
    pts[3]=left-bottom  → (0,H-1) ← correct y-position but wrong end
  This flips the crop horizontally AND distorts it, completely scrambling keypoint positions relative to training expectations. The YOLO-pose model predicts keypoints in scrambled positions → spline interpolation with assumed t-values (nose=0, tail=1) produces V-shapes.
  implication: CONFIRMED root cause.

- timestamp: 2026-03-01T00:30:00Z
  checked: unit test suite after fix
  found: 637 pass, 20 pre-existing failures (test_build_yolo_training_data + test_pipeline — not touching changed files). No new failures.
  implication: Fix does not break existing test suite.

## Resolution

root_cause: In both pose_estimation.py and segmentation.py `_extract_crop`, the OBB corner selection `src = [pts[0], pts[1], pts[3]]` was WRONG for the Ultralytics obb_points convention.

  Ultralytics obb_points ordering from xywhr2xyxyxyxy:
    pts[0] = right-bottom
    pts[1] = right-top
    pts[2] = left-top     ← TRUE top-left (TL)
    pts[3] = left-bottom  ← TRUE bottom-left (BL)

  Old (wrong): src = [pts[0]=right-bottom, pts[1]=right-top, pts[3]=left-bottom]
  Fixed: src = [pts[2]=left-top, pts[1]=right-top, pts[3]=left-bottom]

  The wrong mapping caused right-bottom to go to crop origin (0,0), producing a flipped/scrambled crop. The YOLO-pose model predicted keypoints in positions consistent with its training distribution but wildly wrong relative to the scrambled crop geometry. When interpolated with assumed t-values (nose=0.0→tail=1.0), the spline folded back on itself → V-shapes.

fix: Changed `src = np.array([pts[0], pts[1], pts[3]], ...)` to `src = np.array([pts[2], pts[1], pts[3]], ...)` in both:
  - src/aquapose/core/midline/backends/pose_estimation.py (line 342)
  - src/aquapose/core/midline/backends/segmentation.py (line 246)
  Also added explanatory comment documenting Ultralytics corner ordering convention.

verification: Unit tests pass (637/657, 20 pre-existing failures unrelated to changed files).

files_changed:
  - src/aquapose/core/midline/backends/pose_estimation.py
  - src/aquapose/core/midline/backends/segmentation.py
