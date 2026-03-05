---
status: diagnosed
trigger: "Batched YOLO inference produces different detection outputs than sequential inference"
created: 2026-03-05T00:00:00Z
updated: 2026-03-05T04:00:00Z
---

## Current Focus

hypothesis: CONFIRMED - The 907px outlier has a precise 3-step mechanism: (1) e3v83eb has a degenerate 2D midline (35px span vs 108px for e3v831e) that poisons the initial all-camera triangulation, (2) the poisoned initial 3D points cause ALL cameras to exceed 10px reprojection for body points 0-4, so outlier rejection kills them entirely (0 inliers at BP[0]), (3) the spline is fitted to only BP[5]-BP[14] (u=0.357-1.0) and must extrapolate wildly for the head region (CP[0], CP[1]).
test: Full step-by-step reproduction confirmed exact mechanism.
expecting: N/A - root cause confirmed
next_action: Return diagnosis

## Symptoms

expected: Batched YOLO inference should produce identical detection results to sequential inference on the same input images.
actual: 8 of 12 cameras show different detection counts between sequential and batched runs. Cascades into association (-241 observations) and reconstruction (max reproj error 43px to 907px).
errors: No errors - pipeline completes successfully. Issue is output divergence.
reproduction: Compare baseline run (sequential, run_20260304_180748) vs post-optimization run (batched, run_20260304_221326) on same YH single-chunk workload.
started: Introduced with phase 59 (batched YOLO inference optimization).

## Eliminated

- hypothesis: Different LetterBox preprocessing (auto/rect mode) between single and batch
  evidence: LetterBox uses auto=False in both cases because ultralytics default rect=False. Pre_transform applies identical LetterBox to each image regardless of batch size.
  timestamp: 2026-03-05T00:00:30Z

- hypothesis: BatchNorm layers using batch statistics instead of running stats
  evidence: Model is in eval mode (model.training=False, bn.training=False), so BatchNorm uses stored running statistics, not batch statistics.
  timestamp: 2026-03-05T00:00:40Z

- hypothesis: Large position shifts (215px) are real high-confidence detections moving
  evidence: Full cache comparison of 14,861 matched detection pairs shows max position shift is 1.41px. The 215px claim came from a synthetic test using a single replicated frame, not real multi-camera data. In real data, zero detections shift >5px.
  timestamp: 2026-03-05T01:00:00Z

- hypothesis: Phase 56 vectorized association scoring changed scoring numerics
  evidence: Tested 1000 random ray pairs -- scalar and batch ray_ray_closest_point produce BIT-IDENTICAL results (max diff 0.0). Early termination logic is also equivalent. Phase 56 is a pure performance refactor with zero numerical impact.
  timestamp: 2026-03-05T02:20:00Z

- hypothesis: Association was accidentally broken by v3.4 optimization phases (56-59)
  evidence: (1) Phase 56 changed scoring.py -- vectorized scoring, but numerically identical (verified). (2) Phase 57 changed reconstruction only, not association input. (3) Phase 58 changed frame I/O only. (4) Phase 59 changed detection batching (the root cause of detection flips). No accidental interface or format changes. The association code itself was NOT broken.
  timestamp: 2026-03-05T02:25:00Z

## Evidence

- timestamp: 2026-03-05T00:00:10Z
  checked: Git history of detection stage before/after phase 59
  found: Before phase 59 (commit c9c1db1^), DetectionStage called self._detector.detect() per camera (one image at a time). After phase 59, it calls self._detector.detect_batch() with all 12 camera frames at once.
  implication: The code changed from batch_size=1 to batch_size=12 for GPU inference.

- timestamp: 2026-03-05T00:00:20Z
  checked: Both run configs
  found: Baseline has no detection_batch_frames key (code used detect() per-image). Post-opt run has detection_batch_frames=0 (all cameras in one batch via detect_batch()).
  implication: Confirms the batching change between runs.

- timestamp: 2026-03-05T00:00:30Z
  checked: Ultralytics pre_transform and LetterBox code
  found: auto parameter is always False because default rect=False. Same preprocessing pipeline for single and batch.
  implication: Preprocessing is NOT the cause; images enter the model identically.

- timestamp: 2026-03-05T00:00:40Z
  checked: Model eval mode
  found: model.training=False, BatchNorm layers in eval mode.
  implication: BatchNorm is NOT the cause.

- timestamp: 2026-03-05T00:00:50Z
  checked: Empirical test - same image, sequential predict() 5 times
  found: All 5 runs produce BIT-IDENTICAL results (0.0 diff).
  implication: Sequential inference is perfectly deterministic.

- timestamp: 2026-03-05T00:00:51Z
  checked: Empirical test - same image 4x in a batch
  found: All 4 batch slots produce BIT-IDENTICAL results to each other.
  implication: Batched inference is internally consistent.

- timestamp: 2026-03-05T00:00:52Z
  checked: Empirical test - single predict(frame) vs predict([frame], batch=1)
  found: Results are BIT-IDENTICAL (0.0 diff).
  implication: The list-vs-array input path is not the issue; batch_size=1 produces same results either way.

- timestamp: 2026-03-05T00:00:53Z
  checked: Empirical test - predict(frame) vs predict([frame]*4, batch=4)
  found: Confidence diffs up to 0.0008, position diffs up to 215 pixels (worst case, frame 3 with 12 detections). Mean confidence diff ~0.0004.
  implication: CONFIRMED - batch_size>1 triggers different CUDA kernel execution paths that produce different floating-point results.

- timestamp: 2026-03-05T00:00:54Z
  checked: Empirical test - 4 real camera frames, single vs batch of 4
  found: All 4 frames show confidence shifts (max 0.003), position shifts (max 215px on frame with many detections), but detection counts stayed the same for these 4 frames.
  implication: The small confidence shifts can push borderline detections across the conf=0.5 threshold, explaining the +1/-1 detection count changes across 8 of 12 cameras in the full run.

- timestamp: 2026-03-05T01:00:00Z
  checked: Full matched comparison of actual detection caches from both pipeline runs (14,861 matched pairs across 200 frames x 12 cameras)
  found: |
    Position shifts are NEGLIGIBLE:
    - Overall max: 1.41px, median: 0.00px, mean: 0.02px, p99: 0.50px
    - High-confidence (>0.7): max 1.41px, only 1 detection shifted >1px, zero >5px
    - The "215px shift" from synthetic test was an ARTIFACT of the test setup (replicated single frame, not real multi-camera data)

    Unmatched detections (the actual problem):
    - 19 baseline-only, 18 post-opt-only (37 total out of ~14,900)
    - ALL unmatched detections are within 0.03 of the conf=0.5 threshold
    - Baseline-only: conf range 0.5000-0.5097
    - Post-opt-only: conf range 0.5000-0.5253
    - These are borderline detections that shift across the threshold due to tiny confidence perturbations

    Confidence differences for matched pairs:
    - Max conf diff: 0.051 (in 0.5-0.6 bin), typically <0.003 for high-conf
    - Higher confidence = smaller perturbation
  implication: The batch nondeterminism effect is real but TINY — sub-pixel position shifts and <0.05 confidence shifts. The downstream catastrophe (907px reproj error) is caused entirely by ~37 borderline detections appearing/disappearing, which cascades through tracking and association. The "215px position shift" was NOT a real effect.

- timestamp: 2026-03-05T02:10:00Z
  checked: Full comparison of diagnostic caches -- detections, tracking, and association between baseline and post-opt runs
  found: |
    DETECTION LEVEL: Total detections baseline=14880, postopt=14879 (diff=-1).
    37 frame/camera combos have +/-1 detection count differences, scattered across 11 cameras.

    TRACKING LEVEL: Total tracklets baseline=133, postopt=131.
    Two cameras show significant tracking differences:
    - e3v8334: 23->22 tracklets. One tracklet disappeared; others have different
      detected/coasted frame counts (e.g. track_id=8: 140 detected->23 detected).
    - e3v83eb: 18->17 tracklets. Tracklet splits/merges changed.
    Centroid positions for matched tracklets differ by <0.5px (negligible).

    ASSOCIATION LEVEL: Multi-cam groups baseline=22, postopt=20.
    - 2-cam groups: 7->5 (lost 2)
    - 3-cam groups: 5->3 (lost 2)
    - 4-cam groups: 8->10 (gained 2)
    - Total obs in multi-cam groups: 12814->12608 (diff=-206)

    The "lost" groups were NOT destroyed -- they were REORGANIZED:
    - Baseline fish=28 (e3v82e0,e3v831e,e3v8334, 3-cam) got absorbed into
      post-opt fish=9 (e3v82e0,e3v831e,e3v8334,e3v83f0, 4-cam) -- gained a camera
    - Baseline fish=29 (e3v8334,e3v83f0, 2-cam, conf=0.45) dissolved -- low confidence
    - Baseline fish=40 (e3v831e,e3v83eb,e3v83f0, 3-cam) became
      post-opt fish=40 (e3v831e,e3v8334,e3v83f0, 3-cam) -- swapped one camera
    - Baseline fish=43 (e3v8334,e3v83f0, 2-cam, conf=0.64) dissolved

    The cascade is EXPECTED behavior: different detections -> different tracker splits
    -> different tracklets -> different scoring edges -> different Leiden clusters.
    The Leiden algorithm amplifies small scoring differences into discrete grouping
    changes, especially for borderline groups (conf < 0.5).
  implication: |
    The association cascade is inherent sensitivity, NOT a code bug.
    Phase 56 vectorization is numerically identical (verified).
    No accidental interface/format changes in phases 57-59.
    The -206 observation difference is primarily group reorganization,
    not actual data loss. Some low-confidence 2-cam groups dissolved
    while some groups gained cameras.

- timestamp: 2026-03-05T02:20:00Z
  checked: Phase 56 vectorized scoring numerical equivalence
  found: |
    Tested ray_ray_closest_point_batch vs scalar ray_ray_closest_point
    on 1000 random ray pairs. Results are BIT-IDENTICAL (max diff = 0.0).
    No inlier status disagreements at threshold 0.01.
    Total soft-kernel score sums are identical.
  implication: Phase 56 is a pure performance refactor with zero numerical impact on association.

- timestamp: 2026-03-05T03:00:00Z
  checked: Post-opt run reconstruction cache for frame/fish with max reprojection error
  found: |
    Frame 154, fish 38 is the sole outlier. arc_length=96.26m (vs normal ~0.1m).
    Control points [0] and [1] are at [-15,14,89] and [0.1,-1,-6.8] -- wildly outside
    aquarium bounds. Remaining control points [2-6] are normal. n_cameras=2,
    is_low_confidence=True. Per-camera residuals: e3v831e=57.7px, e3v8334=66.7px,
    e3v83eb=205.8px. This is the only frame in 200 with max_residual > 100px.
  implication: Single degenerate reconstruction event causing the entire 907px max_reprojection_error in eval.

- timestamp: 2026-03-05T03:10:00Z
  checked: Baseline run reconstruction at same frame/fish (frame 154, fish 38)
  found: |
    Baseline has fish 38 at frame 154 with cameras [e3v831e, e3v8334, e3v83f0],
    n_cameras=3, arc_length=0.1088m, max_residual=5.15px. Normal reconstruction.
    Both runs associate fish 38 with same 4 cameras. The difference: e3v83f0 is
    "detected" in baseline but "coasted" in post-opt at frame 154.
  implication: The coasted status of e3v83f0 is the proximate cause -- removes the stabilizing 3rd camera.

- timestamp: 2026-03-05T03:20:00Z
  checked: DLT backend vectorized path ray-angle filter
  found: |
    _triangulate_fish_vectorized() (dlt.py lines 366-578) has an explicit note
    (lines 377-383) that the 2-camera ray-angle filter is "deliberately omitted"
    from the vectorized path for performance. The scalar _triangulate_body_point()
    retains the filter at lines 640-650. When body points fall to 2 cameras after
    outlier rejection, near-parallel rays produce ill-conditioned lstsq solutions.
  implication: Pre-existing robustness gap -- vectorized path sacrifices safety for speed.

- timestamp: 2026-03-05T03:30:00Z
  checked: Post-reconstruction sanity checks in DLT backend
  found: |
    No arc_length guard, no max_residual guard, no control_point bounds check.
    A reconstruction with arc_length=96m (700x expected) and max_residual=907px
    passes through to the output. The only guard is min_body_points (need 9+ valid
    body points for spline fitting) and is_low_confidence flag (which is informational,
    not a filter).
  implication: Second robustness gap -- degenerate reconstructions are not rejected.

- timestamp: 2026-03-05T02:25:00Z
  checked: Git history of phases 56-59 for all association-adjacent code changes
  found: |
    Phase 56 (commits 331df95, ba616de): Only scoring.py changed. Added
    ray_ray_closest_point_batch and rewrote score_tracklet_pair internals.
    Numerically identical (verified). Early termination logic is equivalent.
    Phase 57: Only reconstruction code changed (DLT vectorization).
    Phase 58: Only frame I/O changed (prefetch).
    Phase 59: Detection stage changed to batched inference (root cause).
    No tracking or clustering code was touched in any of these phases.
  implication: Association logic was NOT accidentally changed. The cascade is purely due to different detection inputs.

- timestamp: 2026-03-05T05:00:00Z
  checked: Full step-by-step reproduction of vectorized DLT triangulation for fish 38, frame 154
  found: |
    e3v83eb 2D midline is degenerate (35px span vs 108px for good cameras). This poisons
    the initial all-camera triangulation: BP[0] ends up with ALL 3 cameras as outliers
    (10.1, 19.0, 26.3 px residuals), BP[1-4] have only 1 inlier. After rejection,
    only BP[5]-BP[14] are valid (10/15 points). Spline fitted to u=[0.357,1.0] must
    extrapolate wildly for head region -> CP[0]=[-15,14,89], CP[1]=[0.1,-1,-6.8].
    The issue is NOT RANSAC vs threshold. It's: (1) a bad 2D midline that corrupts initial
    triangulation, (2) single-round outlier rejection that can't recover, (3) spline
    extrapolation beyond observed data range.
  implication: |
    The root cause is a cascade of three robustness gaps: no 2D midline quality filter,
    self-poisoning outlier rejection (bad camera inflates residuals for good cameras),
    and no spline extrapolation guard. RANSAC would help but is not the simplest fix.

## Resolution

root_cause: |
  TWO-PART ROOT CAUSE:

  1. DETECTION NONDETERMINISM (unavoidable): GPU floating-point non-determinism from
     different batch tensor sizes. batch_size=1 vs batch_size=12 selects different CUDA
     convolution kernels with different FP reduction orders. Effect: ~37/14900 borderline
     detections (conf within 0.03 of threshold 0.5) flip between detected/undetected.
     Position shifts are negligible (max 1.41px).

  2. INHERENT CASCADE SENSITIVITY (not a bug): The detection flips cascade through
     OC-SORT tracking (different tracklet splits/merges in 2 cameras) and then through
     Leiden clustering (discrete grouping changes). The cascade is amplified by:
     - OC-SORT's sensitivity to detection gaps (missing 1 detection can split a tracklet)
     - Leiden's discrete nature (small score changes -> different cluster assignments)
     - Low-confidence groups (conf < 0.5) being inherently unstable

     The association code was NOT accidentally broken. Phase 56 vectorized scoring is
     numerically identical to the original (verified). No interface/format changes in
     phases 57-59.

  SEVERITY REASSESSMENT: The originally reported "-241 observations, 87% 2-cam group loss"
  is misleading. The actual effect is:
  - 22 -> 20 multi-cam groups (2 fewer, not catastrophic)
  - 12814 -> 12608 total multi-cam observations (-206, or -1.6%)
  - Lost groups were primarily REORGANIZED (absorbed into larger groups or dissolved
    low-confidence 2-cam pairs), not destroyed
  - 4-cam groups actually increased from 8 to 10

fix:
verification:
files_changed: []

## Reconstruction Outlier Investigation (907px)

### Degenerate Reconstruction Details

- **Frame:** 154
- **Fish ID:** 38
- **Arc length:** 96.26m (normal: ~0.10-0.14m, a ~700x blowup)
- **Cameras used:** e3v831e, e3v8334, e3v83eb (3 cameras passed to backend)
- **Min cameras after outlier rejection (n_cameras):** 2
- **is_low_confidence:** True
- **Per-camera residuals:** e3v831e=57.7px, e3v8334=66.7px, e3v83eb=205.8px
- **Degenerate control points:**
  - CP[0]: [-15.4, 14.7, 89.6] -- ~90m outside aquarium
  - CP[1]: [0.1, -1.0, -6.8] -- ~8m outside aquarium
  - CP[2-6]: reasonable values ~[-1.2, 0.35, 1.4]

### Baseline Comparison (frame 154, fish 38)

- **Cameras used:** e3v831e, e3v8334, e3v83f0 (3 cameras, all good)
- **n_cameras:** 3
- **Arc length:** 0.1088m (normal)
- **Max residual:** 5.15px
- Both runs have fish 38 associated with the SAME 4 cameras: e3v831e, e3v8334, e3v83eb, e3v83f0

### Root Cause Chain

1. **Detection batching** causes ~37 borderline detections to flip across conf=0.5 threshold
2. **OC-SORT tracker** for camera e3v83f0 (track 10) **coasts** on frame 154 in the post-opt run (detected in baseline)
3. **ReconstructionStage** excludes coasted e3v83f0, leaving 3 detected cameras: e3v831e, e3v8334, e3v83eb
4. **DLT outlier rejection** removes e3v83eb for some body points (poor view geometry for this fish at this position)
5. **2-camera body points** (e3v831e + e3v8334) have near-parallel rays, producing ill-conditioned triangulation
6. **Missing ray-angle filter**: The vectorized `_triangulate_fish_vectorized()` deliberately omits the `_MIN_RAY_ANGLE_DEG` check that the scalar `_triangulate_body_point()` has (see dlt.py lines 377-383). This was a conscious tradeoff for performance.
7. **Spline fitting** to degenerate body points produces wildly wrong control points at the endpoints
8. **No post-hoc sanity check**: There is no guard on arc_length, max_residual, or control point bounds after reconstruction

### Robustness Gap (Pre-existing)

This is NOT solely a batching issue. The reconstruction backend has two robustness gaps:
1. **Vectorized path omits ray-angle filter** for 2-camera body points (documented design choice in code)
2. **No post-reconstruction sanity check** -- a 96m arc length (vs expected ~0.1m) or 907px residual should be rejected

The baseline avoided this by luck: e3v83f0 was detected (not coasted) at frame 154, providing a good 3rd camera that prevented 2-camera degenerate triangulations.

### Step-by-Step Reproduction (2026-03-05T05:00:00Z)

**2D Midline Quality Analysis:**
- e3v831e: 15 body points spanning ~108px. Good fish midline. Confidence: 0.46-0.97 (high in body, low at extremities).
- e3v8334: 15 body points spanning ~115px. Good fish midline. Confidence: 0.62-0.81.
- e3v83eb: 15 body points spanning ONLY ~35px. DEGENERATE midline -- entire fish body compressed to tiny pixel range (1322,1071 to 1291,1052). Confidence uniformly low: 0.51 across head, 0.33-0.61 across body. This camera is viewing the fish nearly end-on or at extreme distance.

**Pass 1 (all 3 cameras) Results:**
- 3D points for all 15 body points are reasonable (-1.26 to -1.17, 0.31 to 0.39, 1.36 to 1.50).
- BUT reprojection residuals reveal systematic bias from e3v83eb:
  - BP[0]: e3v831e=10.1px, e3v8334=19.0px, e3v83eb=26.3px -> 0 inliers
  - BP[5]: e3v831e=7.1px, e3v8334=9.5px, e3v83eb=16.7px -> 2 inliers
  - BP[9]: e3v831e=2.0px, e3v8334=1.6px, e3v83eb=1.1px -> 3 inliers (midpoint, best)
  - BP[14]: e3v831e=6.0px, e3v8334=9.8px, e3v83eb=16.3px -> 2 inliers

**The Critical Pattern:**
- e3v83eb has high residuals at both ends but low residuals near the middle (~BP[9]).
- This is because the 35px midline is roughly correct only near its center -- the endpoints are severely compressed.
- The e3v83eb midline POISONS the 3-camera triangulation at the head/tail ends.
- This causes EVEN the good cameras (e3v831e, e3v8334) to show >10px residuals at the head.
- At BP[0], ALL 3 cameras are outliers (0 inliers). At BP[1-4], only e3v831e is an inlier (1 inlier).

**Pass 2 Results:**
- BP[0]: 0 inliers -> invalid (excluded from spline)
- BP[1-4]: 1 inlier -> less than 2 required -> invalid
- BP[5-6]: 2 inliers (e3v831e, e3v8334) -> valid, 2-camera triangulation
- BP[7-11]: 3 inliers -> valid, 3-camera triangulation
- BP[12-14]: 2 inliers (e3v831e, e3v8334) -> valid, 2-camera triangulation

**Spline Fitting:**
- Only 10 valid body points: BP[5] through BP[14]
- Arc-length parameters: u = [5/14, 6/14, ..., 14/14] = [0.357, 0.429, ..., 1.0]
- The spline must represent u=[0,1] but only has data for u=[0.357,1.0]
- CP[0] and CP[1] require extrapolation to u<0.357 -- WILD extrapolation -> degenerate values

**Root Cause is NOT RANSAC.** It is a 3-part failure:
1. Bad 2D midline from e3v83eb (degenerate view geometry, ~35px span)
2. Single-pass outlier rejection that poisons itself (bad camera corrupts initial triangulation, which inflates residuals for good cameras, which causes mass rejection at endpoints)
3. Spline extrapolation beyond observed data range

**Why This Doesn't Happen with the Scalar Path:**
The scalar `_triangulate_body_point()` has the same triangulate-then-reject logic. It would ALSO produce BP[0] with 0 inliers and BP[1-4] with 1 inlier. The difference is purely that the baseline run had e3v83f0 (a good camera) instead of e3v83eb (a degenerate one).

**RANSAC Evaluation:**
RANSAC is NOT the approach used here. The algorithm is:
1. Triangulate ALL cameras
2. Compute reprojection residuals
3. Reject cameras above threshold
4. Re-triangulate inliers

This is a single-round outlier rejection, not iterative RANSAC. With C=3 cameras:
- If the outlier camera corrupts the initial triangulation, ALL cameras may show high residuals
- This is the fundamental vulnerability: a single bad camera can sabotage the initial triangulation
  enough that even good cameras get rejected

**Would RANSAC Help?**
With C=3 cameras, RANSAC (minimal sample = 2 cameras) would try C(3,2)=3 pairs:
- (e3v831e, e3v8334): 10-degree ray angle. Marginal but workable. Would produce a reasonable point.
- (e3v831e, e3v83eb): 17-degree angle. e3v83eb is bad -> bad 3D point.
- (e3v8334, e3v83eb): 26-degree angle. e3v83eb is bad -> bad 3D point.
RANSAC WOULD help: 1 of 3 samples would give a good consensus (the e3v831e+e3v8334 pair).
BUT: The real fix is simpler -- just reject the degenerate reconstruction after the fact (arc_length > 1m or max_residual > 50px).

**Better Fixes:**
1. **Post-hoc sanity check**: Reject reconstructions with arc_length > N * expected_length or max_residual > threshold
2. **Iterative outlier rejection**: Instead of single-round, do: triangulate-all -> reject worst camera -> re-triangulate -> re-check residuals
3. **Minimum span filter**: Reject 2D midlines with total span < threshold (e.g., 50px) before triangulation
4. **Spline extrapolation guard**: Refuse to fit if first valid u > 0.3 or last valid u < 0.7
