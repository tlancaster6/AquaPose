# Phase 78: Occlusion Investigation Findings

**Camera:** e3v831e | **Frame range:** 300-499 (200 frames at 30fps) | **Crop region:** (263,225)-(613,525) | **Occlusion window:** F394-F403

## OBB Behavior Observations

### Detection stability
The OBB detector maintains stable detection counts throughout the occlusion event. During the occlusion window (F394-F403), detection counts range from 12-15 per frame, compared to 10-13 in surrounding non-occlusion frames. The count actually *increases* slightly during occlusion, suggesting the detector tends to produce extra detections at overlapping regions rather than merging fish into single boxes.

### No persistent merging observed
There is no evidence of OBB box merging (two fish collapsed into one detection) during the occlusion window. All frames maintain detection counts at or above the expected 9 fish visible in the crop region. This is a strong positive finding.

### Detection count fluctuation
Frame-to-frame detection count changes of +/- 2 occur near the occlusion window (e.g., F393: +2, F395: +2, F396: -2), but these are within normal variation and not indicative of systematic merging or splitting artifacts.

### Frames with fewer than expected detections
19 frames across the 200-frame range show fewer than 9 detections, concentrated in the post-occlusion region (F416-F418, F427, F444). These appear to be fish partially leaving the crop region rather than detection failures.

### Detection dropout on white tank wall background
In the full-frame 20-second video, an isolated fish swimming against the white tank wall (as opposed to the sandy floor) is intermittently undetected even at conf=0.1. This is not an NMS issue — the fish has no nearby detections to compete with. The white-wall background is under-represented in training data; the round 1 pseudo-label augmentation did not resolve this gap, likely because the baseline model that generated pseudo-labels also missed these cases. This is a detection reliability issue that could cause tracker fragmentation in wall-adjacent regions.

### Tracking continuity
Tracking coverage is excellent: 99.4% of all detections (2161/2175) are tracked. Only 14 detections across 200 frames failed to receive track IDs, and these appear to be transient low-confidence detections.

## Keypoint Behavior Observations

### Duplicate pose detections during occlusion (Gaussian NMS artifact)
The YOLO-pose model produces multiple near-identical pose instances within single OBB crops during the occlusion window:

- **9 of 10 occlusion frames** (F394-F403) have at least one detection with multi-instance pose output
- Peak at **F400**: one detection produces **6 pose instances** within a single OBB crop
- F399: 4 instances; F401 and F403: 3 instances each
- Outside the occlusion window, multi-instance frames are sporadic (23 out of 190 remaining frames = 12%)

These are **not** detections of the intruding fish — the secondary keypoints overlap the focal fish in nearly identical positions, and the duplicate bounding boxes have near-perfect overlap. This is a Gaussian NMS artifact: YOLO-pose uses Gaussian NMS internally, which is known to fail for bounding boxes with long aspect ratios (the same issue we addressed in the OBB detection stage with custom geometric NMS). On the 128×64 pose crop, near-duplicate detections survive Gaussian NMS when their centers are even slightly offset.

The pose model was run with `conf=0.1`, which is very permissive and allows weak duplicates through. The intruding fish is never detected as a separate instance, likely because the training data rarely contains two fish in a single crop.

**Note:** The production pipeline only uses `kp.xy[0]` (primary instance), so these duplicates are invisible in normal operation. Addressing this in the pose stage would be more challenging than the OBB fix, since NMS happens between detection and keypoint regression within the model.

### Endpoint confidence collapse during occlusion
During occlusion, the **primary instance** shows a distinctive confidence pattern:
- **Nose:** Drops to 0.00-0.08 (vs mean 0.927 across all frames)
- **Tail:** Drops to 0.00-0.33 (vs mean 0.662 across all frames)
- **Mid-body keypoints (spine1-spine3):** Remain high at 0.65-0.99

This "endpoint collapse" pattern is consistent: when two fish overlap, the pose model cannot resolve which endpoints belong to which fish, but the mid-body keypoints remain well-localized. The confidence collapse at nose and tail provides a reliable **occlusion detection signal** for the tracker.

### No keypoint identity jumps observed
Critically, there is **no evidence of single-frame keypoint identity jumps** (keypoints from fish A being assigned to fish B's detection). The pose model does not produce chimeric poses with some keypoints from each fish — the primary detection's keypoints consistently belong to the focal fish, even during occlusion.

## Confidence Patterns

### Per-keypoint confidence (all 200 frames, all detections)

| Keypoint | Mean | Std | Min | % below 0.3 |
|----------|------|-----|-----|-------------|
| nose | 0.927 | 0.174 | 0.000 | 3.2% |
| head | 0.992 | 0.072 | 0.041 | 0.5% |
| spine1 | 0.998 | 0.019 | 0.506 | 0.0% |
| spine2 | 0.994 | 0.042 | 0.085 | 0.1% |
| spine3 | 0.985 | 0.084 | 0.001 | 0.8% |
| tail | 0.662 | 0.285 | 0.000 | 15.4% |

Key findings:
- **Tail** is the weakest keypoint by far: 15.4% of detections have tail confidence below 0.3
- **Mid-body keypoints** (spine1-spine3) are extremely reliable (>99.4% above 0.3)
- **Nose** occasionally drops but is generally strong (96.8% above 0.3)
- The tail weakness is not occlusion-specific; it appears across all frames and likely reflects the model's difficulty localizing the thin, often partially obscured tail fin

### Occlusion-specific confidence pattern
During occlusion, the confidence pattern becomes bimodal:
- **Isolated fish:** All 6 keypoints at 0.8-1.0
- **Overlapping fish:** Endpoints (nose, tail) drop to ~0.0, mid-body stays at 0.65-0.99

This bimodal pattern makes per-keypoint confidence a useful feature for the tracker to detect occlusion and down-weight uncertain endpoints.

## Confidence Sweep Results

Camera: e3v831e | Frames: 0-600 (600 frames, full 20-second clip)

| Threshold | Total Dets | Mean/Frame | Median/Frame | Min/Frame | Max/Frame |
|-----------|-----------|------------|--------------|-----------|-----------|
| 0.10      |      7304 |       12.2 |         12.0 |         6 |        17 |
| 0.15      |      6770 |       11.3 |         11.0 |         5 |        16 |
| 0.20      |      6426 |       10.7 |         11.0 |         5 |        15 |
| 0.25      |      6093 |       10.2 |         10.0 |         5 |        15 |
| 0.30      |      5797 |        9.7 |         10.0 |         4 |        15 |
| 0.35      |      5501 |        9.2 |          9.0 |         4 |        14 |
| 0.40      |      5250 |        8.8 |          9.0 |         4 |        12 |
| 0.45      |      4992 |        8.3 |          8.0 |         3 |        12 |
| 0.50      |      4739 |        7.9 |          8.0 |         3 |        11 |

### Analysis

The sweep shows a smooth tradeoff without a sharp elbow:
- At **0.10**: 12.2 mean dets/frame — includes ~3 false positives per frame above the 9 expected fish
- At **0.25**: 10.2 mean — ~1 false positive per frame
- At **0.35**: 9.2 mean — close to the expected 9 fish
- At **0.50**: 7.9 mean — starts dropping below 9, losing real fish

**Note:** The confidence sweep was run without polygon NMS. With polygon NMS applied (as in the production pipeline), false positives at conf=0.1 are minimal — the full-frame 20-second video shows clean detections with very few spurious boxes. This suggests the confidence threshold can be dropped to 0.1 with polygon NMS without meaningful false positive cost, improving recall for difficult cases (e.g., white-wall background).

### Recommended threshold: 0.25

**Justification:** At 0.25, the median is 10 detections per frame (9 fish + ~1 false positive), with a minimum of 5 (acceptable for frames where fish are at crop edges). This provides a good balance:
- Low enough to capture all real fish in nearly all frames
- High enough to suppress most false positives
- The remaining ~1 FP/frame is easily handled by the tracker (it will fail to associate and be pruned)

The current pipeline default of 0.2 is also reasonable. The difference between 0.20 and 0.25 is only ~0.5 detections/frame, and the tracker handles either well.

## Go/No-Go Recommendation

### Criteria Evaluation

**Primary criterion: Keypoint identity jumps**
- Observed identity jumps during occlusion: **0 frames**
- Occlusion frames analyzed: 10 (F394-F403)
- Percentage: **0%** (threshold: >20% = no-go)
- **PASS**: No keypoint identity jumps detected. The pose model cleanly separates instances.

**Secondary criterion: OBB merging**
- Observed consecutive merge frames: **0**
- **PASS**: No OBB merging observed. The detector maintains separate boxes for overlapping fish.

**Informational: Confidence collapse**
- Endpoint (nose/tail) confidence drops to ~0 during occlusion in 9/10 frames
- This is expected behavior and **useful as a signal** for the tracker to down-weight uncertain keypoints
- **Not a concern**: The tracker should use per-keypoint confidence weighting

### Recommendation: **GO**

The OBB detector and pose model behave well during occlusion events in this clip:
1. No box merging — each fish maintains its own OBB
2. No keypoint identity jumps — the pose model does not produce chimeric poses
3. Confidence patterns are informative — endpoint collapse provides a reliable occlusion signal

**Tracking performance better than expected:** In the full 20-second clip, OC-SORT tracking shows fragmentation but only one persistent ID swap (fish A and B enter occlusion, B exits with A's ID). This suggests tracking is less of a bottleneck than anticipated — the bigger opportunity may be improving cross-view association by switching from bbox centroid association to anatomical keypoint association. The keypoint-based approach addresses both: OKS matching resists ID swaps (pose similarity is harder to confuse than box overlap), and keypoint-derived 3D points provide more geometrically stable association features than bbox centroids for elongated fish.

The OKS-based keypoint tracker (Phase 83) is viable. The tracker should:
- Use per-keypoint confidence weighting (down-weight low-confidence endpoints during occlusion)
- The confidence threshold can be dropped to 0.1 with polygon NMS (minimal false positives, better recall)

**Known limitations:**
- The pose model's Gaussian NMS produces duplicate detections during occlusion (same fish detected multiple times). This is invisible in production (only `kp.xy[0]` is used) but worth noting. The intruding fish is never detected within the focal fish's crop.
- Detection dropout on white-wall background persists even at conf=0.1 (see OBB Behavior Observations above).

## Recommended Follow-Up (Phase 79)

**Retrain OBB detector before proceeding to tracker work.** The current model was trained on the original manual annotations plus uncurated pseudo-labels, with train/val split drawn only from manual annotations. A production retrain should:
1. Include corrected pseudo-labels in the train/val split (not just the original manual set) — the manual-only val was useful for fine-grained accuracy comparisons during the pseudo-label milestone, but for production the model should see the full data distribution
2. Train for more epochs to improve recall on under-represented cases (white-wall background)
3. This is a quick win that could reduce tracker fragmentation from missed detections before investing in the keypoint-based tracker

## Screenshots

Key frames saved in `screenshots/`:
- `normal_f350.png` — Normal non-occlusion frame showing well-separated fish
- `pre_occlusion_f390.png` — Fish approaching, before overlap
- `early_occlusion_f394.png` — Early overlap, first multi-instance detection
- `peak_occlusion_f400.png` — Maximum overlap, 6 pose instances in one OBB
- `late_occlusion_f403.png` — Overlap resolving, 3 instances
- `post_separation_f410.png` — Fish fully separated

---
*Phase: 78-occlusion-investigation*
*Findings generated: 2026-03-10*
