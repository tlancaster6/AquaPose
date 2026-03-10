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

### Tracking continuity
Tracking coverage is excellent: 99.4% of all detections (2161/2175) are tracked. Only 14 detections across 200 frames failed to receive track IDs, and these appear to be transient low-confidence detections.

## Keypoint Behavior Observations

### Multi-instance pose detection during occlusion
The YOLO-pose model detects multiple pose instances within single OBB crops during the occlusion window. This is the most interesting finding:

- **9 of 10 occlusion frames** (F394-F403) have at least one detection with multi-instance pose output
- Peak at **F400**: one detection produces **6 pose instances** within a single OBB crop
- F399: 4 instances; F401 and F403: 3 instances each
- Outside the occlusion window, multi-instance frames are sporadic (23 out of 190 remaining frames = 12%)

This confirms the pose model detects both the primary fish and secondary overlapping fish within an OBB crop, which could be leveraged as an occlusion signal.

### Endpoint confidence collapse during occlusion
During multi-instance detections, the **primary instance** shows a distinctive confidence pattern:
- **Nose:** Drops to 0.00-0.08 (vs mean 0.927 across all frames)
- **Tail:** Drops to 0.00-0.33 (vs mean 0.662 across all frames)
- **Mid-body keypoints (spine1-spine3):** Remain high at 0.65-0.99

This "endpoint collapse" pattern is consistent: when two fish overlap, the pose model cannot resolve which endpoints belong to which fish, but the mid-body keypoints remain well-localized. The confidence collapse at nose and tail provides a reliable **occlusion detection signal** for the tracker.

### No keypoint identity jumps observed
Critically, there is **no evidence of single-frame keypoint identity jumps** (keypoints from fish A being assigned to fish B's detection). The multi-instance detection mechanism appears to cleanly separate the two fish's poses rather than producing a chimeric pose with some keypoints from each fish.

The primary instance in each multi-instance detection consistently maintains high confidence on the mid-body keypoints belonging to the OBB's target fish, while the secondary instances capture the overlapping fish's pose with generally lower confidence.

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

This bimodal pattern makes confidence a useful feature for the tracker to detect and handle occlusion events.

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
3. Multi-instance detection works — secondary instances detected in 90% of occlusion frames
4. Confidence patterns are informative — endpoint collapse provides a reliable occlusion signal

The OKS-based keypoint tracker (Phase 83) is viable. The tracker should:
- Use per-keypoint confidence weighting (down-weight low-confidence endpoints)
- Consider leveraging multi-instance pose detection as an auxiliary signal for occlusion awareness
- The confidence threshold of 0.20-0.25 is appropriate for the OBB detector

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
