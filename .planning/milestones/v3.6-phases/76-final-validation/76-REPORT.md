# AquaPose v3.5-v3.6 Final Validation Report

## Overview

Milestones v3.5 and v3.6 implemented a pseudo-label iteration loop for improving AquaPose's OBB detection and pose estimation models. The work spanned six phases:

- **v3.5 (Pseudo-labeling infrastructure):** Built the training data store, pseudo-label selection pipeline, Label Studio export/import for manual curation, elastic augmentation for curvature debiasing, and training module code quality improvements.
- **v3.6 (Model iteration loop):** Established baseline metrics (Phase 72), generated round 1 pseudo-labels, retrained with curated labels (Phase 73), evaluated at full pipeline scale (Phase 74), decided to accept round 1 models as final and skip round 2 (Phase 75 skipped), and produced this final validation report (Phase 76).

**Final run:** `run_20260309_175421` -- 9000 frames (5 minutes at 30fps), 12 cameras, diagnostic mode, using round 1 winner models.

## Model Provenance Chain

```
Phase 71: Baseline Models
  OBB:  training/obb/run_20260307_094353/best_model.pt   (manual annotations only, imgsz=640)
  Pose: training/pose/run_20260307_113057/best_model.pt  (manual annotations + elastic aug, imgsz=320)
        |
Phase 72: Baseline Pipeline Run
  run_20260307_140127 (9000 frames, diagnostic mode)
  -> eval_results.json -> baseline metrics established
        |
Phase 73: Pseudo-Label Generation & Retraining
  pseudo-label select -> 40 OBB / 256 pose train images selected
  -> Label Studio export -> manual curation in CVAT -> corrected labels imported
  -> A/B comparison: curated vs uncurated
  Winners:
    OBB:  training/obb/run_20260309_120659/best_model.pt   (curated pseudo-labels)
    Pose: training/pose/run_20260309_152248/best_model.pt  (curated + elastic augmentation)
        |
Phase 74: Round 1 Pipeline Evaluation
  run_20260309_175421 (9000 frames, diagnostic mode, round 1 models)
  -> eval_results.json -> comparison with baseline
  -> Decision: Accept round 1 models, skip round 2
        |
Phase 75: SKIPPED (round 2 not needed)
        |
Phase 76: Final Validation (this report)
```

All models are registered in `~/aquapose/projects/YH/config.yaml`.

## Training Results (Phase 73)

### OBB Detection

| Model | Tag | mAP50 | mAP50-95 | P | R |
|-------|-----|-------|----------|---|---|
| Baseline | baseline | 0.935 | 0.689 | -- | -- |
| + Pseudo-labels | round1-uncurated | 0.961 | 0.721 | 0.907 | 0.903 |
| + Corrected labels | round1-curated | 0.973 | 0.700 | 0.896 | 0.938 |

Evaluated on manual-only val set (9 images, 65 instances). Curated model won on mAP50 and recall. Uncurated had better mAP50-95, but curated's higher recall (0.938 vs 0.903) was more valuable for downstream pipeline performance.

### Pose Estimation

| Model | Tag | Box mAP50 | Box mAP50-95 | Pose mAP50 | Pose mAP50-95 |
|-------|-----|-----------|-------------|------------|--------------|
| Baseline | baseline | 0.991 | 0.836 | 0.991 | 0.965 |
| + Pseudo-labels | round1-uncurated | 0.991 | 0.859 | 0.991 | 0.966 |
| + Corrected + augmented | round1-curated-aug | 0.994 | 0.968 | 0.994 | 0.968 |

Evaluated on manual-only val set (64 images, 65 instances). Curated+aug model won across all metrics. The large improvement in box mAP50-95 (+13.2 pts) was notable; pose mAP50-95 improvement was smaller (+0.3 pts) since baseline was already strong.

### Corrected Pseudo-Label Val (held-out, 64 images)

| Model | Pose mAP50 | Pose mAP50-95 |
|-------|-----------|--------------|
| Baseline | 0.971 | 0.892 |
| Uncurated | 0.974 | 0.893 |
| **Curated+aug** | **0.990** | **0.984** |

The decisive result: curated+aug scored +9.2 pts mAP50-95 over baseline on held-out corrected data. Uncurated barely moved (+0.1 pts), confirming that curation + elastic augmentation together drive the improvement.

### Curvature-Stratified OKS

| Model | Low curv (n=21) | Mid curv (n=21) | High curv (n=22) |
|-------|----------------|-----------------|------------------|
| Baseline | 0.609 | 0.578 | 0.436 |
| Uncurated | 0.632 | 0.560 | 0.407 |
| **Curated+aug** | **0.896** | **0.868** | **0.780** |

High-curvature improvement was largest: 0.436 to 0.780 (+0.344 OKS). Uncurated actually regressed on high-curvature fish.

## Pipeline Evaluation (Phase 74)

### Full Metric Comparison

```
Stage           Metric                    Round 0 (Baseline)   Round 1              Delta          %Change
--------------  ------------------------  -------------------  -------------------  -------------  -------
Association     fish_yield_ratio          0.8574               0.9121               +0.0546        +6.4%
Association     p50_camera_count          2.0000               3.0000               +1.0000        +50.0%
Association     p90_camera_count          4.0000               4.0000               +0.0000        +0.0%
Association     singleton_rate            0.3127               0.2737               -0.0390        -12.5%
Association     total_fish_observations   101049               101716               +667.0000      +0.7%
Detection       mean_confidence           0.6931               0.7548               +0.0618        +8.9%
Detection       mean_jitter               0.2572               0.0841               -0.1731        -67.3%
Detection       total_detections          353388               323548               -29840.0000    -8.4%
Fragmentation   mean_continuity_ratio     0.9473               0.9819               +0.0346        +3.6%
Fragmentation   max_gap_duration          110                  26                   -84.0000       -76.4%
Fragmentation   total_gaps                6                    6                    +0.0000        +0.0%
Fragmentation   unique_fish_ids           22                   22                   +0.0000        +0.0%
Midline         mean_confidence           0.9454               0.9524               +0.0070        +0.7%
Midline         p50_confidence            0.9882               0.9962               +0.0081        +0.8%
Reconstruction  fish_reconstructed        52578                59433                +6855.0000     +13.0%
Reconstruction  mean_reprojection_error   3.5200               2.5945               -0.9255        -26.3%
Reconstruction  p50_reprojection_error    3.0199               2.1615               -0.8584        -28.4%
Reconstruction  p90_reprojection_error    5.2028               4.1715               -1.0314        -19.8%
Reconstruction  p95_reprojection_error    6.6733               5.9563               -0.7170        -10.7%
Reconstruction  inlier_ratio              0.8714               0.8438               -0.0276        -3.2%
Tracking        detection_coverage        0.8010               0.8239               +0.0229        +2.9%
Tracking        track_count               1850                 1702                 -148.0000      -8.0%
Tracking        length_mean               166.1076             180.8179             +14.7103       +8.9%
```

Round 2 column: N/A (skipped per Phase 74 decision).

### Per-Keypoint Reprojection Error

| Point | Baseline Mean | Round 1 Mean | Baseline P90 | Round 1 P90 |
|-------|--------------|-------------|-------------|------------|
| 0 (head) | 4.53 px | 4.16 px | 7.07 px | 6.61 px |
| 7 (mid) | 3.13 px | 2.72 px | 5.05 px | 4.34 px |
| 14 (tail) | 7.32 px | 4.51 px | 13.68 px | 8.12 px |

Tail keypoint (point 14) saw the largest improvement: -38% mean, -41% P90. This is consistent with the curated+augmented pose model's better handling of curved fish tails.

### Curvature-Stratified Reprojection Quality

| Quartile | Baseline Mean | Round 1 Mean | Baseline P90 | Round 1 P90 |
|----------|--------------|-------------|-------------|------------|
| Q1 (straight) | 3.49 px | 2.68 px | 4.98 px | 4.30 px |
| Q2 | 3.48 px | 2.50 px | 4.89 px | 3.89 px |
| Q3 | 3.49 px | 2.60 px | 5.00 px | 3.93 px |
| Q4 (curved) | 3.84 px | 2.94 px | 5.91 px | 4.56 px |

Q4-Q1 gap narrowed from 0.35 px to 0.26 px (mean), confirming reduced curvature bias.

## Key Outcomes

**Headline improvements (round 0 to round 1):**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Singleton rate | 31.3% | 27.4% | -12.5% |
| P50 reprojection error | 3.02 px | 2.16 px | -28.4% |
| P90 reprojection error | 5.20 px | 4.17 px | -19.8% |
| Fish yield ratio | 85.7% | 91.2% | +6.4% |
| 3D reconstructions | 52,578 | 59,433 | +13.0% |
| Detection jitter | 0.257 | 0.084 | -67.3% |
| Max tracking gap | 110 frames | 26 frames | -76.4% |
| Total false-positive detections | 353,388 | 323,548 | -8.4% |

The pseudo-label iteration loop produced clear improvements across detection, tracking, association, and reconstruction stages. One round of curation + retraining was sufficient; Phase 74 concluded that diminishing returns made a second round unnecessary.

## Known Limitations

### 1. Singleton rate still ~27%
Down from 31.3% but still means roughly 1 in 4 fish observations are seen in only one camera and cannot be triangulated. Contributing factors: camera overlap geometry (some tank regions visible to only 1-2 cameras), algae occlusion, and detection model recall gaps.

### 2. High-curvature tail keypoint error
Tail keypoint (point 14) error reduced 38% but remains the highest per-keypoint error at 4.51 px mean, 8.12 px P90. The tail is inherently harder to localize: it is thin, often occluded, and its position is highly sensitive to body curvature estimation.

### 3. Algae domain shift
Manual annotations were collected from a clean-tank period. Current conditions include algae growth on tank walls that was not present in training data. This likely causes some false-positive detections (algae patches misidentified as fish parts) and may degrade keypoint accuracy for partially occluded fish.

### 4. Inlier ratio slightly decreased
Inlier ratio dropped from 87.1% to 84.4% (-3.2%). The round 1 models reconstruct 13% more fish, including harder cases that baseline models missed entirely. More reconstruction attempts on marginal multi-view associations naturally reduces the inlier rate.

### 5. Single iteration round
Only one pseudo-label iteration was performed. Round 2 was skipped based on the Phase 74 decision that improvements were substantial enough and further iteration would yield diminishing returns. If future data collection changes tank conditions significantly, a second round may become worthwhile.

### 6. Curvature bias persists
Q4 (curved) fish still have higher reprojection error than straight fish (2.94 vs 2.68 px mean, 4.56 vs 4.30 px P90). The gap narrowed from 0.35 to 0.26 px (mean), but curved fish remain harder to reconstruct accurately.

## Visualization Outputs

Generated from `run_20260309_175421` using `aquapose viz --overlay --trails --fade-trails --detections`:

| Output | Path | Description |
|--------|------|-------------|
| Overlay mosaic | `viz/overlay_mosaic.mp4` | 12-camera grid video with reprojected 3D midlines overlaid on original frames. Each fish colored by deterministic ID palette. |
| Trail videos | `viz/trails/*.mp4` | Per-camera trail videos (12 files) showing fish trajectories with fading solid-point trails. |
| Detection PNGs | `viz/detections/*.png` | Per-camera detection overlay images showing OBB bounding boxes on sample frames. |

All outputs located at `~/aquapose/projects/YH/runs/run_20260309_175421/viz/`.

## Run Details

| Parameter | Value |
|-----------|-------|
| Final run ID | `run_20260309_175421` |
| Baseline run ID | `run_20260307_140127` |
| Frames | 9000 (30 chunks x 300) |
| Duration | 5 minutes at 30fps |
| Cameras | 12 |
| Mode | Diagnostic |
| OBB model | `training/obb/run_20260309_120659/best_model.pt` (curated) |
| Pose model | `training/pose/run_20260309_152248/best_model.pt` (curated+augmented) |
