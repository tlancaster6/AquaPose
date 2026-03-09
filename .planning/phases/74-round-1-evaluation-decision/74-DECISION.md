# Round 1 Evaluation Decision

## Model Provenance

| Model | Source | Training Data |
|-------|--------|---------------|
| OBB (detection) | `training/obb/run_20260309_120659/best_model.pt` | Curated pseudo-labels (Phase 73 winner) |
| Pose (midline) | `training/pose/run_20260309_152248/best_model.pt` | Curated+augmented pseudo-labels (Phase 73 winner) |

**A/B arm used:** Curated (OBB) and Curated+Augmented (Pose) -- winners from Phase 73 A/B comparison.

## Pipeline Run Details

| Parameter | Baseline (Round 0) | Round 1 |
|-----------|-------------------|---------|
| Run ID | `run_20260307_140127` | `run_20260309_175421` |
| Frames | 9000 (30 chunks x 300) | 9000 (30 chunks x 300) |
| Mode | diagnostic | diagnostic |
| Total time | ~100s/chunk | ~130s/chunk |

## Metric Comparison Table

```
Stage           Metric                    run_20260307_140127  run_20260309_175421  Delta          %Change
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

## Primary Metrics Summary

All primary metrics improved in round 1:

| Metric | Baseline | Round 1 | Delta | Direction |
|--------|----------|---------|-------|-----------|
| **singleton_rate** | 31.3% | 27.4% | -3.9 pp | IMPROVED (lower is better) |
| **p50 reprojection error** | 3.02 px | 2.16 px | -0.86 px | IMPROVED (-28.4%) |
| **p90 reprojection error** | 5.20 px | 4.17 px | -1.03 px | IMPROVED (-19.8%) |

## Secondary Metrics of Interest

| Metric | Baseline | Round 1 | Delta | Notes |
|--------|----------|---------|-------|-------|
| fish_yield_ratio | 85.7% | 91.2% | +5.5 pp | More fish successfully associated |
| continuity_ratio | 94.7% | 98.2% | +3.5 pp | Fewer tracking gaps |
| fish_reconstructed | 52,578 | 59,433 | +6,855 | 13% more 3D reconstructions |
| detection jitter | 0.257 | 0.084 | -0.173 | 67% less detection noise |
| p50_camera_count | 2.0 | 3.0 | +1.0 | Median cameras per fish increased |
| detection confidence | 0.693 | 0.755 | +0.062 | Higher confidence detections |
| total_detections | 353,388 | 323,548 | -29,840 | Fewer false positive detections |
| inlier_ratio | 87.1% | 84.4% | -2.8 pp | Slight decrease (more attempts, stricter) |

### Per-Keypoint Reprojection Error

| Point | Baseline Mean | Round 1 Mean | Baseline P90 | Round 1 P90 |
|-------|--------------|-------------|-------------|------------|
| 0 (head) | 4.53 px | 4.16 px | 7.07 px | 6.61 px |
| 7 (mid) | 3.13 px | 2.72 px | 5.05 px | 4.34 px |
| 14 (tail) | 7.32 px | 4.51 px | 13.68 px | 8.12 px |

Notable: Tail keypoint error reduced by 38% (mean) and 41% (P90) -- the largest improvement, consistent with the curated+augmented pose model's better handling of curved fish.

### Curvature-Stratified Quality

| Quartile | Baseline Mean | Round 1 Mean | Baseline P90 | Round 1 P90 |
|----------|--------------|-------------|-------------|------------|
| Q1 (straight) | 3.49 px | 2.68 px | 4.98 px | 4.30 px |
| Q2 | 3.48 px | 2.50 px | 4.89 px | 3.89 px |
| Q3 | 3.49 px | 2.60 px | 5.00 px | 3.93 px |
| Q4 (curved) | 3.84 px | 2.94 px | 5.91 px | 4.56 px |

The Q4-Q1 gap narrowed from 0.35 px to 0.26 px (mean), confirming reduced curvature bias.

## Go/No-Go Verdict

_To be filled during decision checkpoint review._
