# Phase 92: Parameter Tuning Pass Results

**Date:** 2026-03-12
**Data:** YH project, 900 frames (3 chunks), 12 cameras

## Methodology

**Sweep tool:** `hatch run aquapose tune -p YH --stage association`

The sweep runs a two-phase grid search over the association stage configuration:

1. **Joint grid (Phase 1):** Exhaustive 3D grid over the three parameters that interact most tightly: `ray_distance_threshold`, `score_min`, and `keypoint_confidence_floor`. Each combo runs the full association stage on cached diagnostic data and measures fish yield ratio and mean centroid reprojection error.

2. **Carry-forward (Phase 2):** The joint-grid winner is held fixed while each remaining parameter (`eviction_reproj_threshold`, `leiden_resolution`, `early_k`) is swept independently.

**Baseline:** Current v3.8 defaults (which were set during the v3.1 tuning pass in Phase 43.1 and refined in subsequent phases). The v3.7 baseline singleton rate was 27%; the v3.8 defaults had already reduced this to ~5% via multi-keypoint scoring, group validation, and singleton recovery.

## Grid Ranges

| Parameter | Values | Default |
|-----------|--------|---------|
| `ray_distance_threshold` | 0.01, 0.02, 0.03 | 0.01 |
| `score_min` | 0.03, 0.15, 0.30 | 0.30 |
| `keypoint_confidence_floor` | 0.20, 0.30, 0.40 | 0.20 |
| `eviction_reproj_threshold` | 0.02, 0.03, 0.05 | 0.02 |
| `leiden_resolution` | 0.5, 1.0, 2.0 | 1.0 |
| `early_k` | 5, 10, 30 | 10 |

## Joint Grid Results (27 combos)

| ray_dist | score_min | kpt_floor | yield | error (px) |
|----------|-----------|-----------|-------|------------|
| 0.010 | 0.030 | 0.20 | 97.4% | 2.94 |
| 0.010 | 0.030 | 0.30 | 97.4% | 2.95 |
| 0.010 | 0.030 | 0.40 | 97.4% | 2.95 |
| 0.010 | 0.150 | 0.20 | 99.5% | 2.99 |
| 0.010 | 0.150 | 0.30 | 99.5% | 3.01 |
| 0.010 | 0.150 | 0.40 | 99.5% | 3.01 |
| **0.010** | **0.300** | **0.20** | **102.6%** | **2.85** |
| 0.010 | 0.300 | 0.30 | 102.6% | 2.89 |
| 0.010 | 0.300 | 0.40 | 102.6% | 2.89 |
| 0.020 | 0.030 | 0.20 | 97.4% | 2.94 |
| 0.020 | 0.030 | 0.30 | 97.4% | 2.95 |
| 0.020 | 0.030 | 0.40 | 97.4% | 2.95 |
| 0.020 | 0.150 | 0.20 | 97.4% | 2.96 |
| 0.020 | 0.150 | 0.30 | 97.4% | 2.98 |
| 0.020 | 0.150 | 0.40 | 97.4% | 2.98 |
| 0.020 | 0.300 | 0.20 | 98.9% | 2.97 |
| 0.020 | 0.300 | 0.30 | 98.9% | 2.98 |
| 0.020 | 0.300 | 0.40 | 98.9% | 2.98 |
| 0.030 | 0.030 | 0.20 | 97.4% | 2.92 |
| 0.030 | 0.030 | 0.30 | 97.4% | 2.94 |
| 0.030 | 0.030 | 0.40 | 97.4% | 2.94 |
| 0.030 | 0.150 | 0.20 | 97.4% | 2.95 |
| 0.030 | 0.150 | 0.30 | 97.4% | 2.97 |
| 0.030 | 0.150 | 0.40 | 97.4% | 2.97 |
| 0.030 | 0.300 | 0.20 | 94.0% | 2.88 |
| 0.030 | 0.300 | 0.30 | 94.0% | 2.90 |
| 0.030 | 0.300 | 0.40 | 94.0% | 2.90 |

**Joint grid winner:** ray_dist=0.010, score_min=0.300, kpt_floor=0.20 (102.6% yield, 2.85px error)

## Carry-Forward Results

| Parameter | Value | Yield | Error (px) |
|-----------|-------|-------|------------|
| eviction_reproj_threshold | 0.02 | 102.6% | 2.85 |
| eviction_reproj_threshold | 0.03 | 102.6% | 2.85 |
| eviction_reproj_threshold | 0.05 | 102.6% | 2.85 |
| leiden_resolution | 0.5 | 97.0% | 2.96 |
| leiden_resolution | **1.0** | **102.6%** | **2.85** |
| leiden_resolution | 2.0 | 56.6% | 3.04 |
| early_k | 5.0 | 102.6% | 2.85 |
| early_k | **10.0** | **102.6%** | **2.85** |
| early_k | 30.0 | 99.8% | 2.91 |

**Notable observations:**
- `eviction_reproj_threshold` is insensitive in the tested range (all three values produce identical metrics)
- `leiden_resolution=2.0` is catastrophic (56.6% yield) due to over-clustering
- `leiden_resolution=0.5` under-clusters slightly (97.0% yield vs 102.6%)
- `early_k=5` and `early_k=10` are equivalent; `early_k=30` slightly degrades yield

## Winner vs Baseline Comparison

| Metric | Baseline | Winner | Delta |
|--------|----------|--------|-------|
| fish_yield_ratio | 102.6% | 102.6% | +0.0% |
| mean_reprojection_error | 2.855 px | 2.855 px | +0.000 px |
| max_reprojection_error | 160.295 px | 160.295 px | +0.000 px |
| singleton_rate | 5.4% | 5.4% | +0.0% |

**The sweep winner is identical to the current defaults.** No parameter changes are needed.

## Conclusion

The current v3.8 association defaults are already optimal across the tested grid. The parameters set during earlier development phases (ray_distance_threshold=0.01, score_min=0.30, keypoint_confidence_floor=0.20, eviction_reproj_threshold=0.02, leiden_resolution=1.0, early_k=10) represent the best configuration found by the sweep.

**v3.7 to v3.8 improvement (for reference):**
- Singleton rate: 27% (v3.7) -> 5.4% (v3.8) = 80% reduction
- Reprojection error: ~3.0px (v3.7) -> 2.85px (v3.8) = 5% improvement
- Fish yield: stable at ~100%

These gains come from the architectural improvements (multi-keypoint scoring, group validation with changepoint detection, singleton recovery) rather than parameter tuning, which confirms that the algorithm design was sound and the defaults were well-chosen during development.

## E2E Validation

A 3-chunk pipeline run (run_20260312_151712) with current defaults confirmed end-to-end correctness. Overlay visualizations show correct fish associations across all cameras.

## Final Configuration (unchanged)

```yaml
# AssociationConfig defaults (no changes from sweep)
ray_distance_threshold: 0.01
score_min: 0.30
keypoint_confidence_floor: 0.20
eviction_reproj_threshold: 0.02
leiden_resolution: 1.0
early_k: 10
```
