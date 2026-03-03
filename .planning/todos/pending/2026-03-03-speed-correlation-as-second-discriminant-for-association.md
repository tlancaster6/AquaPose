---
created: 2026-03-03T00:00:00Z
title: Speed correlation as second discriminant for cross-view association
area: association
files:
  - src/aquapose/core/association/scoring.py
---

## Problem

Cross-view association recovers 7/9 fish but leaves ~72% of tracklets as singletons.
The root cause is that correct and incorrect tracklet pairs have overlapping ray-ray
distance distributions in the densely-packed tank. The centroid-based ray distance
threshold (0.03m) can't be widened without admitting false matches that poison Leiden
clustering — yield drops from 22% to 3-10% at wider thresholds (see
`.planning/debug/association-singletons.md`).

The missing singletons are stable, long-lived tracklets whose correct cross-view
partners produce ray-ray distances just above the current threshold. We need a second
independent signal that lets us **safely widen the ray distance gate** by filtering out
the false matches that currently flood in at wider thresholds.

## Proposed Solution

Add **speed profile correlation** as a third multiplicative factor in the existing
score formula.

### Formulation

Current score:
```
score = ray_affinity * overlap_weight
```

New score:
```
score = ray_affinity * speed_affinity * overlap_weight
```

Where:
- `ray_affinity`: existing soft linear kernel `max(0, 1 - distance/threshold)`, with
  `ray_distance_threshold` widened (tuned, likely 0.05-0.08m vs current 0.03m)
- `speed_affinity`: `max(0, pearson_r(speed_a, speed_b))`, where speed is scalar
  frame-to-frame centroid displacement magnitude over shared frames. Negative
  correlations map to 0 (anti-correlated motion = definitely not the same fish).
  `nan` from zero-variance inputs maps to 1.0 (pass-through — two stationary
  tracklets are consistent, not suspicious).
- `overlap_weight`: unchanged (`min(t_shared, t_saturate) / t_saturate`)

### Why this formulation

- **No zone boundaries**: no inner/outer threshold split. A single wider
  `ray_distance_threshold` controls ray scoring. Speed correlation naturally
  suppresses false matches that the wider threshold admits.
- **No weighting parameters**: all three factors are in [0, 1] and multiply directly.
  Each acts as a gate — a pair must have reasonable ray distance AND correlated speed
  AND sufficient overlap. No relative weights to tune.
- **One new parameter: none**. `speed_affinity` is just `max(0, r)`. The only
  parameters to tune are `ray_distance_threshold` and `score_min`, both of which
  already exist.

### Speed computation

Scalar speed per frame = magnitude of centroid displacement from previous frame.
Computed from existing `Tracklet2D.centroids` via finite differences — no new data,
no boxmot internals, no pipeline changes.

### Noise handling

- **No explicit smoothing**: Pearson correlation over 50+ shared frames implicitly
  averages out uncorrelated jitter. Explicit smoothing (e.g. moving average) would
  smear the sharp speed changes that are most discriminative.
- **No minimum variance check**: two tracklets both showing near-zero speed is
  confirming evidence (both agree the fish is stationary). `numpy.corrcoef` returns
  `nan` for zero-variance inputs; we map `nan` → `speed_affinity = 1.0`
  (pass-through), letting ray distance decide as it does today.

### Why speed, not velocity direction

Scalar speed is naturally view-invariant — a fish's speed changes are driven by its
actual movement and appear correlated across all cameras regardless of viewing angle.
Direction requires projecting 2D velocities to 3D, adding complexity and reintroducing
calibration sensitivity.

### Why not midline shape

Midlines aren't available at the association stage (computed after association).
Reordering the pipeline is non-trivial, would lose the current efficiency gain of only
running pose estimation on associated tracklets, and the noise-to-signal ratio on
midline shape may be too high for reliable discrimination.

## Risks

- **Schooling behavior**: Fish in a dense tank may exhibit correlated speeds. In this
  case speed affinity would be high for both correct and incorrect pairs — neutral,
  not harmful. The worst case is no improvement over ray-only scoring, not regression.
- **Short overlaps**: Pearson r is noisy with few samples. The existing `t_min` and
  `overlap_weight` already penalize short overlaps, which limits exposure here.

## Evaluation Strategy

Use existing `tune_association.py` harness metrics:
- **Fish recovered** (baseline: 7/9)
- **Yield** (baseline: 23.9%)
- **Mean reprojection error** (baseline: 10.08px)
- **Singleton rate** (baseline: 72%)

Success = fish count or yield increases without error regression. Even if we don't
recover fish 8 and 9, adding cameras to existing groups (e.g. 3-4 → 5-6 cameras per
fish) is a real win — more views improve downstream midline triangulation.

## Tuning Strategy

2D grid sweep over `ray_distance_threshold` x `score_min` using existing
`tune_association.py` infrastructure. No new sweep dimensions needed since speed
affinity has no tunable parameters.

- `ray_distance_threshold`: sweep wider than current 0.03m (e.g. 0.03-0.10m)
- `score_min`: retune since score distribution shifts with the speed factor
- Hold `leiden_resolution` and `early_k` fixed at current tuned values (1.5 and 20)

## Implementation Notes

- Modify `score_tracklet_pair()` in `scoring.py` to compute speed arrays from
  centroids over shared frames, compute Pearson r, and multiply into the score.
- Speed arrays are computed per-call from the centroid tuples already on Tracklet2D.
  If profiling shows this is a bottleneck, pre-compute and cache on Tracklet2D later.
- No changes to Tracklet2D, pipeline ordering, or clustering logic.
