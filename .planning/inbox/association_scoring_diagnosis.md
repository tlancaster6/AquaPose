# Association Scoring Diagnosis

## Problem
Cross-view association produces 45/49 singleton groups. Only 4 multi-view groups formed from ~52 tracklets across 10 cameras with 9 fish.

## Investigation Summary

### What's NOT broken
- **Detection/Midline/Tracking**: All working well. ~41.5 dets/frame, 90.7% midline yield, 11 tracks across 10 cameras.
- **Overlap graph**: All 66 camera pairs connected (all > 100 shared voxels). Every pair is being scored.
- **Ray-ray distances for correct pairs**: Median 0.15–2.0cm, well under the 3cm threshold. Correct pairs get 100% inlier rate.
- **Scoring produces 87 edges** that survive `score_min=0.3`. The edges form a single 45-node connected component across 7 cameras.

### Root Cause: Binary inlier counting discards distance magnitude
The scoring function (`scoring.py:200`) uses a hard threshold: `if dist < ray_distance_threshold` → inlier. A correct pair at 0.41cm and a wrong pair at 1.86cm both count as inliers equally. The final score `f = inlier_count / t_shared` is 1.0 for both.

**Concrete example** (e3v831e track 6):
- Correct match `(e3v83f0, 8)`: median ray-ray distance **0.41cm** → score **0.98**
- Wrong match `(e3v83f0, 1)`: median ray-ray distance **1.86cm** → score **0.98**
- The 4.5x distance difference is invisible to the scoring function.

This means Leiden receives a graph where correct and incorrect edges have indistinguishable weights, making community detection impossible. Must-not-link enforcement then fragments the oversized communities into singletons.

### Ghost penalty is ineffective
- For wrong pairs, the 3D midpoint lands in a region where other cameras DO have nearby detections (from other fish), so `ghost_ratio ≈ 0` even for wrong pairs.
- For some correct pairs, the midpoint falls outside the inverse LUT volume, so `n_visible_other = 0` and ghost penalty can't fire.
- With 9 fish densely packed, there's almost always a detection within the 50px `ghost_pixel_threshold`.

### XY-Z anisotropy confirmed
- Reconstruction jitter is **4–10x larger in Z than XY** (σZ/σXY = 4.0–9.8x).
- All cameras are mounted at the same height (Origin Z = 1.031) looking steeply downward (Dir Z ≈ 0.99).
- The ray-ray miss vector is inherently horizontal (XY) for this camera geometry, so 3D miss distance ≈ XY miss distance for all pairs.
- Z anisotropy doesn't affect pairwise scoring (miss vector is already XY), but does affect downstream triangulated point accuracy.

### Per-track separability varies
For e3v831e ↔ e3v83f0 (11×10 = 110 pairs):
- ~6 of 11 tracks have clear best-match separation (gap 0.56–2.14cm, ratio 1.8–4.5x)
- ~3 of 11 tracks have ambiguous best/2nd matches (gap < 0.05cm) — these fish are too close for this camera pair alone, but multi-camera consensus should resolve them since the full graph has 87 edges across 7 cameras.

## Proposed Fix: Soft Scoring Kernel

Replace binary inlier counting with distance-weighted scoring:

```python
# Current (binary):
if dist < config.ray_distance_threshold:
    inlier_count += 1

# Proposed (soft kernel):
if dist < config.ray_distance_threshold:
    contribution = 1.0 - (dist / config.ray_distance_threshold)  # linear
    # OR: contribution = exp(-dist² / (2 * sigma²))              # gaussian
    score_sum += contribution
    inlier_count += 1
```

Final score becomes `score_sum / t_shared` instead of `inlier_count / t_shared`.

**Expected effect**: Correct pair at 0.41cm gets contribution ~0.86, wrong pair at 1.86cm gets contribution ~0.38. The 4.5x distance difference translates to a ~2.3x score difference, giving Leiden clear community structure.

## Future Consideration: XY Projection Method
Since Z reconstruction error is 4–10x higher than XY, and fish separation is primarily in XY, a projection-based scoring variant could be explored. Project the 3D midpoint to XY only and cluster in 2D. Z information would then be used downstream (e.g., during refinement or reconstruction) where multi-view consensus reduces Z uncertainty, potentially resolving ID swaps from XY-ambiguous cases (occlusions, fish crossing paths in XY).
