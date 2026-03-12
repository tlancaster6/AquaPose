# Association Stage Rework: Multi-Keypoint Scoring

## Problem

The association stage (Stage 3) establishes cross-camera fish identities by scoring pairwise tracklet affinity using ray-ray geometry, then clustering with Leiden community detection. The current implementation has a fundamental discrimination bottleneck: it casts a **single ray per tracklet per frame** from one centroid point (keypoint index 2, spine1). This discards the rich per-detection pose information that upstream models already provide (6 anatomical keypoints at mAP50-95=0.964).

### Consequences of single-centroid scoring

- **Weak signal-to-noise**: Correct and incorrect tracklet pairs can produce similar ray-ray distances from a single point, limiting the scorer's ability to distinguish true matches from false ones. Tuning thresholds tighter rejects true matches; tuning looser admits false ones.
- **No redundancy against keypoint jitter**: A single noisy centroid (from partial occlusion, frame-edge clipping, or detector error) corrupts the entire score for that frame.
- **Singleton rate plateau**: ~22% of tracklets remain singletons post-tuning. Many are stable, long-lived tracks that simply can't find cross-view partners under centroid-only geometry — a fundamental discrimination limit, not a parameter tuning problem.

### Upstream ID swaps leak through

Upstream 2D tracking (OC-SORT) is tuned to favor fragmentation over merging to minimize ID swaps, but swaps still occur. A swapped tracklet follows fish A for some frames, then fish B. The current system treats tracklets as atomic units (keep or evict the whole thing), so swapped tracklets either:

- **(Case A)** Fail to associate with either true identity and become unrecoverable singletons — neither half dominates enough to score well against either fish's group.
- **(Case B)** Associate with one fish's group when the swap happens near the start or end of the tracklet, with the long correct segment driving the score above threshold. The short incorrect segment poisons the group's consensus positions for those frames.

### Structural redundancy with refinement

The refinement step (post-clustering) re-casts the same centroid rays, re-triangulates consensus 3D positions, and evicts outlier tracklets — work that largely duplicates what a richer scoring step would already provide. With single-centroid scoring, this separate validation pass is necessary because scoring alone doesn't produce enough geometric evidence. With multi-keypoint scoring, refinement becomes redundant in its current form.

## Proposed Solution

Upgrade association scoring to cast rays from **multiple keypoints per detection per frame**, replacing the single-centroid approach. Add post-clustering group validation with temporal changepoint detection and swap-aware singleton recovery. This is a targeted rework within the existing 2D-track-first architecture, not a pipeline restructuring.

### Pipeline flow

```
1. Multi-keypoint scoring
2. Must-not-link constraints
3. Leiden clustering
4. Group validation (changepoint splitting + outlier eviction)
5. Singleton recovery (with swap-aware split-and-assign)
```

### Step 1: Multi-keypoint pairwise scoring

For each tracklet pair on shared frames, cast rays from K keypoints (matched 1:1 by keypoint index across cameras) instead of 1 centroid. Aggregate per-keypoint ray-ray distances into a richer affinity score that captures geometric consistency across the full body, not just one point.

- Keypoints with low detector confidence on a given frame are excluded, providing natural robustness to partial visibility.
- Mid-body keypoints (head, spine1, spine2) are expected to carry most of the signal; nose and tail are noisy. The optimal subset is an empirical question.
- Scoring produces aggregate scores only (one number per pair). Per-frame ray distances are not stored — steps 4 and 5 compute their own per-frame residuals from scratch against formed groups, which is cheap because they operate on ~9 groups rather than ~1000 pairs.

### Steps 2-3: Constraints and clustering

Unchanged from the current pipeline. Must-not-link constraints (same-camera tracklets with overlapping detected frames cannot share a group) are kept as a cheap safety net. Leiden clustering operates on the richer edge weights from step 1. Must-not-link may become unnecessary in practice with better scoring — candidate for removal if it never triggers.

### Step 4: Group validation

For each tracklet in a multi-view group, cast multi-keypoint rays and compute per-frame residuals against the rest of the group. This requires fresh ray casting (not reuse from step 1) because scoring only covers camera pairs that passed adjacency filtering, leaving gaps for transitive group members. The cost is small — ~9 groups with 4-5 cameras each, vs. ~1000 pairs in scoring. Two checks:

- **Changepoint detection**: Identify temporal transitions in the residual series that indicate an upstream ID swap. The detection method is simple: find the split point that maximizes the difference in mean residual between the two halves, subject to a minimum segment length and a significance threshold. Multi-keypoint residuals produce a sharper discontinuity at swap points than a single centroid would (the entire body geometry changes). Swaps typically occur during proximity/occlusion, so the transition may be gradual rather than a clean step function, but the before/after levels should differ clearly once the fish separate. We don't need to pinpoint the exact swap frame — approximate splitting is sufficient. For multiple swaps within one tracklet, recurse on each half. If a changepoint is found, split the tracklet — the consistent segment stays in the group, the inconsistent segment becomes a singleton candidate for step 5. Error asymmetry favors sensitivity: a false split creates two valid fragments that singleton recovery can reassemble, while a missed swap poisons reconstruction.
- **Outlier eviction**: If a tracklet's overall residual exceeds threshold (no changepoint, just a bad match), evict it to singleton status.

### Step 5: Singleton recovery with swap detection

For each singleton (including those created by step 4), cast multi-keypoint rays and compute per-frame residuals against every existing group. Singletons by definition lack useful stored scoring data — they failed to associate in step 1 — so this is fresh computation. Cost scales with (number of singletons × number of groups), both small. Three outcomes:

- **Strong overall match** to one group: assign to that group.
- **No overall match, but a temporal split produces two segments matching different groups**: this is a case A swap — split and assign each segment to its matching group. Detection works by computing per-frame residuals against all groups once, then sweeping candidate split points across the precomputed residuals to find the best two-group partition. Each half must match its group convincingly (stricter threshold than step 4, where group membership provides a strong prior). This step is lower-confidence than step 4's group-based detection because case A singletons tend to have roughly even splits (if one half dominated, it would have scored well in step 1 and been case B instead), meaning shorter segments and noisier estimates. Controlled by a minimum segment length parameter — setting it very high (e.g., 999999) disables singleton splitting as a natural no-op, avoiding a separate config toggle.
- **No match** even after split analysis: remains a singleton (single-camera-only fish, or genuinely unresolvable).

**Known limitation**: Fish swimming in close parallel for an extended period produce ambiguous residuals against both groups throughout. No temporal signal exists to detect a swap in this case. This is an inherent ambiguity that no method can resolve without appearance features.

### What was dropped and why

- **Fragment merging**: Removed. It existed to rejoin same-camera tracklets separated by small temporal gaps, working against the upstream design intent of favoring fragmentation over merging. The downstream contract only requires group membership (which cameras see which fish in which frames), not continuous per-camera tracklets. If gap-bridging is needed, it belongs in a later stage operating on fully-3D output tracks, not in association.
- **Full refinement re-triangulation**: Replaced by group validation in step 4. Both require ray casting, but step 4 uses multi-keypoint rays (richer signal) and combines outlier eviction with changepoint splitting in a single pass, rather than re-triangulating a single centroid independently of scoring.

### What stays the same

- **2D tracking first, then associate**: The architectural lesson from v1.0 (per-frame 3D association is fragile; tracklet-level temporal evidence is structurally superior) remains valid and is preserved.
- **Downstream contract**: `TrackletGroup` output format is unchanged. Reconstruction (Stage 5) continues to independently triangulate keypoints for midline fitting — its work is complementary, not duplicated.

### Architectural context

An earlier design discussion considered a full overhaul to 3D-first volumetric voting (detect fish in 3D per frame, then back-assign 2D tracklets). This was rejected because it re-introduces the per-frame 3D association fragility that caused the v1.0→v2.0 pipeline restructuring. The current proposal preserves the structural advantage of tracklet-level temporal evidence while addressing the discrimination bottleneck within that architecture.

### Computational cost

Estimated 2-3x increase in scoring wall time (from ~7s to ~15-20s per 300-frame chunk), driven by additional ray casts and distance computations. Mitigated by keypoint confidence filtering (fewer rays on partial-visibility frames) and improved early termination discrimination (clearer signal → faster rejection of bad pairs). Group validation and singleton recovery require fresh ray casting but operate on a much smaller problem (~9 groups × 4-5 cameras, ~20% singletons × ~9 groups) compared to the O(pairs × frames) scoring step, so their cost is minor relative to step 1. Forward LUT is already at pixel-exact resolution (grid_step=1); no LUT changes needed.
