# Phase 88: Multi-Keypoint Pairwise Scoring - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace single-centroid ray casting in association scoring with K-keypoint vectorized scoring. The scorer casts rays from all confident keypoints per detection per frame instead of one centroid, producing richer pairwise affinity scores. Depends on Phase 87 (Tracklet2D keypoint propagation).

</domain>

<decisions>
## Implementation Decisions

### Keypoint selection
- Use all 6 keypoints (nose, head, spine1, spine2, spine3, tail) — no subset exclusion
- All keypoints contribute equally (no confidence weighting on contribution)
- Add `keypoint_confidence_floor` to AssociationConfig, default 0.3 (matches tracking's centroid_confidence_floor, but independently tunable)
- A frame with only 1 confident keypoint still contributes (minimum is 1, not 2+)

### Per-frame aggregation
- Matched keypoint pairing: nose-to-nose, head-to-head, etc. (K distances, not K x K)
- Only score keypoint indices where BOTH tracklets are above the confidence floor on that frame (intersection)
- Arithmetic mean of matched keypoint ray-ray distances as default aggregation method
- Add `aggregation_method` config field (string/enum) defaulting to `"mean"` — only implement mean now, plumbing for future alternatives (median, trimmed_mean)

### Fallback behavior
- No centroid fallback: if either tracklet has `keypoints=None`, return score 0.0 for that pair. Phase 89 removes fragment merging, so None-keypoint tracklets will not occur in normal operation
- Remove centroid scoring code path entirely — clean break, no dead code
- Frames where all keypoints on either tracklet fall below the confidence floor are skipped (don't count toward t_shared or score_sum)

### Soft kernel
- Keep existing soft linear kernel: `max(0, 1 - mean_dist / threshold)`
- Apply kernel AFTER computing mean keypoint distance (not per-keypoint)
- Same `ray_distance_threshold` value (0.01m) — tuning deferred to Phase 92
- Early termination logic unchanged: bail after early_k frames if score_sum == 0

### Claude's Discretion
- Internal vectorization strategy (how to batch K keypoints across N frames efficiently)
- Whether to reshape as (N*K, 2) for a single cast_ray call or iterate per-keypoint
- Test fixture design for round-trip LUT correctness verification

</decisions>

<specifics>
## Specific Ideas

- The user explicitly wants to move away from centroids — no backward-compatibility centroid fallback
- Phase 89 (Fragment Merging Removal) will eliminate the only source of None-keypoint tracklets, so the no-fallback decision is safe in context of the milestone

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ray_ray_closest_point_batch(origins_a, dirs_a, origins_b, dirs_b)` — already vectorized for N ray pairs, returns (N,) distances
- `ForwardLUT.cast_ray(pixels: (N, 2))` — accepts batched pixel coords, returns (N, 3) origins + directions
- `_batch_score_frames()` — current centroid scoring helper; will be replaced with keypoint-aware version
- `AssociationConfigLike` protocol — structural protocol for config; needs new fields added

### Established Patterns
- Scoring uses torch tensors for LUT queries, converts to numpy float64 for ray math
- Config protocol in core/ mirrors frozen dataclass in engine/ (IB-003 import boundary)
- Soft kernel: `np.where(inlier, 1.0 - dists / threshold, 0.0)` with vectorized numpy

### Integration Points
- `Tracklet2D.keypoints` (T, K, 2) and `keypoint_conf` (T, K) — Phase 87 output, direct input to new scorer
- `AssociationConfig` in `engine/config.py` — add `keypoint_confidence_floor`, `aggregation_method`
- `AssociationConfigLike` protocol in `core/association/scoring.py` — mirror new config fields
- `score_all_pairs()` and `score_tracklet_pair()` — public API unchanged, internal implementation changes

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 88-multi-keypoint-pairwise-scoring*
*Context gathered: 2026-03-11*
