# Phase 90: Group Validation with Changepoint Detection - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

After Leiden clustering, audit each tracklet group for temporal ID swaps and persistent outliers. Split swapped tracklets at detected changepoints; evict uniformly-bad tracklets. Replace `refinement.py` with `validation.py`. Singleton recovery is Phase 91 — this phase only produces singleton candidates, it does not reassign them.

</domain>

<decisions>
## Implementation Decisions

### Residual computation
- Use matched keypoint rays (Phase 88 pattern): for each frame, cast rays from all confident keypoints on each tracklet, compare matched keypoints across cameras (nose-to-nose, head-to-head, etc.) via ray-ray distance
- All 6 keypoints contribute equally — the per-keypoint confidence floor from Phase 88 already filters unreliable keypoints
- Frames with fewer than 2 cameras having confident keypoints are skipped (excluded from residual series)
- Residuals are computed per-tracklet vs rest of group: each tracklet gets one residual time series representing its mean keypoint ray distance against all other tracklets in the group for that frame

### Changepoint detection
- Simple threshold + run classification: each frame is classified as "consistent" (residual < threshold) or "inconsistent" based on the eviction threshold
- Find the longest consistent run; the transition point is the changepoint
- Single changepoint per pass, but the validation function should be composable for iterative passes (split, recompute residuals against updated group, check again)
- Minimum segment length after split: configurable, default ~10 frames (~0.3s at 30fps)
- Changepoint threshold reuses `eviction_reproj_threshold` — one parameter, tunable in Phase 92

### Eviction vs splitting
- Decision tree: (1) Compute residual series. (2) If mostly consistent (>50% of frames below threshold), keep as-is. (3) If changepoint found with both segments >= min_length, split: consistent segment stays, inconsistent segment becomes singleton. (4) If no clear changepoint (uniformly high residual), evict entire tracklet as singleton.
- Inconsistent segments become immediate singleton candidates — Phase 91 handles recovery/reassignment
- Thin group handling (group drops to 1 camera after validation): Claude's discretion

### TrackletGroup output contract
- Keep same fields: `per_frame_confidence` and `consensus_centroids` are populated by validation just as refinement did
- Internal computation changes (multi-keypoint instead of centroid-only) but output shape is unchanged
- After splits/evictions, recompute per_frame_confidence and overall confidence from the cleaned group's residuals
- Split tracklets become new Tracklet2D instances with sliced arrays (frames, centroids, keypoints, keypoint_conf, bboxes, frame_status)
- Split segments get new unique local track_ids to prevent duplicate IDs within the same camera

### Claude's Discretion
- Thin group dissolution policy (keep single-camera groups vs dissolve to singletons)
- Internal implementation of the consistent-run detection algorithm
- How to recompute consensus_centroids after membership changes (can reuse existing triangulation helpers)
- Whether to log/emit diagnostic info about splits and evictions for debugging

</decisions>

<specifics>
## Specific Ideas

- Validation function should be composable for iterative splitting: split once, recompute residuals against updated group, check for more changepoints, repeat. The architecture should support this naturally even if Phase 90 only does one pass.
- The Phase 88 multi-keypoint scoring pattern (batched ray casting, confidence intersection mask) should be reusable for residual computation — don't reinvent the wheel.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scoring.py:_score_pair_batch()`: Batched multi-keypoint ray casting with confidence masks. Core pattern for residual computation.
- `scoring.py:ray_ray_closest_point()`: Pairwise ray-ray distance + midpoint. Used by current refinement and reusable for consensus.
- `refinement.py:_compute_frame_consensus()`: Per-frame 3D consensus triangulation from tracklet rays. Can be adapted for multi-keypoint consensus.
- `refinement.py:_compute_per_frame_confidence()`: Per-frame confidence from ray convergence quality. Output format matches what TrackletGroup expects.
- `Tracklet2D`: Has `keypoints: (T, K, 2)` and `keypoint_conf: (T, K)` fields from Phase 87. All the data needed for multi-keypoint residuals.

### Established Patterns
- Config protocol pattern (`RefinementConfigLike`): core/ modules define a Protocol for config, satisfied by engine's `AssociationConfig` without import. validation.py should follow the same pattern.
- `TrackletGroup` is a frozen dataclass — new instances must be constructed, not mutated.
- Stage integration: `stage.py` calls refinement at Step 5 via lazy import. Same pattern for validation.

### Integration Points
- `stage.py:AssociationStage.run()` line 104-108: Replace `refine_clusters()` call with `validate_groups()` (or equivalent)
- `AssociationConfig` in engine/config.py: Add new config fields (min_segment_length) alongside existing eviction_reproj_threshold
- `association/__init__.py`: Update public API exports (remove refinement, add validation)
- Downstream consumers of `per_frame_confidence` and `consensus_centroids`: midline and reconstruction stages

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 90-group-validation-with-changepoint-detection*
*Context gathered: 2026-03-11*
