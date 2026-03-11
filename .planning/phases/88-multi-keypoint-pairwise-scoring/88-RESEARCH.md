# Phase 88: Multi-Keypoint Pairwise Scoring - Research

**Researched:** 2026-03-11
**Domain:** Association scoring refactor — centroid to multi-keypoint ray casting
**Confidence:** HIGH

## Summary

Phase 88 replaces the single-centroid ray casting in `_batch_score_frames()` with matched-keypoint ray casting using the `Tracklet2D.keypoints` and `keypoint_conf` arrays added in Phase 87. The change is well-scoped: the public API (`score_tracklet_pair`, `score_all_pairs`) does not change signatures, only the internal scoring logic and config protocol gain new fields.

The existing vectorized infrastructure (`ForwardLUT.cast_ray`, `ray_ray_closest_point_batch`) already handles batched (N, 2) pixel inputs and returns (N, 3) arrays, so extending to K keypoints per frame is a matter of reshaping (N*K, 2) inputs, computing distances, then reshaping back to (N, K) for per-frame aggregation. The confidence masking adds a boolean intersection step per frame to exclude low-confidence keypoints.

**Primary recommendation:** Replace `_batch_score_frames()` with a keypoint-aware version that reshapes keypoints to (N*K, 2), casts rays in one call per camera, computes matched-keypoint distances, masks by confidence intersection, aggregates per-frame via mean, then applies the soft kernel. Remove centroid code path entirely.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use all 6 keypoints (nose, head, spine1, spine2, spine3, tail) — no subset exclusion
- All keypoints contribute equally (no confidence weighting on contribution)
- Add `keypoint_confidence_floor` to AssociationConfig, default 0.3
- A frame with only 1 confident keypoint still contributes (minimum is 1, not 2+)
- Matched keypoint pairing: nose-to-nose, head-to-head, etc. (K distances, not K x K)
- Only score keypoint indices where BOTH tracklets are above the confidence floor on that frame (intersection)
- Arithmetic mean of matched keypoint ray-ray distances as default aggregation method
- Add `aggregation_method` config field (string/enum) defaulting to `"mean"` — only implement mean now
- No centroid fallback: if either tracklet has `keypoints=None`, return score 0.0
- Remove centroid scoring code path entirely — clean break, no dead code
- Keep existing soft linear kernel: `max(0, 1 - mean_dist / threshold)`
- Apply kernel AFTER computing mean keypoint distance (not per-keypoint)
- Same `ray_distance_threshold` value (0.01m)
- Early termination logic unchanged: bail after early_k frames if score_sum == 0
- Frames where all keypoints on either tracklet fall below the confidence floor are skipped (don't count toward t_shared or score_sum)

### Claude's Discretion
- Internal vectorization strategy (how to batch K keypoints across N frames efficiently)
- Whether to reshape as (N*K, 2) for a single cast_ray call or iterate per-keypoint
- Test fixture design for round-trip LUT correctness verification

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SCORE-01 | Association scorer casts rays from multiple keypoints per detection per frame, not just one centroid | Replace `_batch_score_frames()` centroid logic with keypoint extraction from `Tracklet2D.keypoints` |
| SCORE-02 | Low-confidence keypoints (below configurable threshold) are excluded from scoring per frame | Add `keypoint_confidence_floor` to config protocol + dataclass; build boolean mask from `keypoint_conf` intersection |
| SCORE-03 | Per-keypoint ray-ray distances are aggregated into a single pairwise score via configurable method (mean, trimmed mean) | Add `aggregation_method` field; implement mean; plumb for future alternatives |
| SCORE-04 | Multi-keypoint scoring is vectorized (NumPy broadcasting, no per-pair Python loop) | Reshape (N, K, 2) -> (N*K, 2) for single `cast_ray` call; `ray_ray_closest_point_batch` handles all distances in one call |
</phase_requirements>

## Architecture Patterns

### Current Scoring Flow (centroid-based)
```
score_tracklet_pair()
  -> split shared_frames into early + remaining
  -> _batch_score_frames(early_frames, ...)
     -> extract centroids (N, 2)
     -> cast_ray per camera -> origins (N, 3), dirs (N, 3)
     -> ray_ray_closest_point_batch -> dists (N,)
     -> soft kernel -> contributions (N,)
     -> sum contributions
  -> early termination check
  -> _batch_score_frames(remaining_frames, ...)
  -> combine: f = score_sum / t_shared, w = overlap_reliability, score = f * w
```

### New Scoring Flow (keypoint-based)
```
score_tracklet_pair()
  -> early exit if either tracklet has keypoints=None -> return 0.0
  -> split shared_frames into early + remaining
  -> _batch_score_frames_keypoint(early_frames, ...)
     -> extract keypoints for both tracklets: (N, K, 2)
     -> extract confidences for both: (N, K)
     -> build valid mask: both_confident = (conf_a >= floor) & (conf_b >= floor)  # (N, K)
     -> count valid_per_frame = both_confident.sum(axis=1)  # (N,)
     -> skip_mask = valid_per_frame == 0  # frames to exclude entirely
     -> reshape valid keypoints to (M, 2) where M = sum of valid keypoints across all frames
     -> cast_ray per camera -> origins (M, 3), dirs (M, 3)
     -> ray_ray_closest_point_batch -> dists (M,)
     -> scatter back to per-frame groups, compute mean per frame
     -> apply soft kernel per frame: max(0, 1 - mean_dist / threshold)
     -> return (sum of contributions, count of non-skipped frames)
  -> early termination check (using effective frame count)
  -> _batch_score_frames_keypoint(remaining_frames, ...)
  -> combine: f = score_sum / effective_t_shared, w = overlap_reliability
```

### Key Design Decision: Reshape (N*K, 2) vs Per-Keypoint Iteration

**Recommendation: Reshape approach** — flatten all valid keypoints across all frames into a single (M, 2) array, make one `cast_ray` call per camera, one `ray_ray_closest_point_batch` call, then scatter results back.

Rationale:
- `ForwardLUT.cast_ray()` already accepts arbitrary (N, 2) batches
- `ray_ray_closest_point_batch()` is pure NumPy vectorized
- One call with M points is faster than K calls with N points each (less Python overhead, better cache locality)
- The scatter-back step is simple index arithmetic with `np.split` or cumulative sum indexing

### Config Changes

Two new fields on `AssociationConfig` (frozen dataclass) and `AssociationConfigLike` (protocol):

1. `keypoint_confidence_floor: float = 0.3` — minimum confidence for a keypoint to participate in scoring
2. `aggregation_method: str = "mean"` — how to combine per-keypoint distances within a frame

### Frame Skipping and t_shared Adjustment

When all keypoints on either tracklet fall below the confidence floor for a frame, that frame is "skipped" — it does not count toward `t_shared` or `score_sum`. This changes the semantics slightly:

- **Before:** `t_shared = len(shared_frames)`, always equal to the frame intersection count
- **After:** `effective_t_shared = t_shared - n_skipped_frames`, may be less

The `t_min` check should still use the raw `t_shared` (do they share enough frames?), but `f = score_sum / effective_t_shared` uses the adjusted count. If `effective_t_shared == 0`, return 0.0.

### Soft Kernel Application Point

The user decision is clear: apply the soft kernel AFTER computing the mean keypoint distance per frame. This means:
1. Compute matched distances for each confident keypoint pair
2. Take the arithmetic mean of those distances
3. Apply `max(0, 1 - mean_dist / threshold)` to get the frame's contribution

This differs from applying the kernel per-keypoint then averaging, which would be more lenient (a mix of inlier and outlier keypoints could still produce positive contributions).

## Common Pitfalls

### Pitfall 1: Confidence Mask Shape Mismatch
**What goes wrong:** `keypoint_conf` is (T, K) over the full tracklet, but we index by shared frames. Off-by-one or wrong indexing into the confidence array.
**How to avoid:** Use the same `idx_a`/`idx_b` frame-to-index mapping already used for centroids. Extract `conf_a = tracklet_a.keypoint_conf[idx_a]` which gives (N, K).

### Pitfall 2: Empty Valid Keypoints After Masking
**What goes wrong:** If the confidence floor is high and all keypoints are low-confidence on every frame, M=0. Passing empty arrays to `cast_ray` or `ray_ray_closest_point_batch`.
**How to avoid:** Check M > 0 before calling ray functions. Return (0.0, 0) if no valid keypoints exist.

### Pitfall 3: CUDA Tensor to NumPy
**What goes wrong:** `cast_ray` returns torch tensors potentially on CUDA. Calling `.numpy()` without `.cpu()` crashes.
**How to avoid:** Always use `.cpu().numpy()` as the existing code already does. This is documented in CLAUDE.md.

### Pitfall 4: Early Termination Frame Count
**What goes wrong:** Early termination logic uses `t_shared >= early_k` to decide whether to check. With frame skipping, `effective_t_shared` after early frames may be less than `early_k` even though we checked `early_k` frames.
**How to avoid:** Keep early termination based on raw frame count (not effective). The early_k frames are checked; if no score contributions at all (including due to skipped frames), bail out.

### Pitfall 5: Frozen Dataclass Field Addition
**What goes wrong:** Adding fields to `AssociationConfig` (frozen dataclass) requires updating `_filter_fields` validation. Existing YAML configs without the new fields must still load.
**How to avoid:** New fields have defaults, so existing configs load fine. `_filter_fields` already handles this correctly — it only rejects *unknown* fields, not missing ones.

## Code Examples

### Keypoint Extraction and Confidence Masking
```python
# Extract keypoints and confidences for shared frames
kpts_a = tracklet_a.keypoints[idx_a]  # (N, K, 2)
kpts_b = tracklet_b.keypoints[idx_b]  # (N, K, 2)
conf_a = tracklet_a.keypoint_conf[idx_a]  # (N, K)
conf_b = tracklet_b.keypoint_conf[idx_b]  # (N, K)

# Intersection mask: both must be confident
floor = config.keypoint_confidence_floor
valid = (conf_a >= floor) & (conf_b >= floor)  # (N, K) bool

# Count valid keypoints per frame
n_valid = valid.sum(axis=1)  # (N,)
```

### Reshape for Batched Ray Casting
```python
# Flatten valid keypoints for batched cast_ray
# Use valid mask to select only confident keypoint pairs
pixels_a = kpts_a[valid]  # (M, 2) where M = valid.sum()
pixels_b = kpts_b[valid]  # (M, 2)

# Single cast_ray call per camera
pix_a_t = torch.tensor(pixels_a, dtype=torch.float32)
origins_a, dirs_a = lut_a.cast_ray(pix_a_t)
# ... same for camera B
```

### Per-Frame Mean Distance Aggregation
```python
# After getting dists (M,), scatter back to per-frame means
# Use n_valid to split: frame i has n_valid[i] distances
frame_offsets = np.cumsum(n_valid[:-1])  # split points
frame_groups = np.split(dists, frame_offsets)
mean_dists = np.array([g.mean() for g in frame_groups if len(g) > 0])
```

### Optimized Per-Frame Mean via np.add.reduceat
```python
# More efficient than np.split for large arrays
# Only for frames with n_valid > 0
active = n_valid > 0  # (N,) bool
if not active.any():
    return 0.0, 0

cumsum = np.cumsum(n_valid[active])
offsets = np.concatenate([[0], cumsum[:-1]])
sums = np.add.reduceat(dists, offsets)
mean_dists_active = sums / n_valid[active]
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ray-ray distance | Custom loop | `ray_ray_closest_point_batch` | Already vectorized, tested, handles edge cases (parallel rays) |
| Batched ray casting | Per-pixel loop | `ForwardLUT.cast_ray` | Bilinear interpolation over precomputed grid, handles batches |
| Config validation | Manual field checking | `_filter_fields` + frozen dataclass | Existing pattern catches unknown fields with hints |

## Open Questions

1. **Frame-skip effect on early termination semantics**
   - What we know: Early termination checks `score_sum == 0` after `early_k` frames. With frame skipping, some of those frames may have been skipped.
   - What's unclear: Should we still terminate if score_sum == 0 but all early frames were skipped (no keypoints at all)?
   - Recommendation: Yes, still terminate. If early_k frames produce no usable keypoints, the pair is unscorable. This is consistent with the "no centroid fallback" decision.

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `src/aquapose/core/association/scoring.py` — current implementation
- Codebase inspection: `src/aquapose/engine/config.py` — AssociationConfig dataclass
- Codebase inspection: `src/aquapose/core/tracking/types.py` — Tracklet2D with keypoints/keypoint_conf
- Codebase inspection: `src/aquapose/calibration/luts.py` — ForwardLUT.cast_ray API
- Codebase inspection: `tests/unit/core/association/test_scoring.py` — existing test patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all changes are within existing codebase patterns
- Architecture: HIGH — straightforward extension of existing vectorized scoring
- Pitfalls: HIGH — identified from direct code inspection

**Research date:** 2026-03-11
**Valid until:** 2026-04-11 (stable internal codebase)
