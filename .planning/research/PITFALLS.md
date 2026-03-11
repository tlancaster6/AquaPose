# Pitfalls Research

**Domain:** Adding multi-keypoint association scoring, temporal changepoint detection, and singleton recovery to an existing multi-view fish tracking pipeline (v3.8 Improved Association milestone)
**Researched:** 2026-03-11
**Confidence:** HIGH — pitfalls derived from direct codebase inspection of scoring.py, refinement.py, keypoint_tracker.py, and tracking/types.py; v3.7 project memory (ForwardLUT coordinate-space mismatch post-mortem); design document in association_multikey_rework.md; and structural analysis of the vectorized broadcasting patterns already in production.

> **Scope note:** This file covers pitfalls specific to the v3.8 milestone: extending vectorized centroid scoring to multi-keypoint scoring, adding changepoint-based group validation, singleton recovery with swap splitting, and removing refinement. Prior milestone pitfalls (v3.6, v3.4) are preserved in the git history of PITFALLS.md and are not repeated here.

---

## Critical Pitfalls

### Pitfall P1: Tracklet2D Does Not Carry Keypoints — Silent Data Loss

**What goes wrong:**
The `KeypointTracker` builder accumulates per-frame keypoints (`shape (6, 2)`) in `_TrackletBuilder.keypoints`, but `_TrackletBuilder.to_tracklet2d()` constructs `Tracklet2D` with only `frames`, `centroids`, `bboxes`, and `frame_status`. Keypoints are dropped. `Tracklet2D` has no `keypoints` attribute.

Multi-keypoint scoring requires `K` keypoints per frame per tracklet. If the implementation reads `tracklet.keypoints` or tries to build a `(T, K, 2)` array from tracklets, it will get an `AttributeError` or silently use only centroids.

**Why it happens:**
`Tracklet2D` was designed when centroid-only scoring was the contract. The builder always had keypoints for internal OKS cost computation, but `to_tracklet2d()` was never updated to expose them. The design document acknowledges the multi-keypoint requirement but does not mention that `Tracklet2D` must be extended first.

**How to avoid:**
Extend `Tracklet2D` with a `keypoints` field (e.g., `tuple[tuple[tuple[float, float], ...], ...]` — one tuple of `(u, v)` per keypoint per frame) and a matching `keypoint_confidences` field (one float per keypoint per frame, from the detection confidence). Update `_TrackletBuilder.to_tracklet2d()` to populate both from `builder.keypoints` and the per-frame detection confidence arrays. This is a prerequisite change that must be done before any multi-keypoint scoring code runs. Also extend `ChunkHandoff` serialization if it round-trips `Tracklet2D` state across chunk boundaries.

**Warning signs:**
- `AttributeError: 'Tracklet2D' object has no attribute 'keypoints'` in association stage
- Tests pass with centroid-only scoring but integration tests fail when switching to multi-keypoint
- Multi-keypoint scoring silently falls back to single-centroid if it reads `centroids` as a proxy without asserting the keypoints field exists

**Phase to address:**
First implementation phase — extend `Tracklet2D` before writing any scoring code. This is a contract-breaking change that will require updating all test fixtures, golden data checks, and any code that constructs `Tracklet2D` directly.

---

### Pitfall P2: LUT Coordinate Space Mismatch — The Exact Prior Bug Returns

**What goes wrong:**
This bug already destroyed the v3.7 association quality (86% singleton rate before fix). `ForwardLUT.cast_ray()` expects pixel coordinates in the undistorted space (K_new, fx~1364). If keypoint pixel coordinates from `Tracklet2D` are in distorted space (K, fx~1587), the ~14% focal length mismatch inflates ray-ray distances by a factor that prevents most associations, producing near-100% singletons.

With multi-keypoint scoring, the risk multiplies: there are now 6 pixel coordinates per frame instead of 1. Any one of those keypoints being in the wrong coordinate space will corrupt the per-keypoint ray and contaminate the aggregate score. Keypoints from coasted frames (interpolated by the KF) are in KF state space, which is undistorted. Keypoints from directly-observed frames come from `detection.keypoints`, which must be verified to be in undistorted space.

**Why it happens:**
The original bug was that `generate_forward_luts` was called without `undistortion_maps`, so it used the raw distorted K matrix. The fix passed `undistortion_maps`. But when keypoints are added to `Tracklet2D`, any new code path that obtains pixel positions from `Detection.keypoints` and stores them in the tracklet must ensure those positions are already in undistorted space. If `DetectionStage` or `PoseEstimationStage` returns keypoints in raw camera/distorted pixel coordinates, and the undistortion only happens inside `ForwardLUT.cast_ray()`, then there is no mismatch. But if the detection pipeline applies distortion correction internally and keypoints are already corrected to a different K, casting them through the LUT again will double-correct.

**How to avoid:**
- Before implementing multi-keypoint scoring, document the coordinate space of `Detection.keypoints` at the output of the `PoseEstimationStage`. Trace the YOLO inference output through to `Detection.keypoints` and confirm whether these are raw (distorted) or undistorted pixel coordinates.
- Write a unit test that: takes a known 3D keypoint, projects it to 2D via the calibration forward model, stores it as a fake `Tracklet2D` keypoint, calls `ForwardLUT.cast_ray()`, and verifies the back-cast ray passes within 2mm of the original 3D point. This tests the full keypoint→ray round-trip for multi-keypoint scoring.
- Add a bounds check: keypoints from a 1600x1200 camera with valid undistorted K should fall within [0, 1600] x [0, 1200]. Keypoints outside this range (or outside the OBB bbox) at rate > 1% indicate coordinate space confusion.

**Warning signs:**
- Singleton rate after v3.8 implementation is > 40% on the benchmark clip (expect target ~15% vs v3.7 ~27%)
- Per-keypoint ray-ray distances are uniformly 14% larger than expected (same pattern as the original LUT bug)
- Association quality degrades on keypoints further from centroid (head, tail) while spine keypoints perform better — inconsistent spatial scaling is a sign of coordinate mismatch

**Phase to address:**
Multi-keypoint scoring implementation phase — write and run the round-trip test before any end-to-end evaluation.

---

### Pitfall P3: Broadcasting Shape Mismatch When Extending to K Keypoints

**What goes wrong:**
The current `_batch_score_frames` vectorizes across `N` shared frames: `pix_a` is `(N, 2)`, `origins_a` is `(N, 3)`, `dists` is `(N,)`. Extending to `K` keypoints changes the shape to `(N, K, 2)` pixels, `(N, K, 3)` origins and directions, and `(N, K)` distances. If the implementation naively reshapes to `(N*K, 2)` before `cast_ray`, the LUT call works, but masking by keypoint confidence (not all K keypoints are valid on every frame) must happen before the reshape. Doing it after would require inverse-mapping masked indices back to `(frame, keypoint)` pairs — complex and error-prone.

The confidence filtering also introduces a ragged structure: some frames contribute 2 keypoints, some contribute 6. A dense `(N, K, 2)` representation requires filling invalid slots with sentinel values and masking after the fact. `ray_ray_closest_point_batch` is correct for any input shape but will compute ray-ray distances for sentinel pixels and produce valid-looking but meaningless distance values unless the mask is applied before aggregation.

**Why it happens:**
NumPy broadcasting makes it easy to write code that runs without error but operates on the wrong elements. Confidence masks that are computed correctly but applied one step too late (after ray casting instead of before) will pass all shape checks and produce numerical outputs — just wrong ones. This is especially dangerous because the failure mode (slightly inflated ray distances for low-confidence keypoints) will look like a modest degradation rather than a crash.

**How to avoid:**
- Decide on representation before writing code: either flatten to `(N*K_valid, 2)` with a frame+keypoint index array tracking which `(frame, kpt)` each ray belongs to, or keep the dense `(N, K, 2)` representation with NaN-filled invalid slots and mask after distance computation. The flattened approach is simpler to reason about but requires building the index arrays.
- Write a reference implementation using per-frame Python loops first, verify it produces the correct output on a synthetic example with known ground truth, then vectorize and compare. The loop version is the correctness oracle; the vectorized version is the performance optimization.
- Unit-test the shape transitions explicitly: for `N=5` frames with `K=6` keypoints and `confidence=[1,1,0,1,0,0]` on one frame, verify that the per-frame score correctly uses only the 4 valid keypoints on that frame, not 6.

**Warning signs:**
- Score values increase uniformly when K is increased from 1 to 6 — if low-confidence keypoints are not filtered, every extra keypoint adds noise that should not contribute
- Association quality improves with K=3 but degrades with K=6 — more keypoints adding noise, indicating confidence filtering is not working
- Unit tests on synthetic data pass but real-data association quality is worse than single-centroid baseline

**Phase to address:**
Multi-keypoint scoring implementation phase — write unit tests against synthetic ground truth before touching real data.

---

### Pitfall P4: Aggregation Function Choices Across K Keypoints Are Load-Bearing

**What goes wrong:**
The design document specifies "aggregate per-keypoint ray-ray distances into a richer affinity score" but does not specify the aggregation function. The current single-centroid scoring uses a soft linear kernel `1 - dist / threshold` summed across frames and normalized. With K keypoints, a natural extension sums the soft kernel across all `(frame, keypoint)` pairs and normalizes by the total number of valid pairs. But this treats all keypoints equally, and the design document notes that "mid-body keypoints (head, spine1, spine2) are expected to carry most of the signal; nose and tail are noisy."

If equal-weight aggregation is used, a fish pair where tail keypoints give high distance but spine keypoints give low distance will score poorly even though the spine evidence is strong. Per-keypoint weighting requires either learned weights (too complex) or anatomically motivated priors (tail has higher variance → lower weight). Using the wrong aggregation will not crash — it will silently produce weaker discrimination than single-centroid, defeating the purpose of the entire milestone.

Separately, the reliability-weighting factor `w = min(t_shared, t_saturate) / t_saturate` is computed once per pair at the frame level. With multi-keypoint scoring, each frame now contributes up to K evidence points. The overlap window `t_shared` counts frames, not frame×keypoint pairs. Reusing the same `w` is correct if aggregation is per-frame (average K keypoints per frame, then sum frames). If aggregation is across all `(frame, keypoint)` pairs without per-frame averaging, the effective `t_shared` should be weighted by average keypoints-per-frame, not raw frame count. Using raw frame count with flat aggregation will overweight frames with high keypoint coverage relative to frames where only 2 keypoints were visible.

**How to avoid:**
- Start with mean-of-valid-keypoints per frame, then sum frames and normalize by `t_shared`. This is the direct extension of the current per-frame soft kernel and preserves the `w` weighting semantics exactly. Every frame contributes exactly 1 aggregated score regardless of how many keypoints were valid on that frame.
- After the mean-per-frame baseline is working and evaluated against v3.7, consider per-keypoint weighting as a tuning step. Do not implement weighting before establishing the baseline multi-keypoint result.
- Keep the old single-centroid scoring path runnable via config flag during the transition, so A/B comparison is possible without re-running the full pipeline from scratch.

**Warning signs:**
- Multi-keypoint scoring gives worse association quality than single-centroid on the benchmark clip — indicates aggregation is compounding noise rather than averaging it out
- Score distribution shifts right (higher scores for all pairs) with K=6 vs K=1 — indicates valid and invalid keypoints are being counted equally
- Head-end pairs score differently from tail-end pairs for the same ground-truth fish pair — indicates tail keypoint noise is dominating

**Phase to address:**
Multi-keypoint scoring implementation phase — establish aggregation approach before implementation and document the choice.

---

### Pitfall P5: Changepoint Detection False Positives on Short Tracklets

**What goes wrong:**
The design document's changepoint detection "finds the split point that maximizes the difference in mean residual between the two halves, subject to a minimum segment length." On a 200-frame chunk, a typical tracklet spans 50-150 frames. The residual series for a correctly associated tracklet has non-zero variance from keypoint jitter, partial occlusion, and fish proximity events. A "best split" search over a 50-frame residual series will always find some split that looks like a local mean difference — even for random noise, the expected maximum mean difference is O(σ / sqrt(min_segment)). With min_segment=5 frames and σ≈0.5 threshold units, the expected false-positive maximum is ~0.22 threshold units. If the significance threshold is set too permissively, correctly associated tracklets will be split in half.

The consequences are worse than missed swaps: a false split evicts a valid tracklet segment to singleton status. In singleton recovery, this orphaned segment must compete against all groups. If it matches back to its original group (which it should, since it was correct), the round-trip adds no value and the group is identical to before splitting. But if singleton recovery is less sensitive than group validation (as stated in the design document: "stricter threshold"), the orphaned segment may fail to rejoin and become a permanent singleton, reducing the group's temporal coverage.

**Why it happens:**
The CUSUM-style "maximum mean difference split" is a naive changepoint test. It has no null distribution calibration — it always returns a "best" split point regardless of whether any split is statistically significant. The significance threshold (the minimum acceptable mean difference between halves) must be calibrated against the expected residual noise on a correctly-associated tracklet, which depends on the number of keypoints used, the fish density, and the distance threshold. Using a fixed threshold without calibration guarantees either too many false splits (permissive) or too few true detections (conservative).

**How to avoid:**
- Compute an expected noise baseline: for 100 confirmed-correct tracklets from the v3.7 benchmark run, compute the per-frame residual series and measure the typical within-tracklet residual variance. Use 2×σ_noise as the minimum mean-difference threshold for changepoint detection.
- Enforce a minimum segment length of at least 10 frames (not the design document's unspecified "minimum segment length"). With 30fps video and typical fish swap events lasting ~1-3 seconds, a 10-frame minimum filters noise while still catching swaps.
- After implementing, measure the false positive rate explicitly: count how many segments produced by changepoint splitting successfully rejoin their original group in singleton recovery. A rate > 30% indicates too many false splits.
- Consider using a permutation test on the residual series: shuffle the per-frame residuals, compute the best split for 1000 shuffles, and only accept a split if the observed max-mean-difference exceeds the 95th percentile of the shuffled distribution. This is more expensive but provides proper statistical calibration.

**Warning signs:**
- Group count increases dramatically after group validation (e.g., from 9 groups to 25 post-validation) — indicates mass false splitting
- Short tracklets (< 20 frames) are split more often than long tracklets — short series are more susceptible to noise-driven apparent changepoints
- After full pipeline, more singletons exist post-v3.8 than post-v3.7 — net effect is negative despite the intended improvement

**Phase to address:**
Group validation implementation phase — calibrate significance threshold against confirmed-correct tracklets before deploying.

---

### Pitfall P6: Singleton Recovery Assigns Singletons to Wrong Group Under Fish Proximity

**What goes wrong:**
The design document acknowledges: "Fish swimming in close parallel for an extended period produce ambiguous residuals against both groups throughout." This is not the only ambiguous case. A singleton tracklet from a camera with narrow field of view angle to two fish may produce similar ray-ray distances to both fish groups because the projection geometry is poor (near-parallel rays from that camera to both fish). The singleton recovery assigns the singleton to the group with the lowest median residual — which may be determined by 2-3 frames where the fish briefly separated, not by the full-tracklet evidence.

If singleton recovery assigns a tracklet to the wrong group, reconstruction for that group in those frames gets a cross-keypoint signal from a different fish. The error propagates through DLT triangulation as outlier measurements, reducing reconstruction confidence and potentially shifting the estimated midline toward the wrong fish. This is worse than leaving the tracklet as a singleton (no contribution to reconstruction) because a wrong assignment actively corrupts reconstruction.

**Why it happens:**
The "strong overall match to one group" criterion requires computing per-frame residuals against all groups and identifying a clear winner. The threshold for "strong match" is set per the design document without specifying calibration. If the threshold is too loose, a singleton with moderately low residuals to two different groups (due to geometric ambiguity) gets assigned to whichever group happens to have slightly lower average residual — a statistically meaningless distinction.

**How to avoid:**
- Add a gap criterion: only assign a singleton to a group if the best-group residual is significantly lower than the second-best-group residual (e.g., the margin exceeds 30% of the threshold). If two groups are within 30% of each other, leave the singleton unassigned.
- Measure the rate of "competitive assignments" (singleton has residuals within 30% to two groups) on the benchmark clip. If this rate is > 10%, the geometry of this rig makes singleton recovery ambiguous for many tracklets and a more conservative threshold is warranted.
- After implementing singleton recovery, verify reconstruction quality does not degrade: compare per-frame reprojection error for groups with recovered singletons vs. groups without. If groups with recovered singletons have worse reprojection, singleton recovery is adding more noise than signal and the threshold should be tightened.

**Warning signs:**
- Reprojection error increases in groups that gained singletons vs. groups that did not
- The same camera-tracklet is recovered into different groups on different benchmark runs (instability under noise)
- Reconstruction shows brief "jumps" in the 3D midline at frame boundaries that correspond to frames covered by a recovered singleton

**Phase to address:**
Singleton recovery implementation phase — verify reconstruction quality before and after recovery via the existing `aquapose eval` infrastructure.

---

### Pitfall P7: Removing Refinement Without an Equivalent Gate Breaks Downstream Confidence

**What goes wrong:**
The current `refine_clusters()` in `refinement.py` produces three outputs consumed downstream: (1) evicted tracklets are made singletons (group membership change), (2) `per_frame_confidence` is populated for each group, and (3) `consensus_centroids` are set. The reconstruction stage and evaluation stage may read `per_frame_confidence` and `consensus_centroids` from `TrackletGroup`. If refinement is removed but no equivalent computation fills these fields, downstream code sees `None` where it previously saw valid data.

The design document says refinement is "replaced by group validation in step 4," but group validation produces evicted singletons, not per-frame confidence or consensus centroids. If reconstruction reads `consensus_centroids` to weight DLT triangulation inputs (or uses `per_frame_confidence` to filter frames), setting these to `None` in all groups will silently degrade reconstruction.

**Why it happens:**
The design document focuses on the algorithmic replacement (group validation subsumes refinement's eviction function) but does not address the data contract. `TrackletGroup` is a frozen dataclass with optional `per_frame_confidence` and `consensus_centroids` fields that were populated by refinement. Removing the computation without tracing all consumers of those fields creates a silent API break.

**How to avoid:**
- Before removing `refinement.py`, grep the codebase for all reads of `TrackletGroup.per_frame_confidence` and `TrackletGroup.consensus_centroids`. Determine whether any downstream code has conditional logic on these fields being non-None.
- If reconstruction or evaluation uses these fields, group validation must populate equivalent values. The design document's "fresh ray casting for each group" in step 4 could produce `per_frame_confidence` as a side effect without a separate pass.
- Consider deprecating but not removing refinement first (set `refinement_enabled=False` as the default config) to confirm the downstream impact before deleting the code.

**Warning signs:**
- Reconstruction quality degrades after removing refinement, even when group membership is equivalent
- `aquapose eval` association evaluator produces different metrics (not just singleton rate) compared to v3.7 baseline — indicates downstream consumers of refinement outputs changed behavior
- `TrackletGroup.consensus_centroids` is None for all groups and code that previously branched on this shows unexpected None-path behavior

**Phase to address:**
Refinement removal phase — audit all consumers of `TrackletGroup.per_frame_confidence` and `consensus_centroids` before deletion.

---

### Pitfall P8: Must-Not-Link Constraints Break When Tracklets Are Split

**What goes wrong:**
Must-not-link constraints prevent same-camera tracklets with overlapping frames from sharing a Leiden cluster. The constraints are stored as a set of `(key_a, key_b)` pairs where each key is `(camera_id, track_id)`. If group validation splits a tracklet into two segments (each becoming a new tracklet with its own ID), the must-not-link constraints from the original split tracklet do not automatically propagate to both fragments.

Two scenarios: (a) The split produces two fragments from the same camera. They now share a camera and may have overlapping-frame windows from the original tracklet's neighbors. They should inherit the same must-not-link constraints as the original. (b) The split tracklet's key `(cam, track_id)` was in a must-not-link pair with another tracklet. Both new fragments should inherit this constraint. If neither case is handled, the Leiden graph used in singleton recovery will be missing edges that should block incorrect cluster merges.

**Why it happens:**
Must-not-link constraints are computed once in `score_all_pairs` (or its equivalent) based on the original tracklet set. The design document says "constraints and clustering are unchanged" but that refers to the initial clustering pass. Group validation happens after Leiden clustering and creates new tracklet objects (fragments). These new objects were not part of the original constraint computation and have no entries in the scored pairs dictionary.

However, the design document says singleton recovery does not re-run Leiden — it assigns singletons to existing groups directly via residual comparison. In that case, must-not-link constraints are not explicitly enforced during singleton recovery. This means a fragment from camera C could be assigned to a group that already has a tracklet from camera C for the same frames — violating the fundamental constraint.

**How to avoid:**
- Track the origin of each fragment: when splitting tracklet `(cam, old_id)` into segments, the new fragments should carry `origin_camera = cam` and `origin_frames` so that the singleton recovery step can check the same-camera overlap constraint directly before assigning.
- In singleton recovery, before assigning a singleton to a group, verify that no tracklet in that group has the same `camera_id` and overlapping `frames`. This check is O(cameras × frames) and cheap.
- Unit-test this: construct a scenario where a split fragment from camera C is offered to a group that already has a different tracklet from camera C with overlapping frames. Verify it is not assigned.

**Warning signs:**
- Post-recovery groups have two tracklets from the same camera with overlapping frames
- `TrackletGroup.tracklets` contains pairs `(cam_id, frame_set)` that overlap — detectable with a group validity check
- 3D reconstruction errors spike on frames where two same-camera tracklets are active in the same group

**Phase to address:**
Singleton recovery implementation phase — add a group validity assertion as a post-recovery invariant check.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip extending `Tracklet2D` with keypoints; compute them freshly from detection cache in association stage | Avoids modifying the core type contract | Association stage becomes coupled to detection cache format; breaks the clean stage-output-as-input contract; fails when cache is unavailable | Never — extend `Tracklet2D` |
| Use the same threshold for changepoint detection in group validation and singleton split detection | Single parameter to tune | Group validation operates on tracklets with strong group prior (lower FP cost); singleton detection has weak signal; same threshold leads to either over-splitting in validation or under-splitting in singleton recovery | Only as a starting point; split into two params after calibration |
| Disable must-not-link constraints because multi-keypoint scoring makes them "unnecessary" | Simpler graph; faster Leiden | Constraints serve as a cheap safety net for near-parallel camera geometries where scoring is ambiguous; removing them risks rare but catastrophic multi-fish merges | Only after empirically confirming zero constraint violations on a full benchmark run |
| Keep `refinement.py` as dead code with `refinement_enabled=False` | Easy rollback if removal causes issues | Dead code accrues maintenance burden; future changes to `TrackletGroup` must keep `refinement.py` consistent; creates confusion about which code path is active | Acceptable during the same milestone; must be deleted before v3.8 ships |
| Tune changepoint threshold on the same clip used for development | Fast feedback | Overfitting to specific fish behavior on one clip; threshold that works on development clip may not generalize to held-out clips or different tank conditions | Never for final threshold; use at least two independent clips |

---

## Integration Gotchas

Common mistakes when connecting the new components to the existing pipeline.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `Tracklet2D` keypoint extension | Add `keypoints` field without adding `keypoint_confidences`; confidence filtering requires both position and confidence | Add both fields together; confidence is used to determine which keypoints to cast rays from |
| `ForwardLUT.cast_ray` with keypoints | Pass `(N*K, 2)` flat tensor without tracking which entries are masked as invalid | Build an index array of valid `(frame, kpt)` pairs first; only pass valid pixels to `cast_ray`; use index array to scatter results back |
| Group validation residual computation | Re-use scoring-phase ray data (stored somewhere) instead of re-computing fresh rays | Scoring only covers adjacent camera pairs; transitive group members may not have scored against each other; group validation requires fresh rays for all cameras in the group |
| Must-not-link inheritance after splitting | Generate new tracklet IDs for fragments without checking constraint inheritance | When splitting, check whether either fragment's `(camera_id, frames)` conflicts with any other tracklet in any group; enforce the constraint inline during singleton recovery |
| Refinement removal and `per_frame_confidence` | Delete `refine_clusters` call; groups have `per_frame_confidence=None`; reconstruction silently uses all frames equally | Check all reads of `per_frame_confidence` in reconstruction and evaluation; either populate it from group validation or update consumers to handle `None` correctly |
| Singleton recovery threshold vs. group validation threshold | Use same `eviction_reproj_threshold` for both | Group validation evicts from an established group (strong prior); singleton recovery assigns to a group (no prior); recovery threshold should be stricter; use separate config parameters |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-frame `cast_ray` call in group validation | Group validation becomes 4x slower than scoring (scoring already vectorizes across frames; group validation naively loops per-frame) | Vectorize across the frame dimension in group validation the same way `_batch_score_frames` does; all frames for one tracklet-group pair in one `cast_ray` call | Chunks > 100 frames with 9 groups × 5 cameras |
| Dense `(N, K, 6, 6)` pairwise distance matrix | For K=6 keypoints, building a pairwise distance matrix across all keypoints of both tracklets is O(K²) per frame; if also across all T frames, it is O(T×K²) | Use `ray_ray_closest_point_batch` which is already vectorized across frame pairs; extend it to take `(T, K, 3)` inputs and return `(T, K)` distances; mean-reduce over K | Chunks > 200 frames; K > 6 |
| Repeated LUT lookups for the same pixel in different pipeline steps | Scoring, group validation, and singleton recovery all call `cast_ray` for the same tracklet pixels | Cache `(origins, dirs)` per tracklet per step; within one pipeline run, re-casting the same centroid/keypoints three times wastes GPU time | Any chunk with > 50 tracklets |
| Changepoint search O(T²) over full tracklet | Naive "find split maximizing mean difference" is O(T) per candidate split × T candidates = O(T²) | Use the O(T) CUSUM prefix-sum trick: precompute cumulative sum and count; compute all split mean-differences in O(T) | Tracklets > 100 frames |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **`Tracklet2D` extension:** Keypoints field added — verify `keypoint_confidences` is also added and both are populated for coasted frames (interpolated KF state, not raw detection confidence, for coasted-frame keypoints).
- [ ] **Multi-keypoint scoring:** Scores computed and passing tests — verify the aggregation correctly excludes low-confidence keypoints (not just the ones from coasted frames); cross-check with known-correct pair from the benchmark clip.
- [ ] **Group validation:** Changepoint splits being produced — verify false positive rate by counting how many split fragments successfully rejoin their original group in singleton recovery; rate > 30% means threshold is too permissive.
- [ ] **Singleton recovery:** Singletons being assigned to groups — verify no group ends up with two same-camera tracklets covering overlapping frames; add a post-recovery group validity assertion.
- [ ] **Refinement removal:** `refine_clusters` removed — verify `TrackletGroup.per_frame_confidence` and `consensus_centroids` are either populated by the new pipeline or all consumers of these fields handle `None` correctly.
- [ ] **Must-not-link propagation:** Constraints still applied — verify that fragments created by splitting inherit the same-camera overlap constraint and that singleton recovery enforces it inline.
- [ ] **Benchmark regression:** Singleton rate reported — verify against v3.7 baseline (27%) with the same clip; multi-keypoint scoring should move toward the ~15% target, not above 27%.
- [ ] **Evaluation metrics unchanged:** `aquapose eval` runs clean — verify the association evaluator and reconstruction evaluator produce valid output with the new `TrackletGroup` structure (especially if `per_frame_confidence` is now differently populated).

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| `Tracklet2D` missing keypoints field discovered mid-implementation | MEDIUM | Extend `Tracklet2D`; update `_TrackletBuilder.to_tracklet2d()`; re-run all unit tests; re-generate any golden data fixtures that include `Tracklet2D` serialization |
| LUT coordinate space mismatch (second occurrence) | HIGH | Run the `cast_ray` round-trip unit test to confirm mismatch; identify where the coordinate space diverges (detection output vs. LUT expectation); regenerate LUT cache after verifying correct K matrix is used |
| Changepoint detection producing mass false splits | LOW | Increase the significance threshold by 2x; re-run group validation; measure how split count changes; if still too many, add a minimum-tracklet-length filter (only attempt changepoint detection on tracklets > 30 frames) |
| Singleton recovery corrupting groups (reprojection error increases) | LOW | Disable singleton recovery (`min_segment_length=999999`); report that singleton rate did not improve; re-enable only after re-calibrating the assignment margin gap criterion |
| Refinement removal breaks downstream metrics | MEDIUM | Re-enable `refinement_enabled=True` temporarily; compare which downstream code paths depend on `per_frame_confidence` and `consensus_centroids`; populate those fields from group validation before re-removing |
| Must-not-link violation in post-recovery groups | LOW | Add inline constraint check in singleton recovery assignment; evict the conflicting tracklet back to singleton; re-run eval to confirm group validity |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1: `Tracklet2D` missing keypoints | `Tracklet2D` extension phase (prerequisite) | Unit test: construct `Tracklet2D` with keypoints, verify field is accessible and correct dtype |
| P2: LUT coordinate space mismatch | Multi-keypoint scoring phase | Round-trip test: 3D keypoint → project → `cast_ray` → verify ray passes within 2mm of source |
| P3: Broadcasting shape mismatch | Multi-keypoint scoring phase | Unit test with `N=5, K=6, confidence=[1,1,0,1,0,0]` on one frame; verify invalid keypoints are excluded from score |
| P4: Wrong aggregation function | Multi-keypoint scoring design | A/B comparison: single-centroid vs. mean-per-frame multi-keypoint on benchmark clip; multi-keypoint should be >= single-centroid |
| P5: Changepoint false positives | Group validation phase | Measure false-positive rate: count split fragments that rejoin original group in singleton recovery; target < 30% |
| P6: Singleton to wrong group | Singleton recovery phase | Compare per-group reconstruction error before and after recovery; groups with recovered singletons should not be worse |
| P7: Refinement removal breaks downstream | Refinement removal phase | Grep all reads of `per_frame_confidence` and `consensus_centroids`; run full `aquapose eval` before and after removal |
| P8: Must-not-link constraint violation | Singleton recovery phase | Post-recovery group validity assertion: no group has two same-camera tracklets with overlapping frames |

---

## Sources

- AquaPose codebase direct inspection: `core/association/scoring.py` (vectorized `_batch_score_frames`, `ray_ray_closest_point_batch`), `core/association/refinement.py` (downstream `TrackletGroup` field population), `core/tracking/types.py` (current `Tracklet2D` fields — no `keypoints`), `core/tracking/keypoint_tracker.py` (`_TrackletBuilder.to_tracklet2d()` drops keypoints, `builder.keypoints` field exists), `core/tracking/stage.py` (confirms tracking stage does not add keypoints to `Tracklet2D`)
- AquaPose project memory: ForwardLUT coordinate space mismatch post-mortem (2026-03-04) — direct precedent for P2
- `association_multikey_rework.md` design document — primary source for intended design; P1, P3, P4, P5, P6, P7, P8 derived from analyzing gaps and edge cases in the design
- CUSUM changepoint detection literature context: standard O(T) prefix-sum implementation; significance threshold calibration requirements for non-stationary tracking residuals

---
*Pitfalls research for: Multi-keypoint association scoring, changepoint detection, and singleton recovery (v3.8 Improved Association)*
*Researched: 2026-03-11*
