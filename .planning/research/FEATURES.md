# Feature Landscape

**Domain:** Multi-keypoint cross-view association scoring, changepoint-based ID swap detection, and singleton recovery for multi-view fish pose estimation (AquaPose v3.8)
**Researched:** 2026-03-11
**Confidence:** MEDIUM — scoring aggregation choices and changepoint algorithm selection are grounded in domain literature and Python library documentation; specific empirical tradeoffs for very short tracking sequences (~200 frames, ~9 fish) are not directly studied in literature, so recommendations are derived from first principles and codebase constraints.

---

## Context: What Already Exists (v3.7 Baseline)

| Existing Component | What It Does | Relevance to v3.8 |
|-------------------|--------------|-------------------|
| `scoring.py` / `score_all_pairs()` | Vectorized single-centroid ray-ray scoring with soft linear kernel, early termination, overlap reliability weighting | Will be replaced — multi-keypoint scoring replaces single-centroid |
| `score_tracklet_pair()` | Per-pair frame loop using centroid (kpt index 2, spine1 only), casts 1 ray/frame from each camera | Will be replaced — needs K rays/frame per keypoint |
| `ray_ray_closest_point_batch()` | Vectorized NumPy closest-point distance for N ray pairs | Reused — multi-keypoint expands N from T to K*T per pair |
| `clustering.py` | Leiden graph clustering with must-not-link constraints | Unchanged |
| `refinement.py` | Post-clustering re-triangulation and centroid outlier eviction | Removed — replaced by group validation with changepoint detection |
| Fragment merging | Same-camera gap-bridging (working against upstream tracker design) | Removed — see design doc |
| `AssociationConfig` | Frozen dataclass: ray_distance_threshold, score_min, t_min, t_saturate, early_k, min_shared_voxels | Extended — new params for keypoint confidence threshold, min segment length, changepoint significance |
| `TrackletGroup` | frozen dataclass: fish_id, tracklets, confidence, per_frame_confidence, consensus_centroids | Downstream contract unchanged — reconstruction consumes this |
| Forward LUT / `cast_ray()` | pixel-exact (grid_step=1) refractive ray casting | Reused — no LUT changes needed |
| `Tracklet2D` | Has centroids[], frames[], camera_id, track_id, but keypoints not yet in scoring path | Extended — multi-keypoint scoring requires accessing per-frame keypoint arrays from Tracklet2D |

---

## Table Stakes

Features required to declare v3.8 successful. Missing these means the milestone goal (reduce 27% singleton rate) cannot be achieved or validated.

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|-------------|
| **Multi-keypoint pairwise scoring** | Single-centroid scoring is the root-cause bottleneck. With 6 anatomical keypoints available at mAP50-95=0.974, discarding 5/6 is a fundamental signal waste. Multi-keypoint scoring is the stated goal of this milestone. | MEDIUM | Requires Tracklet2D to carry per-frame keypoint arrays and confidence scores into the association stage. Currently centroids[] is the only positional array stored. |
| **Confidence-filtered keypoint selection** | Partially occluded detections produce unreliable keypoints. Without confidence filtering, including low-confidence keypoints (e.g., tail obscured by another fish) adds noise that degrades scoring accuracy rather than improving it. This is standard practice in multi-view pose literature. | LOW | Needs per-frame per-keypoint confidence values in Tracklet2D. Already produced by YOLO-pose but currently only used in the 2D tracker. |
| **Aggregation of per-keypoint ray distances into a single score** | The scoring pipeline produces one affinity score per pair per frame (used as a graph edge weight for Leiden). Multi-keypoint scoring produces K distances per frame rather than 1. These must be collapsed to a scalar. The aggregation method determines robustness. | LOW | Depends on multi-keypoint scoring being implemented first. |
| **Group validation with outlier eviction** | Replaces refinement.py. For each group, compute per-frame multi-keypoint residuals against the group consensus and evict tracklets that consistently disagree. Without this, bad matches that survived scoring (due to borderline edge weights) corrupt reconstruction. | MEDIUM | Depends on multi-keypoint scoring (re-uses same ray casting logic). Requires per-frame residual computation against a group rather than a pair. |
| **Temporal changepoint detection within group validation** | Enables splitting swapped tracklets rather than evicting them whole. A tracklet that is correct for 80 frames and swapped for 20 frames should contribute its 80 good frames — eviction discards all 100. Changepoint detection is the correct tool for this: identify the transition in residual mean and split at that boundary. | MEDIUM | Depends on group validation producing a per-frame residual series. Standard algorithm (see Differentiators for algorithm choice). |
| **Singleton recovery — simple assignment** | Re-evaluate each singleton against all formed groups and assign if a strong match is found. Many singletons are stable long-lived tracks that scored just below threshold due to marginal adjacency graph coverage. Single assignment is the dominant recovery case. | LOW | Depends on groups being finalized after group validation. Uses same multi-keypoint ray-residual logic. |
| **Updated AssociationConfig with new parameters** | New config fields: keypoint_conf_threshold (which keypoints qualify per frame), changepoint_significance_threshold, min_segment_frames (guards against splitting very short segments), singleton_assignment_threshold (stricter than group validation eviction threshold). Without config exposure, parameters cannot be tuned. | LOW | Frozen dataclass extension. Must not break existing YAML configs (add fields with defaults). |
| **Singleton recovery — split-and-assign for ID swaps** | Case A singletons (failed to associate because the correct half and swapped half are roughly equal length) require sweep-based split analysis. Without this, case A swaps remain as singletons even after groups are formed. Controlled by min_segment_frames to disable when segment evidence is too thin. | MEDIUM | Depends on simple singleton assignment being implemented first. Shares per-frame residual infrastructure. |

---

## Differentiators

Features that improve quality or robustness beyond the minimum needed to reduce singleton rate. These are worth building in v3.8 if the table stakes are delivered cleanly.

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|-------------|
| **Trimmed mean aggregation for keypoint distances** | A single occluded or detector-error keypoint can produce a large outlier ray distance on a given frame. Arithmetic mean propagates this outlier to the score. Trimmed mean (drop top 1-2 distances of K=6) is more robust without the complexity of a full RANSAC loop. Literature on multi-view pose confidence aggregation (e.g., SelfPose3d CVPR 2024, confidence-weighted DLT) consistently finds that robustifying the aggregator improves outlier tolerance. In the AquaPose context, 6 keypoints is a small enough K that dropping the top 1 and computing mean of the remaining 5 is the natural robust choice. | LOW | Drop-in replacement for mean aggregation within multi-keypoint scoring. Adds one line of numpy sorting. |
| **Confidence-weighted mean (instead of or in addition to trimming)** | Weight each keypoint's ray distance by its detector confidence score before averaging. High-confidence keypoints get more influence than marginally-visible ones. More principled than binary thresholding (confidence >= threshold: include; else: exclude) because it preserves gradient information. Literature supports confidence-weighted triangulation: the existing DLT backend already uses confidence weights, and maDLC (DeepLabCut) uses confidence-weighted part affinity field scoring. | LOW | Requires per-frame per-keypoint confidence array in Tracklet2D (same requirement as binary thresholding). Can be combined with binary threshold (confidence-weighted among those above threshold). |
| **PELT (ruptures library) for changepoint detection instead of custom binary split** | The design doc proposes a custom "find split maximizing mean residual difference" — effectively binary segmentation with one split. This works for the simple case. The `ruptures` Python library (mature, PyPI, O(n log n) binary segmentation and O(Kn) PELT) provides tested implementations with multiple cost functions. For 100-200 frame sequences, both are fast enough that choosing PELT over custom code is free robustness and allows multi-swap detection (recurse on halves) without writing the recursion. Using `ruptures` with `model="l2"` (mean shift in L2 norm) and `n_bkps=None` (let penalty determine count) matches the stated goal of detecting mean-level shifts in the residual series. | LOW | Add `ruptures` to project dependencies (pure Python, no C extension issues). Swap custom split logic in group validation for `ruptures.Pelt(model="l2").fit_predict(signal, pen=penalty)`. |
| **Minimum-keypoint-count guard per frame** | If fewer than M keypoints pass the confidence threshold on a given frame (e.g., only 1 of 6 passes), that frame contributes a very uncertain ray distance. Excluding frames where fewer than, say, 2 keypoints qualify prevents low-observation frames from adding noise. This is distinct from the existing t_min (minimum shared frames between pair) — it is a per-frame quality gate. | LOW | Single config parameter: min_kpts_per_frame. Filters frames from scoring within a pair's overlap. |
| **Body-keypoint subsets: head+spine only scoring mode** | Keypoints 0-2 (nose, head, spine1) are the most anatomically stable and least likely to be occluded. Tail keypoints (spine4, spine5) are frequently partially out of frame on edge-of-tank fish. An optional scoring mode that uses only the anterior 3 keypoints trades coverage for signal quality. May perform better than all-6 for fish near the tank wall. | LOW | Config flag: keypoint_subset (list of indices). Defaults to all 6. |
| **Updated association evaluator: per-group camera count and residual reporting** | With multi-keypoint scoring, the group validation residuals are a richer diagnostic. Adding per-group mean multi-keypoint residual and camera count distribution to AssociationMetrics gives the tuning harness more signal to differentiate good parameter sets. | LOW | Extension to existing `AssociationStageEvaluator`. Uses output already computed during group validation. |

---

## Anti-Features

Features to explicitly NOT build in v3.8.

| Anti-Feature | Why It Might Be Requested | Why Avoid | What to Do Instead |
|--------------|--------------------------|-----------|-------------------|
| **Per-frame 3D position consensus during scoring** | "Compute 3D position estimates during scoring to get richer affinity" | Re-introduces per-frame 3D reasoning into scoring — the architectural pattern that failed in v1.0. The design doc explicitly rejected this: tracklet-level temporal evidence is structurally superior to per-frame 3D association. Triangulating during scoring would also require storing per-frame midpoints (memory) and produce noisy Z estimates (132x Z/XY anisotropy). | Use ray-ray distance (pure geometry) as the scoring primitive. 3D consensus is computed downstream in reconstruction. |
| **Learned/neural affinity scoring** | "Train a small network to predict whether two tracklets match, using keypoint motion features" | Requires labeled multi-view association ground truth (which does not exist), training infrastructure, and inference overhead. The geometric signal (ray-ray distance) is physically principled and sufficient for rigid-body fish. Learned scoring is appropriate when geometric features fail (appearance-identical objects, non-rigid deformation) — neither applies here. | Use geometric ray-ray scoring. Tune thresholds via the existing `aquapose tune` infrastructure. |
| **Fragment merging** | "Reconnect same-camera tracklets separated by brief occlusion gaps" | Explicitly removed per the design doc. Works against the upstream tracker's design intent (fragmentation over merging). The downstream contract only requires group membership, not continuous per-camera tracklets. | Leave gap-bridging to a later 3D trajectory post-processing stage if needed. |
| **Appearance-based re-identification (ReID)** | "Use fish appearance features (color, texture) to resolve ambiguous close-proximity swaps" | Fish in this dataset are visually similar (same species, controlled lighting). Appearance features would require training a ReID model on fish-specific data that does not exist. The inherent ambiguity when fish swim in close parallel is acknowledged in the design doc as unresolvable without appearance — but building ReID infrastructure for a marginal improvement case is out of scope. | Accept that sustained close-parallel swimming produces unresolvable ambiguity. Flag these frames in diagnostics. |
| **Bidirectional changepoint scoring** | "Score the split in both temporal directions to handle swaps at chunk boundaries" | Chunk boundary handling already exists via ChunkHandoff. Bidirectional changepoint analysis was tried in v3.7 tracking refinement (bidirectional merge) and produced no benefit (44 vs 42 tracks). Adds complexity without clear gain for ~200-frame chunks. | Use forward-only changepoint detection with sufficient min_segment_frames to avoid spurious splits near boundaries. |
| **Exhaustive all-pairs keypoint matching** | "Try all K! keypoint orderings to find the best geometric alignment, in case keypoint identity is confused" | The 2D tracker already maintains keypoint identity from frame to frame (24-dim KF tracks each keypoint independently). Cross-view keypoint identity is by index (same anatomical landmark). Permutation search is O(K!) = 720 for K=6, applied to every pair-frame — completely unnecessary and expensive. | Match keypoints 1:1 by index. Confidence filtering handles the case where one camera's detection is missing a keypoint. |

---

## Feature Dependencies

```
[Tracklet2D keypoint arrays in association path]
    (prerequisite: currently centroids[] only; keypoints[] and confidences[] must reach AssociationStage)
    └──required by──> [Multi-keypoint pairwise scoring]
                          └──required by──> [Confidence-filtered keypoint selection]
                          └──required by──> [Aggregation method choice (mean/trimmed/weighted)]
                          └──required by──> [Group validation with residuals]
                                                └──required by──> [Changepoint detection]
                                                └──required by──> [Outlier eviction]
                          └──required by──> [Singleton recovery — assignment]
                                                └──required by──> [Singleton recovery — split-and-assign]

[Updated AssociationConfig]
    └──required by──> [All new parameter controls]

[ruptures library] (optional dependency for changepoint)
    └──enhances──> [Changepoint detection]
    (can be replaced by custom binary split if dependency undesirable)

[Group validation complete]
    └──required by──> [Singleton recovery — assignment]
                          └──required by──> [Singleton recovery — split-and-assign]

[Fragment merging] ──REMOVED──> not a dependency of anything in v3.8
[refinement.py] ──REPLACED BY──> [Group validation]
```

### Dependency Notes

- **Tracklet2D keypoint propagation is the critical prerequisite.** The `TrackingStage` builds `Tracklet2D` objects, and currently stores `centroids[]` per frame. The `AssociationStage` receives the full `tracks_2d: dict[str, list[Tracklet2D]]` from `PipelineContext`. Multi-keypoint scoring requires that `Tracklet2D` also carries `keypoints: list[np.ndarray]` (shape `(6, 2)` per frame) and `keypoint_confidences: list[np.ndarray]` (shape `(6,)` per frame). This is the only cross-stage data contract change in v3.8.

- **Aggregation method is a tuning parameter, not a hard dependency.** Mean is the safe default. Trimmed mean and confidence-weighted mean can be added as config options once the basic pipeline works. Do not over-engineer the aggregator before seeing whether plain mean already solves the discrimination problem.

- **Changepoint detection is independent of singleton recovery.** Group validation (step 4) and singleton recovery (step 5) share the same per-frame residual computation infrastructure, but each can be implemented and tested independently. Implement group validation + changepoint first; add singleton recovery after.

- **min_segment_frames simultaneously controls:** (a) minimum segment length for changepoint splitting in group validation, (b) minimum half-length for singleton split-and-assign. Setting it high (e.g., 999999) disables split-and-assign without a separate config toggle, per the design doc.

- **Refinement removal simplifies the stage graph.** `refinement.py` currently runs after Leiden clustering. Removing it and replacing with group validation does not change the Leiden step or the TrackletGroup output format — it is a direct substitution in the association stage's internal pipeline.

---

## MVP Definition

### Launch With (v3.8 core)

Minimum viable to achieve the milestone goal (reduce singleton rate, enable downstream reconstruction improvement):

- [ ] Tracklet2D carries per-frame keypoints[] and keypoint_confidences[] to AssociationStage — essential prerequisite
- [ ] Multi-keypoint pairwise scoring with binary confidence threshold filtering and arithmetic mean aggregation — core scoring upgrade
- [ ] Group validation: per-frame multi-keypoint residuals against group consensus, outlier eviction — replaces refinement.py
- [ ] Temporal changepoint detection in group validation using binary split (custom or ruptures) — swap recovery in groups
- [ ] Singleton recovery: simple assignment (sweep singletons against groups) — primary recovery mechanism
- [ ] Fragment merging removed, refinement.py removed — simplification
- [ ] Updated AssociationConfig with new parameters and defaults — enables tuning pass
- [ ] Parameter tuning pass against v3.7 baseline using existing `aquapose tune` infrastructure — validation that singleton rate improved

### Add After Core Validation (v3.8.x)

Features to add once basic multi-keypoint scoring is confirmed to reduce singleton rate:

- [ ] Trimmed mean aggregation (drop top 1 of 6 distances) — add if residual outliers visible in diagnostics
- [ ] Confidence-weighted mean — add if binary threshold creates too many frame dropouts
- [ ] Singleton recovery split-and-assign — add if case A swaps are measurable in eval output
- [ ] Minimum-keypoint-count guard per frame — add if low-kpt frames degrade scoring in practice

### Future Consideration (v3.9+)

- [ ] Appearance-based ReID for close-proximity sustained swimming — requires fish-specific ReID training data
- [ ] 3D trajectory gap-bridging post-reconstruction — belongs in a post-processing stage, not association

---

## Scoring Aggregation Methods: Detailed Analysis

This section addresses the downstream consumer's specific question: which aggregation method for multi-keypoint ray distances.

### Option A: Arithmetic mean of all qualifying keypoints
**When to use:** Default starting point. Simple to implement and debug.
**Confidence:** HIGH — this is the natural baseline.
**Weakness:** A single outlier keypoint (e.g., tail partially occluded) inflates the mean for the whole frame.

### Option B: Trimmed mean (drop top 1 distance, mean of remaining 5)
**When to use:** After confirming that outlier keypoints occur in practice (visible in per-frame residual diagnostics).
**Confidence:** MEDIUM — standard robust statistics practice; the specific benefit for K=6 is not empirically verified for this dataset.
**Advantage:** O(K log K) sort is negligible. Handles the most common single-keypoint failure mode.
**Implementation:** `np.sort(dists)[:-1].mean()` for K=6.

### Option C: Confidence-weighted mean
**When to use:** When binary thresholding creates too many per-frame dropouts (e.g., fish near edge-of-tank frequently have all keypoints below threshold).
**Confidence:** MEDIUM — supported by analogous use in DLT triangulation (already in codebase) and multi-view pose literature (SelfPose3d 2024, maDLC 2022).
**Advantage:** Preserves gradient information vs. hard threshold. Naturally handles partial visibility.
**Implementation:** `np.average(dists, weights=confs)` where confs are YOLO confidence scores for each keypoint.

### Option D: Median
**When to use:** When more than half the keypoints are expected to be noise (not the case here — fish body is well-modeled by 6 anatomical points with ordered correspondence).
**Confidence:** LOW for this specific use case — median is appropriate when outlier fraction exceeds 50%. For K=6 with at most 1-2 unreliable keypoints, median is overly conservative and discards too much signal.
**Recommendation:** Do not use as primary aggregator. Could be used for group-level residual aggregation over many frames where outlier frames are common.

### Recommended order of implementation:
1. Start with arithmetic mean + binary confidence threshold (Option A).
2. Add trimmed mean as config option if diagnostics show single-keypoint outliers (Option B).
3. Consider confidence-weighted mean only if binary threshold creates systematic frame dropouts (Option C).
4. Never use median as primary frame-level aggregator for K=6 (Option D).

---

## Changepoint Detection: Detailed Analysis

This section addresses the downstream consumer's specific question: which changepoint algorithm for ~200-frame sequences.

### Problem characteristics
- Sequence length: 50-200 frames per chunk (ChunkOrchestrator default 200)
- Signal: per-frame mean multi-keypoint residual (scalar), computed against group consensus
- Expected pattern: step function (low residual for consistent segment, high residual for swapped segment)
- Expected number of changepoints: 0 or 1 per tracklet (single swap); recursion handles rare double swaps
- Required speed: per-tracklet, called for each tracklet in each group — ~9 groups x ~5 tracklets = ~45 calls per chunk

### Option A: Custom binary split (design doc proposal)
**Algorithm:** Sweep all candidate split points; find split maximizing difference in mean residual between left and right halves; apply threshold on difference magnitude and minimum segment length.
**Complexity:** O(n) for a single pass; O(n) total.
**Confidence:** HIGH — analytically correct for detecting a single mean shift; the design doc's stated approach.
**Weakness:** Only detects one changepoint per call. Multiple swaps require caller-implemented recursion. No statistical penalty framework (must hand-tune significance threshold).
**Implementation:** ~20 lines of NumPy. No new dependency.

### Option B: ruptures PELT with l2 cost
**Algorithm:** Pruned Exact Linear Time dynamic program. Detects optimal K changepoints under a penalized likelihood criterion.
**Complexity:** O(Kn) average, where K is changepoints detected (typically 0-2 here).
**Confidence:** MEDIUM — ruptures is a well-maintained library (2018, actively updated per PyPI 2025), l2 cost is the correct model for mean shift detection in a noisy residual series.
**Advantage:** Handles multiple changepoints natively; built-in penalty framework (BIC-like pen parameter) avoids manual significance threshold tuning; tested implementation.
**Implementation:** `ruptures.Pelt(model="l2").fit_predict(signal.reshape(-1,1), pen=penalty)`.
**Dependency cost:** `pip install ruptures` — pure Python, no C extensions, no GPU dependencies.

### Option C: ruptures Binary Segmentation with l2 cost
**Algorithm:** Greedy top-down binary search.
**Complexity:** O(n log n) average.
**Confidence:** MEDIUM — same caveats as PELT. Faster per call than PELT but not materially different for n=200.
**Advantage:** Slightly simpler parameter interpretation (n_bkps vs pen).
**Note:** On short sequences (n=200) the difference between PELT and Binary Segmentation is negligible. PELT is preferred because the penalty-based interface is more natural for "detect only real changepoints" than the fixed-count interface.

### Option D: CUSUM
**Algorithm:** Online cumulative sum control chart.
**Confidence:** LOW for this application — CUSUM is designed for online (streaming) detection of a single sustained shift. Here, the goal is offline batch analysis to find where a swap occurred within an already-complete tracklet. CUSUM's sequential design adds complexity without benefit for the batch use case.
**Recommendation:** Do not use.

### Recommendation: Start with custom binary split (Option A)
The custom binary split is adequate for the single-swap case, has no new dependency, and is fully transparent. Add `ruptures` PELT (Option B) as an upgrade path if multiple-swap tracklets are observed in practice or if the manual significance threshold proves difficult to tune. The design doc's approach is correct; ruptures is a quality-of-life improvement, not a correctness requirement.

---

## Singleton Recovery Strategies: Detailed Analysis

### Context in multi-view tracking literature

Track re-identification in multi-view systems (Ma et al., Information Fusion 2024; LRFS filtering Mahler 2024) generally approaches singleton recovery as a global re-initialization problem: compare lost tracks against all current group hypotheses using appearance or motion features. This is over-engineered for AquaPose's case, where:
- The number of fish is known (9)
- The number of groups is small (~9)
- Keypoint geometry provides a direct physical matching signal
- The session is a fixed-length video clip (not an infinite stream)

### AquaPose-specific recovery cases

**Case B (easy):** Singleton mostly matches one group but was marginally below score_min or outside adjacency graph. Mean multi-keypoint residual against that group is consistently low. Simple assignment handles this. This case is expected to dominate.

**Case A (hard):** Singleton matches no group well because it is a 50/50 swap. Per-frame residual series shows two distinct levels corresponding to two different groups. Sweep-based split-and-assign handles this. This case is rarer.

**No match:** Fish is genuinely single-camera (tank geometry blind spot), or the detection is a false positive. Remains singleton. This is the correct outcome.

### Implementation order

1. Implement simple assignment first: for each singleton, compute mean multi-keypoint residual against each group over shared frames; assign if below singleton_assignment_threshold.
2. Validate that singleton rate drops before implementing split-and-assign.
3. Add split-and-assign only if measurable case A swaps remain after simple assignment.

### Key parameter: singleton_assignment_threshold

Should be stricter than the group validation eviction threshold. The rationale: eviction in group validation has the prior that the tracklet is probably a group member (it survived scoring); the bar for rejecting it should be lower than the bar for admitting a cold-start singleton. The eviction threshold governs "should we remove this?"; the assignment threshold governs "should we add this?" — these should not be the same value.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Tracklet2D keypoint propagation | HIGH | LOW | P1 |
| Multi-keypoint pairwise scoring (mean aggregation) | HIGH | MEDIUM | P1 |
| Confidence threshold filtering | HIGH | LOW | P1 |
| Group validation + outlier eviction | HIGH | MEDIUM | P1 |
| Changepoint detection (custom binary split) | HIGH | LOW | P1 |
| Singleton recovery — simple assignment | HIGH | LOW | P1 |
| Fragment merging removal | MEDIUM (cleanup) | LOW | P1 |
| Refinement.py removal | MEDIUM (cleanup) | LOW | P1 |
| Updated AssociationConfig | HIGH | LOW | P1 |
| Parameter tuning pass | HIGH | LOW | P1 |
| Trimmed mean aggregation | MEDIUM | LOW | P2 |
| Confidence-weighted mean | MEDIUM | LOW | P2 |
| Singleton recovery — split-and-assign | MEDIUM | MEDIUM | P2 |
| Min-keypoint-count guard | LOW | LOW | P2 |
| ruptures PELT changepoint upgrade | LOW | LOW | P3 |
| Body-keypoint anterior subset mode | LOW | LOW | P3 |
| Extended association evaluator metrics | MEDIUM | LOW | P2 |

**Priority key:**
- P1: Must have for launch — directly enables singleton rate reduction
- P2: Should have — improves robustness or adds tuning flexibility; add after P1 validated
- P3: Nice to have — quality-of-life or deferred edge cases

---

## Sources

- `.planning/inbox/association_multikey_rework.md` — primary design document (HIGH confidence)
- `.planning/PROJECT.md` — existing capabilities and v3.8 milestone definition (HIGH confidence)
- Direct codebase inspection: `src/aquapose/core/association/scoring.py`, `types.py` — confirmed current scoring uses centroids[] only, single ray per frame (HIGH confidence)
- [ruptures documentation](https://centre-borelli.github.io/ruptures-docs/) — PELT and Binary Segmentation algorithm properties (HIGH confidence for algorithm behavior; MEDIUM for short-sequence performance)
- [Ma et al. 2024, Information Fusion](https://arxiv.org/abs/2405.18606) — track re-identification in 3D multi-view tracking (MEDIUM confidence — abstract only; specific recovery methods not detailed)
- [Multi-animal DeepLabCut, Nature Methods 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9007739/) — keypoint affinity scoring, tracklet graph matching with multiple cost functions including Hausdorff keypoint distance (MEDIUM confidence)
- [SelfPose3d CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Srivastav_SelfPose3d_Self-Supervised_Multi-Person_Multi-View_3d_Pose_Estimation_CVPR_2024_paper.pdf) — confidence-weighted pseudo-label aggregation in multi-view pose estimation (MEDIUM confidence)
- [vmTracking PLOS Biology 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11845028/) — multi-animal tracking; uses virtual markers rather than geometric scoring, confirming that geometric scoring is an AquaPose-specific design choice (MEDIUM confidence)
- [Fish Tracking Challenge 2024](https://arxiv.org/html/2409.00339v1) — evaluation methodology for fish multi-object tracking (MEDIUM confidence)
- Standard robust statistics: trimmed mean, winsorized mean (HIGH confidence — well-established)
- Existing AquaPose DLT backend (confidence-weighted triangulation) as evidence that confidence-weighted aggregation is already the project standard (HIGH confidence — codebase)

---

*Feature research for: AquaPose v3.8 Improved Association*
*Researched: 2026-03-11*
