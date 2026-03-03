# Cross-Camera Association: Tuning Opportunities & Complexity Audit

> Deep-dive analysis of the SPECSEED association pipeline, focused on why
> reconstruction receives so few multi-camera fish and what can be done about it.
> Includes an evaluation approach that requires no manual annotation.

---

## Context

Evaluation of 15 frames from the YH dataset shows 30-37 fish entries per frame
in the MidlineSet, but only 6-7 distinct fish have multi-camera coverage (≥2
cameras). The remaining 24-30 entries are single-camera singletons that cannot
be triangulated. The reconstruction rate is 12-15% of fish-frames. The 2D
midlines look good in visualizations, so the bottleneck is association, not
detection or midline extraction.

---

## 1. Algorithm Summary

SPECSEED runs in 5 steps:

1. **Camera overlap graph** — filter camera pairs by shared voxel count
2. **Pairwise scoring** — ray-ray distance + ghost penalty + overlap weighting
3. **Leiden clustering** — graph partition into fish identities
4. **Fragment merging** — stitch same-camera OC-SORT fragments within clusters
5. **3D refinement** — evict tracklets whose rays don't converge to consensus

---

## 2. Tunable Parameters

All exposed via `AssociationConfig` in `engine/config.py`.

### Scoring (Step 1)

| Parameter | Default | Effect | Tuning Direction |
|-----------|---------|--------|-----------------|
| `ray_distance_threshold` | 0.03 m | Max ray-ray distance to count as inlier | **Loosen (increase)** if fish are farther from camera or calibration has systematic bias. Current value assumes ~2cm fish width; real ray error may be higher due to refraction model error. |
| `ghost_pixel_threshold` | 50.0 px | Max pixel distance for a detection to "support" a ghost point | Decrease to make ghost penalty stricter; increase to be more lenient. |
| `t_min` | 3 | Min shared frames to even attempt scoring a pair | Lowering to 2 scores more pairs but risks noisy matches. |
| `t_saturate` | 100 | Frame count where overlap reliability weight saturates | Unlikely to matter much; 100 is reasonable. |
| `early_k` | 10 | Frames checked before early-termination on zero inliers | **Increase** if fish move slowly and rays converge gradually. |

### Clustering (Steps 2-3)

| Parameter | Default | Effect | Tuning Direction |
|-----------|---------|--------|-----------------|
| `score_min` | 0.3 | Minimum affinity to create a graph edge | **Lower** to admit weaker but real associations. This is the primary gatekeeper: pairs scoring below 0.3 are invisible to Leiden. |
| `leiden_resolution` | 1.0 | Leiden partition resolution | Lower = fewer, larger clusters (merge more). Higher = more, smaller clusters (split more). |

### Fragment Merging (Step 4)

| Parameter | Default | Effect | Tuning Direction |
|-----------|---------|--------|-----------------|
| `max_merge_gap` | 30 | Max frame gap for stitching fragments | Increase if OC-SORT produces longer gaps. |

### Refinement (Step 5)

| Parameter | Default | Effect | Tuning Direction |
|-----------|---------|--------|-----------------|
| `eviction_reproj_threshold` | 0.025 m | Max median ray-to-consensus distance before eviction | **Loosen (increase)** if calibration has systematic error. Too tight = evicts valid tracklets. |
| `min_cameras_refine` | 3 | Skip refinement for clusters with fewer cameras | Lower to 2 to refine more clusters. |

---

## 3. Likely Causes of Low Multi-Camera Yield

### 3a. `score_min` and `ray_distance_threshold` are the primary gatekeepers

The combined score formula is:

```
score = inlier_fraction × (1 - mean_ghost_ratio) × overlap_weight
```

where `inlier_fraction = (frames with ray_distance < threshold) / shared_frames`.

If the ray-distance threshold (0.03 m) is too tight, inlier_fraction drops,
the score drops below score_min (0.3), and the edge never enters the graph.
Leiden can only cluster what it can see — missing edges produce singletons.

**This is the most likely root cause.** Refractive calibration error, LUT
interpolation error, and centroid extraction noise all contribute to ray-ray
distances. If the combined error budget exceeds 3cm, legitimate matches score
as non-inliers.

### 3b. Ghost penalty may suppress valid matches

The ghost penalty checks whether *other* cameras that see a triangulated voxel
also have nearby detections. If a fish is only detected in 3 of 8 cameras that
can see the voxel, the ghost ratio is 5/6 ≈ 0.83, and the score is multiplied
by 0.17. This penalizes fish that are genuinely visible to few cameras (e.g.,
fish near tank edges or partially occluded).

### 3c. Refinement eviction may be too aggressive

The 2.5cm eviction threshold in Step 5 removes tracklets from clusters
post-clustering. If systematic calibration bias exceeds 2.5cm, valid tracklets
get evicted to permanent singletons (they are never reassigned).

### 3d. OC-SORT fragmentation + merge gap limit

If OC-SORT produces many short fragments with gaps >30 frames, fragment merging
can't stitch them, and each fragment is a separate node in the clustering graph.
Short fragments have few shared frames with other cameras, so they score below
`t_min` and never enter the graph.

---

## 4. Potential Over-Engineering / Bandaids

### 4a. Must-not-link post-processing (clustering.py:195-241)

Leiden doesn't support hard constraints natively. The workaround detects
violations after clustering and resolves them by evicting the weaker tracklet
to a singleton with confidence=0.0. This runs up to 3 iterations.

**Assessment:** Necessary given the algorithm choice, but the eviction-only
strategy (no reassignment) means mistakes compound. If the evicted tracklet
actually belonged to the cluster, it's permanently lost. The 3-iteration cap
is arbitrary.

### 4b. Ghost penalty (scoring.py:203-232)

The ghost penalty is conceptually sound (negative evidence from cameras that
*should* see the fish but don't detect it), but the implementation has
structural issues:

- It uses Stage 1 detections (bounding box centroids), not Stage 2 tracklet
  positions. A fish that was detected but not tracked still counts as
  "supporting."
- The `ghost_pixel_threshold` (50px) is generous — at typical camera
  resolution, 50px could match a different fish entirely.
- The penalty is multiplicative with no floor, so even a moderate ghost ratio
  (0.5) halves the score. This interacts badly with `score_min`: a pair with
  perfect ray convergence (inlier_fraction=1.0) but ghost_ratio=0.6 scores
  only 0.4, barely above the 0.3 cutoff.

**Assessment:** The ghost penalty is doing double duty as both a false-positive
filter and an implicit camera-count requirement. It may be over-penalizing fish
that are genuinely visible to only a few cameras.

### 4c. Robust consensus (refinement.py:270) — "keep best 50%"

The refinement step triangulates a consensus 3D point by keeping the best 50%
of pairwise ray-ray distances. This is a form of robust estimation, but the
50% threshold is hardcoded with no justification.

**Assessment:** Minor. The 50% threshold is reasonable as a robust median
estimator. Not likely causing issues.

### 4d. Fragment merging with linear interpolation (clustering.py:464-482)

Gap frames between merged fragments are filled with linearly interpolated
centroids and bboxes. This assumes constant-velocity motion during the gap.

**Assessment:** Acceptable for short gaps (≤30 frames at 30fps = 1 second).
The interpolated frames are tagged with `frame_status="interpolated"` so
downstream can distinguish them. Not a bandaid — this is a reasonable design.

---

## 5. Suggested Evaluation Approach (No Manual Annotation)

The key insight: we don't need ground-truth associations to measure whether
parameter changes improve reconstruction yield and quality. The evaluation
harness already provides the metrics we need.

### Metric 1: Reconstruction Coverage Rate

**What:** `reconstructed_fish_frames / total_input_fish_frames`

Already computed by the evaluation harness. Currently 12-15%. Higher is better
(assuming quality doesn't degrade).

### Metric 2: Multi-Camera Yield

**What:** For each frame, count fish with ≥2, ≥3, ≥4 cameras in the
MidlineSet after association. No annotation needed — just count the association
output structure.

**Script:** Iterate `tracklet_groups`, count cameras per group per frame.
Report distribution histograms.

### Metric 3: Reprojection Error (Self-Consistency)

**What:** Triangulate each multi-camera fish, reproject to all cameras, measure
pixel error. Already computed as Tier 1 in the evaluation harness.

**Interpretation:** Lower mean reprojection error after loosening thresholds
means the newly-admitted associations are geometrically valid. If error
increases dramatically, the loosened thresholds are admitting false matches.

### Metric 4: Leave-One-Out Stability (Already Tier 2)

**What:** Drop one camera, re-triangulate, measure 3D displacement. Already
computed. Stable (low displacement) means the fish identity is well-constrained
by multiple cameras. Unstable (high displacement) means the association depends
on a single critical camera.

### Metric 5: Singleton Rate

**What:** `singleton_groups / total_groups` from association output. No
annotation needed. Lower singleton rate = better association coverage.

### Metric 6: Cluster Consistency Score

**What:** For each cluster, compute the mean pairwise affinity score of its
edges. Already stored as `TrackletGroup.confidence`. Compare distributions
across parameter settings.

### Suggested Evaluation Protocol

1. **Baseline:** Run current parameters, record all 6 metrics.
2. **Sweep `ray_distance_threshold`:** Try 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10 m.
   For each, record metrics 1-6. Plot reconstruction rate vs. reprojection
   error to find the Pareto frontier.
3. **Sweep `score_min`:** Try 0.15, 0.20, 0.25, 0.30, 0.35. Same metric
   collection.
4. **Sweep `eviction_reproj_threshold`:** Try 0.02, 0.03, 0.04, 0.05, 0.08 m.
5. **Combined:** Take best individual settings, run together, verify no
   negative interaction.
6. **Ghost penalty ablation:** Run with ghost penalty disabled (set
   `ghost_pixel_threshold` to a very large value like 10000) to measure its
   net effect on yield vs. false positive rate.

All of this uses only the existing evaluation harness and association output
structure — no manual annotation required. The self-consistency metrics
(reprojection error, leave-one-out stability) serve as proxy ground truth.

---

## 6. Recommended Priority Order

1. **Loosen `ray_distance_threshold`** (0.03 → 0.05-0.06 m) — most likely to
   recover missing associations
2. **Lower `score_min`** (0.3 → 0.2) — admits weaker but potentially valid edges
3. **Ablate ghost penalty** — determine if it's helping or hurting net yield
4. **Loosen `eviction_reproj_threshold`** (0.025 → 0.04 m) — recover tracklets
   evicted by refinement
5. **Increase `early_k`** (10 → 20) — give slow-converging pairs more chances
6. **Lower `leiden_resolution`** (1.0 → 0.8) — bias toward merging

Items 1-3 address the most likely bottlenecks identified in Section 3. Items
4-6 are secondary and may not be needed if 1-3 suffice.

---

*Analysis based on code review of `src/aquapose/core/association/` (scoring.py,
clustering.py, refinement.py, stage.py, types.py) and `engine/config.py`.*
