# Cross-Camera Fish Tracklet Association

## Problem Statement

Given N per-camera 2D tracklets (each a time-series of bounding-box centroids), determine which tracklets across cameras correspond to the same physical fish, producing up to 9 groups with global IDs.

### Core Challenges

**Projection ambiguity.** A 2D point in one camera maps to a ray in 3D space (refracted at the air–water interface), not to a point in another camera's image plane. Cross-camera comparison must go through 3D.

**Combinatorial cost.** ~108 tracklets (12 cameras × ~9 fish). Naive all-pairs comparison across all shared frames is wasteful — most camera pairs have no overlapping field of view, and most cross-camera tracklet pairs are obviously different fish.

**Partial observability.** A fish visible in one region of the tank may appear in only 4–5 of 12 cameras. Groups are incomplete subsets, not full 12-way matches. Two tracklets from non-overlapping cameras may still belong to the same fish, linked transitively through a third camera.

**Fragmentation.** Brief occlusions or detection dropouts can split a single fish's trajectory into multiple tracklets within one camera. A camera may contribute 0, 1, or >1 tracklets per fish.

---

## Prerequisites

Camera intrinsics, extrinsics, and water surface position have been calculated via full refraction-aware calibration optimization. Each camera has a known projection model that accounts for the air→water optical path via Snell's law at the interface.

---

## Precomputed Lookup Tables

Two complementary LUTs eliminate repeated refraction math at query time. Both are computed once during setup using the calibrated refraction model.

### Forward LUT: Pixel → Ray (per camera)

For each camera, map a grid of 2D pixel coordinates to 3D ray (origin, direction) in water-space. At query time, bilinear interpolation over the grid recovers the ray for any detection centroid. This accelerates Step 1 (ray back-projection).

### Inverse LUT: 3D Voxel → Pixel (all cameras)

Discretize the tank volume into a regular voxel grid (default 2 cm resolution, configurable via `voxel_resolution_cm`). For each voxel center, project through the refraction model into every camera and record:

- Whether the voxel is within the camera's image bounds (visible).
- The 2D pixel coordinates (u, v) where the voxel projects.

This produces a structure: `voxel_map[x, y, z]` → list of `(camera_id, u, v)` for all cameras that see that voxel.

**Uses:**
- **Camera overlap graph (Step 0):** Two cameras are adjacent if they share a sufficient number of visible voxels — this falls out directly from the voxel map without separate frustum intersection tests.
- **Ghost point penalty (Step 1):** After triangulating a candidate 3D point, snap to the nearest voxel and instantly look up which other cameras should see it and where.
- **Reprojection in refinement (Step 5):** Project a triangulated 3D point into any camera via voxel lookup instead of running the refraction model again.

**Storage:** At 2 cm default resolution in a 1m × 1m × 0.5m tank, this is ~625K voxels × 12 cameras × (1 bool + 2 floats) ≈ 37 MB. The visibility boolean can be stored as a 12-bit mask per voxel for compact querying. Resolution is a configurable parameter (`voxel_resolution_cm`, default 2). Finer grids (1 cm, ~300 MB) are available but unlikely to help — the precision-limiting factor is always the triangulated 3D point feeding into the lookup, which has centimeter-scale uncertainty due to the top-down rig geometry.

Note that the voxel LUT is only used during association (this pipeline). Downstream 3D reconstruction operates on continuous coordinates using full-precision ray casting and triangulation, independent of the voxel grid. Coarse voxel resolution here does not limit the precision of the final 3D output.

---

## Per-Camera 2D Tracking

Each camera's detections are independently tracked over time to produce 2D tracklets using **OC-SORT** (Observation-Centric SORT). OC-SORT extends the standard SORT framework (Kalman filter + Hungarian matching on IoU) with three improvements relevant to our setting:

- **Observation-centric re-update:** When a track reappears after occlusion, the Kalman state is retroactively corrected using the new observation, reducing drift accumulated during coasting.
- **Observation-centric momentum:** Motion prediction uses the direction of actual observations rather than the drifted Kalman velocity, improving prediction accuracy after occlusions.
- **Lost track recovery:** Better matching of detections to recently-lost tracks, reducing fragmentation.

No appearance model or re-identification features are used. Within a single top-down camera view, fish trajectories are typically distinguishable by position and velocity alone. The harder cross-camera identity problem is handled by the association pipeline (Steps 0–5), not by the per-camera tracker.

### Tracklet Output Format

Each tracklet produced by OC-SORT must include the following fields, which are consumed by downstream pipeline steps:

- **`camera_id`**: Which camera this tracklet belongs to.
- **`track_id`**: Unique identifier within this camera (local, not global).
- **`frames`**: Ordered list of frame indices where this tracklet is active.
- **`centroids`**: Per-frame 2D bounding-box centroid coordinates (u, v) in pixel space.
- **`frame_status`**: Per-frame tag — `"detected"` (matched to a real detection) or `"coasted"` (Kalman prediction only, no matched detection). This distinction is required by Step 3.1 (must-not-link constraints) and Step 4 (fragment merging).
- **`bboxes`**: Per-frame bounding boxes (x, y, w, h). Used for IoU-based comparisons if needed.

The Kalman state (position, velocity estimates) is maintained internally by OC-SORT and is available for fragment stitching in Step 4, but is not required in the tracklet output format.

---

## Pipeline Overview

```
Per-Camera 2D Tracking              — OC-SORT: detections → tracklets per camera
Step 0: Camera Overlap Graph        — derived from voxel visibility masks
Step 1: Pairwise Tracklet Scoring   — ray–ray distance + ghost point penalty
Step 2: Tracklet Graph Construction — weighted graph over all tracklets
Step 3: Clustering                  — connected components + Leiden with must-not-link constraints
Step 4: Fragment Merging            — stitch same-camera, same-fish tracklet fragments
Step 5: 3D Consistency Refinement   — triangulate per-cluster, evict outlier tracklets
```

---

## Step 0: Camera Overlap Graph

Before any tracklet comparison, determine which camera pairs can potentially co-observe the same fish.

Using the inverse voxel map, count shared-visible voxels for each camera pair: a voxel is shared-visible if its camera visibility mask includes both cameras. If the count exceeds a threshold, mark the pair as adjacent.

The result is a **camera adjacency graph** — typically ~30–40 edges out of C(12,2) = 66 possible pairs, depending on camera placement and tank geometry. All later pairwise work is restricted to adjacent cameras. This computation is a byproduct of the voxel map and requires no additional ray-tracing.

---

## Step 1: Pairwise Tracklet Scoring

For each adjacent camera pair (i, j), and each tracklet pair (a from camera i, b from camera j) with overlapping time intervals:

### 1.1 Ray Back-Projection

For each shared frame t, back-project the 2D centroid from each camera into a 3D ray in water-space using the **forward pixel→ray LUT** (bilinear interpolation over the precomputed grid). No per-frame refraction math is needed.

### 1.2 Ray–Ray Distance

For each shared frame t, compute the closest point of approach between the two refracted rays. Since the medium below the water surface is homogeneous, refracted rays are straight lines — the existing `cast_ray()` returns a surface origin and a direction vector, and the ray is `origin + d * direction`. The closest-point-of-approach for two 3D lines has a standard analytic closed-form solution, yielding both the minimum distance d(t) and the 3D midpoint P(t) (reused in Step 1.3 for the ghost penalty). This is the two-ray special case of the least-squares formulation already implemented in `triangulate_rays()`.

This yields a per-frame distance d(t) and candidate 3D point P(t) for each tracklet pair.

### 1.3 Ghost Point Penalty (Negative Evidence)

A low ray-ray distance is necessary but not sufficient — two rays can pass close to each other by coincidence. The ghost point penalty adds a second signal: consistency with the rest of the camera network.

For each frame t where d(t) < τ:

1. **Triangulate** the candidate 3D point P (midpoint of closest approach between the two rays).
2. **Snap to nearest voxel** in the inverse LUT and retrieve the list of other cameras that see this voxel, along with expected pixel coordinates.
3. For each such camera c:
   - If camera c has a detection within a pixel-distance threshold of the expected (u, v) at frame t → **supporting evidence** (another camera agrees).
   - If camera c has detections at frame t but none near the expected (u, v) → **negative evidence** (camera sees the region but no fish is there). This is the ghost penalty.
   - If camera c has no detections at frame t in this region → **neutral** (camera might not cover this area, or fish might be occluded).

4. Compute a per-frame ghost ratio: g(t) = (number of cameras with negative evidence) / (number of cameras with visibility of the voxel, excluding the original pair).

A high ghost ratio across frames indicates a spurious match — the triangulated point is a "ghost" that other cameras contradict.

### 1.4 Aggregation and Scoring

Given T_shared overlapping frames, the distance series d(t), and the ghost ratio series g(t):

- **Inlier fraction:** f = (number of frames where d(t) < τ) / T_shared, where τ is calibrated to fish body size (e.g., 2 cm).
- **Median distance:** robust central tendency, less sensitive to detection jitter.
- **Mean ghost ratio:** ḡ = mean of g(t) over inlier frames. High values indicate a spurious match.
- **Combined score:** s(a, b) = f × (1 − ḡ), weighted by temporal overlap length. The ghost penalty multiplicatively suppresses pairs that look good geometrically from two cameras but are contradicted by the rest of the network.

### 1.5 Early Termination

- Skip pairs with temporal overlap < T_min (e.g., 10 frames).
- After evaluating the first K frames (e.g., 20), if the inlier fraction is already below 0.1, abandon the pair.

### 1.6 Cost Estimate

~40 adjacent camera pairs × ~81 tracklet pairs per camera pair (9 × 9) × T_shared frames per pair. With early termination, effective computation is substantially below the theoretical upper bound.

---

## Step 2: Tracklet Graph Construction

Build a weighted undirected graph:

- **Nodes:** All ~108 tracklets.
- **Edges:** Scored pairs from Step 1, thresholded at s(a, b) > s_min (e.g., 0.3).
- **Edge weights:** The score s(a, b).

Properties of this graph:

- Edges exist only between tracklets from adjacent cameras (by construction).
- A "true fish" corresponds to a dense subgraph (clique or near-clique).
- Tracklets from non-adjacent cameras can be linked transitively — camera A overlaps B, B overlaps C, so a fish visible in A and C is connected through B.

---

## Step 3: Clustering with Physical Constraints

### 3.1 Hard Constraints

**Must-not-link constraints** are maintained as a separate list of tracklet-pair tuples, independent of the affinity graph. This is algorithm-agnostic — constraints are enforced as a post-processing check after clustering, regardless of which clustering method is used. This decouples the constraint representation from the clustering algorithm choice, making it easy to swap methods without changing how constraints are stored or checked.

**Definition:** Two tracklets from the same camera are must-not-link if they overlap in **detection-backed frames** — frames where both tracklets have matched detections (not coasted predictions). Most Hungarian-based trackers maintain a tracklet for several frames after losing detections, coasting on predicted bounding boxes. This commonly produces brief temporal overlap between a dying tracklet's coasted tail and a new tracklet's first real detections, even when both correspond to the same fish. Overlap during coasted frames does not trigger a must-not-link constraint; instead, such pairs are treated as **fragment candidates** eligible for merging in Step 4.

This requires the tracker to tag each frame in a tracklet as "detected" (matched to a real detection) or "coasted" (predicted, no matched detection). This metadata should be preserved in the tracklet output format.

### 3.2 Clustering: Connected Components + Leiden

**Preprocessing:** Run connected-components on the affinity graph. Fish visible in completely non-overlapping camera subsets will form separate components, splitting them trivially before the heavier optimization. Each component is then clustered independently.

**Clustering:** Apply the Leiden algorithm (via `leidenalg` or `igraph`) to each connected component. Leiden optimizes modularity, automatically discovers the number of clusters (no need to specify k=9), and handles the sparse, roughly-equal-sized community structure of our graph well. Its resolution parameter can be tuned if clusters are over- or under-split.

**Constraint enforcement:** After Leiden produces clusters, check each cluster against the must-not-link constraint set (Step 3.1). If a cluster contains a forbidden pair, split it by removing the weakest internal edge connecting the conflicting tracklets and re-running Leiden on the affected subgraph. On a graph of ~108 nodes, this post-processing is negligible in cost.

---

## Step 4: Fragment Merging

Within each cluster, identify tracklets from the same camera:

- **Non-overlapping in time:** These are fragments of the same fish's trajectory. Merge into a single logical tracklet for that camera.
- **Overlap only in coasted frames:** One tracklet's coasted tail overlaps the other's first real detections (or vice versa). Trim the coasted frames from the dying tracklet, then merge. This is the most common fragmentation pattern from Hungarian-based trackers.
- **Overlap in detection-backed frames:** This should not occur within a cluster — it would have been prevented by the must-not-link constraint in Step 3.1. If it occurs due to a bug, flag for review.

---

## Step 5: 3D Consistency Refinement

For each cluster (candidate global fish ID):

1. At each frame where ≥2 member tracklets have detections, **triangulate** the 3D position using all available rays (least-squares closest point to the set of refracted rays).
2. **Reproject** the triangulated 3D point back into each camera and compute reprojection error against the observed 2D centroid.
3. If a tracklet consistently shows high reprojection error across frames, **evict** it from the cluster.
4. Evicted tracklets are left as **singletons** (unassociated, no global ID). A tracklet that failed geometric consistency with its highest-affinity cluster is unlikely to fit a different cluster better — reassignment would more often introduce wrong associations than recover correct ones. A high orphan rate (>10% of tracklets) is a diagnostic signal of upstream issues (detection quality, scoring thresholds, camera coverage) rather than something to patch with greedy reassignment.

This step catches errors that pairwise scores miss — a tracklet might score well individually with two cluster members but be geometrically inconsistent with the cluster as a whole.

**Per-frame confidence output:** After refinement, each global ID at each frame carries a confidence estimate derived from the reprojection residuals and the number of contributing cameras. Frames where two global IDs triangulate to nearby 3D positions (close encounters) should also be flagged as low-confidence — the identities are correct on average but may be locally ambiguous. Downstream consumers (3D reconstruction, behavior analysis) can use these flags to widen error bars or exclude uncertain frames.

---

## Chunk-Aware Design

In production, videos will be long (minutes to hours). The tracklet count scales linearly with duration — a long video can produce tens of thousands of tracklets, far too many for a single global graph optimization. The pipeline is therefore designed to process **temporal chunks** independently, with a lightweight handoff mechanism to maintain persistent global IDs across chunk boundaries.

The high-level chunking machinery (choosing chunk boundaries, feeding chunks into the pipeline, managing the overall video lifecycle) is out of scope for this stage and will be built later. However, the within-chunk pipeline described in Steps 0–5 is designed now to accept prior-chunk context as input and emit handoff information as output, so that chunk-based operation requires no changes to the core logic.

### Chunk Input: Prior Context (Optional)

When processing a chunk that is not the first in a video, the pipeline accepts an optional **prior context** structure containing:

- **Active global IDs** from the previous chunk (up to 9).
- **Per-ID 3D state at handoff:** 3D position and velocity (finite-difference estimate) at the last few frames of the previous chunk.
- **Per-ID per-camera 2D state at handoff:** the 2D centroid and camera ID of each tracklet contributing to each global ID at the end of the previous chunk.

When prior context is provided, Step 3 (clustering) uses it to **seed** cluster identities rather than discovering them from scratch. Tracklets in the new chunk that are spatially consistent with a prior-chunk global ID (in 3D or in shared 2D camera views) inherit that ID. Tracklets that don't match any prior ID form new clusters as usual.

### Chunk Output: Handoff State

After completing Steps 0–5, the pipeline emits a **handoff state** structure containing:

- **Active global IDs** in this chunk (up to 9).
- **Per-ID 3D state:** position and velocity at the final frames of the chunk.
- **Per-ID per-camera 2D state:** centroid and camera ID of each contributing tracklet at the final frames.
- **Per-ID confidence:** a scalar reflecting association reliability at the handoff boundary. This should be low when:
  - The cluster's average internal edge weight is weak (few strong pairwise matches).
  - The 3D refinement residuals (Step 5) are high at the final frames.
  - Another global ID has a similar 3D position at the final frames (close encounter — the identities may be ambiguous or swapped). Specifically, if two IDs are within a distance threshold (e.g., 2× the triangulation uncertainty) at the boundary, both should have their confidence reduced.

  The next chunk should treat low-confidence IDs as soft seeds — preferring to match them but willing to override if the new chunk's data strongly disagrees.

This is sufficient for the next chunk to initialize its clustering with prior context.

### Overlap Strategy

Chunks should overlap temporally (e.g., 1 second) so that tracklets spanning a chunk boundary appear in both chunks with enough temporal context to be associated correctly. The inter-chunk handoff resolves the duplication — the same physical tracklet in both chunks maps to the same persistent global ID. The exact overlap duration is a tuning parameter to be determined during the chunk-focused testing stage.

### What This Means for Implementation Now

The within-chunk pipeline (Steps 0–5) is implemented and tested as a **single-chunk, full-video processor** — it takes all tracklets and produces global IDs. The only concession to the chunk-aware future is structural:

- Step 3 accepts an optional `prior_context` argument. When absent (full-video mode, or first chunk), clustering starts from scratch. When present, it seeds cluster identities from the prior chunk's handoff state.
- The pipeline returns a `handoff_state` alongside the global ID assignments, even in full-video mode (where it simply won't be consumed by a subsequent chunk).
- Must-not-link constraint sets are scoped to the current chunk's tracklets only — they never reference tracklets from prior chunks.

No chunk-boundary logic, chunk-size selection, or multi-chunk orchestration is built at this stage. These are deferred to a dedicated chunking stage once the single-chunk pipeline is validated.

---

## Design Rationale

| Decision | Rationale |
|---|---|
| Camera-pair pruning (Step 0) | Eliminates ~50%+ of pair comparisons before any ray casting |
| Ray–ray distance, not reprojection | Avoids committing to a 3D point before multi-view consensus exists |
| Ghost point penalty | Catches coincidental ray intersections by checking consistency with the wider camera network — a false match from two cameras is contradicted by others |
| Median + inlier fraction | Robust to the 10–20% of frames with noisy detections |
| Graph clustering, not Hungarian matching | Hungarian assumes 1:1; reality has fragmentation and partial visibility |
| Explicit constraint sets (not negative-∞ edges) | Algorithm-agnostic; constraints are checked in post-processing regardless of clustering method, decoupling constraint enforcement from algorithm choice |
| Detected vs. coasted frame distinction | Prevents false must-not-link constraints between fragments of the same fish caused by tracker coasting behavior |
| 3D refinement as a post-processing step | Catches globally inconsistent associations that pairwise scores miss |
| Forward LUT (pixel→ray) | Moves expensive per-camera refraction computation offline; query-time cost is bilinear interpolation |
| Inverse LUT (voxel→pixel) | Enables instant "which cameras see this 3D point?" queries for ghost penalty, camera overlap, and reprojection — no refraction math at query time |
| Chunk-aware input/output contracts | Pipeline accepts optional prior context and always emits handoff state, enabling future chunk-based processing without modifying core logic |

---

## Performance Considerations

The following bottlenecks are listed in priority order. These optimizations should only be implemented once the basic pipeline logic is validated and producing correct associations — premature optimization here would obscure bugs in the core geometric and clustering logic.

### Bottleneck 1: Ghost Penalty Detection Lookups

The ghost penalty (Step 1.3) is the most expensive per-frame operation. For each inlier frame, after triangulating and snapping to a voxel, you must check 3–5 additional cameras for nearby detections. If per-camera detections are stored as flat lists, each check is a linear scan over all detections in that camera at that frame — repeated millions of times across all tracklet pairs and frames.

**Mitigation:** Build a spatial index of detections per camera per frame. Options in order of implementation simplicity:

- **2D grid binning:** Divide each camera's image plane into coarse cells (e.g., 32×32 pixels). Store detections in their grid cell. Lookup becomes O(1): snap the expected (u, v) to a cell and check that cell plus immediate neighbors. This is the recommended first approach — simple, cache-friendly, and sufficient given that detection density is low (~9 fish per camera).
- **KD-tree per camera per frame:** More precise but heavier. Only justified if detection counts per frame are much higher than expected (e.g., due to false positives).

### Bottleneck 2: Early Termination Tuning

Step 1 processes ~3.2M frame-level evaluations (40 camera pairs × 81 tracklet pairs × ~1000 frames). Most tracklet pairs are non-matching, so aggressive early termination is the single biggest lever on total runtime.

**Mitigation:**

- **Temporal subsampling:** Instead of evaluating every frame, sample every Nth frame (e.g., N=5) for the initial pass. Only proceed to full-frame evaluation for pairs that pass the subsampled check. Fish move smoothly, so subsampling at moderate rates loses little signal.
- **Adaptive thresholds:** Start with a generous K=10 frame evaluation window. If the inlier fraction is exactly 0 (no frames with d(t) < τ), abandon immediately. If it's between 0 and 0.1, extend to K=20 before deciding. This avoids both wasted work on obvious non-matches and premature abandonment of borderline cases.
- **Sort tracklet pairs by spatial plausibility:** Within each camera pair, use the mean 2D centroid of each tracklet to get a rough expected 3D region. Compare rough regions between tracklet pairs and evaluate the most spatially plausible pairs first. Once 9 strong matches are found across the full pipeline, remaining low-plausibility pairs can be skipped or deprioritized.

### Bottleneck 3: Voxel Map Memory and Cache Locality

The inverse voxel map is accessed with random 3D coordinates (the triangulated point from each tracklet pair at each frame). Since different tracklet pairs correspond to different fish in different parts of the tank, access patterns jump around the voxel volume, producing poor cache locality.

**Mitigation:**

- **Grid resolution is already 2 cm by default,** which keeps storage at ~37 MB. If ghost penalty accuracy is empirically insufficient, the resolution can be refined to 1 cm (~300 MB) via the `voxel_resolution_cm` config parameter, at the cost of increased memory and reduced cache performance.
- **Process by spatial region:** When iterating over tracklet pairs, group them by approximate tank region (based on mean centroid back-projection). This clusters voxel accesses spatially, improving cache behavior. This is a second-order optimization — only worth implementing if profiling confirms cache misses are a real bottleneck.

### Bottleneck 4: LUT Precomputation (One-Time)

The inverse voxel map requires projecting ~625K voxel centers (at the default 2 cm resolution) through the refraction model for each of 12 cameras — ~7.5M refraction calculations. At ~1 μs each, this is roughly 8 seconds.

**Mitigation:** This is embarrassingly parallel across voxels and cameras. A GPU implementation or multicore CPU parallelization reduces it to under a second. Since this is a one-time cost per camera setup, even the serial runtime is acceptable for batch workflows. Serialize the result to disk and reload on subsequent runs.

---

## Alternatives and Fallbacks

The following approaches were considered during design and are documented here as fallbacks if the primary methods prove insufficient.

### Per-Camera Tracking: SORT

Standard SORT (Kalman filter + Hungarian matching on IoU) without OC-SORT's occlusion-handling improvements. Simpler to implement from scratch but produces more fragmentation and Kalman drift during occlusions. Use only if OC-SORT's implementation proves problematic.

### Per-Camera Tracking: DeepSORT / BoT-SORT

Adds appearance-based re-identification embeddings to the matching cost, which helps maintain identity through close encounters and long occlusions. Requires training or fine-tuning a re-ID model on fish imagery, which is a significant additional investment. Worth revisiting if within-camera identity swaps during close encounters are a major source of errors that OC-SORT's motion model alone cannot resolve.

### Clustering: Spectral Clustering

Spectral clustering on the graph Laplacian, targeting k=9 clusters. Requires specifying the cluster count upfront, which is a disadvantage — if fewer than 9 fish are visible, forced clusters will be low-quality. Could be useful as a cross-check against Leiden results, or if Leiden's resolution parameter proves difficult to tune.

### Clustering: Correlation Clustering

Directly optimizes agreement with positive (affinity) and negative (must-not-link) edges, discovering the cluster count automatically. Natively handles the constraint structure without post-processing. However, the problem is NP-hard; practical implementations use greedy or LP-relaxation approximations. Worth revisiting if must-not-link constraint violations from Leiden post-processing are frequent or hard to resolve.

### Clustering: Rank-1 Tensor Approximation

Treats multi-camera association as a higher-order matching problem, capturing 3-way (or higher) consistency that pairwise edges miss. More theoretically principled for the multi-view setting but significantly more complex to implement and less supported by standard libraries. Consider if pairwise scoring + Leiden produces too many errors from coincidental pairwise agreements that are globally inconsistent.

---

## Open Questions / Future Work

- **Appearance features:** If fish are visually distinguishable (color, size, markings), appearance similarity can augment geometric scores. This converts the problem from pure geometry to geometry + re-identification.
- **Temporal smoothness:** A fish's 3D trajectory should be smooth. Penalizing large frame-to-frame 3D jumps within a cluster could improve association quality.
- **Online / streaming operation:** The pipeline as described is batch. For real-time use, incremental graph updates and cluster maintenance would be needed.
- **Confidence calibration:** Translating raw scores into posterior probabilities of same-fish identity, enabling principled thresholding.
