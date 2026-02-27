# v2.1 Prospective: AquaPose

**Status:** Draft — ready for `/gsd:new-milestone` review

**Scope:** Pipeline reorder (2D tracking → tracklet association → deferred midline → reconstruction) plus evaluation foundations and diagnostic tooling. Detection improvements (OBB) and keypoint pose estimation are deferred to a future milestone.

**Design spec:** Detailed algorithmic design for the cross-camera association pipeline is in `MS3-SPECSEED.md` (same directory). Requirements below reference that document; they define *what*, not *how*.

---

## Bottleneck Ordering

1. **Evaluation infrastructure** — regression suite in CI, measurement pipelines. Must come first so improvements are measurable.
2. **Pipeline reordering** — move 2D tracking before midline extraction, associate on tracklets instead of per-frame. Fixes the root cause of 3D reconstruction failure (broken cross-view correspondence).

The pipeline reorder is the single biggest change. 2D midlines and tracking already look decent, but 3D reconstruction is broken despite both backends (triangulation and curve optimizer) being available. The problem is structural, not algorithmic.

---

## Pipeline Reordering

### Current pipeline (v2.0)
```
Detection → Midline → Association(RANSAC 3D) → Tracking(3D claim) → Reconstruction
```

### Target pipeline (v2.1)
```
Detection → 2D Tracking → Cross-Camera Association → Midline → Reconstruction
```

**This is a replacement, not an addition.** The v2.0 pipeline stages (RANSAC centroid association, 3D bundle-claiming tracker) are deleted. The existing `FishTracker`, `ransac_centroid_cluster()`, and associated code are removed — recoverable from git history if needed in a future milestone. Keeping the codebase lean during iteration is the priority.

**Why:** The current pipeline extracts midlines before knowing which fish it's looking at, then tries to associate noisy midlines across 12 cameras per frame. This produces broken cross-view correspondence, which makes 3D reconstruction a mess regardless of backend.

**What's broken now:** 3D splines are in the right general area of the tank but jump wildly frame-to-frame. Triangulation produces multi-meter splines that appear and disappear. Curve optimizer produces reasonable-length splines but they never land on fish and jump randomly. Both symptoms point to cross-view correspondence failure.

**The fix:** Track in 2D first (easy, well-solved problem — OC-SORT on bboxes within a single camera view). Then associate *tracklets* across cameras using ray-ray geometric consistency tested over many shared frames, with ghost-point penalties from the wider camera network. This uses many frames of evidence instead of single-frame geometry, making it far more robust.

**Key benefits of reordering:**
- 2D tracking within one camera is nearly trivial — smooth motion, rare identity confusion
- Tracklet association uses trajectory-level evidence (hundreds of frames) instead of single-frame centroids
- Midline extraction happens only for confirmed tracklet-groups, not every detection
- Cross-camera association can inform head-tail ordering consistency
- 3D reconstruction inherits temporal smoothness from 2D tracklets by construction
- One bad frame doesn't break association because the tracklet has hundreds of frames of evidence

### PipelineContext data flow (new)

| Stage | Reads | Writes |
|-------|-------|--------|
| 1. Detection | frames, calibration | `detections` — per-camera per-frame bbox + confidence |
| 2. 2D Tracking | `detections` | `tracks_2d` — per-camera tracklets with local IDs |
| 3. Cross-Camera Association | `tracks_2d`, calibration | `tracklet_groups` — global fish ID → {cam: tracklet} |
| 4. Midline | `tracklet_groups`, `detections`, frames | `annotated_detections` — midlines for grouped detections only |
| 5. Reconstruction | `tracklet_groups`, `annotated_detections`, calibration | `midlines_3d` — per-fish 3D splines |

### Identity model

- **Stage 2** assigns **local per-camera IDs** — tracklet IDs meaningful only within a single camera's timeline. No cross-camera awareness.
- **Stage 3** assigns **global fish IDs** — the association stage determines which local tracklets across cameras correspond to the same physical fish. Global IDs are authoritative from this point forward.
- Local IDs are internal bookkeeping; downstream stages reference global fish IDs only.

### CarryForward (new)

Per-camera 2D track state (positions, velocities, lifecycle per local tracklet). Each camera's tracker is independent — no cross-camera state in carry. Cross-camera association operates on complete tracklets within the batch, not incrementally.

### Chunk-aware design

The association pipeline (Steps 0–5 in the design spec) is built as a single-chunk processor. The only concession to future chunking is structural: Step 3 accepts an optional `prior_context` argument for seeding cluster identities from a previous chunk, and the pipeline always emits a `handoff_state` alongside global ID assignments. No chunk-boundary logic, chunk-size selection, or multi-chunk orchestration is built in this milestone.

### Code disposition

Deleted (not archived):
- `FishTracker` and 3D bundle-claiming logic (`tracker.py`)
- `ransac_centroid_cluster()` and RANSAC association (`associate.py`)
- `AssociationStage` (replaced by new cross-camera association stage)
- `TrackingStage` (replaced by new 2D tracking stage)
- Related tests for deleted code

Recoverable from git history. Reintegration of 3D tracking as an alternative backend is a possible future task, not a v2.1 concern.

---

## Requirements

### Foundations

**EVAL-01: Regression suite in CI**
7 regression tests skip without production video data. Provide a synthetic fixture so numerical equivalence is verified automatically. Fix the 1 xfail (midline golden data regeneration). CI already exists (test.yml, slow-tests.yml) — the gap is that regression tests silently skip.

### Refractive Lookup Tables

**LUT-01: Forward lookup table (pixel → ray)**
For each camera, precompute a grid mapping 2D pixel coordinates to 3D ray (origin, direction) in water-space using the calibrated refraction model. At query time, bilinear interpolation over the grid recovers the ray for any detection centroid. Eliminates per-frame refraction math during association.

Serialize to disk and reload on subsequent runs. One-time cost per camera setup.

**LUT-02: Inverse lookup table (voxel → pixel)**
Discretize the tank volume into a regular voxel grid (default 2 cm resolution, configurable). For each voxel center, project through the refraction model into every camera and record: visibility (boolean) and projected pixel coordinates (u, v).

Produces:
- Camera overlap graph (which camera pairs share visible voxels) — consumed by association Step 0
- Ghost-point lookup (which cameras should see a given 3D point and where) — consumed by association Step 1.3
- Fast reprojection for refinement — consumed by association Step 5

Storage: ~37 MB at default resolution. Serialize to disk.

### Per-Camera 2D Tracking

**TRACK-01: OC-SORT 2D tracking stage**
Replace the existing `TrackingStage` (3D bundle claiming) with a new stage that runs OC-SORT independently per camera, immediately after detection.

Delete `FishTracker`, `ransac_centroid_cluster()`, and the old `TrackingStage`/`AssociationStage`.

Each tracklet must include: `camera_id`, `track_id` (local), `frames`, `centroids`, `frame_status` (detected vs. coasted), `bboxes`. See design spec "Tracklet Output Format" for details.

Output: `tracks_2d` on PipelineContext.

### Cross-Camera Association

**ASSOC-01: Pairwise tracklet scoring**
For each adjacent camera pair (from LUT-02's overlap graph), score all tracklet pairs with temporal overlap. Scoring uses ray-ray closest-point distance (from LUT-01 rays) aggregated across shared frames, plus a ghost-point penalty (from LUT-02 voxel lookups) that checks consistency with the wider camera network.

Produces per-pair affinity scores. Includes early termination for obvious non-matches.

See design spec Steps 1.1–1.6.

**ASSOC-02: Graph clustering and global ID assignment**
Build a weighted affinity graph from pairwise scores (ASSOC-01). Cluster using connected components + Leiden algorithm with must-not-link constraints (same-camera tracklets overlapping in detected frames cannot share a cluster).

Assign global fish IDs to each cluster. Merge same-camera tracklet fragments within clusters (non-overlapping or coasted-only overlap).

See design spec Steps 2–4.

**ASSOC-03: 3D consistency refinement**
For each cluster, triangulate 3D positions across frames using member tracklets. Reproject into each camera and check reprojection error. Evict tracklets with consistently high error. Evicted tracklets become singletons (no global ID).

Emit per-frame confidence estimates (reprojection residuals, camera count, close-encounter flags).

See design spec Step 5.

Output: `tracklet_groups` on PipelineContext, plus `handoff_state` for future chunk-aware operation.

### Pipeline Integration

**PIPE-01: PipelineContext and CarryForward update**
Update `PipelineContext` fields to reflect the new stage ordering (`tracks_2d`, `tracklet_groups`). Update `CarryForward` to carry per-camera 2D track state instead of 3D track state. Update `build_stages()` to wire the new stage order. This is a prerequisite for the new stages — the context fields must exist before TRACK-01 and ASSOC-* can write to them.

**PIPE-02: Deferred midline extraction**
Move midline extraction to Stage 4 (after association). Extract midlines only for detections belonging to confirmed tracklet-groups. Cross-camera group membership provides a head-tail consistency signal — if most cameras agree on head direction, flip outliers.

Update `MidlineStage` to read from `tracklet_groups` instead of raw `detections`.

**PIPE-03: Reconstruction from associated tracklets**
Update `ReconstructionStage` to read from `tracklet_groups` and `annotated_detections`. Triangulate using only the cameras known to observe each fish per frame (correspondence pre-established). No RANSAC needed for cross-view matching.

### Diagnostic Tooling

**DIAG-01: Tracklet and association visualization**
Visualize 2D tracklets per camera (centroid trails on video) and cross-camera associations (color-coded by global fish ID). Build incrementally as each stage lands — essential for validating correctness at every step.

---

## Key Design Decisions

**Why reorder the pipeline?**
3D reconstruction is broken despite decent 2D midlines and working 2D tracking. The symptoms (multi-meter splines, frame-to-frame jumping, neither backend producing usable output) all point to cross-view correspondence failure. The current pipeline tries to associate per-frame across 12 cameras — a hard problem. Tracking in 2D first (easy) then associating tracklets (robust, uses trajectory-level evidence) fixes the correspondence problem structurally rather than trying to tune RANSAC parameters.

**Why replace rather than preserve the old pipeline?**
Maintaining two pipeline orderings (3D-first and 2D-first) doubles the surface area for every change to PipelineContext, CarryForward, or stage interfaces. The v2.0 tracking code is recoverable from git history. Lean iteration now; reintegration as a backend is a possible future task if needed.

**Why OC-SORT for 2D tracking?**
OC-SORT extends standard SORT with observation-centric re-update (reducing Kalman drift during occlusion), observation-centric momentum (better post-occlusion prediction), and improved lost-track recovery. No appearance model needed — within a single top-down camera view, fish are distinguishable by position and velocity alone. Fallback: plain SORT if OC-SORT proves problematic.

**Why ray-ray distance + ghost penalty for association?**
Ray-ray distance is the natural geometric primitive — it avoids committing to a 3D point before multi-view consensus exists. The ghost penalty catches coincidental ray intersections by checking whether the wider camera network agrees. Together they produce a robust pairwise score that leverages the full camera rig.

**Why Leiden clustering?**
Leiden discovers cluster count automatically (no need to specify k=9), handles sparse community structure well, and has a tunable resolution parameter. Must-not-link constraints are enforced in post-processing, keeping the clustering algorithm-agnostic. Fallbacks: spectral clustering (requires specifying k), correlation clustering (NP-hard approximation).

**Why precomputed LUTs?**
The forward (pixel→ray) and inverse (voxel→pixel) LUTs eliminate per-frame refraction math during association. The inverse LUT is particularly valuable — it provides instant camera-overlap queries, ghost-point lookups, and fast reprojection, all from a single 37 MB precomputed structure.

---

## Deferred to Future Milestones

**Detection improvement:**
- OBB detector for tighter crops (DET-01) — benefits both segmentation and keypoint backends

**Keypoint pose estimation:**
- Keypoint regression head on shared encoder (KP-01)
- Keypoint training data / manual annotation (KP-02)
- Direct pose midline backend (KP-03)
- Flexible keypoint count through pipeline (FLEX-01) — only needed when keypoint head (6 points) plugs in

**Post-evaluation:**
- Segmentation improvements — pursue only if keypoint path doesn't meet accuracy needs
- 3D reconstruction benchmark — synthetic ground truth for Chamfer/point-to-curve error
- Curve optimizer vs. triangulation benchmark
- Refractive calibration validation
- HDF5 schema documentation
- 3D tracking backend reintegration
- Chunk-based processing orchestration (chunk boundary logic, chunk size selection)

---

## New Dependencies

- **`leidenalg`** (or `igraph`): Leiden community detection for tracklet graph clustering (ASSOC-02)
- **OC-SORT implementation**: Either a third-party package or vendored implementation for per-camera 2D tracking (TRACK-01). Evaluate available Python packages during implementation.

---

## Out of Scope for v2.1

- Preserving v2.0 pipeline as alternative configuration
- Chunk orchestration (contracts are built, orchestration is deferred)
- Appearance-based re-identification features
- GUI or web interface
- Real-time processing
- Dataset collection (hardware)
- Deployment/packaging (PyPI, Docker)
- New output formats beyond HDF5
- Multi-species generalization
- MOG2 backend validation
