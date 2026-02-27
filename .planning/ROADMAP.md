# Roadmap: AquaPose

## Milestones

- ✅ **v1.0 MVP** — Phases 1-9 (shipped 2026-02-25)
- ✅ **v2.0 Alpha** — Phases 13-21 (shipped 2026-02-27)
- 🚧 **v2.1 Identity** — Phases 22-27 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-9) — SHIPPED 2026-02-25</summary>

- [x] Phase 1: Calibration and Refractive Geometry (2/2 plans) — complete
- [x] Phase 2: Segmentation Pipeline (4/4 plans) — complete
- [x] Phase 02.1: Segmentation Troubleshooting (3/3 plans) — complete (INSERTED)
- [x] Phase 02.1.1: Object-detection alternative to MOG2 (3/3 plans) — complete (INSERTED)
- [x] Phase 3: Fish Mesh Model and 3D Initialization (2/2 plans) — complete
- [x] Phase 4: Per-Fish Reconstruction (3/3 plans) — shelved (ABS too slow)
- [x] Phase 04.1: Isolate phase4-specific code (1/1 plan) — complete (INSERTED)
- [x] Phase 5: Cross-View Identity and 3D Tracking (3/3 plans) — complete
- [x] Phase 6: 2D Medial Axis and Arc-Length Sampling (1/1 plan) — complete
- [x] Phase 7: Multi-View Triangulation (1/1 plan) — complete
- [x] Phase 8: End-to-End Integration Testing (3 plans, 2 summaries) — complete
- [x] Phase 9: Curve-Based Optimization (2/2 plans) — complete

**12 phases, 28 plans total**
Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>✅ v2.0 Alpha (Phases 13-21) — SHIPPED 2026-02-27</summary>

- [x] Phase 13: Engine Core (4/4 plans) — completed 2026-02-25
- [x] Phase 14: Golden Data and Verification Framework (2/2 plans) — completed 2026-02-25
- [x] Phase 14.1: Fix Critical Mismatch (2/2 plans) — completed 2026-02-25 (INSERTED)
- [x] Phase 15: Stage Migrations (5/5 plans) — completed 2026-02-26
- [x] Phase 16: Numerical Verification and Legacy Cleanup (2/2 plans) — completed 2026-02-26
- [x] Phase 17: Observers (5/5 plans) — completed 2026-02-26
- [x] Phase 18: CLI and Execution Modes (3/3 plans) — completed 2026-02-26
- [x] Phase 19: Alpha Refactor Audit (4/4 plans) — completed 2026-02-26
- [x] Phase 20: Post-Refactor Loose Ends (5/5 plans) — completed 2026-02-27
- [x] Phase 21: Retrospective, Prospective (2/2 plans) — completed 2026-02-27

**10 phases, 34 plans total**
Full details: `.planning/milestones/v2.0-ROADMAP.md`

</details>

### 🚧 v2.1 Identity (In Progress)

**Milestone Goal:** Reorder the pipeline to track in 2D first, then associate tracklets across cameras using trajectory-level geometric evidence — fixing the root cause of broken 3D reconstruction.

## Phase Checklist (v2.1)

- [x] **Phase 22: Pipeline Scaffolding** — Reorder PipelineContext, CarryForward, and build_stages(); delete old association and tracking code (completed 2026-02-27)
- [x] **Phase 23: Refractive Lookup Tables** — Build and serialize forward (pixel→ray) and inverse (voxel→pixel) LUTs per camera (completed 2026-02-27)
- [x] **Phase 24: Per-Camera 2D Tracking** — OC-SORT independent per-camera 2D tracking producing structured tracklets (completed 2026-02-27)
- [x] **Phase 25: Association Scoring and Clustering** — Pairwise ray-ray scoring across cameras and Leiden-based global identity clustering (completed 2026-02-27)
- [ ] **Phase 26: Association Refinement and Pipeline Wiring** — 3D refinement of clusters, deferred midline extraction, reconstruction from tracklet groups
- [ ] **Phase 27: Diagnostic Visualization** — Centroid trail overlays per camera and cross-camera color-coded association visualization

## Phase Details

### Phase 22: Pipeline Scaffolding
**Goal**: The engine wires the new 5-stage order (Detection → 2D Tracking → Association → Midline → Reconstruction) with correctly typed PipelineContext and CarryForward; old AssociationStage, TrackingStage, FishTracker, and ransac_centroid_cluster code is deleted
**Depends on**: Phase 21 (v2.0 complete)
**Requirements**: PIPE-01
**Success Criteria** (what must be TRUE):
  1. PipelineContext carries `tracks_2d` (per-camera OC-SORT state) and `tracklet_groups` (cross-camera identity clusters) as typed fields
  2. CarryForward persists per-camera 2D track state across frames without carrying old bundle or 3D cluster fields
  3. `build_stages()` returns the new 5-stage sequence and the pipeline executes without runtime errors on a synthetic frame sequence
  4. The old AssociationStage, TrackingStage, FishTracker, and ransac_centroid_cluster modules are absent from the codebase (no import paths, no dead code)
**Plans**: 2 plans

Plans:
- [ ] 22-01: Define Tracklet2D/TrackletGroup domain types, update PipelineContext, delete legacy tracking/association code
- [ ] 22-02: Create stub stages, rewire build_stages() to new 5-stage order, update observers and tests

### Phase 23: Refractive Lookup Tables
**Goal**: Users can generate and persist forward (pixel→ray) and inverse (voxel→pixel) lookup tables for all cameras, eliminating per-frame refraction math during association
**Depends on**: Phase 22
**Requirements**: LUT-01, LUT-02
**Success Criteria** (what must be TRUE):
  1. User runs a CLI command (or script) that generates a forward LUT for each camera mapping every pixel coordinate to a 3D refracted ray via bilinear interpolation, and saves it to disk
  2. User runs a CLI command (or script) that generates an inverse LUT discretizing the tank volume at configurable resolution, recording per-voxel camera visibility masks and projected pixel coordinates
  3. The inverse LUT produces a valid camera overlap graph and a ghost-point lookup table (voxel → cameras that cannot simultaneously observe a real point there)
  4. LUT files load correctly and lookups return results numerically consistent with the on-the-fly AquaCal refractive projection (within floating-point tolerance)
**Plans**: TBD

Plans:
- [ ] 23-01: Forward LUT (pixel→ray) generation and serialization
- [ ] 23-02: Inverse LUT (voxel→pixel), camera overlap graph, ghost-point lookup

### Phase 24: Per-Camera 2D Tracking
**Goal**: Users can run OC-SORT tracking independently on each camera's detection stream, producing structured tracklets that carry frame-by-frame centroid, bbox, and status information
**Depends on**: Phase 22
**Requirements**: TRACK-01
**Success Criteria** (what must be TRUE):
  1. Detections from each camera are passed through an independent OC-SORT tracker that maintains track identity across frames using IoU and Kalman-predicted motion
  2. Output tracklets each contain `camera_id`, `track_id`, `frames`, `centroids`, `bboxes`, and `frame_status` (detected vs. coasted) as a typed data structure
  3. The DetectionStage emits detections and a new TrackingStage consumes them per-camera, storing tracklets in `PipelineContext.tracks_2d`
  4. Tracks correctly coast (predict without observation) for at least the configured number of frames before being dropped
**Plans**: 1 plan

Plans:
- [ ] 24-01: OC-SORT wrapper, TrackingStage, config expansion, pipeline rewire, unit tests

### Phase 25: Association Scoring and Clustering
**Goal**: Users can score all cross-camera tracklet pairs using ray-ray geometry and cluster them into global fish identity groups via Leiden algorithm with same-camera conflict constraints
**Depends on**: Phases 23, 24
**Requirements**: ASSOC-01, ASSOC-02
**Success Criteria** (what must be TRUE):
  1. For every pair of tracklets from adjacent cameras sharing overlapping frames, a pairwise affinity score is computed from ray-ray closest-point distance using LUT-01, with a ghost-point penalty applied from LUT-02
  2. The scored affinity graph is clustered via connected components followed by Leiden algorithm; same-camera tracklets with detection-backed temporal overlap are prevented from sharing a cluster (must-not-link)
  3. Same-camera tracklet fragments from the same fish (non-overlapping in time) are merged within each cluster
  4. The resulting `tracklet_groups` field in PipelineContext contains at most 9 clusters (one per fish), each listing its constituent tracklets across cameras
**Plans**: 2 plans

Plans:
- [x] 25-01: Pairwise cross-camera affinity scoring (ray-ray distance, ghost-point penalty, AssociationConfig expansion)
- [x] 25-02: Leiden clustering with must-not-link constraints, fragment merging, AssociationStage pipeline wiring

### Phase 26: Association Refinement and Pipeline Wiring
**Goal**: Cross-camera identity clusters are geometrically refined via 3D triangulation error, midline extraction runs only on confirmed tracklet detections, and reconstruction uses known camera membership per fish per frame — completing the end-to-end pipeline
**Depends on**: Phase 25
**Requirements**: ASSOC-03, PIPE-02, PIPE-03
**Success Criteria** (what must be TRUE):
  1. Each tracklet cluster is refined by per-frame 3D triangulation; tracklets with consistently high reprojection error are evicted, and per-frame confidence estimates are emitted alongside the final clusters
  2. Midline extraction (skeletonization + BFS pruning + arc-length sampling) runs only on detections belonging to confirmed tracklet groups, using cross-camera group membership to resolve head-tail ambiguity
  3. Reconstruction triangulates using only the cameras known to observe each fish in each frame (from `tracklet_groups`), with no RANSAC cross-view matching step required
  4. The full pipeline (Detection → 2D Tracking → Association → Midline → Reconstruction) runs end-to-end on a real video clip and produces HDF5 output with correct fish IDs
**Plans**: TBD

Plans:
- [ ] 26-01: Association refinement via 3D triangulation error and tracklet eviction (ASSOC-03)
- [ ] 26-02: Deferred midline extraction from confirmed tracklet groups (PIPE-02)
- [ ] 26-03: Reconstruction from tracklet_groups with known camera membership (PIPE-03)

### Phase 27: Diagnostic Visualization
**Goal**: Users can generate per-camera centroid trail videos and cross-camera association overlays color-coded by global fish ID to inspect tracking and association quality
**Depends on**: Phase 26
**Requirements**: DIAG-01
**Success Criteria** (what must be TRUE):
  1. User can produce a diagnostic video for each camera showing centroid trails for all tracklets, with coasted frames visually distinguished from detected frames
  2. User can produce a diagnostic overlay showing cross-camera associations color-coded by global fish identity cluster, enabling visual inspection of merge and split events
  3. The diagnostic outputs are generated by an Observer (not a stage) and can be enabled via configuration without affecting pipeline computation
**Plans**: 1 plan

Plans:
- [ ] 27-01: TrackletTrailObserver — per-camera centroid trail videos, association mosaic, observer factory wiring, unit tests

## Progress

**Execution Order:** 22 → 23 → 24 → 25 → 26 → 27

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Calibration and Refractive Geometry | v1.0 | 2/2 | Complete | 2026-02-19 |
| 2. Segmentation Pipeline | v1.0 | 4/4 | Complete | 2026-02-20 |
| 02.1 Segmentation Troubleshooting | v1.0 | 3/3 | Complete | 2026-02-20 |
| 02.1.1 Object-detection alternative to MOG2 | v1.0 | 3/3 | Complete | 2026-02-20 |
| 3. Fish Mesh Model and 3D Initialization | v1.0 | 2/2 | Complete | 2026-02-20 |
| 4. Per-Fish Reconstruction | v1.0 | 3/3 | Shelved | 2026-02-21 |
| 04.1 Isolate phase4-specific code | v1.0 | 1/1 | Complete | 2026-02-21 |
| 5. Cross-View Identity and 3D Tracking | v1.0 | 3/3 | Complete | 2026-02-21 |
| 6. 2D Medial Axis and Arc-Length Sampling | v1.0 | 1/1 | Complete | 2026-02-21 |
| 7. Multi-View Triangulation | v1.0 | 1/1 | Complete | 2026-02-22 |
| 8. End-to-End Integration Testing | v1.0 | 2/3 | Complete | 2026-02-23 |
| 9. Curve-Based Optimization | v1.0 | 2/2 | Complete | 2026-02-25 |
| 13. Engine Core | v2.0 | 4/4 | Complete | 2026-02-25 |
| 14. Golden Data and Verification | v2.0 | 2/2 | Complete | 2026-02-25 |
| 14.1 Fix Critical Mismatch | v2.0 | 2/2 | Complete | 2026-02-25 |
| 15. Stage Migrations | v2.0 | 5/5 | Complete | 2026-02-26 |
| 16. Numerical Verification | v2.0 | 2/2 | Complete | 2026-02-26 |
| 17. Observers | v2.0 | 5/5 | Complete | 2026-02-26 |
| 18. CLI and Execution Modes | v2.0 | 3/3 | Complete | 2026-02-26 |
| 19. Alpha Refactor Audit | v2.0 | 4/4 | Complete | 2026-02-26 |
| 20. Post-Refactor Loose Ends | v2.0 | 5/5 | Complete | 2026-02-27 |
| 21. Retrospective, Prospective | v2.0 | 2/2 | Complete | 2026-02-27 |
| 22. Pipeline Scaffolding | 2/2 | Complete    | 2026-02-27 | - |
| 23. Refractive Lookup Tables | 2/2 | Complete    | 2026-02-27 | - |
| 24. Per-Camera 2D Tracking | 1/1 | Complete    | 2026-02-27 | - |
| 25. Association Scoring and Clustering | 2/2 | Complete | 2026-02-27 | - |
| 26. Association Refinement and Pipeline Wiring | 1/3 | In Progress|  | - |
| 27. Diagnostic Visualization | v2.1 | 0/TBD | Not started | - |

### Phase 28: e2e testing

**Goal:** Pipeline runs e2e on real and synthetic data.  Blocking bugs fixed as they are encountered. Complex but non-blocking bugs logged and triaged. Tests in tests/e2e updated to current pipeline and confirmed passing. Final output indicates the pipeline is basically functional -- i.e., at least some fish yield reasonable 3d splines with trajectories that span at least a few contiguous frames. Full optimization and tuning is out of phase scope, we are just looking for pipeline-breaking bugs.
**Requirements**: TBD
**Depends on:** Phase 27
**Plans:** 1/3 plans executed

Plans:
- [ ] TBD (run /gsd:plan-phase 28 to break down)
