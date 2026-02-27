# Requirements: AquaPose

**Defined:** 2026-02-27
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v2.1 Requirements

Requirements for the Identity milestone. Each maps to roadmap phases.

### Refractive Lookup Tables

- [x] **LUT-01**: User can generate a forward lookup table (pixel→ray) per camera that maps 2D pixel coordinates to 3D refracted rays via bilinear interpolation, serialized to disk for reuse
- [x] **LUT-02**: User can generate an inverse lookup table (voxel→pixel) that discretizes the tank volume at configurable resolution and records per-voxel camera visibility masks and projected pixel coordinates, producing camera overlap graph and ghost-point lookup

### Per-Camera 2D Tracking

- [x] **TRACK-01**: User can run OC-SORT 2D tracking independently per camera, producing tracklets with camera_id, track_id, frames, centroids, frame_status (detected/coasted), and bboxes — replacing the old 3D bundle-claiming TrackingStage

### Cross-Camera Association

- [ ] **ASSOC-01**: User can score all tracklet pairs across adjacent cameras using ray-ray closest-point distance (from LUT-01) aggregated over shared frames, with ghost-point penalty (from LUT-02) checking consistency against the wider camera network
- [ ] **ASSOC-02**: User can cluster tracklets into global fish identity groups via connected components + Leiden algorithm with must-not-link constraints (same-camera tracklets with detection-backed temporal overlap cannot share a cluster), with same-camera fragment merging
- [ ] **ASSOC-03**: User can refine clusters via per-frame 3D triangulation and reprojection error checking, evicting tracklets with consistently high error, and emitting per-frame confidence estimates

### Pipeline Integration

- [x] **PIPE-01**: PipelineContext fields reflect new stage ordering (tracks_2d, tracklet_groups), CarryForward carries per-camera 2D track state, and build_stages() wires the new 5-stage order
- [ ] **PIPE-02**: Midline extraction runs after association (Stage 4), processing only detections belonging to confirmed tracklet-groups, with head-tail consistency from cross-camera group membership
- [ ] **PIPE-03**: Reconstruction reads from tracklet_groups and annotated_detections, triangulating using only cameras known to observe each fish per frame (no RANSAC for cross-view matching)

### Diagnostic Tooling

- [x] **DIAG-01**: User can visualize 2D tracklets per camera (centroid trails on video) and cross-camera associations (color-coded by global fish ID)

## Future Requirements

Deferred to a later milestone. Tracked but not in current roadmap.

### Evaluation

- **EVAL-01**: Regression suite runs in CI with synthetic fixtures (deferred — pipeline reorder invalidates existing regression tests; build new regression tests after v2.1 stabilizes)

### Detection

- **DET-01**: OBB detector for tighter crops

### Keypoint Pose Estimation

- **KP-01**: Keypoint regression head on shared encoder
- **KP-02**: Keypoint training data / manual annotation
- **KP-03**: Direct pose midline backend

### Chunking

- **CHUNK-01**: Multi-chunk orchestration (chunk boundary logic, chunk size selection)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Preserving v2.0 pipeline as alternative configuration | Doubles surface area; old code recoverable from git history |
| Chunk orchestration | Contracts built (prior_context, handoff_state), orchestration deferred |
| Appearance-based re-identification | Within-camera tracking relies on position/velocity only |
| MOG2 backend validation | YOLO is primary detector; MOG2 deferred |
| Segmentation improvements | Pursue after pipeline reorder stabilizes |
| Real-time processing | Batch only |
| GUI/web interface | CLI pipeline |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| LUT-01 | Phase 23 | Complete |
| LUT-02 | Phase 23 | Complete |
| TRACK-01 | Phase 24 | Complete |
| ASSOC-01 | Phase 25 | Pending |
| ASSOC-02 | Phase 25 | Pending |
| ASSOC-03 | Phase 26 | Pending |
| PIPE-01 | Phase 22 | Complete |
| PIPE-02 | Phase 26 | Pending |
| PIPE-03 | Phase 26 | Pending |
| DIAG-01 | Phase 27 | Complete |

**Coverage:**
- v2.1 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-02-27*
*Last updated: 2026-02-27 after roadmap creation (Phases 22-27)*
