# Phase 5: Cross-View Identity and 3D Tracking - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Given per-camera fish detections (bounding boxes from YOLO + masks from U-Net), determine which detections across cameras correspond to the same physical fish via RANSAC centroid ray clustering, triangulate 3D centroids, and maintain persistent fish IDs across frames using Hungarian assignment. Output: high-confidence tracklets with per-fish camera sets, 3D centroids, and organized bounding boxes, serialized to HDF5.

Medial axis extraction, midline reconstruction, and trajectory stitching are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Association strategy
- RANSAC centroid ray clustering for cross-view association
- Require minimum 2 cameras for a valid association; single-view detections passed through as flagged low-confidence entries (for track continuity, not reconstruction)
- Prior-guided association: use previous frame's 3D centroids as seed points to bias RANSAC clustering — critical because fish often overlap in many views
- `expected_count` is a configurable parameter (default 9), used as a soft global expectation — not enforced per-camera, not a hard constraint

### Track lifecycle
- Short grace period (5-10 frames) before killing lost tracks — produces high-confidence tracklets rather than long uncertain tracks; trajectory stitching deferred to a later phase with more signal from Phase 6
- 2-3 frame birth confirmation required before a track is considered valid — filters spurious detections
- Constant velocity motion model to predict next-frame position for Hungarian assignment
- First-frame initialization: Claude's discretion (batch vs gradual)

### Quality & thresholds
- Reprojection residual threshold: configurable parameter with a sensible default (informed by Phase 1 Z-uncertainty characterization)
- Low-confidence associations: flag but include in output with confidence score — downstream decides whether to use them
- No per-frame diagnostic logging — adds loop time, not useful at scale for hours-long video; info is recoverable from output
- No warnings on low fish count — silent operation, output speaks for itself

### Output interface
- Primary format: HDF5 serialization (matches existing io module patterns)
- Dataclass API kept accessible for future in-memory pipeline use, but serialization is the focus
- Chunked processing: configurable chunk size for hours-long videos, track state carried across chunks
- Per-fish per-frame data includes: 3D centroid, confidence score, (camera_id, detection_id) → fish_id mapping, AND per-camera bounding boxes organized by fish ID (saves Phase 6 from re-joining detection + identity outputs)

### Claude's Discretion
- First-frame initialization strategy (batch vs gradual ramp)
- Exact RANSAC parameters (iterations, inlier threshold)
- Chunk size default
- HDF5 dataset layout and compression
- Exact constant-velocity model implementation details

</decisions>

<specifics>
## Specific Ideas

- "Short grace produces high-confidence tracklets for easy cases, which we can later stitch into a single optimal trajectory set once we have more to work with from Phase 6"
- Fish frequently swim close together (often overlap) — disambiguation is a first-class concern, not an edge case
- Single-view detections should be hypothetically impossible in the 13-camera rig but worth handling gracefully

</specifics>

<deferred>
## Deferred Ideas

- Trajectory stitching (merging tracklets into continuous tracks) — future phase after Phase 6 provides midline signal
- Per-frame diagnostic summaries — could add later behind a debug flag if needed

</deferred>

---

*Phase: 05-cross-view-identity-and-3d-tracking*
*Context gathered: 2026-02-21*
