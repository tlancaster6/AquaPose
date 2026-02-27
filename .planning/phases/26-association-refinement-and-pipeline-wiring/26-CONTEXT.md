# Phase 26: Association Refinement and Pipeline Wiring - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Geometrically refine cross-camera identity clusters via 3D triangulation error, wire midline extraction to operate only on confirmed tracklet groups with head-tail resolution, and wire reconstruction to use known camera membership per fish — completing the end-to-end pipeline from Detection through Reconstruction with HDF5 output.

</domain>

<decisions>
## Implementation Decisions

### Eviction Policy (ASSOC-03)
- Evicted tracklets become singleton TrackletGroups with low confidence — not discarded
- Eviction metric: median reprojection error across the tracklet's detected frames exceeds a configurable threshold
- All eviction thresholds exposed in AssociationConfig with sensible defaults (researcher/planner determines initial values)
- After eviction, re-triangulate the cleaned cluster to produce updated confidence — single-pass is not sufficient
- Confidence estimates emitted per-frame alongside final clusters

### Head-Tail Resolution (PIPE-02)
- Lives in the Midline stage (Stage 4) as a post-processing step of the skeleton backend
  - Skeleton backend: resolves head-tail before emitting annotated_detections
  - Keypoint backend (future): produces inherently oriented output, resolver is a no-op
- Three signals combined for true head-tail disambiguation (not just consistent ordering):
  1. Cross-camera geometric vote — triangulate both orientations, pick lower total reprojection error
  2. Velocity alignment — midline direction should align with OC-SORT Kalman-filtered velocity vector
  3. Temporal prior — maintain orientation from previous frame; fish don't flip between frames
- Velocity signal gated by speed threshold: below threshold (configurable), ignore velocity and rely on temporal prior only. Handles stationary fish.
- When a camera disagrees with consensus orientation: flip its midline point order to match (don't exclude it)
- Keep "head-tail" naming in codebase — with velocity signal, it's genuinely resolving head vs tail

### Reconstruction (PIPE-03)
- Minimum 3 cameras to attempt triangulation (configurable `min_cameras`, default 3, can lower to 2 for exploratory runs)
- Fish below min_cameras in a frame: frame goes to `dropped` with reason `insufficient_views`
- Interpolate short gaps where a fish drops below min_cameras between measured frames (configurable `max_interp_gap`, default TBD by researcher/planner)
- Interpolated frames flagged via confidence=0 (or NaN), not a separate status field
- Fixed B-spline control point count per fish per frame (set in config, not adaptive)
- No RANSAC for cross-view matching — camera membership is known from tracklet_groups

### HDF5 Output
- Fish-first structure: top-level groups by global fish ID, each containing time-series arrays
  - `spline_controls[T, N, 3]` — B-spline control points per frame
  - `confidence[T]` — per-frame quality score from association refinement (0/NaN for interpolated frames)
- Run metadata as HDF5 root attributes: config hash, video paths, calibration file path, run timestamp
- No separate frame_status array — confidence alone distinguishes measured vs interpolated

### Claude's Discretion
- Exact eviction threshold defaults (reprojection error in pixels)
- Speed threshold default for velocity gating
- Max interpolation gap default
- B-spline control point count default
- HDF5 compression settings
- Internal organization of the orientation resolver within the midline module

</decisions>

<specifics>
## Specific Ideas

- Fish are ~10cm long, ~2cm wide from above — eviction thresholds should be calibrated to this scale
- 9 fish in a closed tank — fish count never changes during a run
- OC-SORT Kalman filter provides smoothed velocity estimates (not raw frame-to-frame differences), suitable for head-tail velocity signal
- The rig has 12 active cameras (e3v8250 skipped) with ~4 camera minimum visibility per tank point — 3-camera minimum for reconstruction is well-supported by rig geometry

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 26-association-refinement-and-pipeline-wiring*
*Context gathered: 2026-02-27*
