# Phase 6: 2D Medial Axis and Arc-Length Sampling - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Extract stable 2D midlines from U-Net segmentation masks and produce fixed-size, arc-length-normalized point correspondences across cameras. This is the 2D input that multi-view triangulation (Stage 3) consumes. Scope covers mask preprocessing, skeletonization, branch pruning, head-to-tail orientation, arc-length resampling, and crop-to-frame coordinate transforms.

Reference document: `.planning/inbox/fish-reconstruction-pivot.md` (Stage 1 + Stage 2) — the implementation should follow this pipeline design closely.

</domain>

<decisions>
## Implementation Decisions

### Midline point count & format
- **15 points** per fish per camera (head=0, tail=14), evenly spaced by normalized arc-length
- **Half-widths included always** — each midline point carries (x, y, half_width)
- Arc-length fractions are **implicit from index** (index/14 gives normalized position)
- Final coordinates in **full-frame pixel space** — transform from crop-space using detection bounding box (scale + translate)

### Head-to-tail orientation
- **Anatomical ordering**: point 0 = snout, point 14 = tail tip
- Use the track's **3D velocity vector** (from Phase 5's `FishTrack`) to determine which skeleton endpoint is the head — the leading edge of motion is the head
- **Ambiguous frames** (near-zero velocity): inherit orientation from previous frame
- **First frame of tracklet**: arbitrary assignment, then **back-correct** early frames once velocity establishes direction
- **Cross-view consistency enforced**: all cameras orient the same way for a given fish on a given frame, driven by the shared 3D velocity

### Skeletonization approach
- **Aggressive morphological smoothing** before skeletonization — closing then opening with larger kernels, appropriate for U-Net masks at ~0.62 IoU
- Adaptive kernel radius proportional to mask minor axis width (per pivot doc: `max(3, minor_axis_width // 8)`)
- Use `skimage.morphology.skeletonize` (morphological thinning), NOT medial axis transform
- **Separate distance transform** (`scipy.ndimage.distance_transform_edt`) on smoothed mask for half-width extraction — sampled at skeleton pixel locations
- **Longest-path BFS pruning** (two-pass): find farthest endpoint from any endpoint, then farthest from that — the path between is the midline. All branches discarded (single path only)

### Edge case handling
- **Masks too small**: skip silently — no midline output for that camera, other cameras compensate
- **Degenerate skeletons** (round mask, skeleton shorter than N points): skip this view entirely
- **Single-camera fish** (only 1 camera sees fish this frame): skip entirely — need ≥2 cameras for triangulation
- **No diagnostic output** from this module — diagnostics via separate scripts if needed

### Claude's Discretion
- Minimum mask area threshold for the "too small" skip
- Exact smoothing kernel sizes and morphological operation sequence
- Implementation of the coordinate transform (scale + translate from crop to frame)
- Internal data structures for midline representation

</decisions>

<specifics>
## Specific Ideas

- The pivot document (`.planning/inbox/fish-reconstruction-pivot.md`) is the authoritative design reference for this phase — Stage 1 (mask preprocessing + medial axis extraction) and Stage 2 (arc-length sampling)
- Back-correction of head-to-tail orientation on early tracklet frames: when writing to disk or passing to downstream consumers, retroactively fix frames that were oriented arbitrarily before velocity was established
- Phase 5's `FishTrack` provides 3D centroid, velocity, and per-camera detection claims — this is the interface for accessing track state

</specifics>

<deferred>
## Deferred Ideas

- **Merged-mask splitting**: overlapping fish producing Y-shaped skeletons — handle upstream in segmentation/instance separation, not in skeletonization. Defer to future phase if it turns out to be common in practice.
- **Epipolar-guided correspondence refinement**: use arc-length as initial guess, refine by finding closest point on each camera's 2D midline to epipolar line from another camera. More robust for highly curved fish but more involved to implement. (Noted in pivot doc as future upgrade.)

</deferred>

---

*Phase: 06-2d-medial-axis-and-arc-length-sampling*
*Context gathered: 2026-02-21*
