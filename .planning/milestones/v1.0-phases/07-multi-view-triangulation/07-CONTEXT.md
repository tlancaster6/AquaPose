# Phase 7: Multi-View Triangulation - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Triangulate corresponding 2D midline points (from Phase 6) across cameras into 3D positions, then fit a B-spline to produce a continuous 3D midline per fish per frame. Covers RECON-03 (triangulation with outlier rejection), RECON-04 (spline fitting + width profile), and RECON-05 (stub interface only for LM refinement). Does NOT include temporal smoothing, HDF5 output, or visualization — those belong in later phases.

</domain>

<decisions>
## Implementation Decisions

### Triangulation strategy (per body point)
- **Exhaustive pairwise search for ≤7 cameras:** Enumerate all (C choose 2) camera pairs, triangulate each via SVD least-squares (`triangulate_rays`), score each candidate by **max reprojection error** across held-out cameras. Keep the candidate with the lowest max held-out error.
- **Re-triangulate with inliers:** After selecting the best pair, re-triangulate using all cameras whose reprojection error against the winning candidate is acceptable — more cameras = better accuracy.
- **Residual rejection for >7 cameras:** Triangulate using all cameras directly, drop any camera with reprojection error > median + 2σ, re-triangulate once with remaining cameras.
- **No view-angle weighting:** Camera axes are nearly orthogonal to fish body axes most of the time; view-angle weighting adds complexity without benefit.
- **No RANSAC random sampling:** Exhaustive search is deterministic and cheap at typical camera counts (4–6 = 6–15 pairs).

### Spline fitting
- **Fixed 7 control points** for all fish, all frames — consistent output shape, no adaptive complexity.
- **Fixed smoothing parameter** tuned once — predictable, reproducible.
- **Width profile stored as discrete array** of N half-width values at sample points — no separate width spline.
- **Total arc length computed at fit time** — every downstream consumer needs it. Other derived quantities (heading, curvature) computed by consumers as needed.

### LM refinement (RECON-05)
- **Stub interface only** — define the function signature and data flow, implement as no-op pass-through (returns spline unchanged). Pipeline runs end-to-end without modification. Full LM implementation deferred to a future phase if baseline quality proves insufficient.

### Quality signals & fallbacks
- **2-camera fish:** Triangulate anyway using the single pair, flag as low-confidence for downstream filtering.
- **Failed body points** (only 1 camera at that arc-length position): Drop the point and let `splprep` handle the gap — spline naturally interpolates through missing positions.
- **Summary stats only** for quality: store mean + max residual per fish, not per-point arrays.
- **Batched/vectorized** processing across all fish in a frame — not per-fish sequential loops.

### Claude's Discretion
- Inlier threshold (pixel distance) for the re-triangulate-with-inliers step
- `splprep` smoothing parameter value
- Minimum number of successfully-triangulated body points required before attempting spline fit
- Internal batching strategy (how to vectorize across variable camera counts per fish)

</decisions>

<specifics>
## Specific Ideas

- Reuse existing `triangulate_rays()` from `calibration/projection.py` — SVD least-squares, already handles arbitrary ray counts
- Reuse existing `RefractiveProjectionModel.project()` for reprojection scoring
- Phase 5's identity mapping provides the (camera_id, detection_id) → fish_id mapping — consume directly
- Phase 6's `MidlineExtractor` provides N arc-length-sampled 2D points per fish per camera — consume directly
- The pivot doc (`.planning/inbox/fish-reconstruction-pivot.md`) has the full pipeline design for reference

</specifics>

<deferred>
## Deferred Ideas

- LM reprojection refinement (RECON-05 full implementation) — add only if baseline triangulation + spline quality is insufficient
- Temporal smoothing / Kalman filtering of control point trajectories — separate phase
- Epipolar-guided correspondence refinement (if arc-length correspondence proves too noisy on highly curved fish)

</deferred>

---

*Phase: 07-multi-view-triangulation*
*Context gathered: 2026-02-21*
