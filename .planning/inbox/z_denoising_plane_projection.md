# Z-Denoising via Plane Projection and Temporal Smoothing

## Context

Empirical analysis of 3D reconstruction output (run_20260305_073212, 200 frames, ~8 fish) reveals that z-direction noise dominates the dorsoventral spine shape signal. Key findings:

- **Median z-range along spine: 1.77 cm** (17% of arc length) vs expected ~1 cm biomechanical arch
- **Frame-to-frame z-profile SNR: 0.12-1.25** (noise >= signal for most fish)
- **Noise is spatially smooth** (autocorr lag-1 = 0.92) — body-scale tilt errors, not point-level jitter
- **Endpoints 45% noisier** than body center (1.5 cm vs 1.0 cm std)
- **More cameras help**: 4-cam median z-range 1.49 cm vs 3.63 cm for 2-cam

Root cause: xy-z anisotropy in the refractive triangulation geometry. Small xy errors propagate to large z shifts through Snell's law.

## Motivation as Milestone Prerequisite

Pseudo-label generation (v3.5) relies on reprojecting 3D splines into camera views to create training data. Noisy z-profiles produce subtly wrong reprojections — the projected midline shifts laterally by a few pixels depending on viewing angle and depth. Clean splines are essential for high-quality pseudo-labels.

## Approach: Decomposed Plane Projection + Temporal Smoothing

A fish's spine lies approximately in a bending plane at any instant. Decompose the 3D midline into:

1. **In-plane shape** — undulating curve (well-conditioned, high SNR)
2. **Plane orientation** — heading/pitch/roll (poorly conditioned in z)
3. **Off-plane deviation** — dorsoventral arch (tiny signal, buried in noise)

Handle each with appropriate filtering via two atomic, independently-testable components. A third component (dorsoventral arch recovery) is deferred as future work — planar splines from A+B are sufficient for pseudo-label generation.

### Component A: Per-Frame Plane Fit + Projection

**What:** After triangulation produces N 3D body points, fit a best-fit plane (IRLS-weighted SVD) and project points onto it before spline fitting. Store the plane normal and centroid alongside each Midline3D for use by Component B.

**Where:** `DltBackend._reconstruct_fish()` in `src/aquapose/core/reconstruction/backends/dlt.py`, between triangulation result and `fit_spline()` call. Adds a pre-processing step before spline fitting; does not modify triangulation or spline fitting internals.

**Config:** Controlled by a reconstruction-stage config toggle (e.g., `plane_projection.enabled`). When disabled, the pipeline behaves exactly as before.

**Algorithm:**
1. Take valid triangulated points pts_3d (M x 3)
2. Compute centroid, subtract it
3. Weighted SVD — weights initialized from per-point camera count and/or reprojection residual to down-weight noisy endpoints
4. Plane normal = smallest singular vector
5. Project: `pts_projected = pts - (pts . normal) * normal`
6. Store signed off-plane residuals per body point, plane normal (3 floats), and centroid (3 floats) in Midline3D
7. Pass projected points to existing `fit_spline()`
8. Optionally run 1 IRLS reweighting iteration based on off-plane residuals for robustness to rare misassociated body points

**Why IRLS-SVD over RANSAC:** RANSAC is overkill here. With only 10-15 body points fitting a plane (2 DOF), the consensus set is always large, so RANSAC degenerates into "fit all points with slightly different subsets" — adding variance without value. More importantly, the "outliers" (noisy endpoints) aren't wrong, they're just noisier, and we already know which points are less reliable from the triangulation step (camera counts, reprojection residuals). IRLS-SVD exploits this directly. Deterministic, fast, and uses information already available.

**No hard bypass.** Always project onto the plane. Store signed off-plane residuals per body point — these carry any real out-of-plane structure (e.g., C-start twist) and are recoverable via temporal consistency in Component B. The off-plane RMS serves as a diagnostic/quality metric, not a gate. A hard bypass would remove the constraint exactly when z-noise is highest, and the threshold boundary would create temporal discontinuities.

**Computational cost:** One weighted M x 3 SVD per fish per frame. Negligible.

**Validation gate:** RMS off-plane scatter drops to near zero (trivially true by construction — confirms implementation correctness). Reprojection residuals do not increase by more than ~0.5 px ("do no harm" check — see note below).

> **Note on reprojection residuals:** Reprojection error is a poor metric for z-accuracy in this geometry. The cameras cannot distinguish a spline with 2 cm of z-noise from one with correct z — both reproject to nearly identical pixels. The current spline fitter *overfits to z-noise* because z has low leverage on reprojection, so removing z-freedom may slightly increase residuals (the fitter loses a degree of freedom it was exploiting). This is expected and correct. Residuals serve as a sanity check (large increase = bad plane fit), not an improvement target. The real validation metrics are z-range and temporal stability.

### Component B: Temporal Smoothing of Plane Orientation

**What:** Post-processing pass that smooths plane normals across frames per fish, then rotates spline control points to match the smoothed plane orientation. Operates on the plane normals and centroids stored during Component A.

**Where:** Global post-processing pass after all frames/chunks are reconstructed. Operates on the final `midlines.h5` output — not per-chunk, so no chunk-boundary artifacts. Adds a post-processing step; does not modify reconstruction internals.

**Config:** Controlled by a separate post-processing config toggle (e.g., `plane_smoothing.enabled`, `plane_smoothing.sigma_frames`). Independent of Component A's toggle — A can run without B, but B requires A's stored normals.

**Algorithm:**
1. Collect per-frame plane normals and centroids for each fish (from Component A output)
2. Enforce consistent sign (dot product with previous frame > 0)
3. Gaussian or Savitzky-Golay filter on normal components (sigma ~ 3-5 frames / 100-170 ms)
4. Re-normalize to unit length
5. For each frame, compute the rotation from the per-frame normal to the smoothed normal
6. Rotate control points (relative to centroid) by this rotation
7. Off-plane residuals stored during Component A transform under the same rotation — recomputation is not needed

**Gap handling:** Filter within continuous track segments only. Fish that drop out of reconstruction for one or more frames create segment boundaries; each segment is filtered independently.

**Computational cost:** Filtering three 1D time series per fish, plus one 3x3 rotation per fish per frame. Trivial.

**Failure modes:**
- Fast maneuvers: detect via angular velocity of raw normal, skip/widen filter for those frames
- Track breaks: only smooth within continuous track segments
- Degenerate normals (straight fish in Component A): interpolate through, don't include in filter

**Validation gate:** Median z-range (world coordinates) drops below ~1 cm. Frame-to-frame z-profile RMS change drops from 0.25-0.47 cm to < 0.1 cm. SNR > 1 for most fish.

## Future Work: Dorsoventral Arch Recovery (Component C — Deferred)

Recovery of the ~0.5 cm dorsoventral arch via temporal averaging of off-plane residuals. Not needed for pseudo-label generation — the arch produces at most ~4 px reprojection shift, and planar splines from A+B are sufficient. Revisit if dorsoventral body shape becomes a research objective. The signed off-plane residuals stored by Component A and transformed by Component B provide the raw data needed for this analysis.

## Assumption-Violating Scenarios

| Scenario | Frequency | Handling |
|----------|-----------|---------|
| C-start / escape response | Rare | Robust plane fit absorbs twist; real out-of-plane structure preserved in signed residuals, recovered via temporal consistency in Component B |
| Feeding strike (rapid pitch) | Occasional | Adaptive filter width in Component B |
| Fish near tank wall (fewer cameras) | Common | Weighted plane fit mitigates; fewer cameras already flagged |
| Straight fish (plane under-determined) | ~5-10% of frames | Component A: degenerate SVD detected via singular value ratio; Component B: interpolate normal from neighboring frames |
| Fish near water surface | Occasional | Existing issue; components don't worsen it |
| Bad cross-view association | Rare post-tuning | Plane-fit residual serves as additional quality check |

## Implementation Notes

- Components A and B are additive — neither modifies existing triangulation, RANSAC, or spline fitting logic. A adds a pre-processing step before `fit_spline()`; B adds a global post-processing pass.
- Component A requires extending Midline3D (or a sidecar structure) with plane normal (3 floats), centroid (3 floats), and signed off-plane residuals (N floats). HDF5 writer needs corresponding schema additions.
- Existing unit tests should pass unchanged after Component A (projection is transparent to downstream).
- New tests needed: weighted plane fit correctness, normal sign consistency, rotation of control points, filter edge cases.
- Components A and B have separate config toggles — A runs at reconstruction time, B runs as a post-processing pass. A can be enabled without B; B requires A.

## Data Dependencies

- Analysis run: `~/aquapose/projects/YH/runs/run_20260305_073212/`
- Reconstruction code: `src/aquapose/core/reconstruction/backends/dlt.py`
- Spline fitting: `src/aquapose/core/reconstruction/utils.py`
- Midline3D type: `src/aquapose/core/types/reconstruction.py`
- HDF5 writer: `src/aquapose/io/midline_writer.py`
