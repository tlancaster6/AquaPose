# Phase 61: Z-Denoising — Decision Record

## Pivot: Plane Fitting → Z-Flattening (2026-03-05)

### Context

Phase 61 originally implemented two-component z-denoising:
- **Component A**: IRLS-weighted SVD plane fit per fish per frame, project points onto fitted plane
- **Component B**: Temporal Gaussian smoothing of plane normals, Rodrigues rotation of control points

### What We Found

After running the pipeline on YH data (200 frames, 10 fish) and analyzing the diagnostic caches:

1. **Plane fit residuals were tiny** (0.035 cm) — points lay flat on their fitted plane
2. **But z-wiggle persisted** — median z-range 2.03 cm, F2F RMS 0.51 cm
3. **The wiggle came from centroid z drift** (0.61 cm std) and **shape instability** (0.74 cm RMS), not plane orientation noise
4. **Normal smoothing (Component B) had negligible effect** because normal jitter was only 3.5°/frame median
5. **95% of fitted planes were tilted >45° from horizontal** — the SVD was rolling to fit z-noise rather than finding the fish's true body plane
6. **SNR was 0.73** — the z-direction signal (real fish tilt) was overwhelmed by triangulation noise

### The Sign Bug

We also found and fixed a sign ambiguity bug: the `smooth-planes` CLI used uncorrected raw normals as the rotation reference, causing destructive ~72° rotations. This was fixed but didn't change the fundamental problem.

### Lock Horizontal Test

Setting all plane normals to [0, 0, 1] (lock horizontal):
- Z-range: 2.01 cm → **0.00 cm** (all flat)
- F2F z-RMS: 0.51 cm → **0.21 cm** (2.4x better)
- Reprojection cost: only +0.24 px (2.28 → 2.52)

The centroid z jitter (0.21 cm) became the limiting factor.

### Decision

**Strip plane fitting entirely. Replace with z-flattening:**

1. After triangulating body points, set all z-coordinates to centroid z
2. Store raw z-offsets for potential future use
3. Replace plane normal smoothing CLI with centroid z temporal smoothing CLI (`smooth-z`)

**Rationale:** The camera geometry cannot resolve z-direction tilt at the current noise level. A biased-but-stable horizontal assumption outperforms a noisy fitted plane. The ~0.24 px reprojection cost is negligible.

### What Was Removed
- `plane_fit.py` (IRLS-weighted SVD, projection)
- `temporal_smoothing.py` old contents (normal smoothing, Rodrigues rotation)
- Midline3D fields: `plane_normal`, `plane_centroid`, `off_plane_residuals`, `is_degenerate_plane`
- HDF5 datasets: same fields
- Config: `PlaneProjectionConfig`, `PlaneSmoothingConfig`
- CLI: `smooth-planes` command

### What Was Added
- Z-flattening in DLT backend (3 lines)
- `temporal_smoothing.py` rewritten: `smooth_centroid_z` function
- Midline3D fields: `centroid_z`, `z_offsets`
- Config: `ZDenoisingConfig(enabled: bool)`
- CLI: `smooth-z` command (temporal centroid z smoothing)
