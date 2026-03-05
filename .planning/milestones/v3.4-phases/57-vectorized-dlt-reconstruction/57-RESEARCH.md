# Phase 57: Vectorized DLT Reconstruction - Research

**Researched:** 2026-03-04
**Domain:** PyTorch batched triangulation within `aquapose.core.reconstruction.backends.dlt`
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Vectorization boundary:**
- Full pipeline batching: ray casting, normal-equation assembly, initial triangulation, reprojection/residuals, outlier rejection, and re-triangulation all vectorized as batch ops across body points
- Batch across body points only — cameras become a stacked tensor dimension within each body point's computation
- Keep `torch.linalg.lstsq` with batched (N, 3, 3) A and (N, 3, 1) b inputs — do not switch to explicit SVD
- Drop the 2-camera ray-angle filter (`_MIN_RAY_ANGLE_DEG` / `_COS_MIN_RAY_ANGLE`) in the vectorized path — rare edge case (2-camera body points are uncommon, and near-parallel rays within those are rarer still). Add a code comment documenting the deliberate omission and rationale.

**Outlier rejection strategy:**
- Masked batch approach: two passes through batched lstsq
  1. First pass: triangulate all body points with all available cameras
  2. Compute per-camera per-body-point reprojection residuals (batch project per camera: C calls instead of N*C)
  3. Build per-point inlier masks based on outlier_threshold
  4. Pre-filter: mask out body points with <2 inlier cameras before second pass
  5. Second pass: re-triangulate with inlier masks applied (zero-weight for outlier cameras)
- Water-surface rejection only after re-triangulation (drop the first-pass water check — above-water initial triangulations are rare and virtually always remain above-water after re-triangulation)

**Fallback & validation:**
- Keep `_triangulate_body_point()` scalar method in the codebase as a reference (no runtime toggle, not called from main path)
- Equivalence tests on both synthetic data (unit tests, CI-friendly) and real YH chunk cache data (integration test)
- Equivalence test documents known differences: 2-camera ray-angle filter dropped, first-pass water-surface check dropped. Test checks these differences are rare and bounded.
- No timing assertions — manual profiling via py-spy or stage timing logs per project convention

**Code structure:**
- Extract vectorized triangulation into `_triangulate_fish_vectorized()` private method on DltBackend
- Returns a structured result object (dataclass/NamedTuple) with: pts_3d (N,3), valid_mask (N,), inlier_masks (N,C), mean_residuals (N,)
- `_reconstruct_fish()` calls `_triangulate_fish_vectorized()` then continues with existing post-processing (half-widths, spline fitting, residuals) unchanged
- Post-triangulation steps (half-width conversion, spline fitting, spline residuals) are NOT vectorized in this phase

### Claude's Discretion

- Exact tensor layout and einsum formulations for normal-equation assembly
- Whether to use a dataclass or NamedTuple for the structured result
- NaN handling strategy in the batched path (masking vs separate valid arrays)
- Exact structure of the equivalence test fixtures

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RECON-01 | DLT first-pass triangulation vectorized across body points via batched SVD (lstsq) | `torch.linalg.lstsq` natively accepts (B, 3, 3) A and (B, 3, 1) b inputs; `RefractiveProjectionModel.cast_ray()` accepts (N, 2) batched pixels; normal-equation assembly `A_i = I - d_i d_i^T` vectorizes via `torch.einsum`; the full two-pass triangulation (first pass + outlier rejection + second pass) can be batched as described in the decisions |
| RECON-02 | Vectorized reconstruction produces numerically equivalent results to per-point loop | Scalar `_triangulate_body_point()` kept as reference; synthetic unit tests compare vectorized vs scalar on same inputs with atol=1e-4m; integration test on real YH chunk cache data verifies unchanged `aquapose eval` metrics; known intentional differences (dropped 2-cam ray-angle filter, dropped first-pass water check) documented and tested to confirm they are rare |
</phase_requirements>

## Summary

Phase 57 replaces the per-body-point Python loop in `DltBackend._reconstruct_fish()` with a single private method `_triangulate_fish_vectorized()` that batches the entire two-pass triangulation (ray casting, normal-equation assembly, lstsq, reprojection, outlier rejection, re-triangulation) across all N body points simultaneously. The existing `_triangulate_body_point()` scalar method is retained as an unreachable reference, not called from the main path.

The vectorized path must handle variable camera availability per body point (NaN pixels exclude a camera for a given point). The natural representation is a (N, C, 2) pixel array with NaN masking, where N = body points and C = cameras. Valid entries drive a (N, C) boolean mask used to zero-weight outlier cameras during the second lstsq pass. `RefractiveProjectionModel.cast_ray()` and `.project()` already accept batched (N, 2) inputs, so ray casting requires only C calls (one per camera over all N body points simultaneously) rather than N*C calls.

Normal-equation assembly (`A = sum_c w_c * (I - d_c d_c^T)`, `b = A_c @ o_c`) can be vectorized with `torch.einsum` to produce (N, 3, 3) A and (N, 3, 1) b tensors in a single pass. `torch.linalg.lstsq` natively supports batched inputs of this shape. The result is `pts_3d` shaped (N, 3). Outlier rejection reads per-camera residuals from a (C, N, 2) reprojection tensor (C calls to `.project()`) and builds an (N, C) inlier mask. The second lstsq pass uses zero-weighted cameras for outliers (equivalent to excluding them). Post-triangulation steps (half-widths, spline fitting, spline residuals) are unchanged.

**Primary recommendation:** Use `torch.einsum('nci,ncj->nij', weighted_dirs, dirs)` for batched outer products in normal-equation assembly; use NaN-masking with `torch.nan_to_num` to convert invalid pixel slots to zero-weight before the first pass; use a dataclass (not NamedTuple) for the structured result so fields can be documented.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | project-pinned | Batched lstsq, einsum, tensor ops | All reconstruction code is torch-based; already imported in dlt.py |
| NumPy | project-pinned | Post-triangulation steps (spline, half-widths) | Used throughout reconstruction; no change to post-processing path |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | project-pinned | Unit tests for vectorized path and equivalence | Existing test infrastructure in `tests/unit/core/reconstruction/` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `torch.linalg.lstsq` batched | explicit SVD (`torch.linalg.svd`) | CONTEXT.md locks lstsq; SVD would require building the system matrix differently and is more complex |
| dataclass for result | NamedTuple | Both work; dataclass allows docstrings per field and `field()` defaults — prefer for clarity |

## Architecture Patterns

### Pattern 1: Batched Normal-Equation Assembly

The scalar per-camera loop in `weighted_triangulate_rays()` (utils.py) and `triangulate_rays()` (calibration/projection.py) accumulates A and b one camera at a time. The vectorized version builds (N, C, 3, 3) per-point-per-camera matrices and sums over the C dimension.

**Concrete tensor layout (N = body points, C = cameras):**

```
pixels_nc:    (N, C, 2)    — NaN where camera c has no valid obs for point n
valid_nc:     (N, C)       — bool, True where pixels_nc is not NaN
weights_nc:   (N, C)       — sqrt(confidence), 0.0 where invalid
origins_nc:   (N, C, 3)    — from C calls to cast_ray, each returning (N, 3)
dirs_nc:      (N, C, 3)    — from C calls to cast_ray
```

Normal equation assembly per body point:
```
M_nc = I[3,3] - einsum('nci,ncj->ncij', dirs_nc, dirs_nc)  # (N, C, 3, 3)
wM_nc = weights_nc[:, :, None, None] * M_nc               # (N, C, 3, 3)
A_n = sum over C: wM_nc                                    # (N, 3, 3)
b_n = sum over C: einsum('ncij,ncj->nci', wM_nc, origins_nc).sum(C)  # (N, 3)
```

Alternatively with explicit loop over C (still O(C) but no Python body-point loop):
```python
A = torch.zeros(N, 3, 3, device=device, dtype=dtype)
b = torch.zeros(N, 3, device=device, dtype=dtype)
eye3 = torch.eye(3, device=device, dtype=dtype)
for cam_idx, cam_id in enumerate(cam_ids):
    d = dirs_nc[:, cam_idx, :]          # (N, 3)
    o = origins_nc[:, cam_idx, :]       # (N, 3)
    w = weights_nc[:, cam_idx]          # (N,)
    # M = I - d d^T per body point
    ddt = torch.einsum('ni,nj->nij', d, d)      # (N, 3, 3)
    M = eye3.unsqueeze(0) - ddt                 # (N, 3, 3)
    wM = w[:, None, None] * M                   # (N, 3, 3)
    A = A + wM
    b = b + torch.einsum('nij,nj->ni', wM, o)  # (N, 3)
```
This outer loop is over C cameras (12 at most), not N body points — acceptable Python overhead.

**Solve:**
```python
result = torch.linalg.lstsq(A, b.unsqueeze(-1))  # A: (N,3,3), b: (N,3,1)
pts_3d = result.solution.squeeze(-1)              # (N, 3)
```

### Pattern 2: Per-Camera Reprojection (C calls, not N*C)

```python
residuals_cn = torch.full((C, N), float('inf'), ...)
for cam_idx, cam_id in enumerate(cam_ids):
    proj_px, valid = models[cam_id].project(pts_3d)  # proj_px: (N, 2), valid: (N,)
    obs_px = pixels_nc[:, cam_idx, :]                 # (N, 2)
    err = torch.linalg.norm(proj_px - obs_px, dim=-1) # (N,)
    residuals_cn[cam_idx] = torch.where(valid & valid_nc[:, cam_idx], err, inf)
```

### Pattern 3: Masked Second Pass

```python
inlier_nc = residuals_cn.T <= outlier_threshold       # (N, C) bool
inlier_count_n = inlier_nc.sum(dim=1)                 # (N,) int
valid_points = inlier_count_n >= 2                    # (N,) — skip points with <2 inliers

# Zero-weight outlier cameras for second lstsq pass
weights_nc_inlier = weights_nc * inlier_nc.float()
# ... rebuild A, b with weights_nc_inlier, solve again for valid_points only
```

### Pattern 4: Structured Result Dataclass

```python
from dataclasses import dataclass

@dataclass
class _TriangulationResult:
    """Result of _triangulate_fish_vectorized()."""
    pts_3d: torch.Tensor        # (N, 3) — NaN for invalid body points
    valid_mask: torch.Tensor    # (N,) bool
    inlier_masks: torch.Tensor  # (N, C) bool — True = inlier camera for that point
    mean_residuals: torch.Tensor  # (N,) — mean inlier reprojection error per point
    inlier_cam_ids: list[list[str]]  # (N,) outer, variable-len inner
```

### Anti-Patterns to Avoid

- **Calling cast_ray in an N-body-point loop**: `cast_ray(pixels)` accepts (N, 2) input. Call it once per camera with all N body points stacked — not once per (body_point, camera) pair.
- **Building (N*C, 3, 3) tensors**: Keep (N, C, 3, 3) or loop over C with (N, 3, 3) accumulation — avoids a large intermediate tensor and keeps the C loop at Python level where C=12.
- **Using `.numpy()` without `.cpu()`**: CLAUDE.md pitfall — always `.detach().cpu().numpy()`.
- **Calling project in a body-point loop**: `project(points)` accepts (N, 3). Call it once per camera over all N points.
- **Assuming NaN propagates through lstsq**: `torch.linalg.lstsq` does not handle NaN inputs cleanly. Must zero-weight invalid cameras before assembly (NaN directions/origins from invalid observations must be replaced with zero-weight contributions, not NaN).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Batched linear solve | Custom SVD decomposition | `torch.linalg.lstsq` batched | Natively supports (B, M, N) inputs; handles near-singular A gracefully |
| Outer product per body point | Manual loop | `torch.einsum('ni,nj->nij', d, d)` | Single fused op; no Python loop |
| Per-camera reprojection loop | Inner (body_point, camera) loop | Loop over C cameras, each call processes N body points | `project()` is already batched over N |

**Key insight:** The existing scalar code has two nested loops (body_points outer, cameras inner). The vectorized path inverts this: cameras become the outer (Python) loop (size C=12), body points become the tensor batch dimension (size N typically 15-30). This eliminates the N-body-point Python loop while keeping the C-camera Python loop, which is acceptable since C is small and fixed.

## Common Pitfalls

### Pitfall 1: NaN in Cast Ray Inputs

**What goes wrong:** Some body points have NaN pixels for certain cameras (missing observations). If NaN pixels are passed to `cast_ray()`, the returned origins and directions will be NaN, corrupting the normal-equation assembly.

**Why it happens:** The scalar path skips NaN body points with an explicit `if np.any(np.isnan(pt)): continue`. The vectorized path stacks all body points including NaN ones.

**How to avoid:** Build `valid_nc` mask before stacking. For each camera, identify rows where `pixels_nc[:, c, :]` has NaN (use `torch.isnan().any(-1)`). Set `weights_nc[:, c] = 0.0` for invalid rows, and replace NaN pixels with zeros before calling `cast_ray()`. Zero-weighted cameras contribute zero to A and b regardless of their origin/direction values.

**Warning signs:** lstsq returning NaN solutions; valid_mask all-False; pts_3d all NaN.

### Pitfall 2: `torch.linalg.lstsq` Driver Selection

**What goes wrong:** `torch.linalg.lstsq` behavior varies by driver (`gels`, `gelsy`, `gelsd`, `gelss`). The default driver on CPU is `gelsd`; on CUDA it may differ. The scalar path uses the same function with no driver argument.

**How to avoid:** Pass no `driver` argument — use the same default as the scalar path. Do not switch to SVD. Document that driver=None is intentional.

**Warning signs:** Numerical differences > 1e-4 m between scalar and vectorized path on the same inputs with no driver mismatch.

### Pitfall 3: TF32 Matmul State

**What goes wrong:** STATE.md flags "Audit TF32 state used when outlier_threshold=10.0 was calibrated." If `torch.backends.cuda.matmul.allow_tf32` differs between the scalar baseline calibration and the vectorized implementation, numerical precision may differ, affecting whether specific body points fall inside/outside the 10.0 px outlier threshold.

**How to avoid:** Verify TF32 state in the YH run metadata before coding. The vectorized tests should use CPU tensors (TF32 is GPU-only) for equivalence validation. Document the TF32 state in the equivalence test.

**Warning signs:** Equivalence test failures on GPU that pass on CPU; eval metrics that differ by a small number of body points near the outlier threshold.

### Pitfall 4: Body Points with Exactly 2 Inlier Cameras

**What goes wrong:** The scalar path applies a ray-angle filter for the 2-camera case and drops the body point if rays are nearly parallel. The vectorized path deliberately drops this filter (CONTEXT.md decision). This means the vectorized path may keep body points that the scalar path drops.

**How to avoid:** The equivalence test must document and account for this known difference. Assert that the total count of differently-handled body points is small (e.g., < 1% of body points on real data). The test should not assert exact set equality of valid body points — only near-equality of 3D positions for points that both paths retain.

**Warning signs:** Equivalence test failing on 2-camera-only observations; unexpected differences in reconstruction for sparse camera frames.

### Pitfall 5: `_reconstruct_fish()` Consumer Expects `list[str]` for Inlier IDs

**What goes wrong:** `_reconstruct_fish()` builds `per_point_inlier_ids: list[list[str]]` from `_triangulate_body_point()` results and passes it to `_convert_half_widths()`. The vectorized result must provide equivalent information (list of inlier camera_id strings per valid body point) for post-processing to work unchanged.

**How to avoid:** The `_TriangulationResult` dataclass should include `inlier_cam_ids: list[list[str]]` computed from the `inlier_masks: (N, C)` bool tensor and the ordered `cam_ids` list. `_reconstruct_fish()` unpacks this to reconstruct the same `per_point_inlier_ids` list it currently builds from `_triangulate_body_point()` calls.

**Warning signs:** `_convert_half_widths()` receiving wrong focal lengths; AttributeError on `inlier_cam_ids`.

## Code Examples

### Batched Normal-Equation Assembly (CPU, dtype=float32)

```python
# Source: analysis of existing triangulate_rays() in calibration/projection.py
# and weighted_triangulate_rays() in reconstruction/utils.py

# cam_ids: list[str] of length C (cameras with at least one valid observation)
# pixels_nc: (N, C, 2) tensor, NaN where camera c has no obs for body point n
# valid_nc: (N, C) bool tensor
# weights_nc: (N, C) float tensor, 0.0 where invalid

# Step 1: Replace NaN pixels with zeros before cast_ray (zero-weight handles them)
pixels_safe = pixels_nc.clone()
pixels_safe[~valid_nc.unsqueeze(-1).expand_as(pixels_safe)] = 0.0

# Step 2: Cast rays — C calls, each processing N body points
origins_nc = torch.zeros(N, C, 3, device=device, dtype=dtype)
dirs_nc = torch.zeros(N, C, 3, device=device, dtype=dtype)
for c, cam_id in enumerate(cam_ids):
    o, d = models[cam_id].cast_ray(pixels_safe[:, c, :])  # (N, 3), (N, 3)
    origins_nc[:, c, :] = o
    dirs_nc[:, c, :] = d

# Step 3: Assemble normal equations — loop over C, accumulate into (N, 3, 3)
eye3 = torch.eye(3, device=device, dtype=dtype)
A = torch.zeros(N, 3, 3, device=device, dtype=dtype)
b = torch.zeros(N, 3, device=device, dtype=dtype)
for c in range(C):
    d = dirs_nc[:, c, :]          # (N, 3)
    o = origins_nc[:, c, :]       # (N, 3)
    w = weights_nc[:, c]          # (N,)
    ddt = torch.einsum('ni,nj->nij', d, d)   # (N, 3, 3)
    M = eye3.unsqueeze(0) - ddt              # (N, 3, 3)
    wM = w[:, None, None] * M               # (N, 3, 3)
    A = A + wM                               # (N, 3, 3)
    b = b + torch.einsum('nij,nj->ni', wM, o)  # (N, 3)

# Step 4: Solve — batched lstsq
result = torch.linalg.lstsq(A, b.unsqueeze(-1))  # (N, 3, 1) solution
pts_3d = result.solution.squeeze(-1)              # (N, 3)
```

### Per-Camera Reprojection in One Pass

```python
# Source: analysis of project() calls in _triangulate_body_point()
inf_val = torch.tensor(float('inf'), device=device, dtype=dtype)
residuals_nc = torch.full((N, C), float('inf'), device=device, dtype=dtype)

for c, cam_id in enumerate(cam_ids):
    proj_px, proj_valid = models[cam_id].project(pts_3d)  # (N,2), (N,)
    obs_px = pixels_nc[:, c, :]                            # (N, 2)
    err = torch.linalg.norm(proj_px - obs_px, dim=-1)      # (N,)
    # Valid only where both projection succeeded and observation existed
    keep = proj_valid & valid_nc[:, c]
    residuals_nc[:, c] = torch.where(keep, err, inf_val)
```

### Building Inlier Masks and Pre-Filtering

```python
inlier_nc = residuals_nc <= outlier_threshold          # (N, C)
inlier_count = inlier_nc.sum(dim=1)                    # (N,) int
point_has_enough = inlier_count >= 2                   # (N,) bool

# Rebuild weights with inlier masking (zero-weight outlier cameras)
weights_nc_inlier = weights_nc * inlier_nc.float()     # (N, C)
# Re-run normal equation assembly for point_has_enough points only
# (can subset or use where(..., 0.0) for excluded points)
```

### Equivalence Test Pattern

```python
# Compare scalar _triangulate_body_point() vs vectorized _triangulate_fish_vectorized()
# on synthetic data where all cameras have valid observations and >=3 cameras.
# Known differences (2-cam ray-angle filter, first-pass water check) are documented.
def test_vectorized_matches_scalar_on_synthetic(dlt_backend, mock_models):
    midline_set = _make_midline_set(n_body_points=15)
    cam_midlines = midline_set[0]
    water_z = 0.0

    # Scalar baseline
    scalar_pts = []
    scalar_valid = []
    for i in range(15):
        result = dlt_backend._triangulate_body_point(i, cam_midlines, water_z)
        if result is not None:
            scalar_pts.append(result[0].detach().cpu().numpy())
            scalar_valid.append(i)

    # Vectorized
    vec_result = dlt_backend._triangulate_fish_vectorized(cam_midlines, water_z)
    vec_valid = [i for i in range(15) if vec_result.valid_mask[i]]
    vec_pts = [vec_result.pts_3d[i].detach().cpu().numpy() for i in vec_valid]

    # Sets should agree (known differences: 2-cam filter, first-pass water check)
    # For synthetic 3-camera data: sets should be identical
    assert set(scalar_valid) == set(vec_valid)
    for i, j in zip(scalar_valid, vec_valid):
        np.testing.assert_allclose(scalar_pts[i], vec_pts[j], atol=1e-4)
```

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Per-body-point Python loop calling `_triangulate_body_point()` | Single `_triangulate_fish_vectorized()` call batching all N body points | Eliminates N-point Python dispatch overhead; ray casting goes from N*C calls to C calls |
| `weighted_triangulate_rays()` loops over cameras with Python `for` | Batched assembly with `torch.einsum` + C-camera Python loop | Camera loop (C=12) replaces body-point loop (N=15-30); same arithmetic, less Python overhead |

## Open Questions

1. **TF32 state during original outlier_threshold calibration**
   - What we know: STATE.md flags this as a research item for Phase 57. `outlier_threshold=10.0` was tuned empirically via `aquapose tune` on YH dataset (2026-03-03).
   - What's unclear: Whether `torch.backends.cuda.matmul.allow_tf32` was True or False during that tuning run. TF32 reduces matmul precision on Ampere+ GPUs, which could affect whether specific near-threshold body points pass or fail the 10.0 px cutoff.
   - Recommendation: Check the YH run metadata (e.g., config, log output from that run) before coding. If TF32 state is unknown, run equivalence tests on CPU only (TF32 is GPU-only). This does not block implementation but should be documented in the equivalence test.

2. **`torch.linalg.lstsq` `driver` default behavior on CUDA vs CPU**
   - What we know: `torch.linalg.lstsq` defaults to `gelsd` on CPU and may use a different path on CUDA.
   - What's unclear: Whether the batched (N, 3, 3) path has the same driver default as the scalar (3, 3) path. Both the scalar and vectorized paths should use no explicit driver argument.
   - Recommendation: Run the equivalence test on CPU where driver is deterministic. Document that CUDA behavior may differ by small floating-point amounts.

3. **Handling body points where ALL cameras return NaN**
   - What we know: The scalar path returns `None` from `_triangulate_body_point()` when `len(cam_ids) < 2`. In the vectorized path, these body points will have `weights_nc[n, :].sum() == 0`, causing A to be zero and lstsq to return an arbitrary or NaN solution.
   - What's unclear: Whether `torch.linalg.lstsq` with an all-zero A matrix raises an error or returns NaN/zero.
   - Recommendation: Use `valid_nc.sum(dim=1) >= 2` as a pre-filter before the first lstsq pass. Body points with fewer than 2 valid cameras should be excluded from the solve (set valid_mask False) and not assigned pts_3d values.

## Validation Architecture

(nyquist_validation not configured — section omitted)

## Sources

### Primary (HIGH confidence)

- Direct code reading: `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/reconstruction/backends/dlt.py` — scalar implementation, full `_triangulate_body_point()` and `_tri_rays()` logic
- Direct code reading: `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/reconstruction/utils.py` — `weighted_triangulate_rays()` scalar loop that the vectorized path replaces
- Direct code reading: `/home/tlancaster6/Projects/AquaPose/src/aquapose/calibration/projection.py` — `RefractiveProjectionModel.cast_ray()` and `.project()` batched signatures
- Direct code reading: `/home/tlancaster6/Projects/AquaPose/tests/unit/core/reconstruction/test_dlt_backend.py` — existing test patterns and `_make_mock_model` / `_make_midline_set` helpers
- PyTorch docs (training knowledge): `torch.linalg.lstsq` batched (B, M, N) support; `torch.einsum` outer product patterns

### Secondary (MEDIUM confidence)

- `.planning/phases/56-vectorized-association-scoring/56-RESEARCH.md` — Phase 56 patterns for torch→numpy conversion, equivalence test structure (same project, adjacent phase)
- `.planning/STATE.md` — TF32 research flag for Phase 57; profiling baseline (reconstruction ~9% of wall time)
- `.planning/phases/57-vectorized-dlt-reconstruction/57-CONTEXT.md` — user decisions on locked choices

### Tertiary (LOW confidence)

None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — same torch/numpy stack already used in dlt.py; no new dependencies
- Architecture patterns: HIGH — derived directly from reading scalar implementation; batched lstsq shape confirmed from torch API knowledge
- Pitfalls: HIGH for NaN handling and consumer interface; MEDIUM for TF32/driver pitfalls (flagged as open questions)

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable torch API; no fast-moving external dependencies)
