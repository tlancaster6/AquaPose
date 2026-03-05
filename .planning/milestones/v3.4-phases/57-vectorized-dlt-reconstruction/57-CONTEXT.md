# Phase 57: Vectorized DLT Reconstruction - Context

**Gathered:** 2026-03-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the per-body-point Python loop in `DltBackend._reconstruct_fish()` with a vectorized path that processes all body points simultaneously via batched `torch.linalg.lstsq`. Produces numerically equivalent 3D midlines (within 1e-4m). Post-triangulation steps (half-widths, spline fitting, residuals) are not vectorized in this phase.

</domain>

<decisions>
## Implementation Decisions

### Vectorization boundary
- Full pipeline batching: ray casting, normal-equation assembly, initial triangulation, reprojection/residuals, outlier rejection, and re-triangulation all vectorized as batch ops across body points
- Batch across body points only — cameras become a stacked tensor dimension within each body point's computation
- Keep `torch.linalg.lstsq` with batched (N, 3, 3) A and (N, 3, 1) b inputs — do not switch to explicit SVD
- Drop the 2-camera ray-angle filter (`_MIN_RAY_ANGLE_DEG` / `_COS_MIN_RAY_ANGLE`) in the vectorized path — rare edge case (2-camera body points are uncommon, and near-parallel rays within those are rarer still). Add a code comment documenting the deliberate omission and rationale.

### Outlier rejection strategy
- Masked batch approach: two passes through batched lstsq
  1. First pass: triangulate all body points with all available cameras
  2. Compute per-camera per-body-point reprojection residuals (batch project per camera: C calls instead of N*C)
  3. Build per-point inlier masks based on outlier_threshold
  4. Pre-filter: mask out body points with <2 inlier cameras before second pass
  5. Second pass: re-triangulate with inlier masks applied (zero-weight for outlier cameras)
- Water-surface rejection only after re-triangulation (drop the first-pass water check — above-water initial triangulations are rare and virtually always remain above-water after re-triangulation)

### Fallback & validation
- Keep `_triangulate_body_point()` scalar method in the codebase as a reference (no runtime toggle, not called from main path)
- Equivalence tests on both synthetic data (unit tests, CI-friendly) and real YH chunk cache data (integration test)
- Equivalence test documents known differences: 2-camera ray-angle filter dropped, first-pass water-surface check dropped. Test checks these differences are rare and bounded.
- No timing assertions — manual profiling via py-spy or stage timing logs per project convention

### Code structure
- Extract vectorized triangulation into `_triangulate_fish_vectorized()` private method on DltBackend
- Returns a structured result object (dataclass/NamedTuple) with: pts_3d (N,3), valid_mask (N,), inlier_masks (N,C), mean_residuals (N,)
- `_reconstruct_fish()` calls `_triangulate_fish_vectorized()` then continues with existing post-processing (half-widths, spline fitting, residuals) unchanged
- Post-triangulation steps (half-width conversion, spline fitting, spline residuals) are NOT vectorized in this phase

### Claude's Discretion
- Exact tensor layout and einsum formulations for normal-equation assembly
- Whether to use a dataclass or NamedTuple for the structured result
- NaN handling strategy in the batched path (masking vs separate valid arrays)
- Exact structure of the equivalence test fixtures

</decisions>

<specifics>
## Specific Ideas

- `RefractiveProjectionModel.cast_ray()` and `.project()` already accept batched inputs — leverage this directly for per-camera batching
- Normal equation assembly `A = I - dd^T` vectorizes cleanly with `torch.einsum` across body points
- The existing `weighted_triangulate_rays()` in utils.py uses a Python loop for the A,b accumulation — the vectorized path should build A,b tensors without Python loops

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `RefractiveProjectionModel.cast_ray(pixels)`: Accepts (N, 2) batched pixels, returns (N, 3) origins and directions — no per-point calls needed
- `RefractiveProjectionModel.project(points_3d)`: Accepts (N, 3) batched 3D points, returns (N, 2) pixel coords + (N,) validity mask
- `torch.linalg.lstsq`: Natively supports batched inputs — (B, 3, 3) @ x = (B, 3, 1) solves all B systems in one call

### Established Patterns
- All torch tensors in reconstruction are float32 on the model's device
- `.detach().cpu().numpy()` for torch-to-numpy conversion (CLAUDE.md pitfall)
- Midline2D.points shape is (N_body_points, 2), point_confidence is (N_body_points,) or None

### Integration Points
- `DltBackend._reconstruct_fish()` is the only caller of `_triangulate_body_point()` — clean replacement point
- `ReconstructionStage` calls `backend.reconstruct_frame()` which calls `_reconstruct_fish()` — no changes needed above DltBackend
- Output types (Midline3D, MidlineSet) are unchanged

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 57-vectorized-dlt-reconstruction*
*Context gathered: 2026-03-04*
