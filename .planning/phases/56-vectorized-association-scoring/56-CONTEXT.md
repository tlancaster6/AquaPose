# Phase 56: Vectorized Association Scoring - Context

**Gathered:** 2026-03-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the per-frame Python loop inside `score_tracklet_pair()` with batched NumPy operations using a new `ray_ray_closest_point_batch()` function. The outer pair loop in `score_all_pairs()` stays as-is. Results must be numerically identical to the scalar baseline. Early-termination semantics are preserved.

</domain>

<decisions>
## Implementation Decisions

### Batch granularity
- Vectorize within a single tracklet pair: batch all shared frames into one `cast_ray` + one `ray_ray_closest_point_batch` call per pair
- Outer pair loop in `score_all_pairs()` remains as Python iteration (pair count is small relative to frame count)
- Single batched `cast_ray` call per camera per pair — stack all shared-frame centroids into one (N,2) tensor, call `cast_ray` once (it already accepts (N,2) input)
- Shared-frame identification (set intersection + sorting) stays as Python set ops — not a bottleneck

### Tensor backend
- NumPy for all batched ray-ray distance computation
- Caller converts torch→numpy after `cast_ray` (one `.cpu().numpy()` per batched call, not per frame)
- `ray_ray_closest_point_batch()` accepts pure numpy arrays — no torch dependency in the function itself

### Early termination
- Two-phase approach: batch the first `early_k` frames, check if score_sum == 0, return 0.0 if so; otherwise batch the remaining frames and finish
- Use `ray_ray_closest_point_batch()` for both phases (not the scalar version)
- If `t_shared <= early_k`, do a single batch call for all frames + early check (no special case needed since all frames fit in the early phase)

### API design
- New `ray_ray_closest_point_batch()` function alongside the existing scalar `ray_ray_closest_point()` — scalar version kept for tests, refinement.py, and single-ray use cases
- Batch version returns distances only: `np.ndarray` shape `(N,)` — `score_tracklet_pair` never uses midpoints (always discards them as `dist, _ = ...`)
- Function lives in `scoring.py` next to the scalar version
- `score_tracklet_pair()` public signature unchanged — same inputs, same output, only internal implementation changes
- Add `ray_ray_closest_point_batch` to `__all__` exports

### Claude's Discretion
- Exact NumPy broadcasting implementation (einsum vs element-wise ops)
- Near-parallel ray handling in the batch path
- Test structure for numerical identity validation

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ray_ray_closest_point()` in `scoring.py`: Scalar reference implementation with near-parallel handling — batch version should match its math exactly
- `ForwardLUT.cast_ray()`: Already accepts (N, 2) pixel tensors and returns (N, 3) origins + directions — no API change needed for batched centroids
- `Tracklet2D.centroids`: Numpy array of (x, y) centroids indexed by frame position

### Established Patterns
- `scoring.py` contains both the geometry function and the scoring functions — keep batch version here
- `core/` modules are pure computation with no engine imports (IB-003 import boundary)
- Existing tests in `tests/unit/core/association/test_scoring.py` use mock LUTs with fixed rays

### Integration Points
- `score_tracklet_pair()` is called by `score_all_pairs()` in the same module — only internal caller
- `ray_ray_closest_point()` scalar version is also used by `refinement.py` — must not be removed
- `__init__.py` exports need updating to include `ray_ray_closest_point_batch`

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 56-vectorized-association-scoring*
*Context gathered: 2026-03-04*
