# Phase 56: Vectorized Association Scoring - Research

**Researched:** 2026-03-04
**Domain:** NumPy vectorization of ray-ray geometry within `aquapose.core.association.scoring`
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Batch granularity:**
- Vectorize within a single tracklet pair: batch all shared frames into one `cast_ray` + one `ray_ray_closest_point_batch` call per pair
- Outer pair loop in `score_all_pairs()` remains as Python iteration (pair count is small relative to frame count)
- Single batched `cast_ray` call per camera per pair — stack all shared-frame centroids into one (N,2) tensor, call `cast_ray` once (it already accepts (N,2) input)
- Shared-frame identification (set intersection + sorting) stays as Python set ops — not a bottleneck

**Tensor backend:**
- NumPy for all batched ray-ray distance computation
- Caller converts torch→numpy after `cast_ray` (one `.cpu().numpy()` per batched call, not per frame)
- `ray_ray_closest_point_batch()` accepts pure numpy arrays — no torch dependency in the function itself

**Early termination:**
- Two-phase approach: batch the first `early_k` frames, check if score_sum == 0, return 0.0 if so; otherwise batch the remaining frames and finish
- Use `ray_ray_closest_point_batch()` for both phases (not the scalar version)
- If `t_shared <= early_k`, do a single batch call for all frames + early check (no special case needed since all frames fit in the early phase)

**API design:**
- New `ray_ray_closest_point_batch()` function alongside the existing scalar `ray_ray_closest_point()` — scalar version kept for tests, refinement.py, and single-ray use cases
- Batch version returns distances only: `np.ndarray` shape `(N,)` — `score_tracklet_pair` never uses midpoints
- Function lives in `scoring.py` next to the scalar version
- `score_tracklet_pair()` public signature unchanged — same inputs, same output, only internal implementation changes
- Add `ray_ray_closest_point_batch` to `__all__` exports

### Claude's Discretion

- Exact NumPy broadcasting implementation (einsum vs element-wise ops)
- Near-parallel ray handling in the batch path
- Test structure for numerical identity validation

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ASSOC-01 | Pairwise ray scoring vectorized via NumPy broadcasting | `ray_ray_closest_point_batch()` replaces per-frame scalar calls inside `score_tracklet_pair()`; `cast_ray` already accepts (N,2) input enabling a single batched call per camera per pair |
| ASSOC-02 | Vectorized scoring produces identical results to per-pair loop | Scalar math in `ray_ray_closest_point()` is standard linear algebra; batch form via broadcasting is algebraically equivalent; near-parallel threshold (1e-12) must be applied element-wise; numerical identity verified by parametric test comparing both paths on same inputs |
</phase_requirements>

## Summary

Phase 56 is a targeted performance optimization that replaces the per-frame Python loop inside `score_tracklet_pair()` with a single batched NumPy call per tracklet pair. The outer loop over camera pairs and tracklet combinations in `score_all_pairs()` is unchanged. The only new artifact is `ray_ray_closest_point_batch()` — a pure-NumPy function that accepts stacked ray arrays and returns a distance vector via broadcasting, matching the scalar math of the existing `ray_ray_closest_point()` exactly.

The change is localized to `scoring.py`. No other modules are affected. `refinement.py` continues using the scalar `ray_ray_closest_point()`. The `ForwardLUT.cast_ray()` signature already accepts (N,2) pixel tensors and returns (N,3) origins and directions, so no changes are needed to the calibration layer. The two-phase early termination (early_k frames, then remaining frames) maps cleanly to two batched calls.

The primary risk is numerical non-identity from floating-point ordering differences between the scalar loop and a batch broadcast. This is mitigated by keeping the same formula and using identical dtype (float32 from `cast_ray` output, float64 during numpy computation as in the scalar path). A parametric test comparing scalar and batch output on the same synthetic rays is the key verification artifact.

**Primary recommendation:** Implement `ray_ray_closest_point_batch()` using element-wise broadcasting (not `einsum`) for readability, matching the scalar formula term-by-term; the test for ASSOC-02 should run both paths on random (non-parallel) rays and assert `np.allclose(batch_dists, scalar_dists, atol=1e-6)`.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| NumPy | project-pinned | Batched ray-ray distance computation | Already used throughout `scoring.py`; no new dependency |
| PyTorch | project-pinned | `cast_ray` returns torch tensors; convert once with `.cpu().numpy()` | Already imported in `scoring.py` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | project-pinned | Parametric numerical identity tests | All new tests; existing test infrastructure in `tests/unit/core/association/test_scoring.py` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Element-wise broadcasting | `np.einsum` | `einsum` is more compact for dot products but harder to read against scalar reference; broadcasting preferred for clarity |
| Two-phase early termination | Single batch then check | Single batch would compute all N frames even when score_sum is 0 after early_k — user decision locked to two-phase |

**Installation:** No new packages required.

## Architecture Patterns

### Recommended Project Structure

No structural changes. All changes are within:

```
src/aquapose/core/association/
├── scoring.py    # Add ray_ray_closest_point_batch(); rewrite score_tracklet_pair() internals
└── __init__.py   # Add ray_ray_closest_point_batch to imports and __all__
tests/unit/core/association/
└── test_scoring.py  # Add tests for batch function and numerical identity
```

### Pattern 1: Scalar-to-Batch Formula Mapping

**What:** The scalar `ray_ray_closest_point()` uses dot products and a 2x2 linear system. The batch version stacks N ray pairs and computes the same quantities with broadcasting.

**When to use:** Any time a scalar O(N) loop over independent computations shares the same formula for each iteration.

**Scalar reference (existing code):**
```python
# Source: src/aquapose/core/association/scoring.py
w0 = origin_a - origin_b                   # (3,)
a = float(np.dot(dir_a, dir_a))
b = float(np.dot(dir_a, dir_b))
c = float(np.dot(dir_b, dir_b))
d = float(np.dot(dir_a, w0))
e = float(np.dot(dir_b, w0))
denom = a * c - b * b
# near-parallel guard: abs(denom) < 1e-12
t_a = (b * e - c * d) / denom
s_b = (a * e - b * d) / denom
pt_a = origin_a + t_a * dir_a
pt_b = origin_b + s_b * dir_b
dist = float(np.linalg.norm(pt_a - pt_b))
```

**Batch equivalent (element-wise broadcasting):**
```python
def ray_ray_closest_point_batch(
    origins_a: np.ndarray,   # (N, 3)
    dirs_a: np.ndarray,      # (N, 3)
    origins_b: np.ndarray,   # (N, 3)
    dirs_b: np.ndarray,      # (N, 3)
) -> np.ndarray:             # (N,) distances
    w0 = origins_a - origins_b                                 # (N, 3)
    a = (dirs_a * dirs_a).sum(axis=1)                         # (N,)
    b = (dirs_a * dirs_b).sum(axis=1)                         # (N,)
    c = (dirs_b * dirs_b).sum(axis=1)                         # (N,)
    d = (dirs_a * w0).sum(axis=1)                             # (N,)
    e = (dirs_b * w0).sum(axis=1)                             # (N,)
    denom = a * c - b * b                                      # (N,)

    parallel_mask = np.abs(denom) < 1e-12                    # (N,) bool

    # Default: near-parallel fallback (t_a=0, s_b=e/c)
    s_b_parallel = np.where(np.abs(c) > 1e-12, e / c, 0.0)
    pt_a_parallel = origins_a + 0.0 * dirs_a                  # origins_a (t_a=0)
    pt_b_parallel = origins_b + s_b_parallel[:, None] * dirs_b
    diff_parallel = pt_a_parallel - pt_b_parallel
    dist_parallel = np.linalg.norm(diff_parallel, axis=1)    # (N,)

    # General case
    safe_denom = np.where(parallel_mask, 1.0, denom)          # avoid /0
    t_a = (b * e - c * d) / safe_denom
    s_b = (a * e - b * d) / safe_denom
    pt_a = origins_a + t_a[:, None] * dirs_a
    pt_b = origins_b + s_b[:, None] * dirs_b
    diff = pt_a - pt_b
    dist_general = np.linalg.norm(diff, axis=1)               # (N,)

    return np.where(parallel_mask, dist_parallel, dist_general)
```

### Pattern 2: Two-Phase Early Termination with Batch Calls

**What:** Split shared_frames into early_frames (first early_k) and remaining_frames. Batch-cast and score early_frames first. If score_sum == 0 after early check, return 0.0 without processing the rest.

**When to use:** When t_shared > early_k — the common case for longer tracklets.

**Example:**
```python
# Phase 1: early_k frames
early_frames = shared_frames[:config.early_k]
remaining_frames = shared_frames[config.early_k:]

# Stack centroids for cam_a and cam_b
cents_a = np.array([tracklet_a.centroids[frames_a[f]] for f in early_frames])
cents_b = np.array([tracklet_b.centroids[frames_b[f]] for f in early_frames])

pix_a = torch.tensor(cents_a, dtype=torch.float32)
pix_b = torch.tensor(cents_b, dtype=torch.float32)

origins_a, dirs_a = lut_a.cast_ray(pix_a)
origins_b, dirs_b = lut_b.cast_ray(pix_b)

dists = ray_ray_closest_point_batch(
    origins_a.cpu().numpy(), dirs_a.cpu().numpy(),
    origins_b.cpu().numpy(), dirs_b.cpu().numpy(),
)

# Soft kernel, vectorized
inlier_mask = dists < config.ray_distance_threshold
score_sum = np.sum(np.where(inlier_mask, 1.0 - dists / config.ray_distance_threshold, 0.0))

# Early termination check
if score_sum == 0.0:
    return 0.0

# Phase 2: remaining frames (if any)
if remaining_frames:
    # same pattern for remaining_frames
    ...
    score_sum += np.sum(np.where(inlier_mask2, ...))

f = score_sum / t_shared
w = min(t_shared, effective_saturate) / effective_saturate
return f * w
```

### Pattern 3: Single-Phase Shortcut When t_shared <= early_k

**What:** If all shared frames fit within the early window, do a single batch call, apply soft kernel, and perform the early check before computing the final score. No second batch call needed.

**Example:**
```python
if t_shared <= config.early_k:
    # Single batch for all frames; early check still applies
    dists = ray_ray_closest_point_batch(...)
    score_sum = np.sum(...)
    if score_sum == 0.0:
        return 0.0
    f = score_sum / t_shared
    w = min(t_shared, effective_saturate) / effective_saturate
    return f * w
```

### Anti-Patterns to Avoid

- **Per-frame `torch.tensor()` construction:** The current code wraps a single centroid in a tensor per frame — the batch version must stack all centroids at once before the single `cast_ray` call.
- **Calling the scalar `ray_ray_closest_point()` from the batch path:** The batch function is a full replacement; never call scalar inside a loop in `score_tracklet_pair`.
- **Importing torch in `ray_ray_closest_point_batch()`:** The batch function must be pure NumPy (no torch dependency), per the locked API design decision. The caller converts torch tensors to numpy.
- **Applying the soft kernel via a Python loop:** After `ray_ray_closest_point_batch()` returns distances, use `np.where` and `np.sum` — not a loop — for the inlier aggregation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Batch dot products (a, b, c, d, e) | Custom dot loop | `(arr * arr).sum(axis=1)` | Standard NumPy reduction; clear correspondence to scalar `np.dot()` |
| Near-parallel masking | Manual for-loop over N | `np.where(parallel_mask, ...)` | Handles all N elements without branching |
| Distance computation | Custom norm loop | `np.linalg.norm(diff, axis=1)` | Vectorized Euclidean norm along last axis |
| Soft kernel accumulation | Python sum over N | `np.where(inlier_mask, contribution, 0.0).sum()` | Single vectorized reduction |

**Key insight:** Every scalar operation in `ray_ray_closest_point()` has a direct broadcasting equivalent; the formula maps 1-to-1 with axis=1 reductions replacing scalar dot products.

## Common Pitfalls

### Pitfall 1: Floating-Point Non-Identity from dtype Promotion

**What goes wrong:** `cast_ray` returns float32 tensors. If the batch function does arithmetic in float32 while the scalar path uses float64 (Python floats from `float(np.dot(...))`), results will differ beyond `atol=1e-6`.

**Why it happens:** The scalar code wraps dot products in `float()`, promoting to float64 for the arithmetic. Batch arrays from `cast_ray.cpu().numpy()` remain float32 unless explicitly cast.

**How to avoid:** Cast input arrays to float64 inside `ray_ray_closest_point_batch()` (or accept float64-promoted input from the caller). The scalar baseline uses float64; the batch function should match.

**Warning signs:** `np.allclose(batch_dists, scalar_dists, atol=1e-6)` fails on skew rays but passes on intersecting rays (intersecting rays have distance ~0, masking precision differences).

### Pitfall 2: Early Termination Applied to Wrong Frame Count

**What goes wrong:** Early termination triggers when `frame_idx_in_shared == config.early_k - 1`. In the batch path, the early check happens after scoring the first `min(early_k, t_shared)` frames. If `t_shared < early_k`, the check must still run on whatever frames exist — don't skip it.

**Why it happens:** The original loop naturally handles `t_shared < early_k` because the loop ends before reaching `early_k - 1`. A two-phase split must explicitly handle this edge case.

**How to avoid:** Use the single-phase path (Pattern 3) when `t_shared <= early_k` — one batch call, one early check, then compute final score.

**Warning signs:** Test with `early_k=5` and `t_shared=3` (divergent rays) — should return 0.0 (early termination), but a broken implementation might return a small positive score if the early check is skipped.

### Pitfall 3: Centroid Tuple Access

**What goes wrong:** `Tracklet2D.centroids` is typed as `tuple` (tuple of tuple[float, float]). Converting to a numpy array requires `np.array([tracklet.centroids[i] for i in indices])`, not `np.array(tracklet.centroids)[indices]` — the latter works but requires creating the full array even for the early-phase subset.

**Why it happens:** Tuples don't support fancy indexing; you must loop or slice to select by list of indices.

**How to avoid:** Use `[frames_a[f] for f in batch_frames]` index list, then list-comprehension into `np.array`.

**Warning signs:** IndexError or TypeError when constructing centroid arrays for the early phase.

### Pitfall 4: `score_sum == 0.0` Floating-Point Comparison

**What goes wrong:** After the batched soft kernel, `score_sum` is a numpy float64 scalar. The comparison `score_sum == 0.0` is safe when all distances exceed the threshold (contributions are exactly 0.0), but could theoretically fail due to tiny rounding errors near zero.

**Why it happens:** The original scalar code also uses `score_sum == 0.0` (it's an exact comparison), which works because contributions are either exactly 0 (excluded by `if dist < threshold`) or strictly positive.

**How to avoid:** Use `np.where(inlier_mask, contribution, 0.0)` — contributions that miss the threshold are exactly 0.0 (not a small float), preserving the exact comparison semantics. Do not use a tolerance check here; the original semantics are intentional.

### Pitfall 5: __all__ and __init__.py Not Updated

**What goes wrong:** `ray_ray_closest_point_batch` is importable from `scoring.py` but missing from `scoring.__all__` and/or `association/__init__.py`, violating the project's public API rules (see `.claude/rules/source-code.md`).

**Why it happens:** Forgetting to update both `__all__` in the module and the re-export in `__init__.py`.

**How to avoid:** Add `"ray_ray_closest_point_batch"` to `scoring.__all__`, add the import in `association/__init__.py`, and add to `association.__all__`. The CONTEXT.md explicitly locks this requirement.

## Code Examples

### Batch Centroid Stacking from Tracklet Tuple

```python
# Source: pattern derived from existing score_tracklet_pair() in scoring.py
# Centroids are tuple[tuple[float, float], ...]; convert batch to (N, 2) array
early_frames = shared_frames[:config.early_k]
idx_a = [frames_a[f] for f in early_frames]
idx_b = [frames_b[f] for f in early_frames]

cents_a = np.array([tracklet_a.centroids[i] for i in idx_a], dtype=np.float64)  # (N, 2)
cents_b = np.array([tracklet_b.centroids[i] for i in idx_b], dtype=np.float64)  # (N, 2)

pix_a = torch.tensor(cents_a, dtype=torch.float32)  # cast_ray expects float32
pix_b = torch.tensor(cents_b, dtype=torch.float32)
```

### Single cast_ray Call Per Camera Per Phase

```python
# Source: ForwardLUT.cast_ray() accepts (N, 2) -> returns (N, 3), (N, 3)
origins_a, dirs_a = lut_a.cast_ray(pix_a)   # one call, not N calls
origins_b, dirs_b = lut_b.cast_ray(pix_b)

# Convert once per batch, not once per frame
oa = origins_a.cpu().numpy().astype(np.float64)  # (N, 3)
da = dirs_a.cpu().numpy().astype(np.float64)      # (N, 3)
ob = origins_b.cpu().numpy().astype(np.float64)   # (N, 3)
db = dirs_b.cpu().numpy().astype(np.float64)      # (N, 3)
```

### Batched Soft Kernel (vectorized)

```python
# Source: derived from scalar soft kernel in score_tracklet_pair()
# Original scalar: if dist < threshold: score_sum += 1.0 - dist/threshold
dists = ray_ray_closest_point_batch(oa, da, ob, db)   # (N,) float64
inlier = dists < config.ray_distance_threshold
contributions = np.where(inlier, 1.0 - dists / config.ray_distance_threshold, 0.0)
score_sum = float(contributions.sum())
```

### Numerical Identity Test Pattern

```python
# Test structure for ASSOC-02 validation
import numpy as np
import pytest
from aquapose.core.association.scoring import ray_ray_closest_point, ray_ray_closest_point_batch

@pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
def test_batch_identical_to_scalar(seed: int) -> None:
    rng = np.random.default_rng(seed)
    N = 20
    origins_a = rng.standard_normal((N, 3))
    dirs_a = rng.standard_normal((N, 3))
    dirs_a /= np.linalg.norm(dirs_a, axis=1, keepdims=True)
    origins_b = rng.standard_normal((N, 3))
    dirs_b = rng.standard_normal((N, 3))
    dirs_b /= np.linalg.norm(dirs_b, axis=1, keepdims=True)

    batch_dists = ray_ray_closest_point_batch(origins_a, dirs_a, origins_b, dirs_b)
    scalar_dists = np.array([
        ray_ray_closest_point(origins_a[i], dirs_a[i], origins_b[i], dirs_b[i])[0]
        for i in range(N)
    ])

    np.testing.assert_allclose(batch_dists, scalar_dists, atol=1e-6,
        err_msg=f"Batch and scalar disagree (seed={seed})")
```

## Open Questions

1. **float64 vs float32 casting in `ray_ray_closest_point_batch()`**
   - What we know: `cast_ray` returns float32; scalar code promotes to float64 via `float(np.dot(...))`. The batch version could cast inside the function or expect the caller to cast.
   - What's unclear: Whether float32 batch arithmetic passes `atol=1e-6` against float64 scalar baseline, or whether explicit float64 cast is needed.
   - Recommendation: Cast to float64 inside `ray_ray_closest_point_batch()` to match scalar precision. If tests pass without it, can relax; but start conservative.

2. **Score identity on real YH data (success criterion 2)**
   - What we know: The phase requires `aquapose eval` association metrics to be identical before and after on a real YH chunk. The runner is in `src/aquapose/evaluation/runner.py` (modified per git status).
   - What's unclear: Whether this eval comparison is automated (part of a CI test) or a manual step before merge.
   - Recommendation: Treat as a manual verification step in the plan; the planner should include it as a final "smoke test" task using a known YH chunk from the Data Paths in MEMORY.md.

## Sources

### Primary (HIGH confidence)

- Direct code inspection: `src/aquapose/core/association/scoring.py` — complete implementation of scalar baseline and function signatures
- Direct code inspection: `src/aquapose/calibration/luts.py::ForwardLUT.cast_ray()` — confirmed (N,2) input / (N,3) output; torch tensors
- Direct code inspection: `tests/unit/core/association/test_scoring.py` — existing test infrastructure, mock LUT pattern
- Direct code inspection: `src/aquapose/core/tracking/types.py` — `Tracklet2D.centroids` is `tuple` of `tuple[float, float]`
- Direct code inspection: `src/aquapose/core/association/refinement.py` — confirms scalar `ray_ray_closest_point` usage that must be preserved
- Direct code inspection: `.planning/phases/56-vectorized-association-scoring/56-CONTEXT.md` — all locked decisions

### Secondary (MEDIUM confidence)

None required — all implementation details derived from direct codebase inspection.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; NumPy broadcasting is well-understood
- Architecture: HIGH — existing module structure is clear; changes are localized to `scoring.py` and its exports
- Pitfalls: HIGH — derived from direct inspection of scalar code, type annotations, and existing test patterns; not from web search

**Research date:** 2026-03-04
**Valid until:** Stable — no third-party library changes; valid until scoring.py is refactored
