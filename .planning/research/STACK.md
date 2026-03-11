# Technology Stack

**Project:** AquaPose v3.8 Improved Association
**Researched:** 2026-03-11
**Scope:** Stack additions for multi-keypoint association scoring, temporal changepoint detection, and singleton recovery
**Confidence:** HIGH

## Key Finding: No New Runtime Dependencies Required

Every feature in the v3.8 milestone can be implemented using libraries already in the dependency set. The changepoint detection problem as specified in the design document — find the single split point that maximizes the difference in mean residuals between two halves — is a max-sweep over a precomputed residual array. This is two lines of NumPy. A library like `ruptures` would be overkill.

**Decision rationale is documented below in the changepoint section.** The rest of the stack changes are pure refactoring of existing NumPy vectorization patterns.

---

## Recommended Stack (Existing — No Changes to pyproject.toml)

### Multi-Keypoint Scoring: NumPy Broadcasting

| Technology | Version | Purpose in v3.8 | Why Sufficient |
|------------|---------|-----------------|----------------|
| NumPy | >=1.24 | Per-keypoint ray casting aggregation, confidence masking, soft-kernel computation | `np.where`, `np.nanmean`, `np.isnan` cover all confidence-filtered aggregation needs. Already used in `_batch_score_frames`. |
| PyTorch | >=2.0 | `ForwardLUT.cast_ray()` — takes `(N, 2)` pixel tensor, returns origins+dirs | Already used in scoring.py. Multi-keypoint extends N from 1 to K per frame. No API changes needed. |

**How multi-keypoint scoring extends the existing pattern:**

Current `_batch_score_frames` stacks one centroid per frame into shape `(T, 2)` then calls `cast_ray` once per camera. Multi-keypoint extends this to shape `(T*K, 2)` — still one `cast_ray` call per camera per batch. The result is reshaped to `(T, K)` distances, then aggregated across the K dimension using `np.nanmean` (NaN for filtered low-confidence keypoints). The existing `ray_ray_closest_point_batch` function handles the `(N,)` input directly — no changes needed to the ray-ray math.

**Confidence filtering with NaN masking:**

```python
# conf_a, conf_b: shape (T, K), confidence per keypoint per frame
# dists: shape (T, K), ray-ray distances
mask = (conf_a < conf_threshold) | (conf_b < conf_threshold)
dists_masked = np.where(mask, np.nan, dists)
per_frame_score = np.nanmean(soft_kernel(dists_masked), axis=1)  # (T,)
```

`np.nanmean` ignores NaN entries, which is exactly the behavior needed: frames where all keypoints are low-confidence produce NaN (treated as 0 contribution downstream). This is the idiomatic NumPy pattern for masked aggregation without boolean indexing overhead.

**Tracklet2D must gain a keypoints field.** The current type only stores `centroids`. Multi-keypoint scoring needs per-frame keypoint positions `(K, 2)` and confidences `(K,)`. This is a type change in `core/tracking/types.py` and the corresponding `_TrackletBuilder.to_tracklet2d()` in `keypoint_tracker.py` — the tracker already accumulates `builder.keypoints` but does not store it on `Tracklet2D`. This is the primary data contract change for the milestone.

---

### Changepoint Detection: NumPy Max-Split Sweep (No Library)

**Decision: Do not add `ruptures` or any changepoint library.**

#### Why not `ruptures` 1.1.10

`ruptures` is a well-maintained library (latest: 1.1.10, Sept 2025, Python 3.9–3.13) with good algorithms for offline changepoint detection: PELT (exact, penalized), Binseg (O(n log n), sequential), BottomUp (hierarchical), and others. It handles multivariate signals natively. For general-purpose changepoint work it is a solid choice.

However, the design document specifies exactly what is needed, and it does not match `ruptures`' strength:

| Criterion | `ruptures` | Max-split sweep |
|-----------|-----------|-----------------|
| API surface | Model-fit + predict, penalty tuning, cost function selection | 10 lines of NumPy |
| Penalty parameter | Required for PELT; Binseg needs explicit `n_bkps` | Not needed — threshold-based |
| Signal structure | Assumes stationary noise within segments | Residual series has non-stationary noise; mean shift is what matters |
| Sequence length | Optimized for long signals; short signals (50–300 points) work but incur import overhead | Trivially fast at any length |
| Dependency cost | New runtime dependency (adds to install size, transitive numpy+scipy already present) | Zero |
| Recursive splitting | Supported via Binseg recursion | Recursion is 3 lines of Python |
| Significance threshold | Needs penalty calibration per signal | Simple absolute threshold on mean difference |

The max-split sweep is:

```python
def find_changepoint(residuals: np.ndarray, min_seg: int) -> int | None:
    """Return split index t* maximizing |mean(residuals[:t]) - mean(residuals[t:])|.

    Returns None if the best split delta is below threshold or sequence is
    too short to meet min_seg constraint on both halves.
    """
    n = len(residuals)
    if n < 2 * min_seg:
        return None
    # Vectorized cumulative mean difference over all valid split points
    cumsum = np.cumsum(residuals)
    t_range = np.arange(min_seg, n - min_seg + 1)  # valid split indices
    mean_left = cumsum[t_range - 1] / t_range
    mean_right = (cumsum[-1] - cumsum[t_range - 1]) / (n - t_range)
    deltas = np.abs(mean_left - mean_right)
    best_t = int(t_range[np.argmax(deltas)])
    return best_t  # caller checks delta against threshold
```

This is O(n) per call, handles minimum segment length constraints, and returns exactly one split index — precisely what the design document calls for. Recursive application handles multiple swaps.

**The design document explicitly states:** "The detection method is simple: find the split point that maximizes the difference in mean residual between the two halves, subject to a minimum segment length and a significance threshold." This description *is* the implementation.

#### Significance threshold without calibration

`ruptures` PELT requires a penalty value that must be calibrated to the signal's noise level. The design document's approach avoids this: the threshold is an absolute delta in mean residual (metres), directly comparable to the existing `eviction_reproj_threshold`. A swap that moves a fish 3cm produces a mean-residual jump of ~0.03m. Setting `changepoint_delta_threshold=0.01` (1cm) catches real swaps and rejects noise. This is calibratable against real data without a library.

---

### Group Validation and Singleton Recovery: Existing Stack

| Technology | Purpose | What's Used |
|------------|---------|-------------|
| NumPy | Per-frame residual computation against group consensus | `np.median`, `np.mean`, `np.cumsum`, `np.argmax` |
| PyTorch / ForwardLUT | Ray casting for group validation rays | Same `.cast_ray()` call as scoring |
| Python dataclasses | Config fields for new thresholds | `changepoint_delta_threshold`, `min_changepoint_segment`, `singleton_recovery_enabled` on `AssociationConfig` |

**No new scipy functions are needed.** The residual computations are mean/median over short arrays (50–300 values). `scipy.stats.ttest_ind` was considered for significance testing but rejected: the design document specifies a deterministic threshold on mean delta, not a statistical test. A t-test p-value would require calibrating alpha and would be sensitive to the assumed distribution of residuals — more complexity for no practical benefit at this signal scale.

---

### What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `ruptures` | Penalty calibration required; overkill for a single mean-split sweep | `np.cumsum` + `np.argmax` (10 lines, already shown above) |
| `bayesian_changepoint_detection` | Bayesian inference overhead; not maintained as a first-class package | Same max-split sweep |
| `changepy` | Last meaningful commit 2017; not maintained | Same max-split sweep |
| `scipy.stats.ttest_ind` for swap detection | Requires distributional assumptions; calibrating alpha adds a free parameter | Absolute delta threshold on mean residuals |
| `pandas` | Tabular aggregation of per-keypoint scores | NumPy structured arrays or plain dicts |
| Any new dependency | The milestone is a scoring rework, not a new domain | All required operations are in the existing stack |

---

## Data Contract Change: Tracklet2D Extended Fields

This is the one structural change the milestone requires — it is a type change, not a dependency change.

**Current `Tracklet2D`** (in `core/tracking/types.py`):
```python
frames: tuple[int, ...]
centroids: tuple[tuple[float, float], ...]
bboxes: tuple[tuple[float, float, float, float], ...]
frame_status: tuple[str, ...]
```

**Required addition for multi-keypoint scoring:**
```python
keypoints: tuple[np.ndarray, ...]   # per-frame (K, 2) float32 arrays
kpt_confs: tuple[np.ndarray, ...]   # per-frame (K,) float32 confidence arrays
```

`_TrackletBuilder` in `keypoint_tracker.py` already accumulates `builder.keypoints` (a list of `(6, 2)` arrays) and `builder.keypoint_conf` (a list of `(6,)` arrays). These are discarded in `to_tracklet2d()`. The fix is a one-line addition per field. The tracker already has the data — it just doesn't pass it through.

**Downstream impact:** `AssociationStage` must read the new fields. All other consumers of `Tracklet2D` (`ClusteringStage`, `ReconstructionStage`, etc.) are unaffected — they only read `centroids` and `frames`.

---

## Integration Points

### 1. `core/tracking/types.py` — Add keypoints/kpt_confs to Tracklet2D

One frozen dataclass field addition per new attribute. The `to_tracklet2d()` conversion in `keypoint_tracker.py` is updated to pass through the accumulated keypoint data.

### 2. `core/association/scoring.py` — Extend `_batch_score_frames`

Replace centroid-only cast with K-keypoint cast. The reshaped input `(T*K, 2)` → `cast_ray` → reshape output to `(T, K)` → apply confidence mask as NaN → `np.nanmean` over K → existing soft kernel over T. The `AssociationConfigLike` protocol gains `conf_threshold: float` and `min_keypoints_per_frame: int`.

### 3. `core/association/refinement.py` — Replace with group_validation.py

The current `refine_clusters` function uses single-centroid consensus. The v3.8 replacement computes multi-keypoint per-frame residuals, runs the changepoint sweep on each tracklet's residual series, splits swapped tracklets, and evicts outliers. The file can be renamed `group_validation.py` or `validation.py` to signal the functional change.

### 4. `core/association/singleton_recovery.py` — New module

Fresh computation: for each singleton, compute per-frame residuals against all existing groups, sweep split points, assign or split. No stored scoring data is reused.

### 5. `engine/config.py` — New config fields on AssociationConfig

```python
conf_threshold: float = 0.3          # min keypoint confidence to include in ray
min_keypoints_per_frame: int = 2     # min valid keypoints for frame to contribute
changepoint_delta_threshold: float = 0.015  # min mean residual delta (m) to split
min_changepoint_segment: int = 10    # min frames per segment after split
singleton_recovery_enabled: bool = True
singleton_split_min_segment: int = 15  # min segment for singleton swap split
```

---

## Installation

No changes to `pyproject.toml`. All features use existing packages:

```toml
# pyproject.toml — NO CHANGES NEEDED
dependencies = [
    "numpy>=1.24",   # nanmean, cumsum, argmax, where
    "torch>=2.0",    # ForwardLUT.cast_ray
    # ... all other existing deps unchanged
]
```

---

## Version Verification

| Library | Required Feature | Current Constraint | Confidence |
|---------|-----------------|-------------------|------------|
| NumPy | `np.nanmean`, `np.cumsum`, `np.argmax`, `np.where` with NaN | >=1.24 | HIGH — all available since NumPy 1.x |
| PyTorch | `ForwardLUT.cast_ray(pix: Tensor[N,2])` | >=2.0 | HIGH — already used in scoring.py |
| Python | Frozen dataclasses, structural Protocol | >=3.11 | HIGH — existing pattern |

---

## Sources

- Codebase direct reading: `core/association/scoring.py`, `core/tracking/types.py`, `core/tracking/keypoint_tracker.py`, `core/association/refinement.py`, `pyproject.toml` — HIGH confidence
- `ruptures` PyPI page (https://pypi.org/project/ruptures/) — version 1.1.10, released Sept 2025 — HIGH confidence
- `ruptures` documentation (https://centre-borelli.github.io/ruptures-docs/) — Binseg, PELT algorithm characteristics — MEDIUM confidence (fetched from official docs)
- Design document `.planning/inbox/association_multikey_rework.md` — defines exact changepoint algorithm — HIGH confidence (primary spec)

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| No new dependencies | HIGH | Direct code reading confirms all operations are in existing stack |
| Max-split sweep correctness | HIGH | O(n) cumsum formulation is standard; directly matches design doc specification |
| `ruptures` rejection | HIGH | Design doc specifies threshold-based splitting, not penalized optimization; library would add complexity without benefit |
| Tracklet2D keypoints field | HIGH | `keypoint_tracker.py` already accumulates the data; it's a pass-through fix |
| NaN masking aggregation | HIGH | `np.nanmean` is the idiomatic pattern; no edge cases for this use |
| Config field ranges | MEDIUM | Initial values are estimates; empirical tuning pass is part of the milestone |
