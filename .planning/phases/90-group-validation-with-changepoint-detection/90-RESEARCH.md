# Phase 90: Group Validation with Changepoint Detection - Research

**Researched:** 2026-03-11
**Domain:** Temporal changepoint detection for multi-view tracklet ID-swap detection; association module refactor
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Residual computation**
- Use matched keypoint rays (Phase 88 pattern): for each frame, cast rays from all confident keypoints on each tracklet, compare matched keypoints across cameras (nose-to-nose, head-to-head, etc.) via ray-ray distance
- All 6 keypoints contribute equally — the per-keypoint confidence floor from Phase 88 already filters unreliable keypoints
- Frames with fewer than 2 cameras having confident keypoints are skipped (excluded from residual series)
- Residuals are computed per-tracklet vs rest of group: each tracklet gets one residual time series representing its mean keypoint ray distance against all other tracklets in the group for that frame

**Changepoint detection**
- Simple threshold + run classification: each frame is classified as "consistent" (residual < threshold) or "inconsistent" based on the eviction threshold
- Find the longest consistent run; the transition point is the changepoint
- Single changepoint per pass, but the validation function should be composable for iterative passes (split, recompute residuals against updated group, check again)
- Minimum segment length after split: configurable, default ~10 frames (~0.3s at 30fps)
- Changepoint threshold reuses `eviction_reproj_threshold` — one parameter, tunable in Phase 92

**Eviction vs splitting decision tree**
1. Compute residual series
2. If mostly consistent (>50% of frames below threshold), keep as-is
3. If changepoint found with both segments >= min_length, split: consistent segment stays, inconsistent segment becomes singleton
4. If no clear changepoint (uniformly high residual), evict entire tracklet as singleton
- Inconsistent segments become immediate singleton candidates — Phase 91 handles recovery/reassignment
- Thin group handling (group drops to 1 camera after validation): Claude's discretion

**TrackletGroup output contract**
- Keep same fields: `per_frame_confidence` and `consensus_centroids` are populated by validation just as refinement did
- Internal computation changes (multi-keypoint instead of centroid-only) but output shape is unchanged
- After splits/evictions, recompute per_frame_confidence and overall confidence from the cleaned group's residuals
- Split tracklets become new Tracklet2D instances with sliced arrays (frames, centroids, keypoints, keypoint_conf, bboxes, frame_status)
- Split segments get new unique local track_ids to prevent duplicate IDs within the same camera

### Claude's Discretion
- Thin group dissolution policy (keep single-camera groups vs dissolve to singletons)
- Internal implementation of the consistent-run detection algorithm
- How to recompute consensus_centroids after membership changes (can reuse existing triangulation helpers)
- Whether to log/emit diagnostic info about splits and evictions for debugging

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VALID-01 | After clustering, each tracklet's multi-keypoint residuals are computed against its group | `_batch_score_frames_kpt` pattern in `scoring.py` is the reusable building block; `_compute_frame_consensus` from `refinement.py` is the starting template |
| VALID-02 | Changepoint detection identifies temporal ID swap points in per-tracklet residual series | Threshold + run classification algorithm is fully defined in CONTEXT.md; no library needed |
| VALID-03 | Swapped tracklets are split at the changepoint — consistent segment stays, inconsistent segment becomes singleton candidate | Tracklet2D is a frozen dataclass; splits require constructing new instances with sliced arrays |
| VALID-04 | Tracklets with high overall residual (no changepoint) are evicted to singleton status | Direct port of `refine_clusters()` eviction pattern with updated residual source |
| CLEAN-02 | Refinement module replaced by group validation (refinement.py deleted after consumer audit) | Consumer audit shows 3 downstream consumers; all require equivalent outputs from `validation.py` |
</phase_requirements>

## Summary

Phase 90 replaces `refinement.py` with `validation.py` in the association module. The core algorithmic change is that residuals are now computed from multi-keypoint ray distances (Phase 88 pattern) rather than centroid-only ray-to-consensus distances, and the eviction-only policy is upgraded to a split-or-evict policy with temporal changepoint detection. The `TrackletGroup` output contract (`per_frame_confidence`, `consensus_centroids`) is unchanged so downstream consumers (reconstruction and evaluation) require no modification.

The changepoint algorithm is a two-step threshold + run classifier: classify each frame as consistent/inconsistent, find the longest consistent run, identify the transition as the changepoint. This is simpler than a max-split sweep — no O(n^2) search, just a linear pass finding run boundaries. The decision tree (keep / split / evict) is based on what fraction of frames are consistent and whether both split segments meet the minimum length requirement.

The consumer audit (performed during this research) confirms: `per_frame_confidence` is referenced only in `reconstruction/stage.py` (docstring only — not actually read at runtime), and `consensus_centroids` is actively consumed by `evaluation/tuning.py`. The `validation.py` module must populate `consensus_centroids` for the group's cleaned-up tracklet set, using the same centroid-based triangulation as `refinement.py` (ray-ray midpoint averaging). The reconstruction stage currently does not read `per_frame_confidence` at runtime but its docstring promises it will — populate it from the group's cleaned residuals as a normalized confidence score.

**Primary recommendation:** Implement `validation.py` as a near drop-in for `refinement.py`: same function signature `validate_groups(groups, forward_luts, config)`, same `TrackletGroup` output shape, upgraded internals (multi-keypoint residuals, split logic). Reuse `_batch_score_frames_kpt` from `scoring.py` for residual computation and `_compute_frame_consensus` / `_compute_per_frame_confidence` from `refinement.py` (copy and adapt).

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| NumPy | >=1.24 (project constraint) | Run-length encoding, prefix sums, boolean masking for changepoint | Already in project; `np.cumsum`, `np.where`, `np.argmax` cover all needs |
| PyTorch | >=2.0 (project constraint) | `ForwardLUT.cast_ray()` for residual computation | Same pattern as scoring.py; already in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Python `dataclasses` (stdlib) | 3.11+ | Frozen dataclass construction for split Tracklet2D | Slicing tracklet arrays into new immutable instances |
| Python `logging` (stdlib) | 3.11+ | Split/eviction diagnostics | Log counts per group for debugging |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom run-classifier changepoint | `ruptures` (PELT) | `ruptures` requires penalty calibration adding a free parameter; custom O(n) is sufficient for single-swap case and uses the same `eviction_reproj_threshold` units |
| Custom run-classifier changepoint | `bayesian_changepoint_detection` | Unmaintained; no advantage over threshold classifier |

**Installation:** No new dependencies. All required packages are already in `pyproject.toml`.

## Architecture Patterns

### Recommended Project Structure
```
src/aquapose/core/association/
├── __init__.py          # Remove refinement exports, add validation exports
├── stage.py             # Replace refine_clusters call with validate_groups call
├── scoring.py           # Unchanged (reused for residual computation)
├── clustering.py        # Unchanged
├── types.py             # Unchanged (TrackletGroup fields are unchanged)
├── validation.py        # NEW: replaces refinement.py
└── refinement.py        # DELETE after validation.py is complete and tests pass

tests/unit/core/association/
├── test_validation.py   # NEW: mirrors test_refinement.py structure
└── test_refinement.py   # KEEP until refinement.py deletion task
```

### Pattern 1: ValidationConfigLike Protocol
**What:** Structural protocol satisfied by `AssociationConfig` without import, following the `RefinementConfigLike` / `AssociationConfigLike` pattern already established in this module.
**When to use:** All validation functions take this as the `config` parameter.

```python
# Source: refinement.py:RefinementConfigLike (adapt for validation)
@runtime_checkable
class ValidationConfigLike(Protocol):
    eviction_reproj_threshold: float
    min_cameras_validate: int     # replaces min_cameras_refine
    validation_enabled: bool      # replaces refinement_enabled
    min_segment_length: int       # new: minimum frames per split segment
```

**Note:** `AssociationConfig` in `engine/config.py` must gain `min_segment_length: int = 10` and the `refinement_enabled` / `min_cameras_refine` fields should be renamed to `validation_enabled` / `min_cameras_validate`. Remove `refinement_enabled` and `min_cameras_refine` from `AssociationConfig`.

### Pattern 2: Multi-Keypoint Residual Computation
**What:** For each group, for each tracklet, compute a per-frame residual series by scoring the target tracklet's keypoint rays against the rest of the group (excluding itself) using the same `_batch_score_frames_kpt` batching pattern as Phase 88.
**When to use:** This is the core inner loop of `validate_groups()`.

The residual for tracklet `t` at frame `f` is: mean ray-ray distance between `t`'s confident keypoints and the corresponding keypoints on all other tracklets in the group that have confident keypoints for frame `f`. If fewer than 2 cameras have confident keypoints for frame `f`, the frame is skipped (excluded from the residual series).

```python
# Conceptual structure (not a direct copy — implement from scratch)
def _compute_tracklet_residuals(
    target: Tracklet2D,
    others: list[Tracklet2D],
    forward_luts: dict[str, ForwardLUT],
    config: ValidationConfigLike,
) -> tuple[list[int], list[float]]:
    """Return (valid_frame_list, residual_per_frame)."""
    ...
```

The key insight: this is a "one vs. many" scoring problem. For each valid frame, cast rays from the target's confident keypoints and from each other tracklet's matching confident keypoints. Compute mean ray-ray distance. This reuses `ray_ray_closest_point_batch` from `scoring.py`.

### Pattern 3: Threshold + Run Classifier Changepoint
**What:** Classify each frame in the residual series as consistent (residual < threshold) or inconsistent. Find the longest contiguous consistent run. If it is separated from another consistent run by an inconsistent segment, the transition between runs is the changepoint.
**When to use:** After computing the per-tracklet residual series.

```python
# Source: user decision in CONTEXT.md — threshold + run classification
def _find_changepoint(
    residuals: list[float],
    frames: list[int],
    threshold: float,
    min_segment_length: int,
) -> int | None:
    """Return the frame index of the changepoint, or None if no valid split.

    Returns the first frame index of the inconsistent run that follows the
    longest consistent run. If the longest consistent run is >= min_segment_length
    AND there is at least min_segment_length frames outside it, returns the
    split frame. Otherwise returns None.
    """
    consistent = [r < threshold for r in residuals]
    # Find the longest consistent run and its boundaries
    # If valid split found (both halves >= min_segment_length), return split point
    # Otherwise return None
    ...
```

**Implementation note:** "Longest consistent run" means the longest contiguous subsequence of `True` values. The "split point" is the last frame of the longest consistent run (the consistent segment is `frames[:split_idx+1]`, the inconsistent segment is `frames[split_idx+1:]`). If >50% of all frames are consistent but there is no clean run separation, keep as-is rather than splitting.

### Pattern 4: Tracklet Splitting
**What:** Construct two new `Tracklet2D` instances by slicing the frozen arrays of the original.
**When to use:** When a changepoint is found.

```python
# Tracklet2D is a frozen dataclass — construct new instances with sliced data
def _split_tracklet(
    tracklet: Tracklet2D,
    split_idx: int,       # index into tracklet.frames where split occurs
    new_id_consistent: int,
    new_id_inconsistent: int,
) -> tuple[Tracklet2D, Tracklet2D]:
    """Split tracklet at split_idx into (consistent, inconsistent) segments."""
    consistent = Tracklet2D(
        camera_id=tracklet.camera_id,
        track_id=new_id_consistent,
        frames=tracklet.frames[:split_idx],
        centroids=tracklet.centroids[:split_idx],
        bboxes=tracklet.bboxes[:split_idx],
        frame_status=tracklet.frame_status[:split_idx],
        keypoints=tracklet.keypoints[:split_idx] if tracklet.keypoints is not None else None,
        keypoint_conf=tracklet.keypoint_conf[:split_idx] if tracklet.keypoint_conf is not None else None,
    )
    # inconsistent segment mirrors consistent, using frames[split_idx:]
    ...
```

**CRITICAL:** `frames`, `centroids`, `bboxes`, `frame_status` are `tuple` fields. Use Python tuple slicing `t[:n]` and `t[n:]`. `keypoints` and `keypoint_conf` are `np.ndarray | None` with shape `(T, K, 2)` and `(T, K)` — use NumPy array slicing `arr[:n]` and `arr[n:]`.

**CRITICAL:** New `track_id` values must be unique within the camera. The validation pass must generate IDs that do not collide with existing tracklet IDs. Recommended approach: assign `new_id = max_existing_id + counter` where `max_existing_id` is computed from all tracklets seen in the groups before validation starts.

### Pattern 5: Output Contract Preservation
**What:** After splitting/evicting, recompute `per_frame_confidence` and `consensus_centroids` for the cleaned group using the same centroid-based triangulation from `refinement.py`.
**When to use:** After any group membership change (split or eviction).

The existing `_compute_frame_consensus` and `_compute_per_frame_confidence` functions in `refinement.py` compute centroid-based consensus. These should be **copied** into `validation.py` (not imported — `refinement.py` will be deleted). The internal computation uses centroids, not keypoints, for the consensus output. This is intentional: `consensus_centroids` is a 3D position field consumed by downstream evaluation; computing it from keypoints would require richer triangulation that is out of scope.

### Pattern 6: Public API (validate_groups)
**What:** Mirror of `refine_clusters()` signature with identical call site in `stage.py`.

```python
# validation.py public API
def validate_groups(
    groups: list[TrackletGroup],
    forward_luts: dict[str, ForwardLUT],
    config: ValidationConfigLike,
) -> list[TrackletGroup]:
    """Validate groups via multi-keypoint residuals, split/evict bad tracklets.

    Returns refined groups plus singleton candidates from splits and evictions.
    """
    ...
```

Stage integration point (`stage.py` lines 100-104):
```python
# Replace:
from aquapose.core.association.refinement import refine_clusters
groups = refine_clusters(groups, forward_luts, self._config.association)

# With:
from aquapose.core.association.validation import validate_groups
groups = validate_groups(groups, forward_luts, self._config.association)
```

### Anti-Patterns to Avoid

- **Computing consensus from keypoints instead of centroids:** `consensus_centroids` must remain centroid-based for downstream compatibility with `evaluation/tuning.py` which projects these 3D points.
- **Mutating TrackletGroup or Tracklet2D:** Both are frozen dataclasses. Always construct new instances.
- **Splitting before checking >50% consistency:** The decision tree requires checking the consistent-frame fraction before attempting to find a changepoint. A mostly-inconsistent tracklet should be evicted, not split.
- **Reusing old track_ids for split segments:** Duplicate `(camera_id, track_id)` pairs within the same run will cause silent errors in downstream scoring or group lookups.
- **Importing `refinement.py` from `validation.py`:** Copy the helper functions you need. `refinement.py` will be deleted; circular helpers are a maintenance risk.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Batched ray casting for residuals | Custom per-frame ray casting loop | `_batch_score_frames_kpt` pattern from `scoring.py` + `ray_ray_closest_point_batch` | Same problem solved in Phase 88 with vectorized broadcasting; avoid per-frame Python loops |
| 3D consensus for `consensus_centroids` | New triangulation implementation | Copy `_compute_frame_consensus` and `_compute_per_frame_confidence` from `refinement.py` | Identical geometry, proven correct, tested — just copy and adapt |
| Changepoint library | `ruptures`, `changepy`, or similar | Custom threshold + run classifier (~30 lines NumPy) | Library requires new dependency and penalty calibration; threshold reuses `eviction_reproj_threshold` in metres |

**Key insight:** `validation.py` is a refactor of `refinement.py` with upgraded internals. Maximize code reuse by copying helpers from `refinement.py` rather than writing from scratch.

## Common Pitfalls

### Pitfall 1: Residual Computation Excludes Target Tracklet From Consensus
**What goes wrong:** When computing the residual of tracklet `t`, the consensus must be computed from the OTHER tracklets in the group, not including `t`. If `t` is included in its own consensus, the residual is artificially low (a tracklet always agrees with itself), causing outliers to appear consistent.
**Why it happens:** Naively passing all group tracklets to the consensus function.
**How to avoid:** For each target tracklet, pass `[t for t in group.tracklets if t is not target]` as the others.
**Warning signs:** All tracklets in a group have near-zero residuals even when the group contains a known-bad tracklet.

### Pitfall 2: tuple vs ndarray Slicing Asymmetry
**What goes wrong:** `frames`, `centroids`, `bboxes`, `frame_status` are `tuple` fields; `keypoints` and `keypoint_conf` are `np.ndarray | None`. Applying `arr[:n]` to a tuple works in Python (tuples are sliceable), but if `keypoints` is `None`, `None[:n]` raises `TypeError`.
**Why it happens:** Assuming all fields have the same type.
**How to avoid:** Guard with `if tracklet.keypoints is not None else None` for numpy fields. Test with both `keypoints=None` and `keypoints=<array>`.
**Warning signs:** `TypeError: 'NoneType' object is not subscriptable` on split operations.

### Pitfall 3: Duplicate track_id After Split
**What goes wrong:** Two Tracklet2D objects with the same `(camera_id, track_id)` pair in the same run. Phase 91 singleton recovery and must-not-link constraint building use `(camera_id, track_id)` as a key.
**Why it happens:** Reusing the original `track_id` for the consistent segment after a split.
**How to avoid:** Compute `next_id = max(t.track_id for groups in all_groups for t in group.tracklets) + 1` before the validation loop; increment for each new segment created.
**Warning signs:** Phase 91 produces groups with two tracklets of the same camera_id and track_id.

### Pitfall 4: Thin Group After Validation (Claude's Discretion)
**What goes wrong:** A group that started with 4 cameras has 3 tracklets evicted/split-away and now has only 1 camera. The group is effectively a singleton but is treated as a valid group.
**Why it happens:** The validation loop evicts independently per tracklet without checking post-eviction group health.
**How to avoid:** After all splits/evictions for a group, check `len({t.camera_id for t in kept_tracklets})`. If 1, dissolve: make the remaining tracklet a singleton candidate too (do not keep as a 1-camera "group" since it holds no cross-view information).
**Warning signs:** `TrackletGroup` objects with 1 tracklet and `fish_id != -1` in the non-singleton list.

### Pitfall 5: per_frame_confidence Consumer Audit
**What goes wrong:** `refinement.py` is deleted but a downstream consumer that reads `per_frame_confidence` now receives `None` and the conditional branch changes behavior silently.
**Why it happens:** Code audit was incomplete.
**How to avoid:** The audit (done during this research) found: `reconstruction/stage.py` line 369 only mentions `per_frame_confidence` in a docstring — not read at runtime in the current code. `evaluation/tuning.py` reads `consensus_centroids` actively (lines 277-291) but NOT `per_frame_confidence`. Therefore `per_frame_confidence` has no active consumer today. However, `validation.py` MUST still populate it (matching the existing `TrackletGroup` contract) because downstream code will use it in future phases and tests assert it is non-None after validation runs.
**Warning signs:** `evaluation/tuning.py` silently skips all groups (returns `inf` reprojection error) because `consensus_centroids` is None.

### Pitfall 6: Changepoint Threshold Calibration (FP Rate Requirement)
**What goes wrong:** The success criterion requires false positive rate on confirmed-correct tracklets < 30%. A threshold too permissive splits correctly-associated tracklets.
**Why it happens:** The threshold reuses `eviction_reproj_threshold` (default 0.025m = 2.5cm). Multi-keypoint residuals have different scale than centroid-based residuals because they aggregate over up to 6 keypoints per frame.
**How to avoid:** During plan execution, measure the FP rate: take confirmed-correct tracklets from the v3.7 benchmark run (tracklets that were part of correctly-identified groups before any validation), run the changepoint classifier on them, count how many are incorrectly split. The FP rate is `(split tracklets that rejoin their original group in Phase 91) / (total correct tracklets)`. If FP rate > 30%, raise `eviction_reproj_threshold`.
**Warning signs:** Group count increases dramatically after validation (e.g., from 9 to 20+). Short tracklets (<20 frames) are split at higher rates than long tracklets.

## Code Examples

### Changepoint: Threshold + Run Classifier
```python
# Source: CONTEXT.md algorithm + project implementation guide
import numpy as np

def _find_changepoint_by_run(
    residuals: list[float],
    frames: list[int],
    threshold: float,
    min_segment_length: int,
) -> int | None:
    """Find split index using longest-consistent-run heuristic.

    Returns:
        Index into ``frames`` of the first frame of the inconsistent run
        after the longest consistent run, or None if no valid split exists.
    """
    n = len(residuals)
    if n < 2 * min_segment_length:
        return None

    consistent = np.array(residuals) < threshold

    # Find run boundaries using diff
    padded = np.concatenate([[False], consistent, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]   # run start indices
    ends = np.where(padded[:-1] & ~padded[1:])[0]       # run end indices (exclusive)

    if len(starts) == 0:
        return None  # no consistent frames at all

    lengths = ends - starts
    best_run_idx = int(np.argmax(lengths))
    best_start = int(starts[best_run_idx])
    best_end = int(ends[best_run_idx])   # exclusive

    # Both segments must meet min_segment_length
    if best_start < min_segment_length:
        return None  # consistent segment is at the start; nothing before it
    if (n - best_end) < min_segment_length and best_end < n:
        return None  # not enough frames after consistent segment

    # The "split" is: consistent segment = frames[:best_end],
    # inconsistent segment = frames[best_end:]
    return best_end  # first frame index of the "outside" segment
```

### Tracklet Slicing
```python
# Source: Tracklet2D definition in tracking/types.py
def _split_tracklet_at(
    t: Tracklet2D,
    split_idx: int,
    id_before: int,
    id_after: int,
) -> tuple[Tracklet2D, Tracklet2D]:
    """Split Tracklet2D at split_idx (frames[:split_idx], frames[split_idx:])."""
    before = Tracklet2D(
        camera_id=t.camera_id,
        track_id=id_before,
        frames=t.frames[:split_idx],
        centroids=t.centroids[:split_idx],
        bboxes=t.bboxes[:split_idx],
        frame_status=t.frame_status[:split_idx],
        keypoints=t.keypoints[:split_idx] if t.keypoints is not None else None,
        keypoint_conf=t.keypoint_conf[:split_idx] if t.keypoint_conf is not None else None,
    )
    after = Tracklet2D(
        camera_id=t.camera_id,
        track_id=id_after,
        frames=t.frames[split_idx:],
        centroids=t.centroids[split_idx:],
        bboxes=t.bboxes[split_idx:],
        frame_status=t.frame_status[split_idx:],
        keypoints=t.keypoints[split_idx:] if t.keypoints is not None else None,
        keypoint_conf=t.keypoint_conf[split_idx:] if t.keypoint_conf is not None else None,
    )
    return before, after
```

### Decision Tree: Keep / Split / Evict
```python
# Source: CONTEXT.md decision tree
def _classify_tracklet(
    residuals: list[float],
    frames: list[int],
    threshold: float,
    min_segment_length: int,
) -> tuple[str, int | None]:
    """Return (action, split_idx) where action is 'keep'/'split'/'evict'.

    'keep'  — majority consistent, no action needed
    'split' — changepoint found, split at split_idx
    'evict' — uniformly inconsistent, evict entire tracklet
    """
    if not residuals:
        return "keep", None

    n_consistent = sum(1 for r in residuals if r < threshold)
    fraction_consistent = n_consistent / len(residuals)

    if fraction_consistent > 0.5:
        return "keep", None  # majority consistent

    # Check for a changepoint
    split_idx = _find_changepoint_by_run(residuals, frames, threshold, min_segment_length)
    if split_idx is not None:
        return "split", split_idx

    # Uniformly inconsistent — evict
    return "evict", None
```

### __init__.py API Update
```python
# Replace in association/__init__.py:
# Remove: from aquapose.core.association.refinement import (RefinementConfigLike, refine_clusters)
# Add:
from aquapose.core.association.validation import (
    ValidationConfigLike,
    validate_groups,
)
# Update __all__ accordingly
```

### AssociationConfig New Fields
```python
# Add to engine/config.py AssociationConfig:
min_segment_length: int = 10         # minimum frames per split segment after changepoint
min_cameras_validate: int = 2        # minimum cameras to run validation (was min_cameras_refine=3)
validation_enabled: bool = True      # replaces refinement_enabled

# Remove from AssociationConfig:
# min_cameras_refine: int = 3
# refinement_enabled: bool = True
```

**Note on min_cameras_validate default:** The old default was 3. With multi-keypoint residuals, even 2 cameras provide meaningful residual signal (multiple keypoint rays from 2 views). Consider defaulting to 2 rather than 3. This is Claude's discretion.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Centroid-only ray-to-consensus eviction in `refinement.py` | Multi-keypoint residual series + split-or-evict in `validation.py` | Phase 90 | Richer signal catches ID swaps that centroid-based eviction misses |
| Single eviction threshold (median distance to consensus) | Per-frame consistent/inconsistent classification + run detection | Phase 90 | Enables temporal splitting instead of whole-tracklet eviction |
| `refine_clusters()` in stage.py | `validate_groups()` in stage.py | Phase 90 | API and call site are structurally identical; consumers unaffected |

**Deprecated/outdated after Phase 90:**
- `refinement.py`: Entirely replaced by `validation.py`. Delete after test_validation.py tests pass and stage.py is updated.
- `RefinementConfigLike`: Remove from `__init__.py` and `__all__`. Replace with `ValidationConfigLike`.
- `refine_clusters`: Remove from `__init__.py` and `__all__`. Replace with `validate_groups`.
- `AssociationConfig.refinement_enabled`: Remove field.
- `AssociationConfig.min_cameras_refine`: Remove field.

## Open Questions

1. **Thin group dissolution policy (1 camera remaining)**
   - What we know: After eviction, a group may have only 1 camera remaining. This is equivalent to a singleton. The design leaves this to Claude's discretion.
   - What's unclear: Should a 1-camera "group" be kept as a low-confidence group or dissolved into a singleton candidate?
   - Recommendation: Dissolve to singleton. A 1-camera group carries no cross-view information and will produce no `consensus_centroids` (requires 2+ cameras). Making it a singleton is consistent with the output contract and gives Phase 91 a chance to recover it properly.

2. **Residual scale: multi-keypoint vs. centroid-only**
   - What we know: `eviction_reproj_threshold` defaults to 0.025m (2.5cm) calibrated for centroid-based residuals. Multi-keypoint residuals aggregate 6 keypoints with different spatial distributions.
   - What's unclear: Whether the 2.5cm threshold is appropriate for multi-keypoint residuals without recalibration.
   - Recommendation: Keep 0.025m as the initial default and measure the FP rate against the v3.7 benchmark as specified in the success criteria. Phase 92 is the official tuning phase. The false-positive check (success criterion 5) serves as the calibration gate.

3. **Handling groups where keypoints are None**
   - What we know: Phase 87 adds keypoints to Tracklet2D, and all tracklets from Phase 88 forward will have keypoints. But the protocol should handle `keypoints=None` gracefully.
   - What's unclear: Whether any tracklets can reach Phase 90 with `keypoints=None`.
   - Recommendation: Fall back to centroid-only residuals (port `_compute_tracklet_distances` logic from `refinement.py`) when `keypoints is None`. Log a warning. This preserves backward compatibility and avoids crashes if keypoint propagation is incomplete.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `src/aquapose/core/association/refinement.py` — full implementation of current eviction logic (centroid-based)
- Direct codebase inspection: `src/aquapose/core/association/scoring.py` — `_batch_score_frames_kpt` batched multi-keypoint pattern; `ray_ray_closest_point_batch` vectorized implementation
- Direct codebase inspection: `src/aquapose/core/association/types.py` — `TrackletGroup` frozen dataclass; `per_frame_confidence` and `consensus_centroids` field contracts
- Direct codebase inspection: `src/aquapose/core/tracking/types.py` — `Tracklet2D` frozen dataclass; `keypoints: np.ndarray | None` shape `(T, K, 2)`, `keypoint_conf: (T, K)`
- Direct codebase inspection: `src/aquapose/engine/config.py` — `AssociationConfig` existing fields; `eviction_reproj_threshold`, `min_cameras_refine`, `refinement_enabled`
- Direct codebase inspection: `src/aquapose/evaluation/tuning.py` lines 277-291 — active consumer of `consensus_centroids`; verified `per_frame_confidence` is NOT actively read
- Direct codebase inspection: `src/aquapose/core/reconstruction/stage.py` line 369 — `per_frame_confidence` appears only in docstring; not read at runtime
- `.planning/phases/90-group-validation-with-changepoint-detection/90-CONTEXT.md` — locked decisions for algorithm design
- `.planning/research/PITFALLS.md` P5, P7, P8 — changepoint FP calibration, refinement consumer audit, must-not-link split constraints

### Secondary (MEDIUM confidence)
- `.planning/research/SUMMARY.md` — milestone-level architecture decisions; confirms no new dependencies; custom O(n) changepoint sufficient

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all libraries already in project with known versions
- Architecture: HIGH — direct codebase inspection of every relevant file; no speculation
- Pitfalls: HIGH — P5 and P7 are from `.planning/research/PITFALLS.md` (milestone research); P1-P4 are from direct code reading

**Research date:** 2026-03-11
**Valid until:** 2026-04-11 (stable domain — no external library changes; internal code is under project control)
