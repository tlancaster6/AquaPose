# Architecture Research

**Domain:** Multi-keypoint association scoring, group validation, and singleton recovery for multi-camera fish tracking
**Researched:** 2026-03-11
**Confidence:** HIGH (based on direct codebase inspection)

## Standard Architecture

### System Overview

```
Current v3.7 Association Module (core/association/)
┌─────────────────────────────────────────────────────────────┐
│ stage.py — AssociationStage.run()                            │
│   1. score_all_pairs()      → scoring.py                     │
│   2. build_must_not_link()  → clustering.py                  │
│   3. cluster_tracklets()    → clustering.py                  │
│   4. merge_fragments()      → clustering.py   REMOVE         │
│   5. refine_clusters()      → refinement.py   REPLACE        │
└─────────────────────────────────────────────────────────────┘

Target v3.8 Association Module (core/association/)
┌─────────────────────────────────────────────────────────────┐
│ stage.py — AssociationStage.run()                            │
│   1. score_all_pairs()      → scoring.py       MODIFY        │
│   2. build_must_not_link()  → clustering.py                  │
│   3. cluster_tracklets()    → clustering.py                  │
│   4. validate_groups()      → validation.py    NEW           │
│   5. recover_singletons()   → validation.py    NEW           │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | v3.7 Role | v3.8 Change |
|-----------|-----------|-------------|
| `scoring.py` | Centroid-only ray-ray pairwise scoring | Extended to multi-keypoint; `_batch_score_frames` casts K rays per frame instead of 1 |
| `clustering.py` | Leiden graph clustering + fragment merge | Fragment merge (`merge_fragments`, `_merge_cam_fragments`, `_try_merge_pair`) removed; clustering unchanged |
| `refinement.py` | Single-centroid triangulation refinement + eviction | Deleted; replaced by `validation.py` |
| `validation.py` | Does not exist | New: `validate_groups()` (changepoint + eviction) and `recover_singletons()` |
| `stage.py` | Orchestrates 5 steps | Steps 4-5 replaced; imports `validation.py` instead of `refinement.py` |
| `types.py` | `TrackletGroup`, `HandoffState`, `AssociationBundle` | No changes — downstream contract unchanged |

The `Tracklet2D` type and its producer also change, as a prerequisite to the scoring changes:

| Component | v3.7 Role | v3.8 Change |
|-----------|-----------|-------------|
| `tracking/types.py` | `Tracklet2D` with centroids only | Add `keypoints: tuple | None` and `keypoint_conf: tuple | None` fields |
| `tracking/keypoint_tracker.py` | `_KptTrackletBuilder.to_tracklet2d()` drops accumulated keypoints | Must populate the two new `Tracklet2D` fields from the builder's existing `keypoints` and `keypoint_conf` lists |

## Recommended Project Structure

```
src/aquapose/core/association/
├── __init__.py          # Public API — update when validation.py added
├── types.py             # TrackletGroup — NO CHANGES
├── scoring.py           # MODIFY: extend _batch_score_frames to multi-keypoint
├── clustering.py        # MODIFY: delete merge_fragments and helpers
├── refinement.py        # DELETE: replaced by validation.py
├── validation.py        # NEW: validate_groups(), recover_singletons()
└── stage.py             # MODIFY: replace merge+refine steps with validate+recover

src/aquapose/core/tracking/
├── types.py             # MODIFY: add keypoints and keypoint_conf fields to Tracklet2D
├── keypoint_tracker.py  # MODIFY: populate new fields in to_tracklet2d()
└── stage.py             # NO CHANGES

src/aquapose/engine/
└── config.py            # MODIFY: add fields to AssociationConfig for new behaviors

src/aquapose/evaluation/stages/
└── association.py       # MODIFY: update DEFAULT_GRID with new tunable params
```

## Architectural Patterns

### Pattern 1: Keypoint Propagation Through Tracklet2D

**What:** `Tracklet2D` currently stores only centroids. Scoring and validation both need per-frame per-keypoint positions and confidences. `_KptTrackletBuilder` in `keypoint_tracker.py` already accumulates these internally (in `self.keypoints` and `self.keypoint_conf` lists) but `to_tracklet2d()` drops them. The fix adds two optional fields to `Tracklet2D` and populates them in the builder.

**When to use:** Step 1 of build order; all other work depends on it.

**Trade-offs:** Small memory increase per tracklet (6 kpts x 2 coords x N frames x float32). For 200-frame chunks with 30 tracklets across 13 cameras: ~200 * 30 * 6 * 2 * 4 bytes ~= 3MB. Acceptable.

**Key implementation note:** The `None` default for both new fields preserves backward compatibility — existing tests that construct `Tracklet2D` without keypoints continue to work. Scoring and validation must check for `None` and fall back to centroid-only behavior when keypoints are absent.

**Type change:**
```python
@dataclass(frozen=True)
class Tracklet2D:
    # existing fields unchanged
    camera_id: str
    track_id: int
    frames: tuple
    centroids: tuple
    bboxes: tuple
    frame_status: tuple
    # new fields
    keypoints: tuple | None = None      # tuple[np.ndarray, ...] shape (K, 2) each frame
    keypoint_conf: tuple | None = None  # tuple[np.ndarray, ...] shape (K,) each frame
```

`interpolate_gaps` in `keypoint_tracker.py` already computes and stores interpolated keypoints in the builder's lists; those will flow through once `to_tracklet2d()` populates the fields.

### Pattern 2: K-Ray Batched Scoring

**What:** The existing `_batch_score_frames` casts one ray per frame from the centroid. The multi-keypoint extension casts K rays per frame — one per visible keypoint pair (matched by index across cameras). The vectorization structure is preserved: instead of N origins/dirs, accumulate N*K_visible entries, then batch the `cast_ray` and `ray_ray_closest_point_batch` calls identically to the current code.

**When to use:** Modifying `_batch_score_frames` and `score_tracklet_pair` in `scoring.py`.

**Trade-offs:** K-fold more entries in the batched arrays. Still fully vectorized. Confidence filtering reduces K on partial-visibility frames. The forward LUT is already at pixel-exact resolution (grid_step=1) so no LUT changes are needed.

**Aggregation:** For each shared frame, compute ray-ray distances for all visible keypoint pairs. Apply the soft kernel `1 - dist / threshold` per keypoint. Average contributions across visible keypoints for the per-frame score. The per-frame aggregation replaces the single distance per frame in the current code; the outer loop structure (early termination, overlap reliability weighting) is unchanged.

**Fallback:** If `tracklet_a.keypoints is None or tracklet_b.keypoints is None`, fall back to centroid-only scoring (existing `_batch_score_frames` behavior). Log at DEBUG level. This makes multi-keypoint scoring opt-in based on data availability.

**New config fields:**
```python
# In AssociationConfig (engine/config.py)
scoring_keypoint_confidence_floor: float = 0.3  # min confidence to include a kpt in scoring
# centroid_keypoint_index and centroid_confidence_floor already exist and are reused for fallback
```

### Pattern 3: Group Validation via Temporal Changepoint Detection

**What:** After Leiden clustering, for each tracklet in a multi-view group, cast multi-keypoint rays and compute per-frame residuals against the group consensus position. Apply two checks: changepoint detection (temporal jump in residuals = upstream ID swap) and outlier eviction (persistently high residuals = wrong group membership).

**When to use:** Implemented in `validate_groups()` in `validation.py`.

**Why fresh ray casting (not reuse from scoring):** Scoring only covers adjacent camera pairs (camera overlap graph filter). A group formed via transitive edges (A-B, B-C) will have missing scoring data for the A-C pair. Fresh casting against the full group consensus covers all members correctly, and the cost is small: ~9 groups x 4-5 cameras vs ~1000 pairs in scoring.

**Changepoint algorithm:** Sweep candidate split points from `min_segment` to `n - min_segment`. At each split, compute `abs(mean(residuals[:split]) - mean(residuals[split:]))`. If the maximum exceeds `changepoint_threshold`, split at that index. The consistent half (lower mean residual against the current group) stays; the inconsistent half becomes a singleton candidate. Recurse on segments if needed (uncommon in practice). Error asymmetry favors sensitivity: a false split produces two valid fragments that singleton recovery can reassemble; a missed swap poisons reconstruction.

**New config fields:**
```python
changepoint_min_segment: int = 10          # minimum frames per half after split
changepoint_threshold: float = 0.015       # minimum residual delta to trigger split (metres)
eviction_reproj_threshold: float = 0.025   # unchanged, reused for outlier eviction
min_cameras_refine: int = 3                # minimum cameras in group to run validation
```

### Pattern 4: Singleton Recovery with Swap-Aware Split-and-Assign

**What:** For each singleton (original + those produced by changepoint splitting), compute per-frame residuals against every existing multi-view group. Three outcomes:
- Strong overall match to one group: assign to that group.
- No overall match, but a temporal split produces two segments each matching a different group: split and assign both (case A swap recovery).
- No match under any split: remains singleton.

**When to use:** Implemented in `recover_singletons()` in `validation.py`, called after `validate_groups()`.

**Split detection for singletons:** Precompute per-frame residuals against all groups in one pass. Then sweep split points, finding the partition that minimizes combined residuals: `min_residual(left, group_i) + min_residual(right, group_j)` for all group pairs. Each half must beat `singleton_match_threshold` (stricter than group validation, because the prior is weaker). Controlled by `min_singleton_split_segment` — setting very high (e.g., 999999) disables splitting as a natural no-op; no separate config toggle needed.

**New config fields:**
```python
singleton_match_threshold: float = 0.015   # max mean residual for a singleton to join a group
min_singleton_split_segment: int = 15      # minimum frames per half for split-assign
```

### Pattern 5: Config Protocol Extensions (IB-003 Preservation)

**What:** `validation.py` accepts configuration via a structural Protocol, the same pattern as `AssociationConfigLike` (scoring.py) and `RefinementConfigLike` (refinement.py). This preserves the core/ import boundary (IB-003): `core/` never imports from `engine/`.

**When to use:** Define `ValidationConfigLike` in `validation.py`. `AssociationConfig` in `engine/config.py` satisfies it structurally.

**Example:**
```python
@runtime_checkable
class ValidationConfigLike(Protocol):
    eviction_reproj_threshold: float
    changepoint_min_segment: int
    changepoint_threshold: float
    singleton_match_threshold: float
    min_singleton_split_segment: int
    min_cameras_refine: int
```

## Data Flow

### v3.8 Association Data Flow

```
PipelineContext.tracks_2d
  dict[cam_id, list[Tracklet2D]]
  NEW: Tracklet2D carries keypoints tuple[np.ndarray(K,2)] and
       keypoint_conf tuple[np.ndarray(K,)]
        |
        v
score_all_pairs()                          [scoring.py — MODIFIED]
  For each adjacent camera pair:
    For each tracklet pair sharing >= t_min frames:
      If both have keypoints: cast K rays per camera (confidence-filtered)
      Else: fall back to centroid ray (existing behavior)
      Compute per-frame scores, aggregate with soft kernel
  Returns: dict[(key_a, key_b), float]     (interface UNCHANGED)
        |
        v
build_must_not_link()                      [clustering.py — UNCHANGED]
cluster_tracklets()                        [clustering.py — UNCHANGED]
  Returns: list[TrackletGroup]
        |
        v
validate_groups()                          [validation.py — NEW]
  For each multi-view group (>= min_cameras_refine):
    For each tracklet in group:
      Cast K rays per frame against group consensus
      Compute per-frame residual series
      Run changepoint detection
      If changepoint: split tracklet — consistent half stays,
                      inconsistent half appended to singletons list
      If no changepoint but high overall residual: evict to singletons
  Returns: (validated_groups, singleton_tracklet_fragments)
        |
        v
recover_singletons()                       [validation.py — NEW]
  For each singleton (original + fragments from validate_groups):
    Compute per-frame residuals against all groups
    If strong overall match: assign to group
    If two-group temporal split found: split and assign each half
    Else: remains singleton (new TrackletGroup with 1 tracklet)
  Returns: list[TrackletGroup]             (interface UNCHANGED)
        |
        v
PipelineContext.tracklet_groups
  list[TrackletGroup]                      (downstream UNCHANGED)
```

### Tracklet2D Evolution

The critical structural change: `Tracklet2D` evolves from a centroid-only record to a full per-frame pose record. This propagates upward through one level:

```
_KptTrackletBuilder (keypoint_tracker.py)
  Already accumulates:
    self.keypoints: list[np.ndarray (K,2)]     per frame
    self.keypoint_conf: list[np.ndarray (K,)]  per frame
  to_tracklet2d() currently DROPS both lists

  [ADD: populate new fields]
        |
        v
Tracklet2D (tracking/types.py)
  ADD: keypoints: tuple | None = None
  ADD: keypoint_conf: tuple | None = None

  [NO CHANGES — passes through]
        |
        v
TrackingStage.run() → context.tracks_2d

  [MODIFIED to consume keypoint fields]
        |
        v
AssociationStage.run() → score + validate + recover
```

## Integration Points

### Files Modified vs New

| File | Change Type | What Changes |
|------|-------------|--------------|
| `core/tracking/types.py` | MODIFY | Add `keypoints: tuple | None = None` and `keypoint_conf: tuple | None = None` fields to `Tracklet2D` (after existing fields for backward compat) |
| `core/tracking/keypoint_tracker.py` | MODIFY | `_KptTrackletBuilder.to_tracklet2d()` populates the two new fields from `self.keypoints` and `self.keypoint_conf` |
| `core/association/scoring.py` | MODIFY | `_batch_score_frames` extended to use keypoint fields when present; `AssociationConfigLike` protocol gets `scoring_keypoint_confidence_floor` |
| `core/association/clustering.py` | MODIFY | Delete `merge_fragments`, `_merge_cam_fragments`, `_try_merge_pair`; remove `max_merge_gap` from `ClusteringConfigLike`; update `__all__` |
| `core/association/refinement.py` | DELETE | Entirely replaced by `validation.py` |
| `core/association/validation.py` | NEW | `ValidationConfigLike` Protocol, `validate_groups()`, `recover_singletons()`, shared internal helpers for multi-keypoint ray casting and residual computation against a group consensus |
| `core/association/stage.py` | MODIFY | Step 4: replace `merge_fragments` + `refine_clusters` calls with `validate_groups` + `recover_singletons`; update imports |
| `core/association/__init__.py` | MODIFY | Add exports from `validation.py`; remove exports from `refinement.py` |
| `engine/config.py` `AssociationConfig` | MODIFY | Add: `scoring_keypoint_confidence_floor`, `changepoint_min_segment`, `changepoint_threshold`, `singleton_match_threshold`, `min_singleton_split_segment`; remove: `max_merge_gap`, `refinement_enabled` (keep `min_cameras_refine`, `eviction_reproj_threshold`) |
| `evaluation/stages/association.py` | MODIFY | Update `DEFAULT_GRID` to include new tunable params; remove old params that no longer exist |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `tracking/types.py` → `association/scoring.py` | `Tracklet2D.keypoints` and `keypoint_conf` fields | `None` default preserves backward compat; scoring checks and falls back to centroid if `None` |
| `association/validation.py` → `calibration/luts` | `ForwardLUT.cast_ray()` | Same interface used by existing `refinement.py`; no changes needed |
| `core/association/` → `engine/config.py` | `ValidationConfigLike` structural protocol | IB-003 preserved: core never imports engine |
| `association/stage.py` → `validation.py` | Direct function calls | Same pattern as existing `refine_clusters` call |

## Suggested Build Order

The dependency chain is strict bottom-up. Each step unblocks the next.

### Step 1: Tracklet2D Keypoint Fields

**Files:** `core/tracking/types.py`, `core/tracking/keypoint_tracker.py`

This is the prerequisite that gates all other work. Without keypoints in `Tracklet2D`, scoring and validation have no data. The change is small: two nullable fields added to the frozen dataclass, one additional assignment in `to_tracklet2d()`.

Tests: unit tests confirm `to_tracklet2d()` output contains keypoint arrays; existing `Tracklet2D` construction tests continue to pass because fields default to `None`.

### Step 2: Multi-Keypoint Scoring

**Files:** `core/association/scoring.py`, `engine/config.py` (partial)

Extend `_batch_score_frames` to use keypoint fields. The function signature and the `score_all_pairs` return type are unchanged, so clustering is unaffected. Can be tested independently by comparing scores against baseline centroid scores.

Tests: unit tests for `_batch_score_frames` with mock tracklets containing known keypoints; verify confidence filtering produces correct visible-keypoint counts; verify fallback to centroid when `keypoints is None`.

### Step 3: Remove Fragment Merging

**Files:** `core/association/clustering.py`, `core/association/stage.py`, `engine/config.py` (partial)

Delete `merge_fragments` and its helpers. Update `stage.py` to remove the call. Remove `max_merge_gap` from `ClusteringConfigLike` and `AssociationConfig`. This is a clean deletion with no new logic. Can be done in parallel with Step 2 since it only touches `clustering.py` and the stage orchestration, which Step 2 does not touch.

Tests: verify `cluster_tracklets` output is unchanged; verify `stage.py` still runs end-to-end.

### Step 4: Group Validation

**Files:** `core/association/validation.py` (partial), `core/association/stage.py`, `engine/config.py` (partial)

Implement `validate_groups()`. Wire into `stage.py` replacing `refine_clusters`. Requires Steps 1 and 2 to be complete because it relies on keypoint data in tracklets and uses the same LUT interaction patterns that multi-keypoint scoring validates.

Tests: unit tests with synthetic tracklets where one has an injected residual jump at a known frame; verify split occurs at correct frame; verify eviction triggers on consistently high-residual tracklets; verify groups with < `min_cameras_refine` cameras pass through unchanged.

### Step 5: Singleton Recovery

**Files:** `core/association/validation.py` (complete), `core/association/stage.py`

Implement `recover_singletons()`. Wire into `stage.py` as the final step. Depends on Step 4's output format (the singleton list from `validate_groups()`).

Tests: unit tests — singleton matches one group; singleton with a temporal split matching two groups; singleton with no match (remains singleton).

### Step 6: Config and Evaluation Cleanup

**Files:** `engine/config.py` (final), `evaluation/stages/association.py`

Add remaining config fields, remove deleted ones. Update `DEFAULT_GRID` in the association evaluator.

### Step 7: Tuning Pass

Run `aquapose tune` on association stage with new parameters using cached tracking outputs from the v3.7 baseline run. Compare `singleton_rate` and `fish_yield_ratio` against v3.7 baseline.

### Dependency Graph

```
Step 1: Tracklet2D keypoints
    |
    +-------> Step 2: Multi-keypoint scoring
    |              |
    |              +------> Step 4: validate_groups()
    |              |              |
    +---> Step 3: Remove merge    +------> Step 5: recover_singletons()
              (no new deps)                    |
                                    Step 6: Config/eval cleanup
                                          |
                                    Step 7: Tuning pass
```

Step 3 (remove fragment merge) can begin immediately after Step 1 and run in parallel with Step 2 since they touch different files. Steps 4 and 5 both require Step 2 to be working so LUT interactions and keypoint confidence filtering are validated before building validation logic on top.

## Anti-Patterns

### Anti-Pattern 1: Reusing Scoring-Step Ray Data for Validation

**What people do:** Store per-frame ray data from `score_all_pairs` and pass it into `validate_groups` to avoid recomputing rays.

**Why it's wrong:** Scoring only covers adjacent camera pairs (camera overlap graph filter). Group members may have transitive connections: A scored against B, B against C, but no direct A-C score. Using stored scoring data leaves gaps in the residual series for transitive members, making changepoint detection unreliable.

**Do this instead:** `validate_groups` casts fresh rays against the full group consensus, covering all members. The cost is negligible: ~9 groups x 4-5 cameras x N frames vs ~1000 pairs in scoring.

### Anti-Pattern 2: Hard-Failing When Keypoints Are Absent

**What people do:** Return 0.0 (or raise) from `score_tracklet_pair` when `tracklet.keypoints is None`.

**Why it's wrong:** Destroys backward compatibility with synthetic test data and any pipeline state that predates the `Tracklet2D` field addition. Silently degrades scoring without a clear signal.

**Do this instead:** Check `keypoints is None` and fall back to centroid-only scoring (existing behavior). Log at DEBUG level. Multi-keypoint scoring is opt-in based on data availability.

### Anti-Pattern 3: Representing Tracklet Splits as New Tracklet2D Objects with Duplicate IDs

**What people do:** Create two new `Tracklet2D` objects by slicing tuple fields at the changepoint index.

**Why it's wrong:** `Tracklet2D` is a frozen dataclass. Slicing is fine, but the `track_id` field will be identical for both halves. Downstream clustering and group membership logic keys on `(camera_id, track_id)` pairs — duplicate keys break set operations and group membership tracking.

**Do this instead:** Represent splits internally in `validation.py` as `(tracklet, start_frame, end_frame)` tuples during changepoint processing. Only convert to new `Tracklet2D` objects if full objects are required downstream, and in that case assign synthetic IDs (e.g., `original_id * 10000 + fragment_idx`) to ensure uniqueness. Since `TrackletGroup` only needs the tracklet reference for ray casting, working with frame-range subsets avoids the problem entirely.

### Anti-Pattern 4: Adding a Separate Config Toggle for Singleton Splitting

**What people do:** Add `singleton_split_enabled: bool = True` to `AssociationConfig`.

**Why it's wrong:** Every boolean toggle increases config surface and requires documentation. "Disabled when segment is too short" is already the natural behavior.

**Do this instead:** Set `min_singleton_split_segment` to a very large value (e.g., 999999) to disable splitting as a natural no-op. No boolean needed.

## Sources

- Direct inspection of `src/aquapose/core/association/scoring.py` — vectorized scoring architecture, `_batch_score_frames` pattern
- Direct inspection of `src/aquapose/core/association/refinement.py` — ray casting patterns, consensus computation, eviction logic (replaced but patterns reused in `validation.py`)
- Direct inspection of `src/aquapose/core/association/clustering.py` — `merge_fragments` structure (to delete)
- Direct inspection of `src/aquapose/core/association/stage.py` — orchestration steps
- Direct inspection of `src/aquapose/core/association/types.py` — `TrackletGroup` (unchanged)
- Direct inspection of `src/aquapose/core/tracking/types.py` — `Tracklet2D` current fields
- Direct inspection of `src/aquapose/core/tracking/keypoint_tracker.py` — `_KptTrackletBuilder` accumulates keypoints but `to_tracklet2d()` drops them
- Direct inspection of `src/aquapose/engine/config.py` — `AssociationConfig` current fields
- Direct inspection of `src/aquapose/evaluation/stages/association.py` — `DEFAULT_GRID`, `AssociationMetrics`
- `.planning/inbox/association_multikey_rework.md` — design document specifying multi-keypoint scoring, changepoint detection, and singleton recovery approaches
- `.planning/PROJECT.md` — v3.8 milestone description, IB-003 import boundary constraint

---
*Architecture research for: AquaPose v3.8 Improved Association*
*Researched: 2026-03-11*
