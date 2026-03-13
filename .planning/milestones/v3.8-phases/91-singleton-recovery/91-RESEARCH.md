# Phase 91: Singleton Recovery - Research

**Researched:** 2026-03-11
**Domain:** Cross-camera tracklet association — post-validation singleton reassignment
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Scoring method**
- Score singletons against groups using ray-to-consensus residuals with multi-keypoint rays (not centroid-only)
- Per-keypoint 3D reference positions are triangulated on-demand from the group's member tracklets for shared frames — no schema changes to TrackletGroup
- Per-keypoint, per-frame ray-to-3D distances are aggregated via arithmetic mean into a single singleton-to-group score
- Consistent with Phase 88's multi-keypoint philosophy — centroid-only scoring is on the path to deprecation

**Assignment logic**
- Greedy best-first matching: sort all (singleton, group) pairs by residual score, assign the best match first, remove that singleton from consideration
- Hard threshold on mean residual (configurable, in metres) — no relative margin requirement
- Single pass: score all singletons against all groups once, then assign greedily; no iterative re-scoring after assignment
- Configurable minimum shared frames required between singleton and group to attempt scoring

**Split-assign behavior (swap-aware recovery)**
- Max-residual binary sweep: try every possible split point, score each segment against all groups, pick the split that maximizes total assignment quality
- Both segments must match different groups — if only one segment matches, the whole tracklet remains a singleton
- Configurable minimum segment length (separate from t_min) — segments shorter than this are not considered
- Binary split only (one split point per singleton) — no recursive splitting

**Constraint enforcement**
- Same-camera overlap check uses detected frames only (coasted/predicted frames do not count as overlap)
- Overlap check happens before scoring — if overlap exists on detected frames, skip scoring entirely for that singleton-group pair
- For split-assign, the same-camera constraint is re-checked per-segment independently against each segment's candidate group

### Claude's Discretion
- Config field names and default values for new parameters (assignment threshold, min segment length, min shared frames)
- Internal data structures for the recovery pass
- Logging and diagnostic output format
- Whether to extract shared utilities from refinement.py or implement standalone

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RECOV-01 | Each singleton is scored against all existing groups using multi-keypoint residuals | Multi-keypoint consensus triangulation pattern from refinement.py `_compute_frame_consensus` and `_compute_tracklet_distances` — adapted to score singleton ray against group's triangulated 3D keypoints |
| RECOV-02 | Singletons with strong overall match to one group are assigned to that group | Greedy best-first assignment over sorted (singleton, group) residual pairs; hard threshold in metres; same-camera detected-frame overlap check before scoring |
| RECOV-03 | Singletons with no overall match but a temporal split matching two different groups are split and assigned (swap-aware recovery) | Binary sweep over all split points; both segments must match different groups below threshold; configurable min_segment_length; per-segment same-camera check |
| RECOV-04 | Same-camera overlap constraint enforced during singleton assignment (detected frames only) | Pattern already established in `build_must_not_link()` in clustering.py using `frame_status == "detected"` sets; recovery re-applies same logic per-pair |
</phase_requirements>

---

## Summary

Phase 91 adds a singleton recovery pass immediately after `validate_groups()` in `AssociationStage.run()`. At that point, groups with `len(tracklets) == 1` are singletons — they were either isolated by Leiden clustering, evicted during validation, or produced as split fragments. Recovery scores each singleton against every multi-tracklet group using multi-keypoint ray-to-3D residuals, attempts whole assignment, then attempts split-assign (swap recovery), and leaves unmatched singletons unchanged.

The core algorithmic pattern is already present in the codebase. `refinement.py` (still present at research time, deleted in Phase 90 Plan 02) has `_compute_frame_consensus()` for on-demand triangulation and `_compute_tracklet_distances()` for ray-to-point distance scoring. Phase 91 adapts these into a recovery-specific module (`recovery.py`) that operates singleton-to-group rather than tracklet-to-own-group. The split-assign sweep is a simple O(N * M * G) loop where N = singleton frames, M = split points, G = groups — tractable for typical chunk sizes (200 frames, ~9 fish).

The integration point is `AssociationStage.run()` between `validate_groups()` and `context.tracklet_groups = groups`. New config fields follow the `ValidationConfigLike` protocol pattern: a `RecoveryConfigLike` protocol defined in `recovery.py` with fields added to `AssociationConfig`. TrackletGroup is a frozen dataclass — group membership updates require constructing new instances. Singleton tracklets with `len(tracklets) == 1` are detected by inspection; no flag field is needed.

**Primary recommendation:** Create `src/aquapose/core/association/recovery.py` with `RecoveryConfigLike` and `recover_singletons()`, add three fields to `AssociationConfig`, wire after `validate_groups()` in `stage.py`, and export from `__init__.py`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | project pin | Ray geometry, distance arrays, split sweep | Already used throughout association; all computations are float64 numpy |
| torch | project pin | `ForwardLUT.cast_ray()` returns tensors — must `.cpu().numpy()` | LUT interface is torch-based; caller converts |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses (stdlib) | stdlib | `RecoveryConfigLike` Protocol, new config fields | Config protocol follows existing pattern |
| typing.Protocol + runtime_checkable | stdlib | Config boundary pattern (core/ must not import engine/) | IB-003 import boundary enforcement |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| On-demand triangulation per (singleton, group) pair | Cache group consensus 3D | Caching adds complexity and TrackletGroup is frozen (no cache field); on-demand is simpler and the group count is small (~9 fish) |
| Binary split sweep | Hungarian assignment over segments | Binary split is O(N) per singleton; Hungarian would require segment enumeration; binary split matches user decision |

**Installation:** No new dependencies required.

---

## Architecture Patterns

### Recommended Project Structure

```
src/aquapose/core/association/
├── recovery.py          # NEW: RecoveryConfigLike protocol + recover_singletons()
├── validation.py        # Phase 90: validate_groups() (replaces refinement.py)
├── scoring.py           # ray_ray_closest_point, ray_ray_closest_point_batch
├── clustering.py        # build_must_not_link, cluster_tracklets
├── stage.py             # AssociationStage.run() — integration point
├── types.py             # TrackletGroup, Tracklet2D (frozen dataclasses)
└── __init__.py          # Public API exports
```

### Pattern 1: Config Protocol (IB-003 Import Boundary)

**What:** Define a `runtime_checkable` Protocol in the core module that `AssociationConfig` satisfies structurally, without importing from `engine/`.
**When to use:** Any new config fields accessed by core/ code.

```python
# Source: existing pattern in scoring.py, refinement.py, clustering.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class RecoveryConfigLike(Protocol):
    recovery_enabled: bool
    recovery_residual_threshold: float
    recovery_min_shared_frames: int
    recovery_min_segment_length: int
    keypoint_confidence_floor: float
```

### Pattern 2: On-Demand Consensus Triangulation

**What:** For each (singleton, group) pair, compute per-frame per-keypoint 3D reference positions from the group's member tracklets for frames shared with the singleton.
**When to use:** Scoring singleton against a candidate group.

```python
# Source: adapted from refinement.py _compute_frame_consensus
# Key adaptation: use keypoints array (T, K, 2) instead of centroids
# for multi-keypoint 3D positions

def _triangulate_group_keypoints(
    shared_frames: list[int],
    group_tracklets: tuple,          # tuple[Tracklet2D, ...]
    forward_luts: dict[str, ForwardLUT],
    keypoint_confidence_floor: float,
) -> dict[int, np.ndarray | None]:   # frame -> (K, 3) or None
    """Triangulate per-keypoint 3D positions for a group on shared frames.

    For each frame, for each keypoint k, cast rays from all group tracklets
    that have confident keypoint k, compute pairwise midpoints, average to
    get consensus_3d[k]. Returns None for frames where fewer than 2 cameras
    have any confident keypoint.
    """
    ...
```

### Pattern 3: Ray-to-3D-Point Distance (per keypoint)

**What:** For each frame, for each confident keypoint in the singleton, compute perpendicular distance from the singleton's ray to the group's triangulated 3D position for that keypoint.
**When to use:** Computing the singleton-to-group residual.

```python
# Source: refinement.py _point_to_ray_distance — used directly
def _point_to_ray_distance(point, origin, direction) -> float:
    w = point - origin
    t = float(np.dot(w, direction))
    closest = origin + t * direction
    return float(np.linalg.norm(point - closest))
```

### Pattern 4: Greedy Assignment with Same-Camera Check

**What:** Build a list of (mean_residual, singleton_idx, group_idx) tuples, sort ascending, assign greedily while enforcing same-camera detected-frame overlap.
**When to use:** Whole-tracklet assignment pass.

```python
# Overlap check pattern from clustering.py build_must_not_link
def _has_camera_overlap(singleton_tracklet, group, *, detected_only=True) -> bool:
    """Return True if singleton's camera already has a detected-frame
    overlap with any tracklet in the group from the same camera."""
    cam = singleton_tracklet.camera_id
    for t in group.tracklets:
        if t.camera_id != cam:
            continue
        # detected frames only
        det_s = {f for f, s in zip(singleton_tracklet.frames, singleton_tracklet.frame_status) if s == "detected"}
        det_t = {f for f, s in zip(t.frames, t.frame_status) if s == "detected"}
        if det_s & det_t:
            return True
    return False
```

### Pattern 5: Binary Split Sweep

**What:** For a singleton with no whole-group match, try every valid split point. Score the "before" segment against all groups and the "after" segment against all groups. Pick the split point that minimizes total residual with both segments matching distinct groups.
**When to use:** Split-assign pass after whole assignment pass finds no match.

```python
# Pseudo-structure (exact implementation at Claude's discretion)
for split_idx in range(min_segment_length, n_frames - min_segment_length + 1):
    seg_before = _slice_tracklet(singleton, 0, split_idx)
    seg_after  = _slice_tracklet(singleton, split_idx, n_frames)
    best_before = _score_singleton_against_groups(seg_before, groups, luts, config)
    best_after  = _score_singleton_against_groups(seg_after,  groups, luts, config)
    if (best_before is not None and best_after is not None
            and best_before.group_idx != best_after.group_idx):
        candidate_splits.append((best_before.residual + best_after.residual, split_idx, best_before, best_after))
```

### Pattern 6: TrackletGroup Reassembly (Frozen Dataclass)

**What:** TrackletGroup is `frozen=True` — adding a singleton requires constructing a new instance with an updated `tracklets` tuple.
**When to use:** Every assignment that adds a singleton (or segment) to a group.

```python
# Source: refinement.py lines 185-191 (same pattern for ID reassignment)
updated_group = TrackletGroup(
    fish_id=group.fish_id,
    tracklets=group.tracklets + (singleton_tracklet,),
    confidence=group.confidence,
    per_frame_confidence=None,   # invalidated by new member
    consensus_centroids=None,    # invalidated by new member
)
```

Note: `per_frame_confidence` and `consensus_centroids` should be set to `None` after assignment since the group's triangulation is now stale. Downstream (Phase 92 eval) reads these fields for diagnostics; they will be `None` for recovered groups. This is acceptable — the fields are declared `| None` and all consumers guard against `None`.

### Pattern 7: Tracklet Slicing for Split-Assign

**What:** Produce two new `Tracklet2D` instances from sliced tuple and numpy array fields.
**When to use:** When a singleton is split and each segment is assigned separately.

```python
# Source: established in Phase 90 validation.py _split_tracklet_at()
# Same slicing pattern applies here
def _slice_tracklet(t: Tracklet2D, start: int, end: int, new_id: int) -> Tracklet2D:
    kpts = t.keypoints[start:end] if t.keypoints is not None else None
    kconf = t.keypoint_conf[start:end] if t.keypoint_conf is not None else None
    return Tracklet2D(
        camera_id=t.camera_id,
        track_id=new_id,
        frames=t.frames[start:end],
        centroids=t.centroids[start:end],
        bboxes=t.bboxes[start:end],
        frame_status=t.frame_status[start:end],
        keypoints=kpts,
        keypoint_conf=kconf,
    )
```

### Anti-Patterns to Avoid

- **Mutating TrackletGroup in place:** It's `frozen=True`; all updates need new instances via `dataclasses.replace()` or the constructor directly.
- **Importing engine/ from core/:** Use the Protocol pattern for config. `recovery.py` must not `import AssociationConfig` directly.
- **Coasted frames in overlap check:** Only `frame_status == "detected"` frames count for the same-camera constraint; coasted frames are Kalman predictions and must not block recovery.
- **Re-scoring after assignment:** Single-pass greedy is the locked decision. Do not iteratively update group consensus and re-score remaining singletons.
- **Bare `.numpy()` on LUT output:** Always `.cpu().numpy()` — LUT tensors may be on CUDA.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ray-to-point perpendicular distance | Custom geometry | `_point_to_ray_distance()` from refinement.py (copy into recovery.py) | Tested, handles edge cases |
| Ray casting from pixel coordinates | Direct LUT math | `forward_lut.cast_ray(pixels_tensor)` | LUT handles refractive geometry; caller converts to numpy |
| Ray-ray midpoint for consensus | Custom | `ray_ray_closest_point()` from scoring.py (import) | Already vectorizable; used in refinement.py |
| Tracklet slicing | New data structure | `Tracklet2D` constructor with sliced fields | frozen dataclass pattern established in Phase 90 |
| Same-camera constraint | New algorithm | Re-use `build_must_not_link()` logic (detected-frames-only pattern from clustering.py) | Pattern already correct and tested |

**Key insight:** The majority of the computation (triangulation, ray geometry, tracklet slicing) is already implemented in the codebase. Phase 91 is primarily about orchestrating existing primitives in a new control flow, not building new algorithms.

---

## Common Pitfalls

### Pitfall 1: Config Field Name Conflicts with Phase 90
**What goes wrong:** Phase 90 adds `min_segment_length` to `AssociationConfig`. Phase 91 also needs a `min_segment_length` for its split-assign pass — but these are the same concept (minimum frames per segment). Using the same field for both is fine; adding a duplicate with a different name causes config bloat.
**Why it happens:** Two phases both deal with segment splitting but in different contexts.
**How to avoid:** Check whether Phase 90's `min_segment_length` field can be reused for Phase 91's recovery split minimum. If the thresholds might differ (validation splits inside groups vs. recovery splits on orphan singletons), use a distinct field name like `recovery_min_segment_length`.
**Warning signs:** AssociationConfig having two identically-named or semantically-identical fields.

### Pitfall 2: Group Consensus Staleness After Assignment
**What goes wrong:** After adding a singleton to a group, the group's `per_frame_confidence` and `consensus_centroids` fields are stale (they were computed without the new tracklet). If downstream code uses these fields without guarding for `None`, it crashes.
**Why it happens:** TrackletGroup fields are populated during validation; assignment doesn't re-triangulate.
**How to avoid:** Set both to `None` in the replacement TrackletGroup after any assignment. Downstream consumers (reconstruction, evaluation) already guard against `None` per existing code.
**Warning signs:** `consensus_centroids` being accessed on groups that had singletons assigned without re-triangulating.

### Pitfall 3: track_id Collision for Split Tracklets
**What goes wrong:** When a singleton is split into two segments, both segments need unique `track_id` values that don't collide with any existing tracklet in the pipeline.
**Why it happens:** `track_id` is local to a camera but must be unique within that camera across the entire group list.
**How to avoid:** Collect all existing track_ids across all cameras before the recovery pass; generate new IDs as `max(existing_ids) + 1`, `+ 2`, etc. Phase 90's `_split_tracklet_at()` establishes this pattern — copy it.
**Warning signs:** Duplicate `(camera_id, track_id)` pairs in the output group list.

### Pitfall 4: Shared Frames Counting — Index vs Frame Number
**What goes wrong:** "Shared frames" between a singleton and a group means shared frame indices (the `frames` tuple values), not shared array positions. Intersecting `set(singleton.frames)` with the union of `set(t.frames) for t in group.tracklets` gives the correct shared set.
**Why it happens:** `frames` is a tuple of frame indices (not necessarily 0-based or contiguous).
**How to avoid:** Always build `frame_map = {f: i for i, f in enumerate(t.frames)}` for index lookups; use set intersection on frame values for overlap detection.
**Warning signs:** Off-by-one in consensus lookups; wrong frame counts triggering `recovery_min_shared_frames` guard.

### Pitfall 5: Split Sweep on Short Singletons
**What goes wrong:** Attempting a split sweep when the singleton is shorter than `2 * min_segment_length` produces no valid split points and should skip gracefully. An off-by-one in the range produces an empty loop or an invalid segment.
**Why it happens:** Range for split_idx must be `[min_segment_length, n_frames - min_segment_length]` inclusive.
**How to avoid:** Guard: `if n_frames < 2 * min_segment_length: return None` before the sweep.
**Warning signs:** `_slice_tracklet` called with empty frame range.

### Pitfall 6: No Keypoints Fallback
**What goes wrong:** A singleton has `keypoints=None` (old-style tracklet or pre-Phase-87 data). Multi-keypoint scoring crashes with `None` index.
**Why it happens:** `keypoints` and `keypoint_conf` are optional fields on `Tracklet2D`.
**How to avoid:** Guard at the top of the scoring function: if `singleton.keypoints is None`, fall back to centroid-only ray distance (same pattern as validation.py's keypoints=None fallback). If the group also has keypoints=None tracklets, use centroids for those too.
**Warning signs:** `TypeError: 'NoneType' object is not subscriptable` in scoring.

---

## Code Examples

### Singleton Detection

```python
# Source: inference from clustering.py and refinement.py patterns
def _partition_groups(
    groups: list[TrackletGroup],
) -> tuple[list[TrackletGroup], list[TrackletGroup]]:
    """Split groups into multi-tracklet groups and singletons."""
    multi = [g for g in groups if len(g.tracklets) > 1]
    singletons = [g for g in groups if len(g.tracklets) == 1]
    return multi, singletons
```

### Consensus Triangulation for a Group (Adapted for Multi-Keypoint)

```python
# Source: adapted from refinement.py _compute_frame_consensus
# Key difference: outputs per-keypoint 3D positions, not single centroid
def _triangulate_group_keypoints_for_frame(
    frame: int,
    group_tracklets: tuple,
    forward_luts: dict[str, ForwardLUT],
    keypoint_confidence_floor: float,
    n_keypoints: int,
) -> np.ndarray | None:
    """Triangulate per-keypoint 3D positions for one frame.

    Returns array of shape (K, 3) or None if fewer than 2 cameras
    have any confident keypoint in this frame.
    """
    # For each keypoint k, collect rays from tracklets that are
    # detected in this frame AND have confident keypoint k.
    # Use ray_ray_closest_point for pairwise midpoints.
    # Return (K, 3) array; None for keypoints with < 2 valid rays.
    ...
```

### Singleton-to-Group Mean Residual

```python
# Source: adapted from refinement.py _compute_tracklet_distances
def _score_singleton_against_group(
    singleton: Tracklet2D,
    group: TrackletGroup,
    forward_luts: dict[str, ForwardLUT],
    config: RecoveryConfigLike,
) -> float | None:
    """Return mean per-keypoint ray-to-3D residual or None if insufficient overlap."""
    # 1. Find shared frames between singleton and group union
    group_frames = set()
    for t in group.tracklets:
        group_frames.update(t.frames)
    shared = sorted(set(singleton.frames) & group_frames)
    if len(shared) < config.recovery_min_shared_frames:
        return None

    # 2. For each shared frame, triangulate group keypoints on-demand
    # 3. Cast singleton rays for its confident keypoints
    # 4. Compute _point_to_ray_distance for matched keypoints
    # 5. Mean over all (frame, keypoint) pairs with valid data
    ...
```

### Stage Integration Point

```python
# Source: stage.py lines 100-106 (current wiring pattern)
# Phase 91 adds recovery pass AFTER validate_groups, BEFORE assignment to context

# Step 4: Group validation via multi-keypoint residuals (Phase 90)
if forward_luts is not None:
    from aquapose.core.association.validation import validate_groups
    groups = validate_groups(groups, forward_luts, self._config.association)

# Step 5: Singleton recovery (Phase 91)
if forward_luts is not None:
    from aquapose.core.association.recovery import recover_singletons
    groups = recover_singletons(groups, forward_luts, self._config.association)

context.tracklet_groups = groups
```

### Config Fields to Add (Recommended Names and Defaults)

```python
# Source: consistent with existing AssociationConfig field style
# Fields to add to AssociationConfig in engine/config.py

recovery_enabled: bool = True
recovery_residual_threshold: float = 0.025   # Same default as eviction_reproj_threshold
recovery_min_shared_frames: int = 3           # Same default as t_min
recovery_min_segment_length: int = 10         # Same default as validation min_segment_length
```

Using `recovery_` prefix avoids collision with Phase 90's `min_segment_length` while making field purpose explicit.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Centroid-only eviction (refinement.py) | Multi-keypoint residuals with changepoint (validation.py) | Phase 90 | Richer signal; surgical splits instead of full evictions |
| No post-eviction recovery | Singleton recovery pass (Phase 91) | Phase 91 | Reduces orphaned singletons from ~27% toward target |
| Pairwise scoring only at initial clustering | Recovery scoring singleton-to-group after validation | Phase 91 | Second chance for fragments from validation |

**Important state at Phase 91 start:**
- `refinement.py` is deleted (Phase 90 Plan 02). The helper functions `_compute_frame_consensus`, `_compute_tracklet_distances`, `_point_to_ray_distance` exist only in `validation.py` at that point. Copy needed functions into `recovery.py` — do NOT import from `validation.py` (keep modules independent for future deletion flexibility).
- `AssociationConfig` has `min_segment_length`, `validation_enabled`, `min_cameras_validate` from Phase 90.
- `stage.py` calls `validate_groups()` at Step 4.

---

## Open Questions

1. **Should recovery re-triangulate group consensus after each assignment?**
   - What we know: User decision is single-pass greedy, no iterative re-scoring.
   - What's unclear: Whether `per_frame_confidence` and `consensus_centroids` on assigned groups should be invalidated (set to None) or recomputed.
   - Recommendation: Set both to None on assigned groups. Recomputation would require re-running full triangulation and adds complexity beyond phase scope. Downstream consumers already guard against None.

2. **Can `recovery_residual_threshold` reuse `eviction_reproj_threshold`?**
   - What we know: Both measure ray-to-3D-point distance in metres. The eviction threshold (0.025m default) was tuned for within-group validation.
   - What's unclear: Whether the same threshold is appropriate for cross-group matching (singletons are noisier by definition).
   - Recommendation: Add a separate `recovery_residual_threshold` field (at Claude's discretion per CONTEXT.md) defaulting to the same 0.025m. This allows independent tuning in Phase 92.

3. **Should `recovery_min_segment_length` be a separate field from Phase 90's `min_segment_length`?**
   - What we know: They serve the same conceptual role (minimum frames per tracklet segment). Phase 90 uses it within validation; Phase 91 uses it for the split sweep on singletons.
   - What's unclear: Whether users will want different values for validation splits vs. recovery splits.
   - Recommendation: Use a separate `recovery_min_segment_length` field (same default 10) to allow independent tuning.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — `src/aquapose/core/association/refinement.py` (triangulation helpers), `scoring.py` (ray geometry), `clustering.py` (same-camera constraint), `stage.py` (integration point), `types.py` (TrackletGroup schema)
- Direct codebase inspection — `src/aquapose/engine/config.py` (AssociationConfig current fields)
- Direct codebase inspection — `src/aquapose/core/tracking/types.py` (Tracklet2D fields including keypoints/keypoint_conf)
- Phase 90 plans 01 and 02 — ValidationConfigLike protocol, config fields being added, `stage.py` wiring after Phase 90

### Secondary (MEDIUM confidence)
- CONTEXT.md locked decisions — algorithm design verified against codebase patterns
- REQUIREMENTS.md RECOV-01 through RECOV-04 — requirement text cross-checked against CONTEXT.md decisions

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all libraries already present
- Architecture patterns: HIGH — all patterns directly observed in existing codebase modules
- Pitfalls: HIGH — derived from direct code inspection of frozen dataclass constraints, existing field semantics, and Phase 90 context

**Research date:** 2026-03-11
**Valid until:** 2026-04-10 (stable codebase; only invalidated if Phase 90 execution deviates from its plans)
