# Phase 87: Tracklet2D Keypoint Propagation - Research

**Researched:** 2026-03-11
**Domain:** Python dataclass extension + NumPy array propagation through tracker output pipeline
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Keypoints stored as `keypoints: np.ndarray | None` with shape `(T, K, 2)`
- Confidences stored as `keypoint_conf: np.ndarray | None` with shape `(T, K)`
- K is inferred from array shape (currently 6 anatomical keypoints), not hardcoded on Tracklet2D
- No validation of array shapes in `to_tracklet2d()` — correct by construction from `_KptTrackletBuilder`
- No `flags.writeable = False` immutability enforcement; frozen dataclass is sufficient
- Coasted frames (both KF coast and gap interpolation): interpolated keypoint positions but confidence = 0.0 for all keypoints
- Frame status rule: `"detected"` → raw YOLO-pose confidence per keypoint; `"coasted"` → 0.0 for all keypoints
- Per-keypoint confidence values pass through raw from YOLO-pose on detected frames — no thresholding at data layer
- `centroids` field stays on Tracklet2D with a deprecation comment; consumers prefer `keypoints[:, centroid_idx, :]` when available
- `keypoints` and `keypoint_conf` default to `None` for type contract correctness
- Phase 88 scoring assumes keypoints are always present — no None-check fallback to centroids

### Claude's Discretion

- Exact field ordering on Tracklet2D dataclass
- Whether `to_tracklet2d()` uses `np.stack()` or `np.array()` for assembly
- Test structure and fixture design for round-trip tests

### Deferred Ideas (OUT OF SCOPE)

- Centroid field removal from Tracklet2D — requires migrating reconstruction, eval, and viz consumers first
- Weighted centroid computation from multiple keypoints
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Tracklet2D carries per-frame keypoints array (T, K, 2) from tracker to association stage | `_KptTrackletBuilder` already accumulates `list[np.ndarray]` per frame; `to_tracklet2d()` must stack them and pass through |
| DATA-02 | Tracklet2D carries per-frame keypoint confidence array (T, K) with 0.0 for coasted frames | `interpolate_gaps()` already sets `np.zeros(n_kp, dtype=np.float32)` on coasted frame confs (line 1012); detected frames get raw YOLO conf passed through |
</phase_requirements>

## Summary

Phase 87 is a surgical data-plumbing change: extend `Tracklet2D` with two new numpy array fields and wire `to_tracklet2d()` to populate them. The core data is already present — `_KptTrackletBuilder` accumulates `keypoints: list[np.ndarray]` and `keypoint_conf: list[np.ndarray]` on every `add_frame()` call, and `interpolate_gaps()` already assigns `np.zeros(n_kp, dtype=np.float32)` as conf for interpolated (coasted) frames. The entire implementation is two file changes plus tests.

The one subtlety is that `interpolate_gaps()` sets conf=0.0 for interpolated frames but KF-coasted frames (missed detections filled by the tracker's `mark_missed()` path) also receive conf=0.0 because the builder's `add_frame()` is called with `kconf` already zeroed by the `_SinglePassTracker` coast path. Both coast sources are already handled correctly by existing code — no logic changes needed in the tracker, only in `to_tracklet2d()`.

**Primary recommendation:** Add `keypoints` and `keypoint_conf` fields to `Tracklet2D`, update `to_tracklet2d()` to stack `builder.keypoints` and `builder.keypoint_conf` into `(T, K, 2)` and `(T, K)` arrays, add deprecation comment to `centroids`, and write round-trip tests via `KeypointTracker`.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | project dep | `np.stack()` to assemble list-of-arrays into `(T, K, 2)` / `(T, K)` | Already used everywhere in the tracker; no new dependency |
| dataclasses | stdlib | `@dataclass(frozen=True)` extension | Existing `Tracklet2D` pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy `Optional` type hint | — | `np.ndarray | None` field typing | Fields default to None for backward compat |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `np.stack(builder.keypoints)` | `np.array(builder.keypoints)` | Both work; `np.stack()` is more explicit about the stacking dimension and raises clearly if shapes are inconsistent — prefer `np.stack()` |
| `(T, K, 2)` numpy array | tuple of tuples | Numpy array enables vectorized slicing by Phase 88 — tuples would force list comprehensions in hot scoring path |

## Architecture Patterns

### Files to Modify

```
src/aquapose/core/tracking/
├── types.py          # Add keypoints + keypoint_conf fields to Tracklet2D
└── keypoint_tracker.py  # Update _KptTrackletBuilder.to_tracklet2d()

tests/unit/core/tracking/
└── test_tracking_stage.py  # Add keypoint round-trip test class
    OR
└── test_keypoint_tracker.py  # Add round-trip via KeypointTracker (preferred)
```

### Pattern 1: Extending the Frozen Dataclass

`Tracklet2D` is `@dataclass(frozen=True)`. Adding new optional fields with `None` defaults is a valid, backward-compatible extension — all existing constructors (clustering merge in `clustering.py:498`, tracker `to_tracklet2d()`) will need to be checked and updated if they construct `Tracklet2D` directly.

**Known direct Tracklet2D constructors:**
1. `_KptTrackletBuilder.to_tracklet2d()` at `keypoint_tracker.py:430` — primary target
2. `_merge_fragments()` in `clustering.py:498` — constructs without keypoint data; must pass `keypoints=None, keypoint_conf=None` (or rely on defaults)

Both must be compatible after the field addition. Since dataclass fields with defaults don't require positional args, existing keyword-argument constructors continue to work. The clustering merge drops keypoints (acceptable — it produces a deprecated fragment-merge path).

### Pattern 2: Stacking in `to_tracklet2d()`

```python
# keypoint_tracker.py — _KptTrackletBuilder.to_tracklet2d()
def to_tracklet2d(self) -> Tracklet2D:
    return Tracklet2D(
        camera_id=self.camera_id,
        track_id=self.track_id,
        frames=tuple(self.frames),
        centroids=tuple(self.centroids),
        bboxes=tuple(self.bboxes),
        frame_status=tuple(self.frame_status),
        keypoints=np.stack(self.keypoints) if self.keypoints else None,
        keypoint_conf=np.stack(self.keypoint_conf) if self.keypoint_conf else None,
    )
```

`np.stack(list_of_arrays)` produces `(T, K, 2)` for keypoints and `(T, K)` for conf when each element is `(K, 2)` / `(K,)` respectively.

### Pattern 3: Deprecation Comment (not runtime warning)

The CONTEXT.md locks: "centroids field stays on Tracklet2D with a deprecation comment". Add a docstring comment, not a `DeprecationWarning`. This is intentional — a runtime warning would spam the pipeline logs.

### Anti-Patterns to Avoid

- **Validating array shapes in `to_tracklet2d()`**: Locked out by user decision. Trust `_KptTrackletBuilder` to maintain consistent shapes.
- **Setting `flags.writeable = False`**: Locked out. Frozen dataclass is sufficient.
- **Adding K as a field on Tracklet2D**: Not needed; infer from `t.keypoints.shape[1]`.
- **Modifying `interpolate_gaps()`**: Already correct — coasted frames get `np.zeros(n_kp, dtype=np.float32)` as conf (verified at `keypoint_tracker.py:1012`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Stacking list of arrays | Loop + preallocate | `np.stack()` | Handles dtype, shape validation, contiguity in one call |
| Coasted-frame zero conf | Post-hoc masking loop | `interpolate_gaps()` already does it | Already sets `np.zeros(n_kp)` — no new logic needed |

**Key insight:** The coasted-confidence rule is already implemented. `interpolate_gaps()` at line 1012 sets `new_kconfs.insert(insert_idx, np.zeros(n_kp, dtype=np.float32))`. KF-coast frames also arrive with zeroed conf because `_SinglePassTracker` calls `add_frame()` with the coasted keypoint positions and their original conf — but actual KF-coast frames in the `_SinglePassTracker.update()` flow use `trk.predict()` without a detection match, so `add_frame()` is NOT called for KF-coast frames; instead they are left as gaps that `interpolate_gaps()` later fills. This is the unified path.

**Verification:** Grepped `add_frame` call sites in `_SinglePassTracker.update()` — `add_frame` is only called on the matched detection path (line 775–781). Unmatched frames are gaps. `interpolate_gaps()` fills them with conf=0.0. No separate KF-coast branch calls `add_frame`.

## Common Pitfalls

### Pitfall 1: Forgotten Tracklet2D Constructor in clustering.py

**What goes wrong:** `_merge_fragments()` at `clustering.py:498` constructs `Tracklet2D(...)` without keyword args for the new fields. If new fields don't have defaults, this breaks. If fields have defaults (`None`) but the function passes only positional args for the old fields, it still works — but it silently produces a merged tracklet with `keypoints=None`.

**Why it happens:** Code search found a second Tracklet2D constructor site that the CONTEXT.md doesn't mention.

**How to avoid:** Check both constructor call sites. The merge function can safely pass `keypoints=None, keypoint_conf=None` explicitly (or rely on defaults if fields have `= None` defaults).

**Warning signs:** `TypeError: Tracklet2D.__init__() got unexpected keyword argument` or missing-field AttributeError at clustering time.

### Pitfall 2: numpy array in frozen dataclass field

**What goes wrong:** `@dataclass(frozen=True)` prevents field reassignment, but does NOT prevent in-place mutation of the numpy array itself (`t.keypoints[0, 0, 0] = 99` still works). This is acceptable per the locked decision — frozen dataclass is sufficient.

**Why it happens:** Python's `frozen=True` only locks `__setattr__`/`__delattr__`.

**How to avoid:** No action needed per locked decision. Document in docstring that `keypoints` should be treated as read-only.

### Pitfall 3: Empty builder edge case

**What goes wrong:** If `builder.keypoints` is an empty list (zero-frame tracklet), `np.stack([])` raises `ValueError: need at least one array to stack`.

**Why it happens:** The `get_tracklets()` guard `if b.detected_count >= self._n_init and b.frames` ensures builders always have at least `n_init` frames before `to_tracklet2d()` is called. So in practice this path is unreachable for valid tracklets. But defensive coding with `if self.keypoints else None` avoids the crash.

**How to avoid:** Use `np.stack(self.keypoints) if self.keypoints else None`.

### Pitfall 4: dtype consistency

**What goes wrong:** `_KptTrackletBuilder.add_frame()` stores both `kpts.copy()` and `kconf.copy()` — the dtypes come from whatever the caller passes. Detected frames get `float32` (from `kpts_arr.astype(np.float32)` in `_SinglePassTracker`). Interpolated frames get `float32` (from `np.zeros(..., dtype=np.float32)` in `interpolate_gaps()`). The stacked array will be `float32` throughout.

**How to avoid:** No special handling needed — dtype is consistently `float32` across both frame sources.

## Code Examples

### Tracklet2D field addition

```python
# src/aquapose/core/tracking/types.py
import numpy as np

@dataclass(frozen=True)
class Tracklet2D:
    """..."""
    camera_id: str
    track_id: int
    frames: tuple
    centroids: tuple  # Deprecated: prefer keypoints[:, centroid_idx, :] when available
    bboxes: tuple
    frame_status: tuple
    keypoints: np.ndarray | None = None     # (T, K, 2), float32
    keypoint_conf: np.ndarray | None = None  # (T, K), float32; 0.0 on coasted frames
```

### to_tracklet2d() update

```python
# src/aquapose/core/tracking/keypoint_tracker.py
def to_tracklet2d(self) -> Tracklet2D:
    return Tracklet2D(
        camera_id=self.camera_id,
        track_id=self.track_id,
        frames=tuple(self.frames),
        centroids=tuple(self.centroids),
        bboxes=tuple(self.bboxes),
        frame_status=tuple(self.frame_status),
        keypoints=np.stack(self.keypoints) if self.keypoints else None,
        keypoint_conf=np.stack(self.keypoint_conf) if self.keypoint_conf else None,
    )
```

### Round-trip test pattern (fits existing test style)

```python
class TestTracklet2DKeypointRoundtrip:
    def test_keypoints_shape_matches_frames(self) -> None:
        """keypoints array shape is (T, K, 2) where T == len(frames)."""
        tracker = _make_keypoint_tracker()
        n_frames = 6  # > n_init=3
        dets = _make_linear_detections_kpt(n_fish=1, n_frames=n_frames)
        for i, frame_dets in enumerate(dets):
            tracker.update(frame_idx=i, detections=frame_dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1
        t = tracklets[0]
        assert t.keypoints is not None
        assert t.keypoints.shape == (len(t.frames), 6, 2)

    def test_keypoint_conf_shape_matches_frames(self) -> None:
        """keypoint_conf array shape is (T, K) where T == len(frames)."""
        # similar structure...

    def test_coasted_frames_have_zero_conf(self) -> None:
        """Interpolated gap frames have conf=0.0 for all keypoints."""
        # feed frames [0,1,2, skip 3, 4,5,6]; gap at 3 gets filled with 0.0 conf
        tracker = _make_keypoint_tracker(max_gap_frames=5, n_init=3)
        dets = _make_linear_detections_kpt(n_fish=1, n_frames=7)
        dets[3] = []  # drop frame 3 — creates a gap
        for i, frame_dets in enumerate(dets):
            tracker.update(frame_idx=i, detections=frame_dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1
        t = tracklets[0]
        assert t.keypoints is not None
        for i, status in enumerate(t.frame_status):
            if status == "coasted":
                assert np.all(t.keypoint_conf[i] == 0.0)

    def test_detected_frames_have_nonzero_conf(self) -> None:
        """Detected frames retain raw YOLO conf > 0."""
        # ...
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via hatch) |
| Config file | pyproject.toml |
| Quick run command | `hatch run test` |
| Full suite command | `hatch run test-all` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | `Tracklet2D.keypoints` is `(T, K, 2)` float32 ndarray after tracker round-trip | unit | `hatch run test tests/unit/core/tracking/test_keypoint_tracker.py -k keypoint_roundtrip` | ❌ Wave 0 |
| DATA-01 | `Tracklet2D.keypoints` is `None` by default (backward compat) | unit | `hatch run test tests/unit/core/tracking/` | ✅ (implicit — new field with None default won't break existing tests) |
| DATA-02 | Coasted frames have `keypoint_conf == 0.0` | unit | `hatch run test tests/unit/core/tracking/test_keypoint_tracker.py -k coasted_conf` | ❌ Wave 0 |
| DATA-02 | Detected frames have raw conf passed through | unit | `hatch run test tests/unit/core/tracking/test_keypoint_tracker.py -k detected_conf` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `hatch run test`
- **Per wave merge:** `hatch run test`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/unit/core/tracking/test_keypoint_tracker.py` — add `TestTracklet2DKeypointRoundtrip` class with 4 tests covering DATA-01 and DATA-02 (file exists, class is new)

## Open Questions

1. **Does the clustering merge path need keypoints?**
   - What we know: `_merge_fragments()` in `clustering.py` constructs `Tracklet2D` directly (line 498) without keypoint fields. Fragment merging will be removed in Phase 89 (CLEAN-01).
   - What's unclear: Whether fragment merging is on the hot path between Phases 87 and 89. During Phase 87, this code still runs.
   - Recommendation: The merged tracklet will have `keypoints=None`. Phase 88 scoring "assumes keypoints are always present" — but merged tracklets come from the clustering pre-processing step, not directly from the tracker. Clarify whether merged tracklets reach the scorer, or if they are edge cases. If they do reach the scorer, the `None` keypoints will cause Phase 88 to fail. This may be acceptable since fragment merging is being removed in Phase 89, but the planner should flag it.

2. **`np.stack()` vs `np.array()` for assembly**
   - What we know: Both produce identical output for uniform shapes. `np.stack()` raises a clear error if any element has a different shape; `np.array()` may silently create an object array.
   - Recommendation: Use `np.stack()` — Claude's discretion, and it's safer.

## Sources

### Primary (HIGH confidence)
- Direct code read of `keypoint_tracker.py` lines 356–437, 793–803, 907–1034, 1127–1157 — `_KptTrackletBuilder`, `to_tracklet2d()`, `interpolate_gaps()`, `KeypointTracker.get_tracklets()`
- Direct code read of `types.py` lines 1–51 — `Tracklet2D` dataclass structure
- Direct code read of `clustering.py` lines 480–505 — second `Tracklet2D` constructor site
- Direct code read of `test_keypoint_tracker.py` lines 485–618 — existing test helper patterns and round-trip test style
- Direct code read of `test_tracking_stage.py` lines 1–333 — existing TrackingStage test patterns

### Secondary (MEDIUM confidence)
- Python dataclass frozen behavior: stdlib `dataclasses` documentation (training knowledge, HIGH confidence — this is stable Python stdlib behavior)
- `np.stack()` shape semantics: numpy stdlib (training knowledge, HIGH confidence)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use, no new dependencies
- Architecture: HIGH — code read confirms exact file/line locations, no speculation
- Pitfalls: HIGH — identified from direct code inspection (second Tracklet2D constructor, empty-list edge case)

**Research date:** 2026-03-11
**Valid until:** 2026-04-10 (stable internal code, no external dependencies)
