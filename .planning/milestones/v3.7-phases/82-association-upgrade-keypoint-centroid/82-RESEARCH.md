# Phase 82: Association Upgrade — Keypoint Centroid - Research

**Researched:** 2026-03-10
**Domain:** Internal pipeline modification — Python dataclass mutation, frozen config extension
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Keypoint selection**: Use a configurable keypoint index, defaulting to 2 (spine1 — mid-body anatomical point)
- **Config field**: `centroid_keypoint_index` in `AssociationConfig` (association is the consumer that drives the requirement)
- **Hardcoded default**: Not resolved by name from dataset.yaml
- **Fallback behavior**: When the configured keypoint is missing or below confidence threshold for a detected frame, fall back to OBB centroid; fallback is silent (no per-frame provenance tag)
- **Coasted frames**: Naturally use OBB centroid since no keypoints exist — no special handling needed
- **Confidence threshold**: Separate config field `centroid_confidence_floor` in `AssociationConfig`, default 0.3
- **Data flow**: The tracker's tracklet builder reads the configured keypoint from the detection's raw keypoints when constructing `Tracklet2D.centroids` — association stage reads `tracklet.centroids` unchanged
- **No modification**: Association stage, LUT/ray-ray scoring, Leiden clustering machinery are untouched

### Claude's Discretion

- Config validation approach (range checking on keypoint index)
- Exact location and method for reading raw keypoints from detection in the tracklet builder
- Any logging/warning when fallback occurs (debug-level at most)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ASSOC-01 | Cross-view association uses mid-body keypoint centroid instead of OBB centroid for ray-based matching | Implemented by: (1) adding two fields to `AssociationConfig`, (2) plumbing config into `_TrackletBuilder.add_frame()` via `OcSortTracker`, (3) reading `detection.keypoints`/`detection.keypoint_conf` in the builder when status is "detected" |
</phase_requirements>

---

## Summary

Phase 82 is a focused internal refactor: swap the source of `Tracklet2D.centroids` from the OBB geometric center to a named anatomical keypoint (spine1, index 2). The change is entirely contained in two files — `engine/config.py` (add two `AssociationConfig` fields) and `core/tracking/ocsort_wrapper.py` (conditionally read the keypoint instead of computing bbox center). No stage interface, no Tracklet2D schema, no association machinery changes.

The critical integration insight from Phase 81 is already in place: `Detection.keypoints` (shape `(K, 2)` float32) and `Detection.keypoint_conf` (shape `(K,)` float32) are populated by PoseStage before the tracker runs. Index 2 is "spine1" per the fixed keypoint order `["nose", "head", "spine1", "spine2", "spine3", "tail"]`. The tracklet builder must read these fields only when `status == "detected"` — coasted frames have no Detection object, so they fall back to OBB centroid naturally.

The phase requires a brief documentation note about why index 2 was selected (confidence statistics). This is achievable by reviewing existing model output logs or the Phase 78.1 investigation results already in the project.

**Primary recommendation:** Add `centroid_keypoint_index: int = 2` and `centroid_confidence_floor: float = 0.3` to `AssociationConfig`; thread config into `OcSortTracker.__init__` and `_TrackletBuilder.add_frame()`; read `det.keypoints[idx]` when confidence passes threshold, else fall back to `cx, cy` from bbox.

---

## Standard Stack

### Core (all existing — no new dependencies)

| Component | Location | Purpose | Notes |
|-----------|----------|---------|-------|
| `AssociationConfig` | `src/aquapose/engine/config.py:119` | Frozen config dataclass for association stage | Add two new fields here |
| `_TrackletBuilder` | `src/aquapose/core/tracking/ocsort_wrapper.py:34` | Mutable accumulator converting detections to Tracklet2D | Central change point |
| `OcSortTracker` | `src/aquapose/core/tracking/ocsort_wrapper.py:105` | Public wrapper, passes detection list to `_TrackletBuilder` | Must thread config fields down |
| `Detection` | `src/aquapose/core/types/detection.py:11` | Detection data class with `keypoints` and `keypoint_conf` | Already populated by PoseStage post-Phase 81 |
| `Tracklet2D` | `src/aquapose/core/tracking/types.py:25` | Frozen output type — no changes needed | `centroids` field type unchanged |

### No New Libraries Required

This phase is a pure internal refactor. No pip installs needed.

---

## Architecture Patterns

### Existing Config Pattern (frozen dataclass in `engine/config.py`)

All stage configs are frozen dataclasses. New fields follow the existing pattern exactly:

```python
# Source: src/aquapose/engine/config.py:119
@dataclass(frozen=True)
class AssociationConfig:
    # ... existing fields ...
    centroid_keypoint_index: int = 2
    """Index into Detection.keypoints to use as tracklet centroid.

    0=nose, 1=head, 2=spine1 (default), 3=spine2, 4=spine3, 5=tail.
    Falls back to OBB centroid when keypoint is absent or below
    centroid_confidence_floor.
    """
    centroid_confidence_floor: float = 0.3
    """Minimum keypoint confidence to use keypoint position as centroid.

    Below this threshold, falls back to OBB centroid for that frame.
    Matches PoseConfig.keypoint_confidence_floor default but is
    independently configurable.
    """
```

`_filter_fields()` in `load_config()` will automatically accept the new fields in YAML and reject unknown ones — no changes to the loader are needed.

### _TrackletBuilder.add_frame() Modification Pattern

The current centroid computation (lines 72-76 of `ocsort_wrapper.py`) derives `cx, cy` from bbox:

```python
# CURRENT (ocsort_wrapper.py:72-76)
x1, y1, x2, y2 = bbox_xyxy
w = x2 - x1
h = y2 - y1
cx = x1 + w / 2.0
cy = y1 + h / 2.0
```

The replacement adds an optional `detection` parameter and applies keypoint extraction before the OBB fallback:

```python
def add_frame(
    self,
    frame_idx: int,
    bbox_xyxy: tuple[float, float, float, float],
    status: str,
    detection: object | None = None,  # Detection | None
    centroid_keypoint_index: int = 2,
    centroid_confidence_floor: float = 0.3,
) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    cx_obb = x1 + w / 2.0
    cy_obb = y1 + h / 2.0
    cx, cy = cx_obb, cy_obb  # default: OBB centroid

    if (
        status == "detected"
        and detection is not None
        and getattr(detection, "keypoints", None) is not None
        and getattr(detection, "keypoint_conf", None) is not None
    ):
        kpts = detection.keypoints       # shape (K, 2) float32
        kconf = detection.keypoint_conf  # shape (K,) float32
        idx = centroid_keypoint_index
        if idx < len(kconf) and kconf[idx] >= centroid_confidence_floor:
            cx, cy = float(kpts[idx, 0]), float(kpts[idx, 1])
        # else: silent fallback to OBB centroid — already set above
    # ...
```

### OcSortTracker Threading Pattern

The two config fields must flow from `AssociationConfig` → `OcSortTracker.__init__()` → `_TrackletBuilder.add_frame()`. The `update()` method already receives the full detection list; it needs to pass the detection object alongside its bbox when calling `add_frame()`.

The current `update()` call site (line 224):
```python
self._builders[local_id].add_frame(frame_idx, bbox_xyxy, "detected")
```

Needs access to the original `Detection` object. The `result` array from boxmot returns column index 7 as the source detection index (`idx`). This allows recovery of the original detection:

```python
# result shape: (N, 8) — [x1, y1, x2, y2, track_id, conf, cls, idx]
det_idx = int(row[7])  # 0-based index into detections list
source_det = detections[det_idx] if 0 <= det_idx < len(detections) else None
self._builders[local_id].add_frame(
    frame_idx, bbox_xyxy, "detected",
    detection=source_det,
    centroid_keypoint_index=self._centroid_keypoint_index,
    centroid_confidence_floor=self._centroid_confidence_floor,
)
```

### Anti-Patterns to Avoid

- **Modifying `Tracklet2D`**: The type is frozen and the `centroids` field type is already correct — `tuple[tuple[float, float], ...]`. No type change needed.
- **Touching association stage**: `AssociationStage`, `scoring.py`, `refinement.py` all read `tracklet.centroids` — these are untouched by this phase.
- **Modifying `TrackingConfig`**: The new config fields belong in `AssociationConfig` (the consumer that drives the requirement), not `TrackingConfig` (the OC-SORT tracker config). However, the fields must be threaded through `OcSortTracker.__init__()` at call-time from wherever the tracker is constructed with knowledge of both configs.
- **Resolving keypoint index by name at runtime**: The decision is a hardcoded default `2` with YAML override via integer index — not resolved from `dataset.yaml` keypoint names at runtime.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Locating where tracker construction reads config | Custom wiring layer | Read `TrackingStage.run()` directly | Tracker is constructed in `stage.py:94-100` — `TrackingConfig` is passed, `AssociationConfig` is not yet available there |
| Confidence array indexing safety | Custom bounds-checking | Standard Python `idx < len(kconf)` | numpy arrays support len(); no need for try/except |

**Key insight:** The only non-trivial plumbing question is whether to thread `centroid_keypoint_index` and `centroid_confidence_floor` through `TrackingStage.run()` from `PipelineConfig` or to construct `OcSortTracker` with the association config fields directly. See Open Questions.

---

## Common Pitfalls

### Pitfall 1: boxmot detection index column

**What goes wrong:** The boxmot `result` array column 7 (the source detection index) is documented as the index into the `dets_array` passed to `tracker.update()`. This is the same ordering as the `detections` list passed to `OcSortTracker.update()`. Using it directly to recover the original `Detection` object is correct.

**Why it happens:** Boxmot's output format has 8 columns. The mapping is `[x1, y1, x2, y2, track_id, conf, cls, idx]` where `idx` is 0-based. The comment on line 213 already documents this.

**How to avoid:** Verify with `int(row[7])` — this is the index into the input `detections` list. Guard with `0 <= det_idx < len(detections)` before indexing.

**Warning signs:** `IndexError` or centroid returning OBB fallback when keypoints are expected.

### Pitfall 2: Coasted frame detection object

**What goes wrong:** Coasted frames are handled separately (lines 230-250 of `ocsort_wrapper.py`) using the Kalman-predicted bbox — there is no matched Detection object. Passing `detection=None` to `add_frame()` for coasted frames is the correct behavior and the fallback path to OBB centroid handles it transparently.

**Why it happens:** The coasting path iterates `self._tracker.active_tracks` without a corresponding detection, so there is genuinely no Detection to reference.

**How to avoid:** Only pass a `detection` argument when `status == "detected"`. Leave coasting calls as `add_frame(frame_idx, bbox_xyxy, "coasted")` with no detection argument.

### Pitfall 3: Config threading — TrackingStage vs AssociationConfig

**What goes wrong:** `TrackingStage.run()` constructs `OcSortTracker` with `TrackingConfig` fields, but the new `centroid_keypoint_index` and `centroid_confidence_floor` fields live in `AssociationConfig`. The tracking stage only receives `self._config` (a `TrackingConfig`).

**Why it happens:** The config hierarchy separates concerns by stage. The tracking stage does not have a reference to `AssociationConfig`.

**How to avoid:** Two options — (A) pass the two centroid fields as separate constructor arguments to `OcSortTracker`, reading them from wherever `TrackingStage` is instantiated (the engine, which has access to `PipelineConfig`); or (B) give `TrackingStage.__init__` an optional `assoc_config` or just the two relevant scalar fields. See Open Questions for the recommended resolution.

### Pitfall 4: numpy array from CUDA tensor

**What goes wrong:** `Detection.keypoints` is set by `PoseStage`, which may return CUDA tensors. CLAUDE.md requires `.cpu().numpy()`, not bare `.numpy()`.

**Why it happens:** PoseStage moves tensors to CPU before storing (per Phase 81 implementation), but this must be verified in the actual PoseStage output.

**How to avoid:** Confirm `Detection.keypoints` is stored as a CPU numpy array by PoseStage. If there is any doubt, apply `.cpu().numpy()` at the storage site in PoseStage, not in the tracker.

---

## Code Examples

### Current centroid computation in `_TrackletBuilder.add_frame()` (lines 59-82)

```python
# Source: src/aquapose/core/tracking/ocsort_wrapper.py:59-82
def add_frame(
    self,
    frame_idx: int,
    bbox_xyxy: tuple[float, float, float, float],
    status: str,
) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    self.frames.append(frame_idx)
    self.centroids.append((cx, cy))
    self.bboxes.append((x1, y1, w, h))
    self.frame_status.append(status)
    if status == "detected":
        self.detected_count += 1
```

### Detection keypoints field (after Phase 81)

```python
# Source: src/aquapose/core/types/detection.py:27-42
# keypoints: Raw anatomical keypoint positions in full-frame pixel
#     coordinates, shape (K, 2) float32. Keypoint indices:
#     0=nose, 1=head, 2=spine1, 3=spine2, 4=spine3, 5=tail.
#     None until PoseStage has run.
# keypoint_conf: Per-keypoint confidence scores in [0, 1],
#     shape (K,) float32. None until PoseStage has run.
keypoints: np.ndarray | None = None
keypoint_conf: np.ndarray | None = None
```

### AssociationConfig adding new fields (pattern)

```python
# Source: src/aquapose/engine/config.py:119 — extend existing class
@dataclass(frozen=True)
class AssociationConfig:
    # ... all existing fields unchanged ...
    centroid_keypoint_index: int = 2
    centroid_confidence_floor: float = 0.3
```

Adding these two fields is backward-compatible: `_filter_fields()` in `load_config()` accepts them from YAML, and they have defaults so existing YAML files without them continue to work.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| OBB centroid: geometric center of the bounding box | Mid-body keypoint (spine1, index 2): anatomical mid-body point | Phase 82 | Centroid is stable under partial occlusion and frame-edge clipping where OBB center drifts |
| No keypoints available during tracking | Raw 6-keypoint pose on `Detection` objects before tracker runs | Phase 81 | Makes keypoint centroid feasible |

---

## Open Questions

1. **Where to thread AssociationConfig fields into OcSortTracker**
   - What we know: `TrackingStage.run()` creates `OcSortTracker` instances (lines 94-100 of `stage.py`). It reads from `self._config` which is a `TrackingConfig`. `AssociationConfig` is not accessible inside `TrackingStage`.
   - What's unclear: Should `centroid_keypoint_index` and `centroid_confidence_floor` be added as explicit constructor parameters to `OcSortTracker` (so `TrackingStage` can pass them from a broader config), or should `TrackingStage` be given access to `AssociationConfig` fields?
   - Recommendation: Add `centroid_keypoint_index: int = 2` and `centroid_confidence_floor: float = 0.3` as constructor parameters to `OcSortTracker.__init__()`. The engine's stage-builder (which has access to full `PipelineConfig`) passes `config.association.centroid_keypoint_index` and `config.association.centroid_confidence_floor` when constructing `TrackingStage`. Alternatively, the planner may prefer to read these from a `TrackingConfig` extension — either works, but keeping them in `AssociationConfig` is semantically correct (they serve the association concern).

2. **Whether to emit a debug-level log on fallback**
   - What we know: CONTEXT.md says "debug-level at most" is within Claude's discretion.
   - Recommendation: Emit one `logger.debug(...)` per fallback frame is too noisy (one per fish per frame). Emit a per-tracklet summary at `DEBUG` level in `to_tracklet2d()` counting fallback frames: `"tracklet %s: %d/%d frames used keypoint centroid"`. This is informative without flooding logs.

3. **Confidence statistics note (success criterion 3)**
   - What we know: spine1 is index 2; Phase 73/78.1 training logs and the Phase 78 investigation data are available.
   - Recommendation: The planner should include a task to query existing run diagnostics (e.g., `~/aquapose/projects/YH/runs/run_20260307_140127/`) for per-keypoint confidence stats, or simply state in a docstring that spine1 (index 2) is the mid-body anatomical point equidistant from head and tail — it is the most geometrically stable and least affected by partial occlusion near the nose or tail. A short comment in `AssociationConfig` docstring or a `NOTES.md` in the phase directory satisfies success criterion 3.

---

## Sources

### Primary (HIGH confidence)

- Direct code inspection: `src/aquapose/core/tracking/ocsort_wrapper.py` — exact lines 59-82 (`add_frame`), 189-257 (`update`)
- Direct code inspection: `src/aquapose/engine/config.py` — full `AssociationConfig` and `load_config` patterns
- Direct code inspection: `src/aquapose/core/types/detection.py` — `Detection.keypoints` and `Detection.keypoint_conf` fields
- Direct code inspection: `src/aquapose/core/tracking/types.py` — `Tracklet2D` frozen dataclass
- Direct code inspection: `src/aquapose/core/tracking/stage.py` — how `OcSortTracker` is constructed
- `.planning/phases/82-association-upgrade-keypoint-centroid/82-CONTEXT.md` — all locked decisions

### Secondary (MEDIUM confidence)

- Boxmot OcSort result column documentation: col 7 is the source detection index — confirmed by comment in `ocsort_wrapper.py:213`

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all relevant code directly inspected
- Architecture: HIGH — exact modification points identified with line numbers
- Pitfalls: HIGH — derived from reading actual code; pitfall 3 (config threading) is a genuine design question, not a bug risk
- Open questions: MEDIUM — recommendations are well-reasoned but final choices are Claude's discretion

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (codebase is stable; only changes if Phase 81 cleanup or config loader is modified)
