# Phase 82: Association Upgrade — Keypoint Centroid - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Cross-view association uses a mid-body keypoint position instead of the OBB centroid for ray-based matching. The swap happens in the tracker's tracklet builder (where `Tracklet2D.centroids` is populated), not in the association stage itself. Association reads centroids as before — it just gets better values.

Phase 81 is a hard dependency: it restructures the pipeline so raw anatomical keypoints (6-point model) are carried through the pipeline, deferring interpolation to dense midline points until reconstruction.

</domain>

<decisions>
## Implementation Decisions

### Keypoint selection
- Use a configurable keypoint index, defaulting to 2 (spine1 — mid-body anatomical point)
- Config field: `centroid_keypoint_index` in association config (association is the consumer that drives the requirement)
- Hardcoded default, not resolved by name from dataset.yaml

### Fallback behavior
- When the configured keypoint is missing or below confidence threshold for a detected frame, fall back to OBB centroid
- Silent fallback — no per-frame provenance tag indicating keypoint vs OBB source
- Coasted frames (Kalman-predicted, no real detection) naturally use OBB centroid since no keypoints exist — no special handling needed

### Confidence filtering
- Minimum keypoint confidence threshold for centroid usage: separate config field `centroid_confidence_floor` in association config, default 0.3
- Matches the pose backend's confidence_floor by default but is independently configurable
- Below threshold → fall back to OBB centroid for that frame

### Data flow
- Phase 81 moves pose estimation before tracking and carries raw 6-keypoint data on detections
- The tracker's tracklet builder reads the configured keypoint from the detection's raw keypoints when constructing `Tracklet2D.centroids`
- Association stage reads `tracklet.centroids` unchanged — no modification to LUT/ray-ray scoring/Leiden clustering machinery

### Claude's Discretion
- Config validation approach (range checking on keypoint index)
- Exact location and method for reading raw keypoints from detection in the tracklet builder
- Any logging/warning when fallback occurs (debug-level at most)

</decisions>

<specifics>
## Specific Ideas

- The interpolation from 6 keypoints to 15 dense midline points should move to reconstruction (Phase 81 concern, not Phase 82) — carry the raw anatomical keypoints as the primary representation through tracking and association
- Success criterion 3 requires documenting which keypoint index was selected and why (confidence statistics) — gather spine1 confidence stats from existing model outputs

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Tracklet2D` (`core/tracking/types.py`): Already has `centroids` field as `tuple[tuple[float, float], ...]` — no type change needed
- `_TrackletBuilder` (`core/tracking/ocsort_wrapper.py:40-98`): Current centroid computation at lines 72-76, swap point for keypoint extraction
- Association scoring (`core/association/scoring.py:305-308`): Reads `tracklet.centroids` — unchanged
- Association refinement (`core/association/refinement.py`): Reads `tracklet.centroids` — unchanged

### Established Patterns
- Frozen config dataclasses per stage (`engine/config.py`) — association config subtree is where new fields go
- `Tracklet2D` uses tuple fields for immutability — centroid population happens in the mutable builder, then freezes via `to_tracklet2d()`
- Keypoint names: `["nose", "head", "spine1", "spine2", "spine3", "tail"]` (indices 0-5), defined in `training/coco_convert.py`

### Integration Points
- `_TrackletBuilder.append()` in `ocsort_wrapper.py` — where OBB center is currently computed (lines 72-76), needs access to detection's raw keypoints
- Phase 81's new data flow must expose raw keypoints on `Detection` or `AnnotatedDetection` before the tracker runs
- Association config in `engine/config.py` — add `centroid_keypoint_index` and `centroid_confidence_floor` fields

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 82-association-upgrade-keypoint-centroid*
*Context gathered: 2026-03-10*
