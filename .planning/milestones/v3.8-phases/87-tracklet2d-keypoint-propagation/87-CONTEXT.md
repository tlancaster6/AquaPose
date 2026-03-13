# Phase 87: Tracklet2D Keypoint Propagation - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend Tracklet2D with per-frame keypoint and confidence arrays so this data flows from the tracking stage to the association stage. The tracker's internal `_KptTrackletBuilder` already stores keypoints and confidences — this phase promotes them onto the public `Tracklet2D` type and updates `to_tracklet2d()` to populate them.

</domain>

<decisions>
## Implementation Decisions

### Array storage format
- Keypoints stored as a single stacked numpy array: `keypoints: np.ndarray | None` with shape `(T, K, 2)`
- Confidences stored as a single stacked numpy array: `keypoint_conf: np.ndarray | None` with shape `(T, K)`
- K is inferred from the array shape (currently 6 anatomical keypoints), not hardcoded on Tracklet2D
- No validation of array shapes in `to_tracklet2d()` — correct by construction from `_KptTrackletBuilder`
- Trust consumers not to mutate arrays (no `flags.writeable = False`); frozen dataclass prevents field reassignment, which is sufficient

### Coasted-frame confidence semantics
- Coasted frames (both KF coast and gap interpolation) store interpolated keypoint positions but confidence = 0.0 for all keypoints
- Simple rule: `frame_status == "detected"` → raw YOLO-pose confidence per keypoint; `frame_status == "coasted"` → 0.0 for all keypoints
- On detected frames, per-keypoint confidence values pass through raw from YOLO-pose — no thresholding or modification at the data layer
- Downstream consumers (Phase 88 scoring) apply their own confidence floor to filter low-confidence keypoints

### Backward compatibility
- `centroids` field stays on Tracklet2D with a deprecation comment noting consumers should prefer `keypoints[:, centroid_idx, :]` when available
- `keypoints` and `keypoint_conf` default to `None` for type contract correctness, but the only existing tracker always populates them
- Phase 88 scoring assumes keypoints are always present — no None-check fallback to centroids
- Low-confidence centroid keypoint (spine1) on individual frames is handled by the per-keypoint confidence floor in scoring — other keypoints still produce rays, so no special centroid fallback logic is needed

### Claude's Discretion
- Exact field ordering on Tracklet2D dataclass
- Whether `to_tracklet2d()` uses `np.stack()` or `np.array()` for assembly
- Test structure and fixture design for round-trip tests

</decisions>

<specifics>
## Specific Ideas

- The `_KptTrackletBuilder` already has `keypoints: list[np.ndarray]` and `keypoint_conf: list[np.ndarray]` — the implementation is mostly wiring `to_tracklet2d()` to stack and pass them through
- User wants centroids deprecated cleanly — downstream migration (eval, viz, reconstruction) is out of scope for this phase but the intent is clear

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_KptTrackletBuilder` (keypoint_tracker.py:357): Already accumulates per-frame keypoints (6,2) and conf (6,) arrays — just needs stacking in `to_tracklet2d()`
- `interpolate_gaps()` (keypoint_tracker.py:907): Already handles keypoint interpolation for gap-filled frames — needs to set conf=0.0 on interpolated frames (may already do this)

### Established Patterns
- Tracklet2D is a frozen dataclass with tuple fields for per-frame data (centroids, bboxes, frame_status)
- Keypoint arrays break the tuple pattern (numpy arrays instead) but this is appropriate for array data that will be sliced/indexed vectorially

### Integration Points
- `to_tracklet2d()` (keypoint_tracker.py:424): Primary conversion point — currently drops keypoints, needs to stack and pass them
- `Tracklet2D` (types.py:18): Add two new fields with None defaults
- Association scoring (scoring.py:304-308): Currently reads `tracklet.centroids` — Phase 88 will switch to `tracklet.keypoints`
- Tests: Round-trip test through `KeypointTracker.update()` → `get_tracklets()` → verify keypoint arrays match input

</code_context>

<deferred>
## Deferred Ideas

- Centroid field removal from Tracklet2D — requires migrating reconstruction, eval, and viz consumers first
- Weighted centroid computation from multiple keypoints — potential improvement for consumers that need a single representative point

</deferred>

---

*Phase: 87-tracklet2d-keypoint-propagation*
*Context gathered: 2026-03-11*
