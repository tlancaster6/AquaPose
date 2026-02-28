# Phase 25: Association Scoring and Clustering - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Score all cross-camera tracklet pairs using ray-ray geometry and cluster them into global fish identity groups via Leiden algorithm with same-camera conflict constraints. Covers SPECSEED Steps 0–4 (camera overlap graph, pairwise scoring, graph construction, Leiden clustering, fragment merging). Step 5 (3D consistency refinement) belongs to Phase 26.

</domain>

<decisions>
## Implementation Decisions

### Scoring Parameters
- All thresholds exposed in `AssociationConfig` dataclass with defaults from SPECSEED; tunable via YAML
- Ray distance threshold τ = 3cm default (fish are ~10cm long, ~2cm wide from above; 3cm accommodates centroid jitter and viewing-angle offset)
- Score minimum s_min = 0.3, T_min = 10 frames, T_saturate = 100 frames (SPECSEED defaults)
- Aggressive early termination: K=10 initial frames, abandon pair immediately at 0 inliers

### TrackletGroup Output
- Aggregate confidence only — no per-frame ray distances, ghost ratios, or camera counts on the group
- Pairwise affinity scores discarded after clustering — not stored on TrackletGroup
- `handoff_state` defined as a typed `HandoffState` dataclass with correct fields, but populated with None/empty (stub). Contracts exist for future chunk orchestration; data deferred.

### Failure & Edge Cases
- `expected_fish_count` configurable in AssociationConfig (default 9, not hardcoded). Fish count is fixed — the tank is closed, fish do not enter or leave.
- Emit diagnostic warning when cluster count ≠ expected_fish_count. More groups = oversplit/false positives; fewer groups = undersplit or detection dropout. Both are quality signals.
- Must-not-link violations: one re-split attempt (remove weakest internal edge, re-run Leiden on subgraph). If violations persist, force-separate — evict the weaker tracklet (lower affinity to cluster) to a singleton. Must-not-link means detection-backed temporal overlap, which is physically impossible for the same fish. Don't propagate contradictory data.
- Singleton (unassociated) tracklets kept as single-tracklet TrackletGroups with low confidence. Visible in output; downstream stages can filter by confidence.

### Fragment Merging
- When coasted frames overlap between fragments: discard all coasted frames from the dying tracklet, keep detected frames from the new tracklet
- If discarding coasted frames creates a gap between the last detected frame of fragment A and the first detected frame of fragment B, linearly interpolate centroids across the gap
- Interpolated frames tagged as `"interpolated"` — a new third frame_status value distinct from `"detected"` and `"coasted"`. Downstream knows these are merge artifacts.
- Configurable `max_merge_gap` in AssociationConfig (default TBD by researcher/planner). Fragments separated by more frames than this limit are kept separate within the cluster, not merged.

### Claude's Discretion
- Ghost penalty detection spatial index strategy (grid binning vs KD-tree)
- Leiden resolution parameter default
- Camera overlap graph adjacency threshold (shared voxel count)
- Module organization within `core/association/`
- Exact `HandoffState` field types and naming
- `max_merge_gap` default value
- Whether to use temporal subsampling for initial scoring pass (SPECSEED performance optimization)

</decisions>

<specifics>
## Specific Ideas

- Fish are ~10cm long, ~2cm wide viewed from above — τ=3cm is calibrated to this
- 9 fish in a 2m diameter × 1m tall cylindrical tank; fish count is always exactly 9, never changes during a run
- Close encounters between fish are frequent from top-down views — this is the primary source of OC-SORT fragmentation and the main challenge for association
- The `"interpolated"` frame tag was chosen specifically because coasting-overlap-then-gap is the most common fragment merging case (OC-SORT drift during close encounters)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 25-association-scoring-and-clustering*
*Context gathered: 2026-02-27*
