# Phase 91: Singleton Recovery - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Singletons (tracklets not assigned to any group after clustering and validation) are scored against existing groups and assigned, split-assigned for swap recovery, or left as true singletons. The same-camera overlap constraint is enforced throughout. This phase does not change clustering, validation, or the scoring infrastructure — it adds a recovery pass after those stages complete.

</domain>

<decisions>
## Implementation Decisions

### Scoring method
- Score singletons against groups using ray-to-consensus residuals with multi-keypoint rays (not centroid-only)
- Per-keypoint 3D reference positions are triangulated on-demand from the group's member tracklets for shared frames — no schema changes to TrackletGroup
- Per-keypoint, per-frame ray-to-3D distances are aggregated via arithmetic mean into a single singleton-to-group score
- Consistent with Phase 88's multi-keypoint philosophy — centroid-only scoring is on the path to deprecation

### Assignment logic
- Greedy best-first matching: sort all (singleton, group) pairs by residual score, assign the best match first, remove that singleton from consideration
- Hard threshold on mean residual (configurable, in metres) — no relative margin requirement
- Single pass: score all singletons against all groups once, then assign greedily; no iterative re-scoring after assignment
- Configurable minimum shared frames required between singleton and group to attempt scoring

### Split-assign behavior (swap-aware recovery)
- Max-residual binary sweep: try every possible split point, score each segment against all groups, pick the split that maximizes total assignment quality
- Both segments must match different groups — if only one segment matches, the whole tracklet remains a singleton
- Configurable minimum segment length (separate from t_min) — segments shorter than this are not considered
- Binary split only (one split point per singleton) — no recursive splitting

### Constraint enforcement
- Same-camera overlap check uses detected frames only (coasted/predicted frames do not count as overlap)
- Overlap check happens before scoring — if overlap exists on detected frames, skip scoring entirely for that singleton-group pair
- For split-assign, the same-camera constraint is re-checked per-segment independently against each segment's candidate group

### Claude's Discretion
- Config field names and default values for new parameters (assignment threshold, min segment length, min shared frames)
- Internal data structures for the recovery pass
- Logging and diagnostic output format
- Whether to extract shared utilities from refinement.py or implement standalone

</decisions>

<specifics>
## Specific Ideas

- The on-demand triangulation approach avoids changing TrackletGroup schema, keeping Phase 91 self-contained
- The "detected frames only" overlap rule reflects that coasted frames are Kalman predictions, not real identity claims — a coasted tracklet in one group shouldn't block a detected singleton from joining
- Binary-only split is consistent with REQUIREMENTS.md ruling out ruptures library and noting recursive split as a future option

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `refinement.py`: `_compute_frame_consensus()` triangulates per-frame 3D centroids from tracklet rays — can be adapted for per-keypoint triangulation
- `refinement.py`: `_compute_tracklet_distances()` computes per-tracklet median ray-to-consensus distance — similar pattern needed for singleton scoring
- `refinement.py`: `_point_to_ray_distance()` utility for point-to-ray perpendicular distance
- `scoring.py`: `ray_ray_closest_point()` and `ray_ray_closest_point_batch()` for ray geometry
- `scoring.py`: `AssociationConfigLike` protocol pattern for config without cross-layer imports
- `clustering.py`: `build_must_not_link()` for same-camera constraint infrastructure

### Established Patterns
- Config protocol pattern (`RefinementConfigLike`, `AssociationConfigLike`) for core/ → engine/ boundary
- Singleton groups created with `fish_id=-1, confidence=0.1` in refinement, then reassigned unique IDs
- TrackletGroup is frozen dataclass — modifications require creating new instances
- ForwardLUT `cast_ray()` returns torch tensors, must `.cpu().numpy()` for computation

### Integration Points
- Recovery runs after refinement in `AssociationStage.run()` — between `refine_clusters()` and `context.tracklet_groups = groups`
- New config fields added to `AssociationConfig` in `engine/config.py` (following existing pattern)
- Tracklet2D `.frames`, `.centroids`, `.camera_id`, `.keypoints`, `.keypoint_conf` fields used for scoring
- Phase 90's validation will produce singletons (evicted tracklets and split fragments) that flow into this recovery pass

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 91-singleton-recovery*
*Context gathered: 2026-03-11*
