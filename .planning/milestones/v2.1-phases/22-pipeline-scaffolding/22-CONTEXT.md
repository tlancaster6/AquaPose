# Phase 22: Pipeline Scaffolding - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Rewire the engine to the new 5-stage order (Detection → 2D Tracking → Association → Midline → Reconstruction). Update PipelineContext and CarryForward with correctly typed fields. Delete all legacy tracking/association code (AssociationStage, TrackingStage, FishTracker, ransac_centroid_cluster). The new TrackingStage and AssociationStage are stubs in this phase — real implementations arrive in Phases 24-25.

</domain>

<decisions>
## Implementation Decisions

### Stub Stage Behavior
- New TrackingStage and AssociationStage are pass-through stubs that write correctly-typed but empty output to PipelineContext
- Stubs emit `logger.warning` when invoked (e.g., "TrackingStage is a stub — producing empty output") to make placeholder status obvious in diagnostic runs
- Stub TrackingStage accepts and returns CarryForward (unchanged), establishing the carry interface plumbing now for Phase 24
- Pipeline executes the full 5-stage chain without error; downstream stages see the right types but no data

### Data Structure Design
- `Tracklet2D` is a frozen dataclass with fields: `camera_id`, `track_id`, `frames`, `centroids`, `bboxes`, `frame_status` (detected vs coasted)
- `TrackletGroup` is a frozen dataclass with fields including `fish_id`, `tracklets`, and a `confidence` field (Optional, None until Phase 25/26 populates it)
- `PipelineContext.tracks_2d` is typed as `dict[CameraId, list[Tracklet2D]]` — grouped by camera, natural for per-camera processing and camera-pair iteration in association
- All new domain types (Tracklet2D, TrackletGroup) live in `core/` — they are domain concepts used by core computation modules, consistent with import discipline (core/ → nothing external; engine/ → core/)

### Legacy Deletion Scope
- Delete source modules: AssociationStage, TrackingStage, FishTracker, ransac_centroid_cluster
- Delete tests that exercise deleted code — no archive, clean removal
- Clean up config references: remove old tracking/association config keys from defaults and YAML presets
- Stub affected observer hooks (DiagnosticsObserver, VisualizationObserver) that referenced old tracking/association data — replace with pass-through no-ops; Phase 27 rewrites them properly
- Adapt existing e2e/integration tests to run the new 5-stage chain with stubs (not delete-and-rebuild)

### Synthetic Validation
- Minimal smoke test scope for this phase (2-3 cameras, few frames, simple detections)
- Reuse existing SyntheticMode infrastructure from v2.0, but include explicit inspection of SyntheticMode output quality: expected fish count, reasonable positions, physically plausible motion
- Assert each stage populates its expected PipelineContext field with the correct type (even if empty/trivial from stubs)
- Synthetic validation runs in the normal `hatch run test` suite (not @slow) — fast enough with stubs
- Future phases (24+) scale up to multi-fish multi-camera scenarios

### Claude's Discretion
- Whether MidlineStage and ReconstructionStage keep v2.0 implementations or become stubs (decide based on codebase state and what compiles cleanly with the new data model)
- Exact internal structure of CarryForward fields for 2D track state
- Specific SyntheticMode inspection assertions (what constitutes "reasonable" positions and motion)

</decisions>

<specifics>
## Specific Ideas

- SyntheticMode was not thoroughly tested for realistic, consistent output in v2.0 — it was only used for smoke testing. Phase 22 should include direct code inspection and basic sanity assertions on synthetic data quality as a secondary goal.
- The synthetic validation scope escalates across the milestone: minimal here, multi-fish multi-camera by Phase 25-26.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 22-pipeline-scaffolding*
*Context gathered: 2026-02-27*
