---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Identity
status: unknown
last_updated: "2026-02-27T22:01:58.382Z"
progress:
  total_phases: 7
  completed_phases: 6
  total_plans: 11
  completed_plans: 11
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-27)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 27 — Diagnostic Visualization (Plan 1 of 1 complete)

## Current Position

Phase: 27 of 27 (v2.1 Identity — Diagnostic Visualization)
Plan: 1 of 1 completed in current phase — Phase 27 Plan 1 COMPLETE
Status: Active — Phase 27-01 done (TrackletTrailObserver, 9 unit tests, observer factory wiring)
Last activity: 2026-02-27 — Plan 27-01 complete (TrackletTrailObserver: per-camera trail videos, association mosaic, diagnostic mode integration)

Progress: [█████░░░░░] 50% (v2.1, 5/10 plans done)

## Performance Metrics

**Velocity (v2.1):**
- Total plans completed: 4
- Average duration: 13 min
- Total execution time: 54 min

**By Phase (v2.1):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 22-pipeline-scaffolding | 2/2 complete | 26 min | 13 min |
| 23-refractive-lookup-tables | 2/2 complete | 28 min | 14 min |
| 24-per-camera-2d-tracking | 1/1 complete | 14 min | 14 min |
| 25-27 | TBD | — | — |
| 27-diagnostic-visualization | 1/1 complete | 7 min | 7 min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Key decisions entering v2.1:

- Pipeline reorder: Detection → 2D Tracking → Association → Midline → Reconstruction (root cause fix for broken 3D)
- Old AssociationStage, TrackingStage, FishTracker, ransac_centroid_cluster deleted (not preserved as alternative)
- PIPE-01 (Phase 22) is a prerequisite for all subsequent stages — scaffolding lands first
- Phase 23 (LUTs) and Phase 24 (OC-SORT Tracking) are independent and can proceed in parallel after Phase 22
- Phase 25 (Association) requires both Phase 23 and Phase 24 to be complete
- Detailed algorithmic specs in .planning/milestones/v2.0-phases/21-retrospective-prospective/MS3-SPECSEED.md (for plan-phase, not roadmapper)
- EVAL-01 regression tests deferred: pipeline reorder invalidates existing regression tests; new tests follow v2.1 stabilization
- [Phase 22-pipeline-scaffolding]: FishTrack/TrackState moved to core/tracking/types.py (not deleted) for reconstruction/visualization compatibility until Phase 26
- [Phase 22-pipeline-scaffolding]: TrackletGroup.tracklets uses generic tuple at runtime (not tuple[Tracklet2D]) to preserve ENG-07 import boundary within core/
- [Phase 22-pipeline-scaffolding]: TrackingStubStage lives in engine/pipeline.py (not core/) — engine-level placeholder dispatched via isinstance check in PosePipeline.run()
- [Phase 22-pipeline-scaffolding]: ReconstructionStage.run() early-returns empty midlines when tracklet_groups==[] (stub path); raises ValueError only when both are None
- [Phase 22-pipeline-scaffolding]: AssociationConfig and TrackingConfig stripped to stubs; _filter_fields() prevents TypeError from stale YAML keys
- [Phase 23-01]: LutConfigLike Protocol instead of TYPE_CHECKING import: IB-003 forbids TYPE_CHECKING backdoors; Protocol with 5 LutConfig fields preserves import boundary while LutConfig satisfies it structurally at runtime
- [Phase 23-01]: ForwardLUT stores grids as numpy float32 arrays (not torch tensors) for zero-copy .npz serialization; cast_ray() converts on-demand via torch.from_numpy()
- [Phase 23-02]: InverseLUT uses O(1) integer grid dict for ghost_point_lookup (no KD-tree): snap point to (ix,iy,iz) via int(round()), dict lookup into voxel array
- [Phase 23-02]: float64 for scalar .npz metadata (voxel_resolution, grid_bounds): float32 precision loss breaks equality comparisons and cache invalidation
- [Phase 23-02]: 1e-6*resolution epsilon on np.arange stop: avoids float32 cumulative overshoot beyond z_max while including exact boundary voxels
- [Phase 24-01]: boxmot OcSort requires 6-column input [x1,y1,x2,y2,conf,cls] not 5-column; cls=0.0 for single-class tracking
- [Phase 24-01]: OcSort does NOT output coasting tracks in update() result — coasting positions captured separately from active_tracks with time_since_update>0
- [Phase 24-01]: TrackingStage uses Any-typed config to avoid circular engine->core import; deferred OcSortTracker import inside run()
- [Phase 24-01]: TrackingStubStage removed entirely; TrackingStage now at Stage 2 in all pipeline modes
- [Phase 27-01]: FISH_COLORS_BGR hardcoded in tracklet_trail_observer.py (not imported) to avoid tight coupling
- [Phase 27-01]: calib_data typed as object in generation method signatures (ENG-07 boundary); cast at VideoSet callsite with type: ignore[arg-type]
- [Phase 27-01]: _draw_trail_scaled separate from _draw_trail to keep scale factors isolated from per-camera path
- [Phase 27-01]: TrackletTrailObserver added to diagnostic mode in observer_factory; registered as "tracklet_trail" for --add-observer

### Roadmap Evolution

- Phase 28 added: e2e testing

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 25 (Association) hard-depends on both Phase 23 (LUTs) and Phase 24 (Tracklets) — BOTH NOW COMPLETE, Phase 25 can proceed
- EVAL-01 deferred: regression test suite skipped with pytestmark; rebuild post-v2.1

## Session Continuity

Last session: 2026-02-27
Stopped at: Phase 28 context gathered
Resume file: .planning/phases/28-e2e-testing/28-CONTEXT.md
