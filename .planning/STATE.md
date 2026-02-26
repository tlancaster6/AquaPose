---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Stage Migrations
status: active
last_updated: "2026-02-26T00:56:30.000Z"
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 9
  completed_plans: 10
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-25)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 15 — Stage Migrations (5-stage pipeline)

## Current Position

Phase: 15 of 18 (Stage Migrations)
Plan: 4 of 5 in current phase — COMPLETE
Status: Phase 15 Plan 04 Complete
Last activity: 2026-02-26 — Completed 15-04 (TrackingStage created in core/tracking/, Hungarian backend wrapping FishTracker)

Progress: [████░░░░░░] 38%

## Performance Metrics

**Velocity:**
- Total plans completed: 0 (v2.0)
- Average duration: — min
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 13-engine-core | 4 | ~20 min | ~5 min |

**Recent Trend:** Active — Phase 13 complete (4 plans)
| Phase 13-engine-core P01 | 6 | 2 tasks | 4 files |
| Phase 13-engine-core P04 | 4 | 2 tasks | 3 files |
| Phase 14.1 P02 | 7 | 2 tasks | 7 files |

## Accumulated Context

### Roadmap Evolution

- Phase 14.1 inserted after Phase 14: Fix Critical Mismatch Between Old and Proposed Pipeline Structures (URGENT)
- Phase 15 plan count corrected from 7 to 5 plans (5-stage canonical pipeline model)

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions entering v2.0:
Phase 15-04 decisions:
- TrackingStage reads context.detections (Stage 1 raw output) NOT context.associated_bundles — FishTracker.update() re-derives cross-camera association internally for v1.0 equivalence; Stage 3 output is a data product for future backends/observers only
- HungarianBackend wraps existing FishTracker with no reimplementation — stateful tracker constructed once at stage init, persists across all frames
- TrackingConfig extended with full tracker parameter set (min_hits, max_age, reprojection_threshold, birth_interval, min_cameras_birth, velocity_damping, velocity_window)

Phase 15-03 decisions:
- AssociationBundle uses fish_idx (per-frame 0-indexed) not fish_id — persistent IDs assigned by Tracking (Stage 4)
- RansacCentroidBackend delegates to existing discover_births() — port behavior, not rewrite
- ALL detections are unclaimed at Stage 3 — tracking has not yet run; v1.0 discover_births only saw unclaimed remainder
- AssociationStage reads annotated_detections preferentially over detections — unwraps AnnotatedDetection to get Detection for RANSAC
- AssociationConfig gains expected_count, min_cameras, reprojection_threshold — no longer empty placeholder

Phase 15-02 decisions:
- MidlineStage extracts midlines for ALL detections (not just tracked fish) — tracking has not run yet; fish_id=-1 placeholder used
- segment_then_extract backend directly calls private helpers from reconstruction.midline — avoids reimplementing the midline pipeline
- AnnotatedDetection wraps Detection + mask + CropRegion + Midline2D — clean wrapper rather than mutating Detection in place
- MidlineConfig gains backend, n_points, min_area — all midline stage parameters in one frozen config

Phase 15-01 decisions:
- DetectionStage uses TYPE_CHECKING guard for PipelineContext annotation — engine/ never imported at runtime (ENG-07)
- Calibration loading deferred to __init__ via local imports — preserves fail-fast while avoiding circular imports
- Backend registry returns YOLOBackend directly; MOG2 deferred to future plan per CONTEXT.md
- DetectionConfig gains model_path and device fields for YOLO weight loading and GPU placement

Phase 14.1-01 decisions:
- 5 canonical stages replace 7: Detection, Midline (segment-then-extract), Association, Tracking, Reconstruction — matches guidebook Section 6
- Midline stage subsumes U-Net/SAM segmentation + skeletonization+BFS as single segment-then-extract backend (not separate stages)
- Reconstruction stage subsumes RANSAC triangulation + B-spline fitting; curve optimizer is planned second backend within same stage
- STG requirements reduced from 7 (STG-01..07) to 5 (STG-01..05); total v2.0 requirement count drops from 24 to 22

Phase 14-02 decisions:
- CropRegion has x1/y1/x2/y2 fields (not x/y/width/height as plan described) — tests written to actual API
- FishTrack uses positions deque (not centroid_3d) — tests assert positions[-1].shape == (3,)
- Low-confidence triangulations exempted from tank-bounds check — RANSAC degenerate outputs expected in v1.0

Phase 14-01 decisions:
- Seeds set before pipeline imports (not just before stage calls) — covers CUDA init triggered by imports
- Each stage output saved as separate .pt file (not bundled) — allows individual stage comparisons
- metadata.pt records generation environment — required to interpret tolerance differences across GPUs
- Camera e3v8250 excluded in script to match orchestrator.py behavior exactly

Key decisions entering v2.0:

- Build order: ENG infrastructure → golden data → stage migrations → verification → observers → CLI
- Golden data MUST be committed before any stage migration begins (VER-01 gates STG-*)
- Frozen dataclasses for config (not Pydantic) — already decided
- Import boundary enforced: engine/ imports computation modules, never reverse (ENG-07)
- Port behavior, not rewrite logic — numerical equivalence is the acceptance bar

Phase 13 decisions:
- Stage Protocol uses typing.Protocol with runtime_checkable — structural typing, no inheritance required
- PipelineContext uses generic stdlib types (list, dict) in fields to maintain ENG-07 import boundary
- Config frozen dataclasses: stage-specific configs composed into PipelineConfig
- load_config() CLI overrides accept both dot-notation strings and nested dicts
- output_dir expanded via Path.expanduser() at load time
- Observer uses structural typing (Protocol) not ABC — any class with on_event satisfies it
- EventBus walks __mro__ for dispatch — subscribing to Event base receives all subtypes
- Fault-tolerant dispatch logs warnings but never re-raises — pipeline determinism preserved
- PosePipeline writes config.yaml before PipelineStart event — config artifact is truly first
- Stage timing keyed by type(stage).__name__ — class name as natural stage identifier
- Observers subscribe to Event base type in constructor — simplest all-events API
- [Phase 14.1-02]: load_config() accepts old YAML keys (segmentation/triangulation) and new keys (midline/reconstruction) for backward compat — new takes precedence
- [Phase 14.1-02]: AssociationConfig is empty placeholder frozen dataclass — Stage 3 parameters unknown until Phase 15 designs the stage
- [Phase 14.1-02]: ReconstructionConfig adds backend field (default triangulation) for future curve_optimizer support without breaking schema change

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-26
Stopped at: Completed 15-04-PLAN.md (TrackingStage created in core/tracking/, HungarianBackend wrapping FishTracker, 9 tests pass)
Resume file: None
