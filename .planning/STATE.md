---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Alpha
status: unknown
last_updated: "2026-02-26T23:00:25.948Z"
progress:
  total_phases: 10
  completed_phases: 7
  total_plans: 27
  completed_plans: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-25)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 17 — Observers

## Current Position

Phase: 19 of 21 (Alpha Refactor Audit) — IN PROGRESS
Plan: 1 of 4 in current phase — COMPLETE
Status: Phase 19-01 Complete — Import boundary checker and pre-commit hook implemented; 7 IB-003 violations cataloged
Last activity: 2026-02-26 - Completed Phase 19 Plan 01: Import boundary checker with pre-commit hook

Progress: [███████░░░] 70%

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
| Phase 16-numerical-verification-and-legacy-cleanup P02 | 15 | 2 tasks | 7 files |
| Phase 19-alpha-refactor-audit P04 | 2 | 1 tasks | 1 files |

## Accumulated Context

### Roadmap Evolution

- Phase 14.1 inserted after Phase 14: Fix Critical Mismatch Between Old and Proposed Pipeline Structures (URGENT)
- Phase 15 plan count corrected from 7 to 5 plans (5-stage canonical pipeline model)
- Phase 19 added: Alpha Refactor Audit
- Phase 20 added: Post-Refactor Loose Ends
- Phase 21 added: Retrospective, Prospective

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions entering v2.0:

Phase 16-01 decisions:
- Midline regression test marked xfail(strict=False): v1.0 golden midlines keyed by fish_id post-tracking, new pipeline extracts midlines pre-tracking — direct comparison requires golden data regeneration with PosePipeline
- pipeline_context fixture is session-scoped: runs full pipeline exactly once and shares PipelineContext across all 7 regression tests
- test_pipeline_determinism runs pipeline twice with same seed and asserts np.array_equal (atol=0) — validates guidebook reproducibility contract
- generate_golden_data.py masks extraction: AnnotatedDetection.mask + .crop_region reformatted to legacy tuple format for golden_segmentation.pt.gz backward compat

Phase 15-05 decisions:
- TriangulationBackend is stateless (delegates to triangulate_midlines()); CurveOptimizerBackend is stateful — single CurveOptimizer persists across frames for warm-starting
- MidlineSet assembly bridges Stage 2 + Stage 4: FishTrack.camera_detections (cam_id→det_idx) used to look up annotated_detections[frame][cam][idx].midline
- build_stages(config) factory lives in engine/pipeline.py alongside PosePipeline — orchestration logic belongs in engine/, not core/
- ReconstructionConfig extended with inlier_threshold, snap_threshold, max_depth extracted from v1.0 hardcoded defaults
- Coasting fish (empty camera_detections) skipped during reconstruction — matches v1.0 behavior

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
- [Phase 16-02]: test_stages.py v1.0 functional tests replaced with importability check — regression suite in tests/regression/ covers the canonical PosePipeline execution path
- [Phase 16-02]: diagnose_triangulation.py was untracked — copied and removed manually since git mv requires tracked files
- [Phase 19-alpha-refactor-audit]: Items 1+2 Open/Warning: FishTracker monolithic association+tracking — remediate together with bundles-aware backend in Phase 20
- [Phase 19-alpha-refactor-audit]: Item 5 Open/Warning: skip_camera_id missing from PipelineConfig and build_stages() — 10 hardcoded occurrences, low-effort Phase 20 fix
- [Phase 19-alpha-refactor-audit]: Items 3+6 Accepted: MidlineSet bridge pattern and CurveOptimizer statefulness are intentional design
- [Phase 19-alpha-refactor-audit]: Items 4+7 Resolved: ReconstructionConfig thresholds and AssociationConfig fields completed as planned in Phase 15
- [Phase 19-01]: IB-003 violations in core/ stage files cataloged (not fixed) -- 7 TYPE_CHECKING backdoors importing aquapose.engine.stages in all 5 stage.py files + core/synthetic.py
- [Phase 19-01]: SR-002 (observer core imports) is warning not error -- some legitimate core imports may be needed by observers; IB-004 applies to legacy computation dirs

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 9 | Add init-config CLI command to generate a default template YAML config file | 2026-02-26 | 54941d7 | [9-add-init-config-cli-command-to-generate-](./quick/9-add-init-config-cli-command-to-generate-/) |

## Session Continuity

Last session: 2026-02-26
Stopped at: Completed Phase 19-01 (import boundary checker + pre-commit hook, 7 IB-003 violations cataloged)
Resume file: None
