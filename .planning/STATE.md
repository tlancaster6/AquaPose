---
gsd_state_version: 1.0
milestone: v3.2
milestone_name: Evaluation Ecosystem
status: unknown
last_updated: "2026-03-03T18:49:38.743Z"
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 6
  completed_plans: 4
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-03)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.2 Evaluation Ecosystem — Phase 47 in progress

## Current Position

Phase: 47 of 50 (Evaluation Primitives) — In Progress
Plan: 1 of 3 complete — advancing to Plan 02
Status: Executing Phase 47
Last activity: 2026-03-03 — Completed 47-01 (detection and tracking stage evaluators, 20 unit tests)

Progress: [████░░░░░░] 20% (4/6 plans complete; phase 47 plan 1/3 done)

## Performance Metrics

**Velocity:**
- Total plans completed: 2 (this milestone)
- Average duration: 5 min
- Total execution time: 0.17 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 46 | 2 complete | 10 min | 5 min |

*Updated after each plan completion*
| Phase 46 P03 | 8 | 5 tasks | 6 files |
| Phase 47 P01 | 4 | 2 tasks | 5 files |

## Accumulated Context

### Decisions

Carried forward from v3.1:
- DLT is the sole reconstruction backend (old triangulation + curve optimizer removed)
- Outlier rejection threshold tuned to 10.0 (from 50.0)
- Association defaults accepted (sweep showed marginal gains; ~70% singleton rate is upstream bottleneck)
- NPZ fixture v2.0 format with CalibBundle (being replaced by per-stage pickle caches in v3.2)

v3.2 design decisions:
- Per-stage pickle files replace monolithic NPZ as evaluation data source (one file per StageComplete event)
- ContextLoader uses shallow copy (not deepcopy) for sweep combo isolation — safe because stage outputs are immutable by convention
- Stage evaluators have zero engine imports — pipeline config passes as explicit function parameters
- No automatic config file mutation — tuning output is a config diff block for manual application
- Legacy evaluation code (harness.py, tune_association.py, tune_threshold.py, measure_baseline.py, pipeline_diagnostics.npz) fully removed, not shimmed
- [Phase 46]: StaleCacheError and cache utilities defined in core/context.py (not separate errors.py); context_fingerprint() made public for DiagnosticObserver use
- [Phase 46]: Envelope format for stage caches: dict with run_id, timestamp, stage_name, version_fingerprint, context keys
- [Phase 46]: Stage name lookup uses type(stage).__name__ matched against _STAGE_OUTPUT_FIELDS dict — pure string matching, no isinstance
- [Phase 46]: DiagnosticObserver captures run_id from PipelineStart event (not __init__) to keep signature stable
- [Phase 46]: Cache envelope format: {run_id, timestamp, stage_name, version_fingerprint, context} written to diagnostics/<stage>_cache.pkl
- [Phase 46]: Inline import of load_stage_cache/StaleCacheError inside CLI run() avoids top-level import coupling
- [Phase 46]: CLI --resume-from uses click.Path(exists=True) for automatic missing-file validation; StaleCacheError converts to ClickException
- [Phase 47]: mean_jitter defined as mean(abs(diff(counts))) per camera averaged over cameras — stable cameras contribute 0.0, flickering cameras raise the mean
- [Phase 47]: detection_coverage = 1.0 - coast_frequency since every Tracklet2D frame is either detected or coasted — no third state
- [Phase 47]: stages/__init__.py left as placeholder; exports deferred to Plan 03 once all evaluators exist

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Phase 49 (TuningOrchestrator) has the most moving parts: cascade config propagation and two-tier frame count logic warrant a pre-implementation sketch during planning
- `stop_after` field presence in PipelineConfig should be confirmed at Phase 49 planning time (ARCHITECTURE.md states it exists; verify before building cascade orchestrator)

## Session Continuity

Last session: 2026-03-03T18:48:37Z
Stopped at: Completed 47-01-PLAN.md — detection and tracking stage evaluators with frozen dataclasses and 20 unit tests
Resume file: None
