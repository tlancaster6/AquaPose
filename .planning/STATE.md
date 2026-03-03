---
gsd_state_version: 1.0
milestone: v3.2
milestone_name: Evaluation Ecosystem
status: unknown
last_updated: "2026-03-03T19:30:18.080Z"
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 8
  completed_plans: 8
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-03)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.2 Evaluation Ecosystem — Phase 48 complete, advancing to Phase 49

## Current Position

Phase: 48 of 50 (EvalRunner and aquapose eval CLI) — Complete
Plan: 2 of 2 complete — 48-01 and 48-02 both done
Status: Phase 48 complete — EvalRunner, formatters, aquapose eval CLI all implemented
Last activity: 2026-03-03 — Completed 48-02 (ASCII/JSON formatters, CLI eval command, deleted measure_baseline.py)

Progress: [████████░░] 50% (8/8 plans for phases 46-48; advancing to phase 49)

## Performance Metrics

**Velocity:**
- Total plans completed: 3 (this milestone)
- Average duration: 5 min
- Total execution time: 0.27 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 46 | 2 complete | 10 min | 5 min |
| 47 | 3 complete | 15 min | 5 min |

*Updated after each plan completion*
| Phase 46 P03 | 8 | 5 tasks | 6 files |
| Phase 47 P01 | 4 | 2 tasks | 5 files |
| Phase 47 P02 | 5 | 2 tasks | 4 files |
| Phase 47 P03 | 6 | 2 tasks | 4 files |
| Phase 48 P01 | 7 | 1 tasks | 3 files |
| Phase 48 P02 | 3 | 2 tasks | 5 files |

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
- [Phase 47 P02]: DEFAULT_GRID early_k values stored as float [5.0, 10.0, ...] to satisfy dict[str, list[float]] type; source uses int
- [Phase 47 P02]: camera_distribution int keys converted to str in to_dict() for JSON compatibility
- [Phase 47 P02]: temporal_smoothness returns 0.0 for fish with only one frame (no consecutive pairs)
- [Phase 47 P02]: point_confidence=None treated as uniform 1.0 for confidence stats and completeness
- [Phase 47 P03]: ReconstructionMetrics is a fresh frozen dataclass, NOT a subclass of Tier1Result — keeps evaluation types independent
- [Phase 47 P03]: tier2_stability = max of all non-None displacement values in Tier2Result.per_fish_dropout (None if all None or empty)
- [Phase 47 P03]: evaluate_reconstruction() wraps compute_tier1() internally; tier2_result is keyword-only param
- [Phase 48-01]: EvalRunner._read_n_animals() uses inline import of load_config to avoid top-level engine coupling
- [Phase 48-01]: MidlineSet construction for evaluate_association uses tracklet_groups + annotated_detections centroid proximity matching (same as DiagnosticObserver)
- [Phase 48-01]: evaluate_midline receives first-camera Midline2D per fish per frame from MidlineSet (single-camera representative)
- [Phase 48]: eval_cmd function name avoids shadowing Python built-in eval; registered as @cli.command('eval')
- [Phase 48]: format_eval_json delegates to result.to_dict() + json.dumps — no duplication of to_dict() logic
- [Phase 48]: eval_results.json always written to run_dir on every eval invocation regardless of --report flag

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Phase 49 (TuningOrchestrator) has the most moving parts: cascade config propagation and two-tier frame count logic warrant a pre-implementation sketch during planning
- `stop_after` field presence in PipelineConfig should be confirmed at Phase 49 planning time (ARCHITECTURE.md states it exists; verify before building cascade orchestrator)

## Session Continuity

Last session: 2026-03-03T19:24:37Z
Stopped at: Completed 48-02-PLAN.md — ASCII/JSON formatters and aquapose eval CLI
Resume file: None
