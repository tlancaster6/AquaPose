---
gsd_state_version: 1.0
milestone: v3.5
milestone_name: Pseudo-Labeling
status: unknown
last_updated: "2026-03-05T21:07:13.967Z"
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 13
  completed_plans: 13
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 66 training run management

## Current Position

Phase: 66 (6 of 6 in v3.5 Pseudo-Labeling)
Plan: 2/2 in current phase
Status: Phase 66 complete. All v3.5 Pseudo-Labeling plans complete.
Last activity: 2026-03-05 - Completed 66-02: Compare command

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 8 (v3.5)
- Average duration: ~9 min
- Total execution time: ~65 min

**Recent Trend:**
- Last 8 plans: 61-01, 61-02, 62-01, 62-02, 65-01, 65-02, 65-03, 66-01 all completed 2026-03-05
- Trend: Stable

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v3.4: GPU non-determinism accepted for batched inference
- v3.5: Component C (dorsoventral arch recovery) deferred --- A+B sufficient for pseudo-label quality
- v3.5 Phase 62: pyyaml load/dump for config round-trip (comments stripped, acceptable)
- v3.5 Phase 62: _LutConfigFromDict avoids training->engine import boundary violation
- v3.5 Phase 63: importlib.import_module for engine.config in pseudo_label_cli.py (AST boundary compliance)
- v3.5 Phase 63: Confidence composite: 50% residual + 30% camera + 20% variance
- v3.5 Phase 65: scipy kmeans2 for curvature-based diversity sampling (avoids sklearn dependency)
- v3.5 Phase 65: Finite-difference curvature from tangent vectors (no scipy.interpolate needed)
- v3.5 Phase 65: Pseudo-label val stored in train/ with JSON sidecar (not separate val dir)
- v3.5 Phase 65: Multi-run filename collision resolved by run_dir.name prefix
- v3.5 Phase 65: Frame index parsed from first 6 chars of stem; runs not in selected_frames kept unfiltered
- v3.5 Phase 65: Dominant gap_reason via Counter.most_common for sidecar metadata
- v3.5 Phase 66: yaml.safe_load for project config in run_manager (no engine imports, boundary compliant)
- v3.5 Phase 66: summary.json schema with run_id, metrics, provenance, training_config
- v3.5 Phase 66: xfail marker for compare command test (anticipating Plan 66-02)
- v3.5 Phase 66: click.style bold/green for best-value highlighting with click.unstyle for column width

### Pending Todos

8 pending todos — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Z-reconstruction noise (median z-range 1.77 cm, SNR 0.12-1.25) must be resolved before pseudo-labels are usable
- Reference analysis run: ~/aquapose/projects/YH/runs/run_20260305_073212/

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 18 | Fix pseudo-label pose output to use OBB-cropped images with crop-space keypoints | 2026-03-05 | 434b81b | Verified | [18-fix-pseudo-label-pose-output-to-use-obb-](./quick/18-fix-pseudo-label-pose-output-to-use-obb-/) |
| 19 | Wire frame selection into pseudo-label assembly CLI | 2026-03-05 | 287a6a8 | Complete | [19-wire-frame-selection-into-pseudo-label-a](./quick/19-wire-frame-selection-into-pseudo-label-a/) |

## Session Continuity

Last session: 2026-03-05
Stopped at: Completed quick task 19 (wire frame selection into assemble CLI)
Resume file: None
