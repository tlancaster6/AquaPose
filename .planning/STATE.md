---
gsd_state_version: 1.0
milestone: v3.5
milestone_name: Pseudo-Labeling
status: unknown
last_updated: "2026-03-05T20:46:17.000Z"
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 13
  completed_plans: 11
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 65 frame selection and dataset assembly

## Current Position

Phase: 65 (5 of 6 in v3.5 Pseudo-Labeling) -- COMPLETE
Plan: 3/3 in current phase
Status: Phase 65 complete (including gap closure). All v3.5 plans executed.
Last activity: 2026-03-05 - Completed 65-03: Wire frame selection and gap_reason sidecar

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 7 (v3.5)
- Average duration: ~10 min
- Total execution time: ~59 min

**Recent Trend:**
- Last 7 plans: 61-01, 61-02, 62-01, 62-02, 65-01, 65-02, 65-03 all completed 2026-03-05
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

### Pending Todos

17 pending todos from v2.2 --- see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Z-reconstruction noise (median z-range 1.77 cm, SNR 0.12-1.25) must be resolved before pseudo-labels are usable
- Reference analysis run: ~/aquapose/projects/YH/runs/run_20260305_073212/

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 18 | Fix pseudo-label pose output to use OBB-cropped images with crop-space keypoints | 2026-03-05 | 434b81b | Verified | [18-fix-pseudo-label-pose-output-to-use-obb-](./quick/18-fix-pseudo-label-pose-output-to-use-obb-/) |

## Session Continuity

Last session: 2026-03-05
Stopped at: Completed 65-03-PLAN.md
Resume file: None
