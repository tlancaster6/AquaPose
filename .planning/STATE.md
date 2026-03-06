---
gsd_state_version: 1.0
milestone: v3.5
milestone_name: Pseudo-Labeling
status: unknown
last_updated: "2026-03-06T20:59:03.618Z"
progress:
  total_phases: 9
  completed_phases: 9
  total_plans: 22
  completed_plans: 22
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 69 CLI workflow cleanup

## Current Position

Phase: 69 (3 of 3 plans complete)
Plan: 3/3 in current phase
Status: Phase 69 complete. All deprecated commands removed, dead code cleaned.
Last activity: 2026-03-06 - Completed 69-03: Deprecated command and dead code removal

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
- Phase 68: JSON columns for tags/provenance/metadata with sqlite json_each for filtering
- Phase 68: PRAGMA user_version for schema versioning with forward-compat check
- Phase 68: Source priority manual(2) > corrected(1) > pseudo(0) for dedup upsert
- Phase 68: Conversion functions moved from scripts/ to coco_convert.py module
- Phase 68: Seg conversion (generate_seg_dataset) not migrated -- not in current workflow
- [Phase 68]: Conversion functions moved from scripts/ to coco_convert.py module
- [Phase 68]: Seg conversion not migrated -- not in current workflow
- [Phase 68]: Relative symlinks for dataset assembly (portable across machines)
- [Phase 68]: Pseudo-labels excluded from val split by default (manual+corrected only in val)
- [Phase 68]: dataset_name from dir basename for model lineage (store-managed or external)
- [Phase 68]: Graceful degradation: model registration failure does not fail training
- [Phase 69]: CWD walk-up stops at home dir for project detection
- [Phase 69]: Lazy ctx.obj caching for project resolution
- [Phase 69]: aquapose.cli_utils treated as shared utility for import boundary compliance
- [Phase 69]: Pseudo-label inspect reworked from --data-dir to run-based auto-discovery
- [Phase 69]: elastic_deform.py kept as library code; write_yolo_dataset and generate_preview_grid deleted (dead code)
- [Phase 21]: compute_curvature docstring generalized to (N, D) for 2D and 3D use
- [Phase 21]: Sidecar metadata flattened from labels[0] at import (confidence, gap_reason, n_source_cameras, raw_metrics, source)

### Pending Todos

8 pending todos — see .planning/todos/pending/ (review for relevance)

### Roadmap Evolution

- Phase 67 added: Elastic midline deformation augmentation for pose training data
- Phase 68 removed: CLI Workflow Cleanup (split into 68+69)
- Phase 68 added: Improved training data storage and tracking
- Phase 69 added: CLI workflow cleanup

### Blockers/Concerns

None active. Z-reconstruction noise resolved via z-flattening and temporal z smoothing (Phase 61).

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 18 | Fix pseudo-label pose output to use OBB-cropped images with crop-space keypoints | 2026-03-05 | 434b81b | Verified | [18-fix-pseudo-label-pose-output-to-use-obb-](./quick/18-fix-pseudo-label-pose-output-to-use-obb-/) |
| 19 | Wire frame selection into pseudo-label assembly CLI | 2026-03-05 | 287a6a8 | Complete | [19-wire-frame-selection-into-pseudo-label-a](./quick/19-wire-frame-selection-into-pseudo-label-a/) |
| 20 | Implement COCO interchange format for pseudo-labels | 2026-03-06 | d2e1195 | Complete | [20-implement-coco-interchange-format-for-ps](./quick/20-implement-coco-interchange-format-for-ps/) |
| 21 | Implement pseudo-label metadata ingestion | 2026-03-06 | 24bd82f | Complete | [21-implement-pseudo-label-metadata-ingestio](./quick/21-implement-pseudo-label-metadata-ingestio/) |
| Phase 68 P02 | 8min | 1 tasks | 8 files |
| Phase 68 P03 | 7min | 2 tasks | 4 files |
| Phase 68 P04 | 6min | 2 tasks | 6 files |
| Phase 69 P02 | 12min | 2 tasks | 11 files |

## Session Continuity

Last session: 2026-03-06
Stopped at: Completed 69-03-PLAN.md (Deprecated Command and Dead Code Removal)
Resume file: .planning/phases/69-cli-workflow-cleanup/69-03-SUMMARY.md
