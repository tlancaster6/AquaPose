# Phase 72: Baseline Pipeline Run & Metrics - Research

**Researched:** 2026-03-07
**Domain:** Pipeline execution + evaluation workflow (no new code)
**Confidence:** HIGH

## Summary

Phase 72 is a pure workflow execution phase -- no source code changes are needed. All required infrastructure (CLI commands, evaluation metrics, diagnostic caching, config overrides) already exists and was verified by examining the codebase. The phase runs `aquapose run` in diagnostic mode on a short clip, then runs `aquapose eval` to produce a full metric report including all Phase 70 extended metrics.

The key dependency is Phase 71 completing model training and registration. After Phase 71, the project config (`config.yaml`) will have updated `detection.weights_path` and `midline.weights_path` pointing to the newly trained baseline models. The config already has `mode: diagnostic` and `chunk_size: 300`, so only `--max-chunks 6` needs to be passed at the CLI.

**Primary recommendation:** Execute two CLI commands (`aquapose run` then `aquapose eval`) and verify that `eval_results.json` contains all expected metric sections. No code changes, no new tooling.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use the first ~1 minute of the YH project video with chunk_size=300 and --max-chunks 6 (~1800 frames, ~60s)
- Diagnostic mode (writes chunk caches needed by Phase 73)
- Baseline models pointed at pipeline via --set config overrides (or config.yaml already updated by Phase 71)
- eval_results.json written to run directory (existing behavior, no changes)
- JSON only -- no markdown summary file
- No special round tagging -- run directory timestamp is the identifier
- Human-readable report goes to stdout only
- Key metrics: reprojection error percentiles (p50/p90/p95), singleton rate, track continuity ratio
- OKS curvature slope tracked separately during training, not in pipeline eval
- Purely directional success thresholds -- no hard go/no-go cutoffs
- Phase 74 is the comparison/decision phase
- Document exact commands (no shell script)

### Claude's Discretion
- Exact --set override syntax for model paths (depends on Phase 71 store output format)
- Whether to add any lightweight validation that the run completed successfully before eval
- Config file adjustments for chunk_size=300

### Deferred Ideas (OUT OF SCOPE)
- `aquapose eval compare RUN_A RUN_B` comparison CLI -- belongs in Phase 74
- Val-set OKS integration into `aquapose eval` -- potential future phase
- Named model profiles (`--models baseline`) -- future improvement
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ITER-01 | Baseline pipeline run on short clip (~1 min) with diagnostic caching produces baseline metric snapshot | Full pipeline infrastructure exists: `aquapose run` with `--max-chunks 6` and diagnostic mode produces chunk caches; `aquapose eval` reads caches and writes eval_results.json with all extended metrics (EVAL-01 through EVAL-06) |
</phase_requirements>

## Standard Stack

### Core
| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| `aquapose run` | Current | Pipeline execution with diagnostic caching | Existing CLI command, fully functional |
| `aquapose eval` | Current | Metric computation and report generation | Existing CLI command, writes eval_results.json |
| YH project config | config.yaml | Pipeline configuration with model paths | Already has diagnostic mode, chunk_size=300, association tuning params |

### Supporting
| Tool | Purpose | When to Use |
|------|---------|-------------|
| `aquapose data status` | Verify store state and model registration | Before running pipeline, to confirm Phase 71 baseline models exist |
| `cat config.yaml` | Verify model paths point to baseline models | Before running pipeline |

## Architecture Patterns

### Pipeline Run Flow
```
aquapose run --project YH --max-chunks 6
    |
    v
~/aquapose/projects/YH/runs/run_YYYYMMDD_HHMMSS/
    ├── config.yaml              # Frozen config snapshot
    ├── config_exhaustive.yaml   # Full config with defaults
    ├── diagnostics/
    │   ├── manifest.json        # Chunk list with frame offsets
    │   ├── chunk_000/cache.pkl  # Per-chunk diagnostic cache
    │   ├── chunk_001/cache.pkl
    │   ├── ...
    │   └── chunk_005/cache.pkl  # 6 chunks total
    └── midlines.h5              # 3D reconstruction output
```

### Eval Output Flow
```
aquapose eval --project YH <run_dir_name>
    |
    v  (reads diagnostics/chunk_*/cache.pkl)
    |
    v  (stdout: human-readable report)
    |
    v  (writes eval_results.json to run directory)
```

### Config Override Mechanics
The `--set` flag uses dot-notation for nested fields:
```
--set detection.weights_path=/path/to/model.pt
--set midline.weights_path=/path/to/model.pt
```
However, Phase 71's `register_trained_model()` auto-updates `config.yaml` with new weights paths. If Phase 71 completes successfully, the config.yaml will already point to baseline models -- no `--set` overrides needed.

### Key Configuration State (YH config.yaml)
The current config already has:
- `mode: diagnostic` -- writes chunk caches
- `chunk_size: 300` -- 10s per chunk at 30fps
- `n_animals: 9` -- 9 fish
- Association params tuned (ray_distance_threshold=0.01, etc.)
- Model paths will be updated by Phase 71 registration

### Anti-Patterns to Avoid
- **Don't modify config.yaml manually** -- Phase 71's model registration handles weights_path updates
- **Don't use `--set chunk_size=300`** -- already in config.yaml
- **Don't use `--set mode=diagnostic`** -- already in config.yaml
- **Don't run `pytest` to verify** -- use `hatch run test` per project conventions

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Metric computation | Custom scripts | `aquapose eval` | All Phase 70 metrics already integrated |
| Diagnostic caching | Manual cache management | `mode: diagnostic` | Orchestrator handles chunk cache lifecycle |
| Model path switching | Manual config editing | `register_trained_model()` | Phase 71 auto-updates config.yaml |
| Run comparison | Side-by-side scripts | Defer to Phase 74 | Explicitly deferred in CONTEXT.md |

## Common Pitfalls

### Pitfall 1: Running Before Phase 71 Completes
**What goes wrong:** Pipeline runs with old (pre-store) model weights, producing metrics that don't reflect the baseline models.
**Why it happens:** Config.yaml currently points to `models/obb3/best_model.pt` and `training_data/augment_experiment/runs/augmented/best_model.pt` -- these are pre-store models.
**How to avoid:** Verify config.yaml model paths point to Phase 71-registered models before running.
**Warning signs:** `data status` shows no registered models, or config paths still point to old locations.

### Pitfall 2: Stale LUT Cache
**What goes wrong:** Association stage uses stale LUTs from prior code versions, producing incorrect cross-view associations.
**Why it happens:** Per project memory, stale LUT cache at `~/aquapose/projects/YH/geometry/luts/` can persist across code changes.
**How to avoid:** If any LUT-related code changed since last successful run, delete and regenerate LUTs.
**Warning signs:** Very high singleton rate (>50%), centroid reprojection errors >10px.

### Pitfall 3: Interpreting run as "latest" When Multiple Exist
**What goes wrong:** `aquapose eval` without explicit run arg picks the latest run, which may not be the baseline run.
**Why it happens:** `resolve_run()` defaults to the most recent run in the project's runs directory.
**How to avoid:** Always pass the explicit run directory name to `aquapose eval`.

### Pitfall 4: Config Exhaustive Not Written
**What goes wrong:** `config_exhaustive.yaml` may not be present if run crashed early.
**Why it happens:** Orchestrator writes it, but only after initialization completes.
**How to avoid:** Check run completed successfully (all 6 chunks processed) before running eval.
**Warning signs:** `diagnostics/manifest.json` missing or incomplete, fewer than 6 chunk directories.

### Pitfall 5: CUDA OOM with 300-Frame Chunks
**What goes wrong:** Larger chunk_size may increase peak GPU memory during batch inference.
**Why it happens:** chunk_size=300 is 50% larger than the prior default of 200.
**How to avoid:** Detection and midline stages have batch controls (`detection_batch_frames`, `midline_batch_crops`) -- defaults of 0 (no limit) should be fine, but if OOM occurs, set these to lower values.
**Warning signs:** CUDA out of memory errors during detection or midline stages.

## Code Examples

### Running the Baseline Pipeline
```bash
# Verify Phase 71 models are registered
aquapose --project YH data status

# Verify config.yaml has correct model paths
cat ~/aquapose/projects/YH/config.yaml

# Run pipeline (diagnostic mode + chunk_size=300 already in config)
aquapose --project YH run --max-chunks 6

# Note the run directory name from output (run_YYYYMMDD_HHMMSS)
```

### Running Evaluation
```bash
# Run eval on the specific baseline run
aquapose --project YH eval run_YYYYMMDD_HHMMSS

# eval_results.json is automatically written to the run directory
cat ~/aquapose/projects/YH/runs/run_YYYYMMDD_HHMMSS/eval_results.json
```

### Validating Run Completion
```bash
# Check manifest for chunk count
cat ~/aquapose/projects/YH/runs/run_YYYYMMDD_HHMMSS/diagnostics/manifest.json | python3 -m json.tool | grep -c '"index"'
# Should show 6

# Check all chunk caches exist
ls ~/aquapose/projects/YH/runs/run_YYYYMMDD_HHMMSS/diagnostics/chunk_*/cache.pkl | wc -l
# Should show 6
```

### Expected eval_results.json Structure
```json
{
  "run_id": "run_YYYYMMDD_HHMMSS",
  "stages_present": ["association", "detection", "midline", "reconstruction", "tracking"],
  "frames_evaluated": 1800,
  "frames_available": 1800,
  "stages": {
    "detection": { "total_detections": ..., "mean_confidence": ... },
    "tracking": { "track_count": ..., "detection_coverage": ... },
    "association": {
      "fish_yield_ratio": ...,
      "singleton_rate": ...,
      "p50_camera_count": ...,
      "p90_camera_count": ...
    },
    "midline": {
      "mean_confidence": ...,
      "p10_confidence": ..., "p50_confidence": ..., "p90_confidence": ...
    },
    "reconstruction": {
      "mean_reprojection_error": ...,
      "p50_reprojection_error": ...,
      "p90_reprojection_error": ...,
      "p95_reprojection_error": ...,
      "per_point_error": { "0": {"mean_px": ..., "p90_px": ...}, ... },
      "curvature_stratified": { "Q1": {...}, "Q2": {...}, "Q3": {...}, "Q4": {...} }
    },
    "fragmentation": {
      "mean_continuity_ratio": ...,
      "total_gaps": ...,
      "mean_gap_duration": ...
    }
  }
}
```

## Key Metrics to Record

| Metric | Source Stage | JSON Path | Purpose |
|--------|-------------|-----------|---------|
| Singleton rate | association | stages.association.singleton_rate | Cross-view association quality |
| Reproj p50 | reconstruction | stages.reconstruction.p50_reprojection_error | Median 3D accuracy |
| Reproj p90 | reconstruction | stages.reconstruction.p90_reprojection_error | Tail 3D accuracy |
| Reproj p95 | reconstruction | stages.reconstruction.p95_reprojection_error | Worst-case 3D accuracy |
| Continuity ratio | fragmentation | stages.fragmentation.mean_continuity_ratio | Track fragmentation |
| Per-keypoint error | reconstruction | stages.reconstruction.per_point_error | Body-point accuracy |
| Curvature-stratified | reconstruction | stages.reconstruction.curvature_stratified | Curvature bias |
| Fish yield | association | stages.association.fish_yield_ratio | Association completeness |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Mean-only reprojection error | Percentile-based (p50/p90/p95) | Phase 70 (v3.6) | Better characterizes error distribution |
| No per-keypoint breakdown | Per-keypoint mean + p90 | Phase 70 (EVAL-04) | Identifies problematic body points |
| No curvature stratification | Curvature-quartile analysis | Phase 70 (EVAL-05) | Quantifies curvature bias |
| No fragmentation analysis | Gap count, duration, continuity | Phase 70 (EVAL-06) | Measures track quality |

## Open Questions

1. **Model paths after Phase 71**
   - What we know: `register_trained_model()` updates config.yaml `detection.weights_path` and `midline.weights_path` automatically
   - What's unclear: Exact paths depend on Phase 71 training run timestamps and directory structure
   - Recommendation: Verify config.yaml after Phase 71 completes, before starting Phase 72

2. **Pipeline runtime estimate**
   - What we know: 6 chunks of 300 frames each. Previous full run (47 chunks) took ~12 hours
   - Estimate: ~1.5 hours for 6 chunks (linear scaling estimate)
   - Recommendation: Use background execution and monitor progress

## Sources

### Primary (HIGH confidence)
- `src/aquapose/cli.py` -- CLI command definitions, `--max-chunks` flag, `--set` override syntax
- `src/aquapose/engine/config.py` -- Config loading, dot-notation override resolution, frozen dataclass hierarchy
- `src/aquapose/evaluation/runner.py` -- EvalRunner orchestration, eval_results.json generation
- `src/aquapose/evaluation/output.py` -- Report formatting, all metric sections
- `src/aquapose/training/run_manager.py` -- Model registration and config.yaml auto-update
- `~/aquapose/projects/YH/config.yaml` -- Current project config state

### Secondary (MEDIUM confidence)
- `.planning/phases/71-data-store-bootstrap/71-02-PLAN.md` -- Phase 71 workflow details
- `.planning/phases/72-baseline-pipeline-run-metrics/72-CONTEXT.md` -- User decisions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all CLI commands exist and are verified in source
- Architecture: HIGH -- diagnostic output structure confirmed from existing runs
- Pitfalls: HIGH -- informed by project memory (stale LUTs, config state)

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable -- no external dependencies)
