# Milestone Seed: Evaluation and Parameter Tuning System

**Status:** Seed (pre-milestone)
**Proposed version:** v3.2 or v4.0 (TBD)
**Date:** 2026-03-03

## Problem Statement

The current evaluation and tuning infrastructure is scattered across standalone scripts (`tune_association.py`, `tune_threshold.py`, `measure_baseline.py`) that each solve a narrow problem with bespoke logic. Key limitations:

1. **Only reconstruction is evaluated** — no stage-specific metrics for detection, tracking, association, or midline quality
2. **No CLI integration** — scripts live in `scripts/` and require manual invocation with long argument lists
3. **Wasteful computation during sweeps** — association tuning regenerates the full 5-stage pipeline per parameter combo, even though only association config changes
4. **Limited parameter coverage** — only association params and DLT outlier threshold are swept; tracking, midline, and detection params are untuned
5. **Metrics are reconstruction-centric** — tuning earlier stages uses downstream reconstruction error as the sole signal, which conflates stage-specific quality with downstream effects
6. **No cascade tuning** — no way to tune stages in sequence, locking in best params at each level before proceeding downstream

## Vision

A unified evaluation and parameter tuning system that:
- Lives under `aquapose eval` and `aquapose tune` CLI subcommands
- Measures stage-specific quality at every pipeline stage using automated proxy metrics (no ground truth required)
- Supports single-stage sweeps and full cascade tuning (tune stage 1 → cache → tune stage 2 → ... → tune stage 5)
- Leverages the diagnostic observer as the caching/serialization layer — any diagnostic-mode run becomes a reusable evaluation target
- Tracks end-to-end performance alongside stage-specific metrics to ensure per-stage improvements don't sacrifice overall quality

## Architecture: Orchestrator over PosePipeline

PosePipeline stays a single-pass executor over a single PipelineContext. Evaluation and tuning are handled by a separate orchestrator that creates and manages multiple pipeline runs.

**Key change to PosePipeline:** `run()` accepts an optional pre-populated `PipelineContext` instead of always creating a fresh one. This enables "resume from stage N" by pre-loading cached upstream outputs into the context and passing a truncated stage list.

```
CLI layer:    aquapose eval / aquapose tune
                    ↓
Orchestrator: EvalRunner / TuningOrchestrator  (sweep loop, metrics, caching, reporting)
                    ↓
Pipeline:     PosePipeline  (unchanged except accepting initial context)
                    ↓
Stages:       Detection → Tracking → Association → Midline → Reconstruction
                    ↓
Observer:     DiagnosticObserver  (per-stage output capture → per-stage files)
```

**Why not an EvaluationPipeline?**
- PosePipeline's job is "execute stages, emit events." Sweep logic (try N configs, evaluate, compare) is an outer loop, not a pipeline feature.
- The diagnostic observer already bridges pipeline execution to evaluation — the orchestrator reads observer output.
- This same pattern applies to future batch/chunk processing: PipelineContext is the building block for a single chunk, and cross-chunk orchestration sits above.
- Keeps PosePipeline testable with simple mocks; orchestrator testable by injecting fake pipeline results.

**New infrastructure required:**
- A context loader that deserializes diagnostic output back into a PipelineContext (the reverse of what the observer does today). Lives in `evaluation/` or `io/`.
- Per-stage diagnostic files (replacing the monolithic `pipeline_diagnostics.npz`) so stages can be loaded selectively.
- Per-stage metric evaluator functions in `evaluation/metrics.py`.

## Design Decisions (Resolved)

| Decision | Resolution | Rationale |
|----------|-----------|-----------|
| Sweep strategy | Grid search, 1D-2D grids | Simple, interpretable; keep parameter grids small rather than going Bayesian |
| Partial pipeline execution | Minimal PosePipeline change + external orchestrator | PosePipeline stays a single-pass executor; orchestrator manages sweep loop |
| Per-stage fixture format | Per-stage files in a run subdirectory | Structure varies depending on when the pipeline stops; monolithic NPZ doesn't accommodate this well |
| Detection tuning | Evaluate only, no sweep | Detection params have limited tunability without model retraining |
| Stage-specific win + E2E regression | Report clearly, user decides | System reports both metrics; no automatic rejection of stage-specific improvements |

## Design Principles

1. **Diagnostic observer is the cache** — intermediate outputs are already serialized by the observer in diagnostic mode. Evaluation and tuning consume these artifacts directly, making the system retroactively applicable to any past diagnostic run.
2. **Stage-specific metrics first** — each stage gets metrics that reflect *its own* quality, not just its downstream effect on reconstruction. Final reconstruction metrics serve as a sanity check, not the primary tuning signal.
3. **One stage at a time, cascade optional** — sweeps target a single stage's parameters. A cascade mode runs stages in sequence, locking in the best config at each level before moving downstream. After each stage tuning, a full pipeline run with the new params (and defaults for untuned stages) validates that overall performance holds.
4. **Reuse upstream, re-run downstream** — when sweeping stage N params, load cached outputs from stages 1..N-1 and only re-run stages N..5. This is the primary efficiency win over the current approach.
5. **Uniform reporting** — consistent report format across stages: per-stage metrics, parameter values tested, best config, and (when cascade) end-to-end validation results.

## Stage-Specific Metrics (Proxy-Based, No Ground Truth)

Since only training data has annotations (and those are evaluated during training), all runtime metrics must be self-consistency and proxy measures.

### Stage 1: Detection (evaluate only — no tuning)
- **Detection yield**: mean detections/frame per camera (should be stable near n_animals)
- **Confidence distribution**: histogram and percentiles of detection confidence scores
- **Yield stability**: frame-to-frame variance in detection count (high variance = missed detections or false positives)
- **Per-camera balance**: coefficient of variation across cameras (poor lighting or angles show as low-yield cameras)

### Stage 2: Tracking (OC-SORT)
- **Track fragmentation**: number of tracks vs expected (9 fish x 13 cameras = ~117 long tracks ideal)
- **Track length distribution**: histogram; long tracks = good, many short tracks = fragmentation
- **Coast event frequency**: how often tracks coast (predict without observation) — high coasting suggests detection gaps
- **Velocity consistency**: sudden large jumps in track position suggest ID switches or false matches

### Stage 3: Association
- **Fish yield**: number of reconstructable fish identities (target: n_animals)
- **Singleton rate**: fraction of tracklets not assigned to any multi-camera group
- **Camera coverage distribution**: per-fish histogram of how many cameras see each fish
- **Cluster quality**: intra-cluster affinity score consistency

### Stage 4: Midline
- **Keypoint confidence**: mean/min confidence across keypoints, per camera and per fish
- **Multi-view consistency**: for fish seen by multiple cameras, reproject 3D midline (from simple triangulation) back to 2D and compare with extracted midlines — measures whether views agree
- **Midline completeness**: fraction of frames where midline extraction succeeds given a detection exists
- **Smoothness**: temporal smoothness of midline keypoints within a track (sudden jumps = extraction failures)

### Stage 5: Reconstruction
- **Tier 1 (existing)**: per-fish, per-camera reprojection error (mean, max, overall)
- **Tier 2 (existing)**: leave-one-out camera dropout stability (max control-point displacement)
- **Inlier ratio**: fraction of cameras used as inliers per body point (low = poor multi-view agreement)
- **Low-confidence flag rate**: fraction of reconstructions flagged as low-confidence

## Tunable Parameters by Stage

### Stage 1: Detection — evaluate only (tuning requires model retraining)

### Stage 2: Tracking
- `tracking.max_coast_frames` (30): how long to predict without observation
- `tracking.n_init` (3): frames to confirm a track
- `tracking.iou_threshold` (0.3): detection-to-track matching
- `tracking.det_thresh` (0.3): minimum detection confidence for tracking

### Stage 3: Association
- `association.ray_distance_threshold` (0.03m)
- `association.score_min` (0.3)
- `association.eviction_reproj_threshold` (0.025m)
- `association.leiden_resolution` (1.0)
- `association.early_k` (10)

### Stage 4: Midline
- `midline.confidence_threshold`: minimum keypoint confidence
- `midline.min_observed_keypoints`: minimum valid keypoints per detection
- `midline.keypoint_confidence_floor`: confidence floor for low-confidence keypoints

### Stage 5: Reconstruction
- `reconstruction.outlier_threshold` (10.0 px)
- `reconstruction.min_cameras` (3)
- `reconstruction.inlier_threshold` (50.0 px)
- `reconstruction.n_control_points` (7)

## CLI Design

```
aquapose eval <run-dir>                     # Evaluate all stages of a diagnostic run
aquapose eval <run-dir> --stage detection   # Evaluate a single stage
aquapose eval <run-dir> --report json       # Machine-readable output

aquapose tune <config.yaml> --stage tracking          # Sweep one stage
aquapose tune <config.yaml> --cascade                 # Sweep all stages in order
aquapose tune <config.yaml> --stage association --param ray_distance_threshold --range 0.01:0.1:0.01
                                                       # Sweep specific param with explicit range
```

### `aquapose eval`
- Input: a diagnostic run directory (contains per-stage diagnostic files)
- Computes stage-specific metrics for all stages (or a specified stage)
- Outputs human-readable report to stdout, optionally JSON for machine consumption
- Works on any past diagnostic-mode run — retroactive analysis

### `aquapose tune`
- Input: project config YAML + stage to tune
- Runs parameter sweeps using stage-specific metrics as primary signal
- Caching: loads cached upstream outputs from diagnostic observer, only re-runs target stage and downstream
- After finding best params: runs full pipeline to validate end-to-end performance
- Cascade mode: tunes stages 1→2→3→4→5 in sequence, each time caching the best result for the next stage
- Outputs: best params, per-param metric table, before/after comparison, config diff

## Caching Strategy (Diagnostic Observer)

The diagnostic observer already captures per-stage outputs. The key changes:

1. **Per-stage files** replace the monolithic `pipeline_diagnostics.npz`. Each stage writes its own artifact to a diagnostics subdirectory within the run. Structure varies naturally depending on `stop_after`.
2. **Context loader** reverses the observer's serialization — reads per-stage files back into a PipelineContext. This is the main new infrastructure.
3. **Sweep workflow:**
   - Full pipeline run in diagnostic mode produces cached per-stage outputs (run D0)
   - Stage N sweep: load stages 1..N-1 from D0, inject into PipelineContext, run stages N..5 with each param combo
   - Best-param validation: full pipeline run with winning params → D1
   - Cascade: D1 becomes the cache for stage N+1 tuning

## Cascade Tuning Flow

```
1. Run full pipeline with current defaults → diagnostic run D0
2. Evaluate D0 (all stages) → baseline metrics
3. Tune Stage 2 (tracking) using D0
   - For each param combo: load stage 1 output from D0, re-run stages 2-5, evaluate stage-2 metrics
   - Pick best stage-2 params
   - Run full pipeline with best stage-2 params → D1
   - Compare D1 end-to-end metrics vs D0 (report regression if any)
4. Tune Stage 3 (association) using D1
   - For each param combo: load stages 1-2 output from D1, re-run stages 3-5, evaluate stage-3 metrics
   - Pick best stage-3 params
   - Run full pipeline with best stage-2 + stage-3 params → D2
   - Compare D2 vs D1
5. ... continue through Stage 5
6. Final report: D0 (baseline) → D_final (fully tuned), per-stage deltas, end-to-end delta
```

Note: Stage 1 (detection) is skipped in cascade tuning since it's evaluate-only.

## Migration Path from Current Scripts

The existing scripts contain domain knowledge that should be preserved:

| Current Script | Becomes |
|---------------|---------|
| `tune_association.py` | Logic migrates into `aquapose tune --stage association` |
| `tune_threshold.py` | Logic migrates into `aquapose tune --stage reconstruction` |
| `measure_baseline.py` | Logic migrates into `aquapose eval` |
| `evaluation/harness.py` | Refactored — reconstruction metrics become one stage evaluator among five |
| `evaluation/metrics.py` | Expanded with per-stage metric functions |
| `evaluation/output.py` | Generalized for multi-stage reporting |

## Success Criteria

1. `aquapose eval <run-dir>` produces a multi-stage report for any diagnostic run (including past runs)
2. `aquapose tune --stage <name>` sweeps that stage's parameters using stage-specific metrics and caches results
3. `aquapose tune --cascade` tunes all stages in sequence with proper caching and end-to-end validation
4. Existing tuning script functionality is fully subsumed (tune_association, tune_threshold, measure_baseline can be retired)
5. Per-stage metrics exist for at least tracking, association, midline, and reconstruction
6. Sweeping a single stage's parameters reuses upstream cached outputs (no redundant computation)
