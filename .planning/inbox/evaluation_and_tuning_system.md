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
- Supports single-stage sweeps and two-stage cascade tuning (association → reconstruction)
- Leverages the diagnostic observer as the caching/serialization layer — any diagnostic-mode run becomes an evaluation target
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
- A context loader that deserializes cached stage outputs back into a PipelineContext (the reverse of what the observer does). Uses pickle for intermediate caches during tuning; structured summary output (human-readable format TBD) for final results.
- Per-stage diagnostic files (replacing the monolithic `pipeline_diagnostics.npz`) so stages can be loaded selectively.
- Per-stage metric evaluator functions in `evaluation/`.

## Design Decisions (Resolved)

| Decision | Resolution | Rationale |
|----------|-----------|-----------|
| Sweep strategy | Grid search, 1D-2D grids | Simple, interpretable; keep parameter grids small rather than going Bayesian |
| Partial pipeline execution | Minimal PosePipeline change + external orchestrator | PosePipeline stays a single-pass executor; orchestrator manages sweep loop |
| Per-stage fixture format | Per-stage files in a run subdirectory | Structure varies depending on when the pipeline stops; monolithic NPZ doesn't accommodate this well |
| Detection tuning | Evaluate only | Detection params have limited tunability without model retraining |
| Tracking tuning | Evaluate only | OC-SORT defaults are well-understood; parameters tightly coupled to detection model output. Add sweep capability later if evaluation reveals tracking as a bottleneck |
| Midline tuning | Evaluate only | Midline params are precision/recall filters; reconstruction's own outlier rejection likely handles bad keypoints already. Add sweep capability later if evaluation shows midline quality is a bottleneck |
| Stage-specific win + E2E regression | Report clearly, user decides | System reports both metrics; no automatic rejection of stage-specific improvements |
| Context loader format | Pickle for intermediate caches, structured summary for final results | Pickle is fast and exact for within-session use (pipeline won't be refactored mid-tuning); structured output for human review and persistence |
| Retroactive compatibility | Not required | Building toward future retroactive evaluation capability, but not a core deliverable |
| Sweep during tuning | Run only target stage per combo, full pipeline for winners only | Stage-specific metrics don't require downstream stages; full runs reserved for top-N validation |
| CarryForward / chunk processing | Out of scope | Tuning operates on first N frames as a single unit; carry protocol is a future batch-processing concern |
| Frame selection during tuning | Configurable, two-tier | Fewer frames during sweep (fast iteration), more frames for winner validation |

## Design Principles

1. **Diagnostic observer is the cache** — intermediate outputs are already serialized by the observer in diagnostic mode. Evaluation and tuning consume these artifacts directly.
2. **Stage-specific metrics first** — each stage gets metrics that reflect *its own* quality, not just its downstream effect on reconstruction. Final reconstruction metrics serve as a sanity check, not the primary tuning signal.
3. **One stage at a time, cascade optional** — sweeps target a single stage's parameters. Cascade mode tunes association then reconstruction in sequence, validating end-to-end performance between stages.
4. **Reuse upstream, re-run only target stage** — when sweeping stage N, load cached outputs from stages 1..N-1 and run only stage N per combo. Full pipeline runs are reserved for validating the top-N winners.
5. **Uniform reporting** — consistent report format across stages: per-stage metrics, parameter values tested, best config, and (when cascade) end-to-end validation results.

## Stage-Specific Metrics (Proxy-Based, No Ground Truth)

Since only training data has annotations (and those are evaluated during training), all runtime metrics must be self-consistency and proxy measures.

### Stage 1: Detection (evaluate only)
- **Detection yield**: mean detections/frame per camera (should be stable near n_animals)
- **Confidence distribution**: histogram and percentiles of detection confidence scores
- **Yield stability**: frame-to-frame variance in detection count (high variance = missed detections or false positives)
- **Per-camera balance**: coefficient of variation across cameras (poor lighting or angles show as low-yield cameras)

### Stage 2: Tracking (evaluate only)
- **Track count per camera**: raw count vs n_animals (significant excess indicates fragmentation)
- **Track length distribution**: histogram; longer tracks = less fragmentation
- **Coast event frequency**: how often tracks coast (predict without observation) — high coasting suggests detection gaps
- **Detection coverage**: fraction of (frame, camera) detection events claimed by a track with status "detected" (not coasted)

### Stage 3: Association (evaluate + tune)
- **Fish yield** (PRIMARY): number of reconstructable fish identities / n_animals. Maximize.
- **Singleton rate**: fraction of tracklets not assigned to any multi-camera group
- **Camera coverage distribution**: per-fish histogram of how many cameras see each fish
- **Cluster quality**: intra-cluster affinity score consistency

### Stage 4: Midline (evaluate only)
- **Keypoint confidence**: mean/min confidence across keypoints, per camera and per fish
- **Midline completeness**: fraction of frames where midline extraction succeeds given a detection exists
- **Smoothness**: temporal smoothness of midline keypoints within a track (sudden jumps = extraction failures)
- **Multi-view consistency**: reserved for full validation runs only (requires triangulation)

### Stage 5: Reconstruction (evaluate + tune)
- **Mean reprojection error** (PRIMARY): Tier 1 overall_mean_px. Minimize.
- **Tier 2 stability**: leave-one-out camera dropout max control-point displacement
- **Inlier ratio**: fraction of cameras used as inliers per body point (low = poor multi-view agreement)
- **Low-confidence flag rate**: fraction of reconstructions flagged as low-confidence

### Primary Metric Scoring

Each tunable stage has a single primary metric for ranking parameter combos during sweeps:

| Stage | Primary Metric | Direction | Tiebreaker |
|-------|---------------|-----------|------------|
| Association | Fish yield ratio | Maximize (closer to 1.0) | Singleton rate (lower is better) |
| Reconstruction | Mean reprojection error (px) | Minimize | Tier 2 max displacement (lower is better) |

Composite scoring is avoided initially. Tiebreakers apply only when primary metrics are tied or near-tied. More complex scoring can be introduced later if single-metric ranking proves insufficient.

## Tunable Parameters

### Stage 3: Association
- `association.ray_distance_threshold` (0.03m)
- `association.score_min` (0.3)
- `association.eviction_reproj_threshold` (0.025m)
- `association.leiden_resolution` (1.0)
- `association.early_k` (10)

### Stage 5: Reconstruction
- `reconstruction.outlier_threshold` (10.0 px)
- `reconstruction.min_cameras` (3)
- `reconstruction.inlier_threshold` (50.0 px)
- `reconstruction.n_control_points` (7)

### Default Sweep Ranges

Default grids live in the evaluation module alongside the metric functions (not in pipeline config). This keeps sweep ranges as an evaluation concern, colocated with the code that judges results. Each stage's evaluator module defines a `DEFAULT_GRIDS` dict mapping parameter names to value lists.

Ranges are overridable via CLI (`--param name --range min:max:step`). Defaults should cover a reasonable neighborhood around the current default values.

## CLI Design

```
aquapose eval <run-dir>                     # Evaluate all stages of a diagnostic run
aquapose eval <run-dir> --stage detection   # Evaluate a single stage
aquapose eval <run-dir> --report json       # Machine-readable output

aquapose tune <config.yaml> --stage association        # Sweep one stage
aquapose tune <config.yaml> --cascade                  # Sweep association then reconstruction
aquapose tune <config.yaml> --stage reconstruction --param outlier_threshold --range 5:50:5
                                                        # Sweep specific param with explicit range
aquapose tune <config.yaml> --stage association --n-frames 30 --n-frames-validate 100
                                                        # Two-tier frame counts
```

### `aquapose eval`
- Input: a diagnostic run directory (contains per-stage diagnostic files)
- Computes stage-specific metrics for all stages (or a specified stage)
- Outputs human-readable report to stdout, optionally JSON for machine consumption

### `aquapose tune`
- Input: project config YAML + stage to tune (or `--cascade`)
- Sweep phase: runs target stage only per param combo using cached upstream, evaluates stage-specific metrics (fast, fewer frames)
- Validation phase: runs full pipeline for top-N winners, evaluates all stages including E2E metrics (slow, more frames)
- Cascade mode: tunes association → validates → tunes reconstruction using validated run → validates
- Outputs: best params, per-param metric table, before/after comparison, config diff

## Caching Strategy

### Hybrid approach: pickle for speed, structured output for persistence

**During tuning (pickle):**
- Per-stage pipeline outputs cached as pickle files in a tuning work directory
- Fast serialization/deserialization of complex Python objects (Detection, Tracklet2D, TrackletGroup, etc.)
- No round-trip fidelity concerns since the pipeline code won't change mid-tuning session
- Discardable after tuning completes

**Final output (structured, human-readable):**
- Summary statistics, metadata, best parameters, metric comparisons
- Format TBD (JSON, YAML, or plain text report) — should be human-reviewable
- Persists alongside the run directory

### Sweep workflow
1. Full pipeline run in diagnostic mode → baseline run D0 (per-stage pickle caches + diagnostic files)
2. Stage N sweep: load stages 1..N-1 from D0 pickle cache, run only stage N per param combo, evaluate stage-N metrics
3. Top-N validation: run full pipeline for each of the N best param combos (more frames), evaluate all stages
4. Pick winner, write structured summary
5. Cascade: winner's full run becomes the cache for the next stage's sweep

### Per-stage diagnostic files
Replace the monolithic `pipeline_diagnostics.npz` with per-stage files in a subdirectory within the run. Structure varies naturally depending on `stop_after`. Each stage writes its own artifact; the context loader reads selectively.

## Cascade Tuning Flow

```
1. Run full pipeline with current defaults → D0
2. Evaluate D0 (all stages) → baseline metrics
3. Tune association using D0
   - For each param combo: load stages 1-2 from D0, run stage 3 only, evaluate association metrics
   - Select top-N by fish yield
   - Run full pipeline for each top-N combo (more frames) → validate E2E
   - Pick winner → D1
   - Report: association metrics improved? E2E regression?
4. Tune reconstruction using D1
   - For each param combo: load stages 1-4 from D1, run stage 5 only, evaluate reconstruction metrics
   - Select top-N by mean reprojection error
   - Run full pipeline for each top-N combo (more frames) → validate E2E
   - Pick winner → D2
   - Report: reconstruction metrics improved? E2E regression?
5. Final report: D0 → D2, per-stage deltas, end-to-end delta, recommended config diff
```

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

1. `aquapose eval <run-dir>` produces a multi-stage report for any diagnostic run
2. `aquapose tune --stage <name>` sweeps that stage's parameters using stage-specific primary metrics and validates top-N winners with full pipeline runs
3. `aquapose tune --cascade` tunes association then reconstruction in sequence with proper caching and end-to-end validation
4. Existing tuning script functionality is fully subsumed (tune_association, tune_threshold, measure_baseline can be retired)
5. Per-stage evaluation metrics exist for all five stages (detection, tracking, association, midline, reconstruction)
6. Sweeping uses only the target stage per param combo (upstream cached, downstream deferred to validation)
7. Two-tier frame selection: configurable fast-sweep and thorough-validation frame counts
