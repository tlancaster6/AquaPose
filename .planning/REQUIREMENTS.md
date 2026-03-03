# Requirements: AquaPose v3.2 Evaluation Ecosystem

**Defined:** 2026-03-03
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v3.2 Requirements

Requirements for the evaluation ecosystem milestone. Each maps to roadmap phases.

### Infrastructure

- [x] **INFRA-01**: DiagnosticObserver writes per-stage pickle cache files on each StageComplete event
- [x] **INFRA-02**: PosePipeline.run() accepts optional pre-populated PipelineContext via initial_context parameter
- [x] **INFRA-03**: ContextLoader deserializes per-stage pickle caches into a fresh PipelineContext for sweep isolation
- [x] **INFRA-04**: StaleCacheError raised with clear message when pickle deserialization fails due to class evolution

### Evaluation

- [x] **EVAL-01**: Detection stage evaluator computes yield, confidence distribution, yield stability, and per-camera balance metrics
- [x] **EVAL-02**: Tracking stage evaluator computes track count, track length distribution, coast frequency, and detection coverage metrics
- [x] **EVAL-03**: Association stage evaluator computes fish yield ratio, singleton rate, camera coverage, and cluster quality metrics
- [x] **EVAL-04**: Midline stage evaluator computes keypoint confidence, midline completeness, and temporal smoothness metrics
- [x] **EVAL-05**: Reconstruction stage evaluator computes mean reprojection error, Tier 2 stability, inlier ratio, and low-confidence flag rate
- [x] **EVAL-06**: `aquapose eval <run-dir>` CLI produces multi-stage human-readable report to stdout
- [x] **EVAL-07**: `aquapose eval <run-dir> --report json` produces machine-readable JSON output

### Tuning

- [x] **TUNE-01**: `aquapose tune --stage association` sweeps association parameters using grid search with fish yield as primary metric
- [x] **TUNE-02**: `aquapose tune --stage reconstruction` sweeps reconstruction parameters using grid search with mean reprojection error as primary metric
- [x] **TUNE-03**: Two-tier frame counts: configurable fast-sweep and thorough-validation frame counts via CLI flags
- [x] **TUNE-04**: Top-N validation runs full pipeline for sweep winners to verify E2E quality
- [x] **TUNE-05**: Tuning output includes before/after metric comparison and recommended config diff block
- [x] **TUNE-06**: DEFAULT_GRIDS for association and reconstruction parameters colocated with stage evaluator modules

### Cleanup

- [x] **CLEAN-01**: `scripts/tune_association.py` retired after `aquapose tune --stage association` achieves feature parity
- [x] **CLEAN-02**: `scripts/tune_threshold.py` retired after `aquapose tune --stage reconstruction` achieves feature parity
- [x] **CLEAN-03**: `scripts/measure_baseline.py` retired after `aquapose eval` achieves feature parity
- [x] **CLEAN-04**: Monolithic `pipeline_diagnostics.npz` machinery removed or fully integrated into per-stage cache system
- [x] **CLEAN-05**: `evaluation/harness.py` removed — functionality consolidated into reconstruction stage evaluator

## Deferred Requirements

### Cascade Tuning

- **CASCADE-01**: `aquapose tune --cascade` tunes association then reconstruction in sequence with proper caching
- **CASCADE-02**: Cascade orchestrator threads association winner's run as upstream cache for reconstruction sweep
- **CASCADE-03**: Combined E2E delta report from baseline through cascade stages

### CLI Extensions

- **CLI-01**: `aquapose eval --stage <name>` filter for single-stage evaluation
- **CLI-02**: `--param name --range min:max:step` CLI override for custom sweep ranges

### Additional Sweep Stages

- **SWEEP-01**: `aquapose tune --stage tracking` — add only if evaluation reveals tracking as a bottleneck
- **SWEEP-02**: `aquapose tune --stage midline` — add only if evaluation reveals midline as a bottleneck

### Advanced Features

- **ADV-01**: Cross-session cache reuse with version tagging
- **ADV-02**: Parallel sweep execution across multiple GPU processes

## Out of Scope

| Feature | Reason |
|---------|--------|
| Bayesian / GP optimization | Parameter spaces are small (2-5 params); grid search is interpretable and sufficient |
| Automatic config file mutation | Destroys reproducibility; researcher must review and apply config diff manually |
| Real-time / streaming evaluation | Evaluation requires full stage output; adds synchronization complexity |
| Pydantic for config validation | Project decision: frozen dataclasses (decided v2.0) |
| Composite weighted scoring | Single primary metric + tiebreaker is more auditable; report all metrics for human review |
| Ground-truth metrics at runtime | No ground truth available; proxy metrics are always available and sufficient |
| Retroactive eval of pre-v3.2 runs | Re-run pipeline in diagnostic mode to get v3.2-compatible output |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 46 | Complete |
| INFRA-02 | Phase 46 | Complete |
| INFRA-03 | Phase 46 | Complete |
| INFRA-04 | Phase 46 | Complete |
| EVAL-01 | Phase 47 | Complete |
| EVAL-02 | Phase 47 | Complete |
| EVAL-03 | Phase 47 | Complete |
| EVAL-04 | Phase 47 | Complete |
| EVAL-05 | Phase 47 | Complete |
| EVAL-06 | Phase 48 | Complete |
| EVAL-07 | Phase 48 | Complete |
| TUNE-01 | Phase 49 | Complete |
| TUNE-02 | Phase 49 | Complete |
| TUNE-03 | Phase 49 | Complete |
| TUNE-04 | Phase 49 | Complete |
| TUNE-05 | Phase 49 | Complete |
| TUNE-06 | Phase 47 | Complete |
| CLEAN-01 | Phase 49 | Complete |
| CLEAN-02 | Phase 49 | Complete |
| CLEAN-03 | Phase 48 | Complete |
| CLEAN-04 | Phase 50 | Complete |
| CLEAN-05 | Phase 50 | Complete |

**Coverage:**
- v3.2 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-03-03*
*Last updated: 2026-03-03 after roadmap creation*
