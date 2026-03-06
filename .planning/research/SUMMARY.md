# Project Research Summary

**Project:** AquaPose v3.6 Model Iteration & QA
**Domain:** Evaluation metrics extension and pseudo-label retraining loop for multi-view fish pose estimation
**Researched:** 2026-03-06
**Confidence:** HIGH

## Executive Summary

AquaPose v3.6 builds the infrastructure to measure and improve model quality through iterative pseudo-label retraining. The milestone has two distinct workstreams: (1) extending the evaluation system with percentile-based metrics, per-keypoint breakdown, curvature-stratified analysis, and track fragmentation; and (2) executing the pseudo-label retraining loop end-to-end through the data store. Research confirms that no new dependencies are needed -- every computation uses existing NumPy/SciPy/stdlib capabilities, and the existing evaluation architecture (pure-function stage evaluators returning frozen dataclasses) provides clean extension points for all new metrics.

The recommended approach is to build metrics infrastructure first (Phase 70) so that all retraining rounds are measured consistently, then bootstrap the data store (Phase 71), run the baseline (Phase 72), and iterate (Phases 73-76). The architecture research identifies a clean two-category split: simple percentile extensions belong in existing evaluator functions, while cross-stage analyses (per-keypoint, curvature, fragmentation) belong in a new `evaluation/analysis.py` module. Per-keypoint analysis should recompute residuals from cached splines rather than modifying `Midline3D` to avoid invalidating all existing pipeline caches.

The dominant risks are not technical but methodological: confirmation bias amplifying model errors across retraining rounds, train/val temporal leakage from adjacent-frame near-duplicates, and elastic augmentation compounding noise when applied to pseudo-labeled (inherently noisy) keypoints. The algae domain shift between clean-tank manual annotations and current conditions is a concrete threat that requires explicit false-positive auditing before pseudo-label generation. All pitfalls have clear prevention strategies tied to specific phases.

## Key Findings

### Recommended Stack

No changes to `pyproject.toml`. All v3.6 features are implementable with the existing dependency set.

**Core technologies (all existing):**
- **NumPy >=1.24**: `np.percentile`, `np.digitize` for all binning and percentile computations
- **SciPy >=1.11**: `BSpline` evaluation for 3D curvature computation from `Midline3D` control points
- **Python stdlib**: `dataclasses` (frozen metric types), `json` (output serialization), `sqlite3` (store queries)
- **Click >=8.1**: Existing CLI commands extend naturally; no new subcommands for metrics

**Explicitly not needed:** pandas, matplotlib, polars, rich, pydantic. The data volumes (hundreds to thousands of fish-frames per eval run) do not justify dataframe libraries, and frozen dataclasses are the project's established pattern.

**Code organization note:** `compute_curvature` must be moved from `training/pseudo_labels.py` to a shared location (e.g., `core/geometry.py`) to avoid evaluation importing from training.

### Expected Features

**Must have (table stakes):**
- Reprojection error percentiles (p50/p90/p95) -- LOW complexity, reveals distributional shape hidden by mean/max
- Midline confidence percentiles (p10/p50/p90) -- LOW complexity, surfaces low-confidence tail
- Camera count percentiles (p50/p90) -- LOW complexity, summarizes coverage
- Per-keypoint reprojection error breakdown -- MEDIUM complexity, reveals systematic head-vs-tail model failures
- COCO-to-store bootstrap workflow (`data convert` CLI subcommand) -- MEDIUM complexity, entry point for the entire iteration loop
- Baseline model training and registration -- LOW complexity, establishes round 0 provenance
- Round-over-round eval comparison -- MEDIUM complexity, the core measurement of iteration success

**Should have (differentiators):**
- Curvature-stratified reconstruction quality -- directly measures whether iteration fixes curvature bias from v3.5
- 3D track fragmentation analysis -- measures trajectory continuity for downstream behavioral analysis
- Detection false positive rate estimation -- heuristic but actionable proxy from singleton analysis
- Per-round provenance summary -- reproducibility and publishability

**Defer (beyond v3.6):**
- Automated iteration loop orchestration -- only 1-2 rounds; human judgment at checkpoints is more valuable
- Ground-truth evaluation against held-out annotations -- insufficient annotation budget (~50 images)
- Segmentation model iteration -- blocked by unresolved cross-camera orientation problem
- Hyperparameter search -- bottleneck is data quality, not training hyperparameters
- Real-time eval dashboard -- training runs are 1-2 hours; text/JSON output suffices

### Architecture Approach

The evaluation system follows a three-layer architecture: CLI -> EvalRunner (orchestrator) -> per-stage evaluator pure functions -> output formatters. New work splits into two categories. **Category A (evaluator extensions)** adds percentile fields to existing frozen dataclasses with default values for backward compatibility -- mechanical, low-risk changes. **Category B (new analysis module)** creates `evaluation/analysis.py` with three frozen dataclasses (`KeypointAnalysis`, `CurvatureAnalysis`, `FragmentationAnalysis`) and three pure analysis functions, wired into `EvalRunnerResult` as optional fields under a top-level `"analyses"` key.

**Major components:**
1. **Existing stage evaluators** (reconstruction, midline, association) -- extend with percentile fields; ~5-10 lines each
2. **`evaluation/analysis.py` (NEW)** -- curvature-stratified quality, track fragmentation, per-keypoint breakdown
3. **`evaluation/runner.py`** -- add CalibBundle loading for per-keypoint analysis; call analysis functions; extend `EvalRunnerResult`
4. **`evaluation/output.py`** -- extend ASCII and JSON formatters for new metrics and analyses

**Critical design decision:** Per-keypoint analysis recomputes residuals from cached B-splines at analysis time rather than storing per-body-point residuals on `Midline3D`. This avoids cache invalidation (`context_fingerprint` mismatch -> `StaleCacheError`) and keeps the core types stable.

**Anti-patterns to avoid:**
- Do not put cross-stage analysis (needs CalibBundle + 2D + 3D data) inside a stage evaluator
- Do not add required (non-default) fields to existing frozen dataclasses -- always use defaults
- Do not let analysis functions load their own data -- pass typed data from EvalRunner
- Do not store per-body-point residuals on Midline3D -- invalidates all existing caches

### Critical Pitfalls

1. **Confirmation bias in pseudo-label retraining (P1)** -- The model reinforces its own errors across rounds, especially for underrepresented poses. Confidence scoring measures geometric consistency, not semantic correctness, so well-triangulated algae patches pass filtering. **Avoid by:** visual audit of 50+ pseudo-labels per round, tracking curvature distribution shift, capping pseudo:manual ratio at 3:1, never allowing pseudo-labels in val.

2. **Train/val temporal leakage (P2)** -- Adjacent video frames are near-duplicates with different content hashes. Random split places near-identical scenes in both train and val. **Avoid by:** temporal holdout (reserve first/last 15 seconds as val-only), generating pseudo-labels only from the training temporal window.

3. **Elastic augmentation on pseudo-labels compounds noise (P3)** -- Pseudo-label keypoints have 2-5px reprojection error. TPS warp assumes ground-truth control points, producing misaligned deformed images. **Avoid by:** applying elastic augmentation only to `source=manual` samples, never to pseudo-labels.

4. **Per-keypoint error conflates z-uncertainty with pose error (P5)** -- Tail keypoints show higher reprojection error due to larger z-oscillation amplitude (132x z-uncertainty), not model inaccuracy. **Avoid by:** reporting per-keypoint z-variance alongside error; using relative per-keypoint improvement between rounds, not absolute values.

5. **Short-run metrics do not predict full-run performance (P6)** -- Tracker drift, chunk boundary artifacts, and behavioral variation are masked in 1-minute clips. **Avoid by:** treating short-run metrics as directional only; planning for metric regression on the full 5-minute validation run.

## Implications for Roadmap

Based on research, the SEED document's phase structure (70-76) is well-designed. Research confirms the ordering and adds specific implementation guidance.

### Phase 70: Metrics and Comparison Infrastructure
**Rationale:** Must exist before any pipeline run so all rounds are measured identically. No external dependencies; extends existing patterns.
**Delivers:** Extended evaluator metrics (percentiles), new analysis module (curvature, fragmentation, per-keypoint), updated formatters.
**Addresses:** All table-stakes metric features; curvature-stratified and fragmentation differentiators.
**Avoids:** P4 (curvature bin instability -- use 3-5 quantile bins, report N per bin), P5 (z-confound -- report z-variance alongside per-keypoint error), P10 (fragmentation ambiguity -- classify gaps by reason).
**Build order:** (1a/1b/1c) Percentile extensions in parallel, then (2a) CurvatureAnalysis, (2b) FragmentationAnalysis, (2c) wire into runner/formatters, (3a) CalibBundle loading, (3b/3c) per-keypoint analysis.

### Phase 71: Data Store Bootstrap
**Rationale:** Critical-path dependency for the iteration loop. Cannot generate pseudo-labels or train models without store-managed data.
**Delivers:** Manual annotations imported, baseline OBB and pose models trained and registered, end-to-end workflow validated.
**Addresses:** COCO-to-store bootstrap, baseline model training, provenance tracking.
**Avoids:** P8 (OBB corner order -- verify `pca_obb` consistency between manual and pseudo-label paths), P9 (store dedup -- import manual first, exclude overlapping frames from pseudo-labels), P11 (visibility collapse -- check COCO annotations for v=1 keypoints), P2 (temporal leakage -- establish temporal split convention at bootstrap time).

### Phase 72: Baseline Pipeline Run and Metrics
**Rationale:** Establishes the quantitative "before" snapshot that all improvement is measured against.
**Delivers:** Baseline metric numbers, overlay video for visual FP assessment, documented false positive rate.
**Addresses:** Round-over-round comparison baseline.
**Avoids:** P7 (algae domain shift -- count algae FPs in overlay video; add negative examples if >5% of detections), P6 (short-run caveat -- explicitly document that metrics are short-run-specific).

### Phase 73: Round 1 Pseudo-Label Generation and Retraining
**Rationale:** First iteration of the loop. Pseudo-labels from baseline run augment training data.
**Delivers:** Round 1 OBB and pose models with pseudo-label augmented training sets.
**Addresses:** Pseudo-label generation, mixed-source dataset assembly, round 1 training.
**Avoids:** P1 (confirmation bias -- visual audit, curvature distribution tracking), P3 (augmentation noise -- augment manual only, not pseudo-labels), P2 (leakage -- exclude val-window frames from pseudo-label export).

### Phase 74: Round 1 Evaluation and Decision
**Rationale:** Checkpoint before committing to round 2. Human judgment required on whether to continue, adjust, or stop.
**Delivers:** Round 0 vs round 1 comparison metrics, decision on whether round 2 is warranted.
**Addresses:** Round-over-round eval comparison.
**Avoids:** P1 (confirmation bias -- stop if val metrics plateau or curvature distribution narrows).

### Phase 75: Round 2 (Conditional)
**Rationale:** Only executed if Phase 74 shows clear improvement and room for more. May be skipped.
**Delivers:** Round 2 models if justified; otherwise documents why iteration stopped.
**Avoids:** P1 (cap at 2 rounds to limit confirmation bias accumulation), P3 (re-evaluate whether augmentation is needed -- pseudo-label curvature diversity may make it redundant).

### Phase 76: Final Validation on Full Run
**Rationale:** Full 5-minute run tests production-scale behavior that short clips cannot capture.
**Delivers:** Production-quality metrics, side-by-side overlay video, final model selection.
**Avoids:** P6 (short vs full divergence -- expect and document metric regression; do not re-iterate based on the gap).

### Phase Ordering Rationale

- **Phase 70 before 71** because metrics infrastructure is independent of the data store and must be ready before the first `aquapose eval` call in Phase 72.
- **Phase 70 and 71 can develop in parallel** since metrics operate on existing cached data while bootstrap operates on annotation files and the store.
- **Phase 72 before 73** because baseline metrics are the comparison target. Without a "before" snapshot, improvement cannot be measured.
- **Phases 73-74 are tightly coupled** (generate, retrain, evaluate, decide) and should be treated as one execution block.
- **Phase 75 is conditional** on Phase 74 results -- the roadmap should not assume it will execute.
- **Phase 76 is always executed** regardless of how many rounds completed; it validates the final model on production-scale data.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 70 (per-keypoint analysis):** CalibBundle loading in EvalRunner is new plumbing; spline-to-camera reprojection logic needs careful implementation to match DLT backend behavior.
- **Phase 73 (pseudo-label filtering):** Temporal-consistency and wall-proximity filters for algae FP exclusion are not yet implemented; need design during phase planning.

Phases with standard patterns (skip research-phase):
- **Phase 70 (percentile extensions):** Mechanical dataclass field additions; no design ambiguity.
- **Phase 71 (data store bootstrap):** All components exist; this is workflow validation, not new development.
- **Phase 72 (baseline run):** Existing CLI commands; no new code needed.
- **Phase 74 (evaluation comparison):** Comparison is JSON diff plus metric table; straightforward.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | No new dependencies. All computations verified against existing codebase APIs. |
| Features | HIGH | Features derived from direct codebase inspection and milestone seed requirements. Clear table-stakes vs differentiator separation. |
| Architecture | HIGH | All extension points verified by reading evaluator source code. Two-category (evaluator extension vs analysis module) split is clean and well-motivated. |
| Pitfalls | HIGH | Pitfalls grounded in codebase inspection (store dedup logic, confidence scoring formula, DLT backend internals) and verified semi-supervised learning literature. |

**Overall confidence:** HIGH

### Gaps to Address

- **Temporal split convention for train/val:** No implementation exists in the store's `assemble()` method. Must be designed during Phase 71 planning. Practical workaround: manually curate frame ranges for train/val windows before import.
- **Algae FP filtering in pseudo-labels:** Temporal-consistency check (stationary detections) and wall-proximity filter are described conceptually but not implemented. Must be designed during Phase 73 planning or handled as manual visual filtering.
- **Per-keypoint z-variance reporting:** The analysis function design includes this but the exact computation (per-keypoint z standard deviation across frames) needs implementation details during Phase 70 per-keypoint task planning.
- **Round-over-round comparison CLI:** Could be a dedicated `aquapose eval --compare` command or manual JSON diff. Design choice deferred to Phase 74 planning; manual approach is sufficient for 1-2 rounds.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection of `evaluation/stages/*.py`, `evaluation/runner.py`, `evaluation/output.py` -- evaluator architecture, available data, extension points
- Direct codebase inspection of `core/types/reconstruction.py`, `core/reconstruction/backends/dlt.py` -- Midline3D schema, per-body-point residual computation
- Direct codebase inspection of `training/store_schema.py`, `training/data_cli.py`, `training/coco_convert.py` -- store workflow, conversion pipeline
- Direct codebase inspection of `training/pseudo_labels.py`, `training/elastic_deform.py` -- pseudo-label confidence scoring, augmentation pipeline
- `.planning/milestones/v3.6-SEED.md` -- milestone scope, phase structure, validation approach

### Secondary (MEDIUM confidence)
- [Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning](https://arxiv.org/pdf/1908.02983) -- confirmation bias in iterative pseudo-labeling
- [Data Leakage in Visual Datasets](https://arxiv.org/html/2508.17416v1) -- near-duplicate and split-level leakage
- Standard practices in multi-view pose estimation evaluation (percentile reporting, curvature stratification, per-keypoint breakdown)

### Tertiary (LOW confidence)
- Curvature bin count recommendation (3-5 bins) -- based on sample size estimation for ~12k fish-frame reconstructions; actual distribution may warrant different binning

---
*Research completed: 2026-03-06*
*Ready for roadmap: yes*
