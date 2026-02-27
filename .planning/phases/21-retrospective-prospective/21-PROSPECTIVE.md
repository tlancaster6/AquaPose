# v2.1 Prospective: AquaPose — Accuracy and Publication Readiness

**Period:** 2026-02-27
**Scope:** Next milestone direction — end-to-end accuracy improvement toward publication-quality results
**Author:** Phase 21 executor, 2026-02-27

---

## Milestone Vision

The v2.0 Alpha refactor delivered a sound, well-tested software architecture: a clean 5-stage pipeline, strict import boundaries, typed event system, observer-based side effects, and a working CLI. The architecture is ready to support significant algorithmic improvements without structural changes. That foundation now enables v2.1 to focus entirely on **quality** — producing 3D fish midline reconstructions that are good enough to publish.

The key driver is research publication. Publication reviewers will scrutinize reconstruction accuracy, robustness across fish behaviors, and quantitative validation against ground truth. The current pipeline produces plausible outputs but has known quality gaps that would be difficult to defend in a paper. v2.1 should close those gaps in priority order: start with the biggest bottleneck, work upstream to downstream.

**What v2.1 achieves:** End-to-end pipeline accuracy sufficient for a methods paper — quantitatively validated 3D midline reconstructions with demonstrated multi-fish tracking robustness, a reproducible evaluation benchmark, and documented quality metrics across the full pipeline.

Directional goals (targets set during requirements phase, not here):
- Segmentation quality meaningfully improved over IoU 0.623
- Regression suite runnable in CI on a synthetic reference dataset
- 3D reconstruction accuracy benchmarked and reported
- Curve optimizer evaluated as an alternative reconstruction backend

---

## Bottleneck Analysis

Each pipeline stage is assessed for current quality, upstream dependency, potential for independent improvement, and expected downstream impact. Ordered by impact on end-to-end reconstruction quality.

### Stage 2: Midline (Segmentation + Extraction) — PRIMARY BOTTLENECK

**Current quality:** U-Net IoU 0.623 on fish masks; downstream 2D midline quality directly limited by mask noise.

**Upstream dependency:** Only calibration data (camera intrinsics/extrinsics). Fully independent of other pipeline stages — can be improved in isolation.

**Nature of limitation:** The current U-Net (MobileNetV3-Small encoder, ~2.5M params, trained on 128x128 crops) was ported as-is from v1.0. The training data quantity and diversity, augmentation strategy, encoder capacity, and training resolution are all improvement levers that were not touched during the v2.0 refactor.

**Expected downstream impact of improvement:** Very high. Every downstream stage — midline extraction quality, cross-view association (which uses 2D centroids from masks), and 3D reconstruction — depends on mask quality. An improvement in segmentation IoU from 0.623 to a higher value will propagate improvements through all five stages. This is the single change with the largest leverage on end-to-end accuracy.

**Independent improvement potential:** High. U-Net improvement requires no changes to other stages. Better masks → better BFS midlines with no code changes to the midline extraction or reconstruction logic.

---

### Evaluation Infrastructure — MUST COME FIRST

**Current quality:** No quantitative evaluation framework exists. Accuracy is currently assessed qualitatively via visualization. Regression tests exist but skip in CI. No ground truth comparison for 3D reconstructions.

**Why this must precede accuracy work:** You cannot systematically improve what you cannot measure. Before investing in segmentation retraining or reconstruction improvements, establish: (a) a repeatable benchmark for each stage, (b) metrics that correspond to publication requirements, and (c) a CI-runnable test fixture so regressions are caught automatically. Without this, accuracy improvements cannot be validated objectively.

**Specific gaps:**
- No IoU measurement pipeline (can only retrain and test U-Net outside the main codebase)
- Regression suite skips without production video data — no way to verify pipeline numerics in CI
- No ground truth for 3D midlines — publication needs a way to validate reconstruction accuracy
- 1 known xfail (midline regression golden data needs `PosePipeline` regeneration)

---

### Stage 1: Detection — MODERATE BOTTLENECK

**Current quality:** YOLO detection is generally reliable, but detection misses or false positives propagate through the entire pipeline. In dense scenes (multiple fish, partial occlusion), detection quality degrades.

**Upstream dependency:** None — depends only on raw video frames.

**Nature of limitation:** YOLO was not retrained during v2.0 (port-behavior doctrine). Detection confidence threshold is fixed. No temporal consistency across frames. No handling of partially occluded fish.

**Expected downstream impact of improvement:** Moderate-to-high. A missed detection eliminates a fish entirely from a frame; a false positive introduces a ghost fish that corrupts association and tracking. Detection improvement before segmentation improvement only helps if segmentation then succeeds on the better detections — so sequencing matters.

**Independent improvement potential:** Moderate. Can be improved via fine-tuning YOLO on more diverse fish data or by adding temporal detection smoothing without touching other stages.

---

### Stage 3: Association — MODERATE BOTTLENECK (with efficiency concern)

**Current quality:** RANSAC centroid clustering is functionally correct but has known limitations with overlapping fish and dense scenes where centroids cluster closely. The Stage 4 coupling fix (Phase 20-04) improved architecture but did not improve the underlying RANSAC algorithm.

**Upstream dependency:** Depends on Stage 2 midlines and Stage 1 detections. Association accuracy is directly limited by mask quality — noisy 2D centroids degrade association decisions. Improving segmentation first will improve association without any association-specific changes.

**Nature of limitation:** O(N² C) per frame for N fish and C cameras. RANSAC-based approach is robust but slow and not designed for dense scenes. No learned cross-view correspondence — purely geometric.

**Expected downstream impact of improvement:** Moderate. Association errors cause tracking ID switches and incorrect reconstruction inputs. However, most association failures are recoverable across subsequent frames by the tracker's population lifecycle. Upstream improvements (better masks) are likely to yield more association accuracy gain than algorithmic changes to RANSAC itself.

**Independent improvement potential:** Moderate. A graph-based or learned association approach could improve both accuracy and efficiency, but requires careful integration with Stage 4's bundle consumption API.

---

### Stage 4: Tracking — LOWER BOTTLENECK

**Current quality:** Hungarian matching with population lifecycle (probationary, confirmed, coasting, dead ID recycling) is well-implemented and handles typical fish behaviors. The Phase 20-04 rewrite to consume Stage 3 bundles improved architectural cleanliness.

**Upstream dependency:** Depends on Stage 3 association bundles. Tracking continuity is fundamentally limited by association quality — if Stage 3 misassociates fish across cameras, Stage 4 sees incoherent bundle inputs.

**Nature of limitation:** Fixed `reprojection_threshold` with no online adaptation. No handling of extended occlusion (fish behind structure or other fish for >max_age frames). MOG2 backend exists but is untested in v2.0 context.

**Expected downstream impact of improvement:** Lower — tracking failures are typically localized (a fish disappears and re-appears with a new ID), whereas segmentation and association failures affect every frame.

**Independent improvement potential:** Moderate. Adaptive thresholds, re-identification, and re-entry logic are all well-scoped improvements. However, these are lower-priority until upstream quality is improved.

---

### Stage 5: Reconstruction — LOWER BOTTLENECK (with benchmark gap)

**Current quality:** RANSAC triangulation + B-spline fitting produces reasonable 3D midlines but has not been benchmarked quantitatively. The curve optimizer backend exists but its advantage over direct triangulation is not characterized.

**Upstream dependency:** Reconstruction is the final stage — all upstream quality issues accumulate here. Improving reconstruction without fixing upstream stages will have limited impact.

**Nature of limitation:** View-angle weighting uses a fixed formula without learned weights. B-spline uses hardcoded `SPLINE_K` and `SPLINE_KNOTS`. Curve optimizer warm-starts from triangulation (adding latency) but has not been benchmarked for accuracy gain. No 3D ground truth to evaluate against.

**Expected downstream impact of improvement:** Only affects output quality — no downstream stages. But this is the stage whose output appears in publication figures, so reconstruction quality is directly visible to reviewers.

**Independent improvement potential:** High for benchmarking (evaluate what we have). Moderate for algorithmic improvements (view-angle weighting, spline parameters). Dependent on evaluation infrastructure existing first.

---

## Candidate Requirements

Requirements are ordered by priority. Higher-priority requirements should be implemented first. Dependencies are noted where a requirement depends on another being complete first.

### Evaluation Infrastructure

---

**ID:** EVAL-01
**Title:** Make regression suite runnable in CI without production video data
**Priority:** Critical
**Rationale:** The 7 regression tests exist but skip in CI (and on any machine without `AQUAPOSE_VIDEO_DIR`). Without a CI-runnable numerical safety net, any future change to the pipeline risks breaking numerical equivalence undetected. This is prerequisite infrastructure for all accuracy improvement work.
**Phase suggestion:** Phase 1 (first plan)
**Depends on:** Nothing — can start immediately
**Details:** Provide a small synthetic dataset (generated from `SyntheticDataStage` or a committed fixture) that the regression suite can run against in CI. Fix the 1 known xfail by regenerating midline golden data with `PosePipeline`. Set up GitHub Actions to run `hatch run test` on push.

---

**ID:** EVAL-02
**Title:** Segmentation quality measurement pipeline
**Priority:** Critical
**Rationale:** U-Net IoU 0.623 is the known primary bottleneck but cannot be systematically improved without a repeatable evaluation pipeline. Need: IoU measurement against a held-out validation set, integrated into the CI/dev workflow so retrained models can be compared objectively.
**Phase suggestion:** Phase 1
**Depends on:** EVAL-01 (CI/CD infrastructure)
**Details:** Script or test to evaluate U-Net on the held-out validation set, reporting per-class and mean IoU. Should integrate with `hatch run` as a named command. Validation set should be committed as a small but representative fixture.

---

**ID:** EVAL-03
**Title:** 3D reconstruction accuracy benchmark
**Priority:** High
**Rationale:** Publication requires quantitative reconstruction accuracy claims. Without ground truth, accuracy must be established via synthetic experiments (known geometry) or inter-rater comparison. Establish the benchmark methodology before optimizing reconstruction.
**Phase suggestion:** Phase 2
**Depends on:** EVAL-01
**Details:** Define and implement a benchmark for 3D midline reconstruction accuracy. Synthetic approach: generate known 3D fish midlines, render through calibration, run reconstruction pipeline, compute Chamfer distance or point-to-curve error. Document the benchmark protocol for the methods section.

---

### Segmentation Improvements

---

**ID:** SEG-01
**Title:** Increase segmentation training data quality and diversity
**Priority:** Critical
**Rationale:** IoU 0.623 on the current model is substantially below what publication-quality masks require. The primary lever for improvement is data: more diverse frames, better coverage of edge cases (partial occlusion, fast motion, dense schooling), and improved annotation quality from the SAM2 pseudo-labeler.
**Phase suggestion:** Phase 2 (after evaluation infrastructure)
**Depends on:** EVAL-02 (measurement pipeline to validate improvement)
**Details:** Expand training data sampling to capture more diverse fish positions, lighting conditions, and inter-fish occlusion scenarios. Consider adding frames from multiple video clips. Re-evaluate SAM2 pseudo-label quality with the current crop-and-box approach and assess if mask quality is sufficient as training signal or if manual correction is needed for a validation subset.

---

**ID:** SEG-02
**Title:** Increase U-Net encoder capacity
**Priority:** High
**Rationale:** MobileNetV3-Small was chosen for speed; at IoU 0.623, speed is less important than quality. A larger encoder (MobileNetV3-Large, EfficientNet-B2, or ResNet-18) will extract richer features and likely improve IoU, especially for partial fish and boundary regions.
**Phase suggestion:** Phase 2
**Depends on:** EVAL-02, SEG-01 (improved data should be evaluated with a better encoder)
**Details:** Benchmark 2-3 encoder options against the improved dataset (SEG-01). Keep 128x128 crop resolution initially; evaluate 256x256 if encoder capacity improvement plateaus. Report IoU on held-out validation set for each configuration.

---

**ID:** SEG-03
**Title:** Segmentation augmentation strategy
**Priority:** High
**Rationale:** Underwater imagery has specific challenges (variable turbidity, caustic lighting, reflection artifacts) that benefit from domain-appropriate augmentation. Current augmentation strategy was not documented or assessed during v2.0 port.
**Phase suggestion:** Phase 2
**Depends on:** EVAL-02
**Details:** Audit current augmentation pipeline. Add underwater-specific augmentations: color jitter calibrated to typical turbidity/caustic ranges, random horizontal flip, brightness/contrast variation, synthetic occlusion (random erasing). Evaluate impact on validation IoU.

---

### Tracking and Association

---

**ID:** TRACK-01
**Title:** Characterize and document tracker sensitivity to association quality
**Priority:** High
**Rationale:** After Stage 3/4 coupling refactor (Phase 20-04), the tracker receives association bundles as input. The relationship between bundle quality (Stage 3) and tracking continuity (Stage 4) has not been characterized. Understanding this dependency is prerequisite to knowing whether to invest in better association algorithms.
**Phase suggestion:** Phase 2
**Depends on:** EVAL-01
**Details:** Run the pipeline on a representative video clip and log Stage 3/4 handoff statistics: bundle count per frame, unmatched bundles (new births), missed bundles (coasting events), ID switches. This diagnostic should inform whether Stage 3 algorithmic improvements are worth the effort.

---

**ID:** TRACK-02
**Title:** Evaluate and tune association RANSAC parameters
**Priority:** Medium
**Rationale:** The current RANSAC centroid clustering parameters were ported from v1.0 without systematic evaluation. Parameter sensitivity analysis may yield accuracy improvements at no algorithmic cost.
**Phase suggestion:** Phase 3
**Depends on:** TRACK-01 (characterization first)
**Details:** Systematic grid search over key RANSAC parameters (inlier threshold, min cameras, expected count) on the representative video clip. Report per-configuration tracking continuity metrics from TRACK-01 diagnostic.

---

### Reconstruction

---

**ID:** RECON-01
**Title:** Benchmark curve optimizer vs. direct triangulation
**Priority:** High
**Rationale:** The curve optimizer exists and passes smoke tests but its accuracy advantage over direct triangulation has never been measured. Publication results should use the better backend — or document why direct triangulation is preferred. This requires EVAL-03 to have a measurement protocol first.
**Phase suggestion:** Phase 3
**Depends on:** EVAL-03 (benchmark methodology)
**Details:** Run both backends on the synthetic benchmark from EVAL-03. Report accuracy (Chamfer/point-to-curve), runtime, and convergence behavior. Recommend a default backend for the production configuration based on accuracy/speed tradeoff.

---

**ID:** RECON-02
**Title:** Finalize and document HDF5 output schema
**Priority:** Medium
**Rationale:** The HDF5 schema is functional but undocumented. Downstream analysis scripts need a stable, versioned schema. Publication data must be interpretable without source code inspection.
**Phase suggestion:** Phase 2 (can be done independently, low effort)
**Depends on:** Nothing
**Details:** Document the current HDF5 schema (groups, datasets, dtypes, units, coordinate conventions). Add a schema version attribute. Write a schema validation utility. Commit as `docs/hdf5_schema.md` and inline docstring in `HDF5ExportObserver`.

---

### CI/CD and Process

---

**ID:** CI-01
**Title:** GitHub Actions CI pipeline
**Priority:** Critical
**Rationale:** 514 tests exist but never run automatically on push. Any change that breaks tests goes undetected until a developer runs the suite locally. CI is foundational infrastructure for all accuracy improvement work.
**Phase suggestion:** Phase 1
**Depends on:** Nothing
**Details:** Set up `.github/workflows/ci.yml` to run `hatch run test` on push to `main` and on pull requests. Include lint check (`hatch run lint`). Exclude `@slow` tests from CI by default — tag as `@slow` and filter with `-m "not slow"`. Ensure the workflow uses Python 3.10+ and installs hatch.

---

**ID:** CI-02
**Title:** Automated model evaluation in CI
**Priority:** Medium
**Rationale:** Once EVAL-02 provides a measurement pipeline, CI should verify that retrained models meet a minimum IoU threshold before they are accepted. This prevents quality regressions in the segmentation model.
**Phase suggestion:** Phase 3 (after segmentation improvements)
**Depends on:** EVAL-02, CI-01
**Details:** Add a CI step that runs `hatch run eval-seg` and asserts IoU above a configured threshold. Gate model acceptance on this check.

---

## Suggested Phase Structure

This is a rough ordering of candidate requirements into implementation phases — not a full roadmap. The `/gsd:new-milestone` discussion cycle will refine phase scope, plan count, and success criteria.

### Phase 1: Foundations for Accuracy Work (EVAL-01, CI-01, RECON-02)

Before any accuracy improvements can be validated, establish the evaluation and CI infrastructure. This phase has no accuracy-improvement deliverables — it enables accurate measurement of improvements in subsequent phases.

- EVAL-01: CI-runnable regression suite (regenerate midline golden data, add CI pipeline)
- CI-01: GitHub Actions for `hatch run test` on push
- RECON-02: Document and version HDF5 schema (low-effort, standalone)

Exit criteria: CI is green, regression suite runs without skipping, HDF5 schema is documented.

### Phase 2: Segmentation Improvement (EVAL-02, SEG-01, SEG-02, SEG-03, EVAL-03, TRACK-01)

The primary bottleneck phase. Improve segmentation from IoU 0.623 toward publication quality by expanding training data, adding augmentation, and evaluating larger encoders. Establish the 3D reconstruction benchmark and characterize Stage 3/4 coupling.

- EVAL-02: Segmentation measurement pipeline (run first, establishes baseline)
- SEG-01: Expand training data quality and diversity
- SEG-03: Underwater augmentation strategy
- SEG-02: Evaluate larger encoder options (run after SEG-01 + SEG-03)
- EVAL-03: 3D reconstruction accuracy benchmark (can run in parallel)
- TRACK-01: Stage 3/4 coupling characterization diagnostic (lightweight, run alongside)

Exit criteria: Segmentation IoU demonstrably improved, benchmark for 3D accuracy exists, Stage 3/4 coupling is characterized.

### Phase 3: Downstream Optimization (RECON-01, TRACK-02, CI-02)

With better upstream quality and measurement infrastructure, evaluate downstream improvement opportunities. Benchmark curve optimizer vs. direct triangulation; tune association parameters; add model quality gate to CI.

- RECON-01: Curve optimizer vs. triangulation benchmark
- TRACK-02: Association parameter evaluation (informed by TRACK-01 characterization)
- CI-02: Automated model evaluation gate in CI

Exit criteria: Best reconstruction backend identified and documented; association parameters tuned; CI enforces model quality.

---

## Out of Scope for v2.1

Keep the next milestone focused on accuracy and evaluation. The following should NOT be included:

- **Architectural changes** — v2.0 architecture is sound; no new stages, no protocol changes, no new module structure
- **New pipeline stages** — the 5-stage canonical model is correct; adding stages is out of scope
- **GUI or web interface** — not relevant for a methods paper
- **Real-time processing** — scientific computing use case; throughput matters but latency does not
- **Dataset collection** — extending the physical aquarium rig or camera count is hardware work, not software
- **Deployment/packaging** — PyPI packaging, Docker images, or distribution infrastructure
- **New output formats** — HDF5 is sufficient; new export formats (e.g., CSV, JSON) are not publication requirements
- **Multi-species generalization** — out of scope for a single-species methods paper
- **MOG2 backend validation** — the MOG2 detector backend is available but untested in v2.0; testing it is not a publication requirement

---

## How to Use This Document in `/gsd:new-milestone`

This prospective is structured as a requirements seed. When starting the next milestone:

1. **Use the Candidate Requirements section** as the starting point for REQUIREMENTS.md. Each entry has an ID, title, priority, and rationale that maps directly to a requirement definition.

2. **Use the Bottleneck Analysis section** to guide the discussion about sequencing. The ordering argument (Evaluation → Segmentation → Downstream) is documented there with rationale.

3. **Use the Suggested Phase Structure section** as the starting point for the roadmap. Phase 1 maps to foundational requirements; Phase 2 maps to the primary bottleneck; Phase 3 maps to downstream optimization.

4. **The Out of Scope section** provides boundary conditions to enforce during the discuss-phase step when scope creep is likely.

5. **Key numeric context:** U-Net IoU 0.623 is the baseline; publication quality is the target; any requirement that does not contribute to measurable accuracy improvement is low-priority or out of scope.

---

*v2.1 prospective document complete.*
*Generated: 2026-02-27*
*Phase: 21-retrospective-prospective, Plan 02*
