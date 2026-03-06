# Feature Landscape

**Domain:** Model iteration evaluation and QA for multi-view fish pose estimation (AquaPose v3.6)
**Researched:** 2026-03-06
**Confidence:** HIGH -- features derived from direct codebase analysis, milestone seed requirements, and domain knowledge of multi-view pose estimation pipelines.

---

## Context: What Already Exists

| Existing Component | What It Does | Relevance to v3.6 |
|-------------------|--------------|-------------------|
| `ReconstructionMetrics` | mean/max reprojection error, per-camera/per-fish breakdown, inlier ratio, low-confidence flag rate, tier2 stability | Needs percentile extension and per-keypoint breakdown |
| `AssociationMetrics` | fish yield ratio, singleton rate, camera distribution histogram | Needs camera count percentile summaries |
| `MidlineMetrics` | mean/std confidence, completeness, temporal smoothness | Needs confidence percentile extension |
| `TrackingMetrics` | track count, length stats (median/mean/min/max), coast frequency | Already captures 2D tracking fragmentation; 3D track fragmentation is a separate concern |
| `EvalRunner` | Loads merged chunk contexts, calls per-stage evaluators, produces `EvalRunnerResult` | Extension point for new analysis functions |
| `format_eval_report` / `format_eval_json` | ASCII and JSON output for `aquapose eval` | Must be extended to render new metrics |
| `Midline3D` | per_camera_residuals (mean per camera), mean_residual, max_residual, control_points, knots | Has per-camera residuals but NOT per-body-point residuals -- per-keypoint breakdown requires new data |
| `_TriangulationResult` (internal) | per-body-point `mean_residuals` shape (N,) and `inlier_cam_ids` | Per-keypoint data exists internally but is NOT surfaced into Midline3D |
| `SampleStore` (SQLite) | content-hash dedup, source priority upsert, provenance tracking, symlink assembly, model lineage | Bootstrap workflow feeds data through this store |
| `coco_convert.py` | COCO-JSON to YOLO-OBB and YOLO-pose format conversion | Exists but may need `data convert` CLI subcommand wiring |
| `data_cli.py` | `data import`, `data assemble` subcommands | Needs `data convert` subcommand to wire coco_convert |
| `train compare` | Cross-run comparison table (mAP50, mAP50-95, Prec, Recall) | Sufficient for training metric comparison |
| `pseudo_labels.py` / `pseudo_label_cli.py` | Source A/B pseudo-label generation with confidence scoring | Already built; v3.6 executes the loop |
| `elastic_deform.py` | TPS-based C-curve/S-curve augmentation | Already built; used during training |
| `ZDenoisingMetrics` | z-range, z-profile RMS, per-fish SNR | Already covers z-quality assessment |

---

## Table Stakes

Features users expect from a "model iteration and QA" milestone. Missing these means the iteration loop cannot be properly evaluated.

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|-------------|
| **Reprojection error percentiles (p50, p90, p95)** | Mean/max alone hide distributional shape. A mean of 3px with p95 of 20px is very different from a mean of 3px with p95 of 5px. Percentiles are the standard way to report error distributions in pose estimation. | LOW | Extends `ReconstructionMetrics`, uses existing `per_camera_residuals` data in `Midline3D` |
| **Midline confidence percentiles (p10, p50, p90)** | Same distributional reasoning. Low p10 reveals a tail of low-confidence predictions that mean/std obscure. | LOW | Extends `MidlineMetrics`, uses existing `point_confidence` arrays |
| **Camera count percentiles for association** | The camera_distribution histogram is hard to scan quickly. p50 and p90 of cameras-per-observation summarize "typical coverage" and "best coverage" in two numbers. | LOW | Extends `AssociationMetrics`, computed from existing `camera_distribution` dict |
| **Per-keypoint reprojection error breakdown** | Head vs tail error profiles reveal systematic model failures. If tail keypoints consistently have 3x the error of head keypoints, the pose model needs targeted improvement. This is standard practice in keypoint-based pose evaluation (COCO OKS reports per-keypoint). | MEDIUM | Requires surfacing `_TriangulationResult.mean_residuals` (per-body-point) into `Midline3D`. Currently discarded after spline fitting. New field: `per_point_residuals: np.ndarray | None` on `Midline3D`. |
| **COCO-JSON to store bootstrap workflow** | The iteration loop cannot start without importing manual annotations into the store. This is the entry point for the entire cycle. Existing `coco_convert.py` functions exist but lack a CLI subcommand. | MEDIUM | Wires `coco_convert.py` into `data convert` CLI subcommand. Needs OBB and pose output paths, then `data import` ingests them. |
| **Baseline model training and registration** | After importing manual annotations, the user must train a baseline model and register it in the store. This establishes "round 0" provenance. The `train` and `data` CLIs exist but the end-to-end workflow needs validation. | LOW | Uses existing `aquapose train {obb, pose}` + store model registration. May need minor CLI ergonomics fixes. |
| **Round-over-round eval comparison** | The whole point of iteration is measuring improvement. The user needs to compare `aquapose eval` results across rounds side-by-side. At minimum: a table showing round 0 vs round 1 (vs round 2) metrics. | MEDIUM | New analysis function or CLI option. Could be as simple as `aquapose eval --compare run_A run_B` producing a diff table. Uses existing `EvalRunnerResult.to_dict()` for both runs. |

---

## Differentiators

Features that add real value but are not strictly required to declare the milestone successful. They make the iteration loop more informative and the results more convincing.

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|-------------|
| **Curvature-stratified reconstruction quality** | The elastic augmentation experiment (v3.5) showed curvature bias in OKS. The same bias likely manifests in 3D reconstruction error: curved fish may have higher reprojection error because the pose model produces worse keypoints for them. Binning reconstructions by 3D spline curvature and reporting error per bin directly measures whether the iteration loop is fixing the curvature bias. | MEDIUM | Compute curvature from `Midline3D.control_points` (evaluate 2nd derivative of B-spline). Bin into quantiles (e.g., quartiles). Report mean/p90 reprojection error per bin. New analysis function, not part of core evaluator. |
| **3D track fragmentation analysis** | After association and reconstruction, "how continuous are the 3D trajectories?" matters for downstream behavioral analysis. Fragmentation = gaps where a fish was visible in >=3 cameras but was not reconstructed (failed association, failed midline, or failed triangulation). High fragmentation means the models or pipeline params need work. | HIGH | Requires cross-referencing: (1) per-frame camera visibility from tracklet_groups, (2) per-frame reconstruction presence from midlines_3d. Must compute: gap count, gap duration distribution, continuity ratio (frames reconstructed / frames with >= 3 camera visibility). New analysis function. |
| **Detection false positive rate estimation** | Algae false positives are the primary detection problem. Estimating FP rate without ground truth requires a proxy: detections that never get associated into multi-view groups (singletons that are not near any fish in other cameras) are likely false positives. | MEDIUM | Analyze singleton observations in association output. Cross-reference with detection confidence. Report estimated FP rate as (low-confidence singletons) / total detections. Heuristic, not ground truth, but actionable. |
| **Per-round provenance summary** | When the iteration loop is complete, a summary showing "round 0: N manual samples, round 1: N manual + M pseudo, round 2: ..." with model lineage makes the result reproducible and publishable. | LOW | Query SQLite store for datasets and models. Format as summary table. Could be `aquapose data summary` subcommand. |
| **Overlay video side-by-side comparison** | The seed doc calls for "side-by-side comparison: baseline overlay vs final overlay on same time segment." This is the visual proof that models improved. | MEDIUM | Stitch two overlay videos horizontally with matching frame numbers. Uses existing `aquapose viz overlay` for each run, then ffmpeg or OpenCV to compose. |

---

## Anti-Features

Features to explicitly NOT build in v3.6.

| Anti-Feature | Why It Might Be Requested | Why Avoid | What to Do Instead |
|--------------|--------------------------|-----------|-------------------|
| **Automated iteration loop** | "Just run `aquapose iterate --rounds 3` and walk away" | The milestone explicitly uses "guided manual execution" with decision checkpoints between rounds. Automating removes the human judgment about whether to proceed, adjust confidence thresholds, or investigate failures. The loop is only 1-2 rounds -- automation overhead exceeds benefit. | Keep the step-by-step CLI workflow. Document the sequence clearly. |
| **Ground-truth evaluation against manual annotations** | Standard ML practice: hold out annotated frames and measure precision/recall | Only ~50 annotated full-frame images exist. Holding out enough for statistically meaningful evaluation would starve training. Pipeline-level metrics (reprojection error, singleton rate, track continuity) are the primary quality signal. | Use pipeline metrics as the evaluation signal. Defer ground-truth eval to v3.7 when more annotations exist. |
| **Active learning for frame selection** | "Select the frames where the model is most uncertain for the next round of annotation" | The pseudo-label confidence scoring already identifies uncertain regions. Active learning adds model-in-the-loop complexity (uncertainty estimation, acquisition function) for marginal gain over confidence-threshold filtering. The manual annotation budget is fixed at ~50 images -- not enough to warrant active learning infrastructure. | Use pseudo-label confidence thresholds and curvature-diversity sampling (already built). |
| **Segmentation model iteration** | "While we're iterating OBB and pose, do seg too" | Cross-camera orientation problem is unresolved for segmentation (explicitly out of scope in seed doc). Including seg would block the milestone on an unsolved upstream problem. | Defer to future milestone after orientation problem is addressed. |
| **Hyperparameter search for training** | "Grid search over learning rate, augmentation params, etc." | Ultralytics defaults are well-tuned. The bottleneck is data quality and quantity, not hyperparameters. One round of training with defaults + elastic augmentation is the right first step. Hyperparameter search adds complexity without addressing the actual bottleneck. | Use Ultralytics defaults. Adjust only if training metrics plateau unexpectedly. |
| **Real-time eval dashboard** | "Plotly/Streamlit dashboard showing metrics updating live during training" | Training runs are 1-2 hours. The existing `train compare` table and `aquapose eval` text report are sufficient for this cadence. A dashboard is engineering effort that does not improve decision quality for 1-2 iteration rounds. | Use `aquapose eval` text/JSON output and `train compare` tables. |

---

## Feature Dependencies

```
[COCO-JSON to store bootstrap] (Phase 71 prerequisite)
    |-- data convert CLI subcommand (wires coco_convert.py)
    |-- data import (existing)
    |-- data assemble (existing)
    |-- aquapose train (existing)
    v
[Baseline model training & registration] (Phase 71)
    |
    v
[Baseline pipeline run] (Phase 72)
    |-- aquapose run (existing)
    |-- aquapose eval with extended metrics (Phase 70)
    |-- aquapose viz overlay (existing)
    v
[Pseudo-label generation] (Phase 73)
    |-- pseudo-label generate (existing)
    |-- data import with source=pseudo (existing)
    |-- data assemble with mixed sources (existing)
    v
[Round 1 training & eval] (Phase 73-74)
    |-- train compare (existing)
    |-- aquapose eval --compare (NEW: round-over-round comparison)
    v
[Final validation] (Phase 76)
    |-- overlay video side-by-side (DIFFERENTIATOR)
    |-- per-round provenance summary (DIFFERENTIATOR)


[Extended metrics] (Phase 70 -- independent, should be done first)
    |-- Reprojection error percentiles -----> extends ReconstructionMetrics
    |-- Midline confidence percentiles -----> extends MidlineMetrics
    |-- Camera count percentiles -----------> extends AssociationMetrics
    |-- Per-keypoint reprojection error ----> requires Midline3D field addition
    |                                         requires DltBackend to surface per-point residuals
    |-- Curvature-stratified quality -------> requires 3D spline curvature computation
    |                                         (standalone analysis function)
    |-- 3D track fragmentation -------------> requires cross-referencing tracklet_groups
                                              and midlines_3d (standalone analysis function)
```

### Dependency Notes

- **Per-keypoint reprojection error is the only metric that requires a core type change.** The `_TriangulationResult.mean_residuals` array (per-body-point) is computed inside `DltBackend` but discarded when building `Midline3D`. Surfacing it requires adding a `per_point_residuals: np.ndarray | None` field to `Midline3D` and populating it from `_TriangulationResult`. This is a small change but touches core types, so it should be done carefully.

- **All other metric extensions operate on data already present** in `Midline3D`, `MidlineSet`, and `PipelineContext`. They are pure additions to evaluator functions and metric dataclasses.

- **The bootstrap workflow (Phase 71) is the critical-path dependency** for the iteration loop. Without it, no round 0 models exist in the store, and the loop cannot start. Phase 70 metrics can be developed in parallel with Phase 71 since they operate on existing cached data.

- **Round-over-round comparison** is needed by Phase 74 (Round 1 Evaluation) but could be as simple as running `aquapose eval` twice and diffing the JSON outputs manually. A dedicated comparison command is a quality-of-life improvement, not a hard blocker.

---

## MVP Recommendation

### Phase 70 (Metrics) -- build first, enables consistent measurement across all rounds:

1. **Reprojection error percentiles** -- LOW complexity, HIGH value. Add p50/p90/p95 to `ReconstructionMetrics`. Two lines of numpy.
2. **Midline confidence percentiles** -- LOW complexity, MEDIUM value. Add p10/p50/p90 to `MidlineMetrics`.
3. **Camera count percentiles** -- LOW complexity, MEDIUM value. Add p50/p90 to `AssociationMetrics`.
4. **Per-keypoint reprojection error** -- MEDIUM complexity, HIGH value. Surface `per_point_residuals` from DltBackend into Midline3D. Add per-keypoint mean/p90 to `ReconstructionMetrics` or a new analysis section.

### Phase 71 (Bootstrap) -- build second, unlocks the iteration loop:

5. **`data convert` CLI subcommand** -- MEDIUM complexity, CRITICAL for loop start. Wires existing `coco_convert.py` functions.
6. **End-to-end bootstrap validation** -- LOW complexity. Run the convert-import-assemble-train-register sequence and verify it works.

### Defer to after Phase 74 evaluation (include only if time permits):

7. **Curvature-stratified reconstruction quality** -- MEDIUM complexity, HIGH diagnostic value for confirming augmentation benefit at the 3D level.
8. **3D track fragmentation** -- HIGH complexity, MEDIUM value. Important for downstream behavioral analysis but not critical for the iteration loop decision (singleton rate and reprojection error already capture most of the signal).
9. **Round-over-round eval comparison command** -- MEDIUM complexity, quality-of-life. Can be done manually with JSON diff until this is built.

### Defer to future milestone:

10. **Detection false positive rate estimation** -- interesting but heuristic. The overlay videos provide visual FP assessment that is more trustworthy.
11. **Overlay video side-by-side** -- nice for presentation but can be done with ffmpeg manually.

---

## Sources

- Direct codebase inspection of `evaluation/stages/*.py`, `evaluation/runner.py`, `evaluation/output.py`, `evaluation/metrics.py` (HIGH confidence)
- Direct inspection of `core/reconstruction/backends/dlt.py` -- confirmed `_TriangulationResult` contains per-body-point residuals that are not surfaced (HIGH confidence)
- Direct inspection of `core/types/reconstruction.py` -- confirmed `Midline3D` schema and available fields (HIGH confidence)
- Direct inspection of `training/store_schema.py`, `training/data_cli.py`, `training/coco_convert.py`, `training/compare.py` (HIGH confidence)
- `.planning/milestones/v3.6-SEED.md` -- milestone scope, phase structure, validation approach (HIGH confidence)
- `.planning/PROJECT.md` -- existing capabilities inventory (HIGH confidence)
- COCO keypoint evaluation protocol (per-keypoint OKS) as domain standard for per-keypoint breakdown (MEDIUM confidence -- training data, not verified against current docs)
- Standard practice in multi-view pose estimation: percentile error reporting, curvature-stratified analysis, track fragmentation metrics (MEDIUM confidence -- domain knowledge)

---

*Feature research for: AquaPose v3.6 Model Iteration & QA*
*Researched: 2026-03-06*
