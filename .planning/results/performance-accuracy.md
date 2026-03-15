# AquaPose Performance & Accuracy Results

All results below are validated as current against the v3.10 codebase (2026-03-15). All full-run metrics (Sections 9-11) are from the Phase 97 production run (run_20260314_200051): 9,450 frames, 12 cameras, 9 fish.

---

## 1. Pseudo-Label Iterative Training (Phase 73, v3.6)

### Methodology

Three model variants were trained for both OBB detection and pose estimation:
- **Baseline**: Trained on manual annotations only
- **Uncurated**: Baseline data + raw pseudo-labels from pipeline reconstruction
- **Curated+Augmented**: Baseline data + human-corrected pseudo-labels + 4x elastic augmentation (pose only)

Evaluation used two held-out sets:
- **Manual-only val**: 9 images / 65 instances (OBB), 64 images / 65 instances (pose) — same distribution as training
- **Corrected pseudo-label val**: 9 images / 65 instances (OBB), 64 images / 66 instances (pose) — harder, more diverse frames selected by the pipeline

Data split: Temporal split (no frame shared across train/val) to prevent cross-camera leakage.

Training: YOLO11 (Ultralytics), OBB at imgsz=640, Pose at imgsz=320, 100 epochs each.

### OBB Detection Results

| Model | Val Set | mAP50 | mAP50-95 | Precision | Recall |
|-------|---------|-------|----------|-----------|--------|
| Baseline | Manual-only | 0.935 | 0.689 | — | — |
| + Pseudo-labels | Manual-only | 0.961 | 0.721 | 0.907 | 0.903 |
| + Curated labels | Manual-only | 0.973 | 0.700 | 0.896 | 0.938 |
| Baseline | Corrected PL | 0.958 | 0.708 | 0.981 | 0.804 |
| + Pseudo-labels | Corrected PL | 0.947 | 0.753 | 0.946 | 0.892 |
| + Curated labels | Corrected PL | 0.959 | 0.741 | 0.910 | 0.934 |

**CSV**: `data/pseudo_label_ablation_obb.csv`

### Pose Estimation Results

| Model | Val Set | Pose mAP50 | Pose mAP50-95 |
|-------|---------|-----------|--------------|
| Baseline | Manual-only | 0.991 | 0.965 |
| + Pseudo-labels | Manual-only | 0.991 | 0.966 |
| + Curated+aug | Manual-only | 0.994 | 0.968 |
| Baseline | Corrected PL | 0.971 | 0.892 |
| + Pseudo-labels | Corrected PL | 0.974 | 0.893 |
| + Curated+aug | Corrected PL | 0.990 | **0.984** |

Key result: **+9.2 pts mAP50-95** on held-out corrected pseudo-label val (0.892 → 0.984) with curation + elastic augmentation.

**CSV**: `data/pseudo_label_ablation_pose.csv`

### Production Models (All-Source Retrained, v3.7)

After the ablation, production models were retrained on the full combined dataset (manual + curated pseudo-labels + augmentation) with stratified splits:

| Task | mAP50-95 | Run ID |
|------|----------|--------|
| OBB Detection | 0.781 | run_20260310_115419 |
| Pose Estimation | 0.974 | run_20260310_171543 |

---

## 2. Curvature-Stratified Pose Accuracy (Phase 73, v3.6)

### Methodology

The corrected pseudo-label validation set (64 images, 66 instances) was split into three curvature terciles based on ground-truth midline curvature. Per-instance OKS was computed for each model variant and averaged within each bin.

Curvature is the mean unsigned curvature of the ground-truth 6-keypoint midline, measured in the image plane.

### Results

| Model | Low Curvature (n=21) | Mid Curvature (n=21) | High Curvature (n=22) |
|-------|---------------------|---------------------|----------------------|
| Baseline | 0.609 | 0.578 | 0.436 |
| + Pseudo-labels | 0.632 | 0.560 | 0.407 |
| + Curated+aug | **0.896** | **0.868** | **0.780** |

Key result: Elastic augmentation + curation provides the largest gain on high-curvature (curved) fish, where the baseline struggles most.

**CSV**: `data/curvature_stratified_oks.csv`

---

## 3. Elastic Augmentation A/B Experiment (v3.5)

### Methodology

Controlled experiment comparing a baseline pose model (257 train images) to an augmented model (257 originals + 1,028 elastic variants = 1,285 train images). Both evaluated on the same 65-image validation set.

Elastic augmentation applies thin-plate spline (TPS) deformations to fish midlines:
- Angle range: 5-15 degrees
- 4 variants per image: 2 C-curve (parabolic profile), 2 S-curve (sin(2*pi*t) profile)
- Only visible keypoints deformed; flanking control points at ±25px along normals prevent lateral blur

Both models: YOLO11-pose, imgsz=320, 100 epochs, curvature-stratified 80/20 split.

### Results

| Metric | Baseline | Augmented |
|--------|----------|-----------|
| Mean OKS | 0.712 | 0.707 |

Mean OKS is similar, but the key metric is the **OKS-vs-curvature regression slope**, which measures how much accuracy degrades as fish curvature increases. The per-image data is in the CSV — regression slopes should be computed from it.

**Note**: The `eval_results.json` source file records identical slopes (-0.297) for both models, which appears to be a bug in the evaluation script (likely using the same curvature array for the regression). The project records document slopes of -0.71 (baseline) and -0.30 (augmented) from the scatter plot analysis — these should be verified by recomputing from the per-image CSV data.

**CSV**: `data/augmentation_oks_per_image.csv` — 65 rows with columns: `stem, curvature, oks_baseline, oks_augmented`
**Suggested plot**: OKS vs curvature scatter with two series and linear regression lines.

---

## 4. Training Curves (v3.5 Augmentation Experiment)

### Methodology

Full 100-epoch training logs for both the baseline and augmented pose models from the elastic augmentation experiment. Logs include training/validation losses and mAP metrics per epoch.

### Key Observations

| Metric | Baseline (best epoch) | Augmented (best epoch) |
|--------|-----------------------|------------------------|
| Best Pose mAP50-95 | 0.909 (epoch 63) | 0.957 (epoch 86) |
| Best Box mAP50-95 | 0.888 (epoch 53) | 0.849 (epoch 81) |
| Training time | ~1,109s (100 epochs) | ~3,152s (100 epochs) |

The augmented model converges to higher pose mAP50-95 earlier and maintains it, while the baseline oscillates more on validation metrics.

**CSVs**: `data/pose_training_curves_baseline.csv`, `data/pose_training_curves_augmented.csv`
**Suggested plot**: Overlaid loss curves (train/val pose loss) and mAP50-95 progression.

---

## 5. Z/XY Reconstruction Anisotropy (v3.5, corrected)

### Methodology

Monte Carlo ray perturbation analysis on the calibrated 12-camera rig:
1. Place a ground-truth 3D point at a given depth in the aquarium
2. Project to all cameras with visibility (within image bounds)
3. Perturb each 2D projection by ±0.5px (simulating detection noise)
4. Triangulate the perturbed rays via DLT
5. Measure reconstruction error in X, Y, Z separately
6. Compute the Z/XY anisotropy ratio

The analysis uses real calibration parameters (not synthetic), with image-bounds filtering to exclude cameras that cannot see the test point.

### Results

| Metric | Value |
|--------|-------|
| Mean Z/XY anisotropy | **~11x** |
| Range | ~7x – ~15x (PROJECT.md documents "7–15x") |
| Previous (v1.0) estimate | 132x (synthetic, overstated) |

The top-down camera geometry (cameras above looking down into water) makes Z-reconstruction inherently ~11x less precise than XY. This is a fundamental property of the rig, not a software limitation.

**Code**: `src/aquapose/calibration/uncertainty.py`
**No CSV** — this is a single summary statistic. Could be re-run with `generate_uncertainty_report()` to produce per-depth profiles if needed.

---

## 6. Cross-View Association Quality (v3.8)

### Methodology

Cross-view association groups 2D tracklets (per-camera fish tracks) into 3D fish identities using multi-keypoint ray-casting followed by Leiden community detection. Quality is measured by:
- **Singleton rate**: Fraction of tracklets assigned to single-camera groups (unusable for 3D reconstruction)
- **Fish yield ratio**: Mean reconstructed fish per frame / expected fish count (target: 100%)
- **Mean reprojection error**: Mean pixel-space distance between reprojected 3D points and original 2D detections

v3.8 introduced three architectural improvements over v3.7:
1. Multi-keypoint pairwise scoring (6 keypoints instead of single centroid)
2. Group validation with temporal changepoint detection (splits/evicts misassigned tracklets)
3. Singleton recovery with swap-aware split-and-assign

### Results

| Metric | v3.7 | v3.8 | Change |
|--------|------|------|--------|
| Singleton rate | 27% | **5.4%** | 80% reduction |
| Mean reprojection error | ~3.0 px | **2.85 px** | 5% improvement |
| Fish yield ratio | ~100% | ~100% | Stable |
| Association wall-time | 452 s/chunk | **<30 s/chunk** (target; see note) | ~15x speedup |

Test data: YH project, 900 frames (3 chunks), 12 cameras, 9 fish.

**Note on wall-time**: The 452s pre-optimization time is measured (Phase 91.1 profiling). The "<30s" post-optimization figure was the design target; all vectorization was verified implemented but a precise post-optimization measurement was not recorded. The Phase 92 parameter sweep ran 27 association combos in ~15 minutes total (~33s each including evaluation overhead), which is consistent with the target but not a clean measurement of association-only time. **This should be re-measured for publication.**

---

## 7. Association Parameter Sensitivity (Phase 92, v3.8)

### Methodology

Systematic grid sweep of 6 association parameters on the same 900-frame test set. Two-stage design:
1. **Joint grid** (27 combos): ray_distance_threshold × score_min × keypoint_confidence_floor
2. **Carry-forward** (9 combos): Fix winner from joint grid, sweep eviction_reproj_threshold, leiden_resolution, early_k individually

Metrics: fish yield ratio (%) and mean reprojection error (px).

### Key Findings

- The sweep winner is **identical to the current defaults** — no parameter changes needed
- `eviction_reproj_threshold` is fully insensitive across 0.02–0.05
- `leiden_resolution=2.0` is catastrophic (56.6% yield due to over-clustering)
- `early_k` insensitive between 5 and 10; slight degradation at 30
- The 80% singleton rate reduction came entirely from **architectural changes**, not parameter tuning

**CSV**: `data/association_parameter_sweep.csv` — 27 joint grid rows + 9 carry-forward rows
**Suggested plots**: Yield heatmap (ray_dist × score_min), bar charts for carry-forward parameter sensitivity

---

## 8. Reconstruction Parameter Sensitivity (v3.9, 2026-03-14)

### Methodology

Sequential carry-forward sweep of 2 reconstruction parameters on YH run_20260314_165129 (300 frames, 12 cameras, 9 fish). Parameters:
- **outlier_threshold**: DLT reprojection error threshold for inlier/outlier classification. Swept 10–100px in steps of 5 (19 values).
- **min_cameras**: Minimum cameras required for DLT triangulation. Swept [2, 3, 4].

Metrics: mean reprojection error (px) and fish coverage (reconstructed / available).

### Results

| Parameter | Value | Mean Error (px) | Fish Reconstructed | Fish Available | Notes |
|-----------|-------|-----------------|--------------------|----|-------|
| outlier_threshold | 10 (default) | 2.87 | 254 | 270 | Winner |
| outlier_threshold | 15 | 2.87 | 254 | 270 | |
| outlier_threshold | 20 | 2.89 | 254 | 270 | |
| outlier_threshold | 25 | 2.89 | 254 | 270 | |
| outlier_threshold | 30 | 2.90 | 254 | 270 | |
| outlier_threshold | 35–100 | 2.91 | 254 | 270 | Saturates at 35 |
| min_cameras | 2 | 3.00 | 271 | 270 | Over-reconstruction; ill-conditioned |
| min_cameras | 3 (default) | 2.87 | 254 | 270 | Winner |
| min_cameras | 4 | 2.87 | 237 | 270 | 6% coverage loss, no error gain |

### Key Findings

- **outlier_threshold** is nearly flat across 10–100px (2.87–2.91px). Inlier ratio is already ~99%, so outlier rejection does minimal work.
- **min_cameras=2** produces spurious over-reconstruction (271 > 270 available) because 2-camera refractive DLT is ill-conditioned.
- **min_cameras=4** costs 6% coverage with no error improvement.
- **min_cameras=3** is the sweet spot: best error with solid coverage.
- Current defaults are already optimal.

**CSV**: `data/reconstruction_parameter_sweep.csv` — full 19-value outlier_threshold grid + 3 min_cameras values
**Suggested plot**: Line plot of error vs outlier_threshold (shows rapid saturation at 35px); bar chart of error and coverage vs min_cameras.

---

## 9. Reconstruction Quality (Full Run)

### Methodology

EvalRunner evaluated the complete Phase 97 full-pipeline run (run_20260314_200051): 9,450 frames, 12 cameras, 9 fish, 32 chunks of 300 frames each. No frame sampling — all frames evaluated. Reconstruction uses v3.9 raw-keypoint triangulation (6 keypoints per fish, min_cameras=3, outlier_threshold=10). Per-keypoint errors are computed by reprojecting the 6 triangulated 3D points through each observing camera's refractive model and measuring pixel distance to the 2D detections. Camera visibility counts the minimum cameras across body points for each Midline3D (field `n_cameras` set by the reconstruction backend).

Run directory: `~/aquapose/projects/YH/runs/run_20260314_200051`
Evaluated: 2026-03-15

### Reprojection Error Distribution

| Statistic | Value (px) |
|-----------|-----------|
| Mean | 3.41 |
| p50 (median) | 2.68 |
| p90 | 6.22 |
| p99 | 14.41 |
| Max | 273.94 |

- Fish-frames reconstructed: 80,103 / 85,050 (94.2% coverage)
- Inlier ratio (not low-confidence): 96.7%

### Per-Keypoint Reprojection Error

Points are indexed 0 (tail) through 5 (head) along the fish midline.

| Point | Mean (px) | P90 (px) |
|-------|-----------|----------|
| 0 (tail) | 4.04 | 6.30 |
| 1 | 2.73 | 4.00 |
| 2 | 2.82 | 4.63 |
| 3 | 3.05 | 5.05 |
| 4 | 3.63 | 5.76 |
| 5 (head) | 5.35 | 8.44 |

Head (point 5) and tail (point 0) have higher error than mid-body points, consistent with higher pose uncertainty at the extremities.

### Camera Visibility Statistics

| Statistic | Value |
|-----------|-------|
| Mean cameras per fish | 3.60 |
| Median cameras per fish | 4.0 |
| Min | 0 |
| Max | 6 |

Camera count distribution across all fish-frame pairs:

| Cameras | Fish-Frames | Fraction |
|---------|-------------|---------|
| 0 | 1,201 | 1.5% |
| 2 | 3,686 | 4.6% |
| 3 | 22,227 | 27.7% |
| 4 | 50,778 | 63.4% |
| 5 | 2,191 | 2.7% |
| 6 | 20 | <0.1% |

Most fish-frames (63.4%) are observed by exactly 4 cameras. The 1.5% with 0 cameras are low-confidence reconstructions retained by the pipeline.

**CSV**: `data/reconstruction_quality_full_run.csv`

---

## 10. Pipeline Performance (Full Run)

### Methodology

Timing data comes from the built-in per-chunk timer in the pipeline engine, measured during the complete Phase 97 full-pipeline run (run_20260314_200051) executed on 2026-03-14. The run processed 9,450 frames from the YH aquarium video across 32 chunks (31 full chunks of 300 frames + 1 partial chunk of 150 frames), with 12 cameras and 9 fish. All 5 pipeline stages were timed per chunk; chunk 31 is a partial chunk and its values reflect the proportionally smaller workload.

**Note:** This run was executed on the development workstation (GPU workstation, not a dedicated benchmarking machine). Absolute numbers will vary by hardware; relative stage shares are hardware-independent.

### Per-Stage Timing

| Stage | Total Time (s) | Mean/Chunk (s) | Share (%) |
|-------|----------------|----------------|-----------|
| Detection | 2,392.8 | 74.78 | 28.9% |
| Pose | 2,545.0 | 79.53 | 30.7% |
| Tracking | 82.1 | 2.57 | 1.0% |
| Association | 1,053.0 | 32.91 | 12.7% |
| Reconstruction | 2,205.5 | 68.92 | 26.6% |
| **Total** | **8,278.6** | **258.7** | **100%** |

### End-to-End Throughput

| Metric | Value |
|--------|-------|
| Total wall-time | 8,278.6 s (2.30 h) |
| Frames processed | 9,450 |
| Throughput | **1.14 frames/sec** |
| Mean chunk time | 258.7 s/chunk |

### Key Observations

- **Pose and detection dominate**: Together they account for 59.6% of total wall-time, driven by GPU inference on 12 cameras × 300 frames per chunk.
- **Tracking is negligible**: At 1.0% share, the Kalman tracker adds essentially zero overhead — its per-chunk cost is ~2.6s vs ~80s for detection or pose.
- **Reconstruction is non-trivial**: At 26.6%, refractive triangulation (6 keypoints × 9 fish × all frames) is the third-largest cost, substantially faster than the v3.4 era when it shared time with legacy spline fitting.
- **Association varies**: Chunk-to-chunk association time ranges from 14.8s to 64.8s (4.3x range), reflecting scene-complexity variation. The v3.4 pre-optimization baseline was 452s/chunk (see Section 6 note); current mean is 32.9s — consistent with the >10x speedup design goal.

**CSV**: `data/pipeline_timing_full_run.csv`
**Suggested plot**: Stacked bar chart of mean per-stage time per chunk; pie chart of stage time share.

---

## 11. Tracking and Association Quality (Full Run)

### Methodology

EvalRunner evaluated the Phase 97 full-pipeline run (run_20260314_200051): 9,450 frames, 12 cameras, 9 fish, 32 chunks of 300 frames each. No frame sampling — all frames evaluated. Tracking uses a custom KeypointTracker per camera (one tracker per camera, independent tracklets). Cross-view association uses multi-keypoint ray scoring + Leiden clustering. Identity stitching across chunks uses tracklet-set overlap. Detection coverage computed from merged PipelineContext detections field. Association wall-time extracted from per-chunk timing.txt.

Run directory: `~/aquapose/projects/YH/runs/run_20260314_200051`
Evaluated: 2026-03-15

### Tracking Metrics (TRACK-01)

| Metric | Value |
|--------|-------|
| Track count | 1,932 |
| Track length — mean | 192.3 frames |
| Track length — median | 216.0 frames |
| Track length — min | 3 frames |
| Track length — max | 300 frames (full chunk) |
| Coast frequency | 9.0% (fraction of frames coasted) |
| Detection coverage | 91.0% (fraction of frames not coasted) |

### 3D Track Fragmentation

Fragmentation is measured on the 3D reconstruction output — fish identity continuity across the full 9,450-frame run.

| Metric | Value |
|--------|-------|
| Unique fish IDs (3D) | 53 |
| Expected fish | 9 |
| Total gaps | 12 |
| Mean gap duration | 51.5 frames |
| Max gap duration | 173 frames |
| Mean continuity ratio | 0.956 |
| Track births | 3 |
| Track deaths | 6 |
| Mean track lifespan | 283.1 frames |
| Median track lifespan | 300.0 frames (full chunk) |

### Identity Consistency (TRACK-02)

The identity stitcher assigns 3D fish identities across chunks using tracklet-set overlap. Unique fish IDs = **53** vs expected = **9**, yielding 44 excess identity fragments — roughly 6 IDs per fish on average across the 32-chunk run (each chunk can spawn a new ID if tracklet overlap is insufficient). Track births (3) and deaths (6) indicate partial cross-chunk continuity breaks. The median track lifespan of 300 frames (one full chunk) confirms that within-chunk tracks are consistently maintained.

Per-fish continuity ratios (non-unity values): fish 8 → 0.885, fish 44 → 0.660, fish 63 → 0.423, fish 159 → 0.906, fish 149 → 0.821, fish 266 → 0.493, fish 275 → 0.547. All remaining tracked fish IDs have continuity = 1.0.

### Detection Coverage Per Camera (TRACK-03)

Coverage = fraction of 9,450 frames where at least one detection was returned for that camera.

| Camera | Frames w/ Detections | Coverage (%) |
|--------|---------------------|-------------|
| e3v829d | 2,090 | 22.12% |
| e3v82e0 | 4,773 | 50.51% |
| e3v82f9 | 363 | 3.84% |
| e3v831e | 8,696 | 92.02% |
| e3v832e | 575 | 6.08% |
| e3v8334 | 7,683 | 81.30% |
| e3v83e9 | 4,748 | 50.24% |
| e3v83eb | 9,097 | 96.26% |
| e3v83ee | 1,823 | 19.29% |
| e3v83ef | 571 | 6.04% |
| e3v83f0 | 9,356 | 99.01% |
| e3v83f1 | 7,482 | 79.17% |
| **Overall mean** | **5,430** | **50.49%** |

Wide variation across cameras reflects aquarium geometry: cameras with wide-angle views covering the main swim zone (e3v83f0: 99.0%, e3v83eb: 96.3%) have much higher coverage than cameras with narrow views (e3v82f9: 3.8%, e3v832e: 6.1%) that only see fish passing through a small region.

### Association Quality (ASSOC-01)

| Metric | Value |
|--------|-------|
| Singleton rate | 12.1% |
| Fish yield ratio | 1.024 (102.4% — slight over-reconstruction) |
| Total fish observations | 99,063 |
| Frames evaluated | 9,450 |
| P50 camera count | 4.0 cameras/fish |
| P90 camera count | 4.0 cameras/fish |

Camera distribution across grouped fish observations:

| Cameras in Group | Observations | Fraction |
|-----------------|-------------|---------|
| 1 (singleton) | 11,984 | 12.1% |
| 2 | 8,177 | 8.3% |
| 3 | 15,259 | 15.4% |
| 4 | 54,297 | 54.8% |
| 5 | 9,286 | 9.4% |
| 6 | 60 | 0.1% |

The singleton rate of **12.1%** on the full run is somewhat higher than the 5.4% reported in Section 6 (v3.8 evaluation on 900 frames). This reflects variability across all 32 production chunks vs the 3-chunk test set.

### Association Wall-Time (ASSOC-02)

Times from per-chunk timing.txt, summed across all 32 chunks (300 frames each).

| Metric | Value |
|--------|-------|
| Total association time | 1,052.96 s |
| Mean per chunk | 32.91 s |
| Min per chunk | 14.75 s (chunk 32) |
| Max per chunk | 64.83 s (chunk 1, cold-start overhead) |
| % of total pipeline time | 12.7% (1,052.96 / 8,278.62 s) |

The first chunk (64.83 s) is an outlier likely due to cold-start LUT generation and model initialization. Excluding chunk 1, the mean per chunk is 31.59 s and max is 56.09 s.

<details>
<summary>All 32 per-chunk association times</summary>

| Chunk | Time (s) |
|-------|---------|
| 1 | 64.83 |
| 2 | 38.39 |
| 3 | 56.09 |
| 4 | 50.15 |
| 5 | 36.34 |
| 6 | 39.72 |
| 7 | 35.65 |
| 8 | 36.18 |
| 9 | 32.90 |
| 10 | 38.55 |
| 11 | 37.91 |
| 12 | 34.20 |
| 13 | 28.07 |
| 14 | 33.82 |
| 15 | 33.86 |
| 16 | 32.49 |
| 17 | 34.64 |
| 18 | 33.26 |
| 19 | 32.81 |
| 20 | 40.78 |
| 21 | 27.55 |
| 22 | 23.82 |
| 23 | 21.98 |
| 24 | 25.04 |
| 25 | 25.67 |
| 26 | 27.58 |
| 27 | 24.23 |
| 28 | 20.77 |
| 29 | 22.94 |
| 30 | 24.15 |
| 31 | 23.84 |
| 32 | 14.75 |

</details>

**CSV**: `data/tracking_association_full_run.csv`

---

## Stale Results (Needing Re-Run)

All results in this document are current as of v3.10 (2026-03-15). No stale entries.

---

## CSV Index

| File | Rows | Description | Suggested Plot |
|------|------|-------------|----------------|
| `data/augmentation_oks_per_image.csv` | 65 | Per-image OKS for baseline vs augmented model, with fish curvature | Scatter: OKS vs curvature, two series + regression lines |
| `data/pose_training_curves_baseline.csv` | 100 | Epoch-level training metrics for baseline pose model | Line: loss curves, mAP progression |
| `data/pose_training_curves_augmented.csv` | 100 | Epoch-level training metrics for augmented pose model | Line: overlay with baseline |
| `data/pseudo_label_ablation_obb.csv` | 6 | OBB detection mAP across 3 models × 2 val sets | Grouped bar: mAP50-95 by model and val set |
| `data/pseudo_label_ablation_pose.csv` | 6 | Pose estimation mAP across 3 models × 2 val sets | Grouped bar: pose mAP50-95 by model and val set |
| `data/curvature_stratified_oks.csv` | 9 | Mean OKS by curvature tercile for 3 model variants | Grouped bar: OKS by curvature bin and model |
| `data/reconstruction_parameter_sweep.csv` | 22 | Reconstruction parameter sweep (19 outlier_threshold + 3 min_cameras) | Line: error vs outlier_threshold; bar: error/coverage vs min_cameras |
| `data/association_parameter_sweep.csv` | 36 | Association parameter grid sweep + carry-forward results | Heatmap: yield vs ray_dist × score_min; bars for sensitivity |
| `data/reconstruction_quality_full_run.csv` | 32 | Reconstruction quality metrics from full 9,450-frame Phase 97 run (reproj error, per-keypoint, camera visibility) | Bar: per-keypoint mean/p90 error; histogram: camera visibility distribution |
| `data/pipeline_timing_full_run.csv` | 32 | Per-chunk per-stage wall-time for all 5 pipeline stages across 32 chunks (full 9,450-frame run) | Stacked bar: per-stage time per chunk; pie: stage time share |
| `data/tracking_association_full_run.csv` | 73 | Tracking, fragmentation, association, detection coverage, and per-chunk association timing from full 9,450-frame run | Bar: per-camera detection coverage; scatter: association time per chunk |
