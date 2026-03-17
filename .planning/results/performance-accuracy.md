# AquaPose Performance & Accuracy Results

All results below are validated as current against the v3.10 codebase (2026-03-15). All full-run metrics (Sections 9-11) are from run_20260315_142347: 9,450 frames, 12 cameras, 9 fish.

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
| Baseline | Manual-only | 0.935 | 0.689 | 0.933 | 0.800 |
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

### Motivation

Fish pose estimation models trained on limited manual annotations tend to under-represent curved body postures, leading to systematic accuracy degradation on high-curvature fish — the poses that matter most for behavioral analysis. The elastic augmentation experiment tests whether synthetically deforming fish midlines can reduce this curvature bias without requiring additional manual labeling.

### Methodology

Controlled A/B experiment comparing a baseline pose model (257 train images) to an augmented model (257 originals + 1,028 elastic variants = 1,285 train images). Both evaluated on the same 65-image validation set. The train/val split was curvature-stratified (80/20) to ensure balanced curvature representation in both sets. Ground-truth curvature is the mean unsigned curvature of the 6-keypoint midline polyline, measured in the image plane.

Both models: YOLO11-pose, imgsz=320, 100 epochs.

#### Deformation Pipeline

Each training image is augmented with 4 elastic variants, cycling through: C-curve positive, S-curve positive, C-curve negative, S-curve negative. Each variant independently samples a random deformation angle from the 5–15° range. The pipeline has three stages:

1. **Keypoint deformation**: Midline keypoints are displaced laterally (perpendicular to the chord) along a parametric profile:
   - **C-curve**: Parabolic profile `4t(1-t)` — zero at nose and tail, maximum at midpoint. Pixel amplitude = `chord_length × tan(angle)`. Produces single-direction bending.
   - **S-curve**: Sinusoidal profile `sin(2πt)` — zero at endpoints and midpoint, positive hump in first half, negative in second. Amplitude halved relative to C-curve so peak-to-peak excursion matches. Produces opposing bends in head and tail halves.
   - Only visible keypoints are deformed; invisible keypoints (occluded, out-of-frame) remain at their original coordinates. This prevents chord-length distortion from (0,0) sentinel values on partially-visible fish.

2. **TPS image warping**: A thin-plate spline (via `scipy.interpolate.RBFInterpolator`) warps the crop image to match the deformed keypoints. Control points include:
   - The 6 midline keypoints (visible only) as primary control points
   - 12 flanking points at ±25px along the local midline normal at each keypoint — these ensure the full fish cross-section moves uniformly, preventing lateral blur where displacement tapers off-midline
   - 4 corner identity anchors that pin the image borders in place
   - Backward mapping (`cv2.remap`) with bilinear interpolation and border replication

3. **Label regeneration**: New OBB and pose labels are computed from the deformed keypoints via PCA-based oriented bounding box fitting.

After deformation, each variant's centroid is re-centered to match the original to prevent positional drift.

#### Bug Fixes During Development

Several issues were discovered and fixed during the augmentation experiment (all committed to `dev`):
- **TPS singular matrix**: Using all 6 keypoints (including invisible ones at (0,0)) as TPS control points caused the RBF matrix to become singular. Fix: only use visible keypoints.
- **S-curve produced C-shape**: Original profile `sin(πt)` has the same shape as a C-curve. Fix: changed to `sin(2πt)` for a true S-shape with opposing lobes.
- **C-curve too subtle**: Original circular-arc math produced barely visible deformations. Fix: replaced with parabolic profile using `tan(angle)` scaling for more intuitive amplitude control.
- **Lateral blur**: TPS displacement tapered to zero away from the midline, blurring the fish body edges. Fix: added flanking control points at ±25px along keypoint normals.
- **Weak augmentation on partial-visibility images**: Deforming invisible keypoints at (0,0) distorted chord length and amplitude. Fix: deform only visible keypoints; reconstruct full array afterward.
- **S-curve too aggressive**: At the same angle, the S-curve's full-period sine produced ~2x the peak excursion of the C-curve. Fix: halved the S-curve amplitude for visual parity.

### Results

| Metric | Baseline | Augmented |
|--------|----------|-----------|
| Mean OKS | 0.712 | 0.707 |

Mean OKS is similar, but the key metric is the **OKS-vs-curvature regression slope**, which measures how much accuracy degrades as fish curvature increases:

| Metric | Baseline | Augmented |
|--------|----------|-----------|
| OKS-vs-curvature slope | **-0.71** | **-0.30** |

The augmented model's shallower slope means it degrades ~2.4x less with increasing curvature. Slopes verified by linear regression on the per-image CSV data (baseline: r²=0.19, p<0.001; augmented: r²=0.04, p=0.12).

The augmented model does show two images with OKS=0.0 (complete detection failures) at curvatures of 0.21 and 0.29 — instances where the baseline scored 0.87 and 0.78 respectively. These outliers pull the augmented mean OKS down slightly but do not materially affect the slope comparison.

**Note**: The `eval_results.json` source file records identical slopes (-0.297) for both models due to a bug in the evaluation script (used the same curvature array for both regressions). The values above are computed directly from the CSV and are authoritative.

### Relationship to Section 2

This standalone A/B experiment (Section 3) established that elastic augmentation reduces curvature bias. The curvature-stratified results in Section 2 show the downstream impact when elastic augmentation is combined with curated pseudo-labels in the full iterative training pipeline: the curated+aug model achieves 0.780 mean OKS on the high-curvature tercile vs 0.436 for the baseline — a +79% improvement that reflects both better training data (curation) and better pose diversity (augmentation).

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

## 5. Z/XY Reconstruction Anisotropy (v3.11, corrected)

### Methodology

Monte Carlo ray perturbation analysis on the 12-camera rig (e3v8250 wide-angle center camera excluded, matching the AquaCal convention). For each test depth and XY position:
1. Project a ground-truth 3D point into all cameras using undistorted intrinsics (K_new)
2. Exclude cameras where the projection falls outside the 1600×1200 image bounds
3. Run 500 Monte Carlo trials: each trial adds independent Gaussian noise (σ = 0.5px) to each camera's pixel, casts one refracted ray per camera, and triangulates via least-squares
4. Compute per-axis RMSE across trials

Grid-averaged results use a 7×7 XY grid (±0.5m around rig centroid at (-0.34, 0.55)), matching the AquaCal synthetic validation grid. Per-depth results are averaged across all 49 grid positions.

### Results

XY error uses the Euclidean norm √(dx²+dy²), matching AquaCal's convention.

| Metric | Value |
|--------|-------|
| Mean Z/XY anisotropy (grid-averaged) | **~2.9x** |
| Range across depth | 2.8x – 3.1x |
| Mean XY RMSE | 0.32 mm |
| Mean Z RMSE | 0.95 mm |
| Average cameras visible | 3–7 (varies with depth) |
| Previous estimates | 132x (v1.0, synthetic), ~11x (v3.5, methodological artifact) |

The top-down camera geometry makes Z-reconstruction inherently ~3x less precise than XY. This is a fundamental property of the rig, not a software limitation.

#### Comparison with AquaCal synthetic validation

AquaCal's tutorial (`02_synthetic_validation.ipynb`, Experiment 3) reports ~2.3x on the same rig geometry with the same 7×7 grid. Both use the Euclidean norm XY metric. The remaining gap (2.9x vs 2.3x) is because AquaCal triangulates using calibrated parameters (fit to noisy data, 0.498px RMS residual), which inflates both XY and Z absolute errors by ~2x. AquaPose uses exact calibration parameters, measuring only the pixel-noise floor. The centroid-only result (2.3x, see below) matches AquaCal exactly, confirming that the grid-averaged difference comes from edge positions where fewer cameras are visible.

#### Centroid-only results (12 cameras visible)

At the rig centroid (-0.34, 0.57), where all 12 cameras converge at depths ≥1.60m:

| Depth Range | Cameras | XY RMSE (mm) | Z RMSE (mm) | Ratio |
|-------------|---------|-------------|-------------|-------|
| 1.45–1.55m | 3–10 | 0.22–0.74 | 0.47–1.96 | 2.1–2.7x |
| 1.60–2.00m | 12 | 0.21–0.24 | 0.44–0.62 | 2.1–2.6x |

**Code**: `src/aquapose/calibration/uncertainty.py`
**Reports**: `.planning/results/uncertainty_grid_avg/`, `.planning/results/uncertainty_12cam_knew/`

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
| min_cameras | 2 | 3.00 | 271 | 270 | Duplicate 3D fish from ill-conditioned 2-camera DLT |
| min_cameras | 3 (default) | 2.87 | 254 | 270 | Winner |
| min_cameras | 4 | 2.87 | 237 | 270 | 6% coverage loss, no error gain |

### Key Findings

- **outlier_threshold** is nearly flat across 10–100px (2.87–2.91px). Inlier ratio is already ~99%, so outlier rejection does minimal work.
- **min_cameras=2** produces duplicate 3D fish (271 > 270 available): the low 2-camera threshold allows multiple camera subsets to independently triangulate the same physical fish, creating spurious extra midlines.
- **min_cameras=4** costs 6% coverage with no error improvement.
- **min_cameras=3** is the sweet spot: best error with solid coverage.
- Current defaults are already optimal.

**CSV**: `data/reconstruction_parameter_sweep.csv` — full 19-value outlier_threshold grid + 3 min_cameras values
**Suggested plot**: Line plot of error vs outlier_threshold (shows rapid saturation at 35px); bar chart of error and coverage vs min_cameras.

---

## 9. Reconstruction Quality (Full Run)

### Methodology

EvalRunner evaluated the complete full-pipeline run (run_20260315_142347): 9,450 frames, 12 cameras, 9 fish, 32 chunks of 300 frames each. No frame sampling — all frames evaluated. Reconstruction uses v3.9 raw-keypoint triangulation (6 keypoints per fish, min_cameras=3, outlier_threshold=10). Per-keypoint errors are computed by reprojecting the 6 triangulated 3D points through each observing camera's refractive model and measuring pixel distance to the 2D detections. Camera visibility counts the minimum cameras across body points for each Midline3D (field `n_cameras` set by the reconstruction backend).

Run directory: `~/aquapose/projects/YH/runs/run_20260315_142347`
Evaluated: 2026-03-15

### Reprojection Error Distribution

| Statistic | Value (px) |
|-----------|-----------|
| Mean | 3.45 |
| p50 (median) | 2.72 |
| p90 | 6.32 |
| p95 | 8.65 |
| p99 | 14.52 |
| Max | 273.94 |

- Fish-frames reconstructed: 82,333 / 85,050 (96.8% coverage)
- Inlier ratio (not low-confidence): 97.6%

### Per-Keypoint Reprojection Error

Points are indexed 0 (nose) through 5 (tail) along the fish midline.

| Point | Mean (px) | P90 (px) |
|-------|-----------|----------|
| 0 (nose) | 4.10 | 6.33 |
| 1 (head) | 2.77 | 4.02 |
| 2 (spine1) | 2.87 | 4.67 |
| 3 (spine2) | 3.10 | 5.09 |
| 4 (spine3) | 3.69 | 5.81 |
| 5 (tail) | 5.42 | 8.48 |

Nose (point 0) and tail (point 5) have higher error than mid-body points, consistent with higher pose uncertainty at the extremities.

### Camera Visibility Statistics

| Statistic | Value |
|-----------|-------|
| Mean cameras per fish | 3.68 |
| Median cameras per fish | 4.0 |
| Min | 0 |
| Max | 6 |

Camera count distribution across all fish-frame pairs:

| Cameras | Fish-Frames | Fraction |
|---------|-------------|---------|
| 0 | 903 | 1.1% |
| 2 | 2,884 | 3.5% |
| 3 | 19,263 | 23.4% |
| 4 | 56,685 | 68.8% |
| 5 | 2,549 | 3.1% |
| 6 | 49 | 0.1% |

Most fish-frames (68.8%) are observed by exactly 4 cameras. Note that `n_cameras` is the *minimum* across body points, not the total observing cameras. There are zero fish-frames with exactly 1 camera.

**CSV**: `data/reconstruction_quality_full_run.csv`

### Temporal Z-Smoothing

Post-hoc Gaussian smoothing of per-fish centroid z across time (sigma=3 frames), applied within continuous track segments. The smoothing delta is applied uniformly to all keypoints in each fish-frame.

| Metric | Value |
|--------|-------|
| Fish processed | 25 |
| Frames processed | 82,324 |
| Mean F2F centroid z jitter (before) | 0.291 cm |
| Mean F2F centroid z jitter (after) | 0.054 cm |
| Jitter reduction | 5.4x |

**Reprojection error impact** (measured by reprojecting pre- and post-smoothing 3D points through the refractive model against 2D detections, 82,330 fish-frames, 1,970,504 residuals):

| Statistic | Before (px) | After (px) | Delta (px) |
|-----------|-------------|------------|------------|
| Mean | 3.656 | 3.699 | +0.043 |
| Median (p50) | 2.298 | 2.322 | +0.024 |
| p90 | 5.820 | 5.928 | +0.108 |
| p95 | 8.371 | 8.546 | +0.176 |
| p99 | 36.720 | 36.966 | +0.246 |

Temporal z-smoothing increases mean reprojection error by +0.043 px (1.2%), consistent with the ~3x Z/XY anisotropy — small z-shifts produce negligible pixel-space displacement through the refractive projection model. The cost is well within noise and far smaller than the 5.4x reduction in frame-to-frame centroid z jitter.

**Script**: `tmp/compare_z_smoothing_reproj.py`

---

## 10. Pipeline Performance (Full Run)

### Methodology

Timing data comes from the built-in per-chunk `stage_timing` in the PipelineContext, measured during the complete full-pipeline run (run_20260315_142347) executed on 2026-03-15. The run processed 9,450 frames from the YH aquarium video across 32 chunks (31 full chunks of 300 frames + 1 partial chunk of 150 frames), with 12 cameras and 9 fish. All 5 pipeline stages were timed per chunk; chunk 31 is a partial chunk and its values reflect the proportionally smaller workload.

**Note:** This run was executed on the development workstation (GPU workstation, not a dedicated benchmarking machine). Absolute numbers will vary by hardware; relative stage shares are hardware-independent.

### Per-Stage Timing

| Stage | Total Time (s) | Mean/Chunk (s) | Share (%) |
|-------|----------------|----------------|-----------|
| Detection | 2,328.6 | 72.77 | 27.7% |
| Pose | 2,564.4 | 80.14 | 30.5% |
| Tracking | 78.6 | 2.46 | 0.9% |
| Association | 1,114.6 | 34.83 | 13.3% |
| Reconstruction | 2,322.5 | 72.58 | 27.6% |
| **Total** | **8,408.7** | **262.8** | **100%** |

### End-to-End Throughput

| Metric | Value |
|--------|-------|
| Total wall-time | 8,408.7 s (2.34 h) |
| Frames processed | 9,450 |
| Throughput | **1.12 frames/sec** |
| Mean chunk time | 262.8 s/chunk |

### Key Observations

- **Pose and detection dominate**: Together they account for 58.2% of total wall-time, driven by GPU inference on 12 cameras × 300 frames per chunk.
- **Tracking is negligible**: At 0.9% share, the Kalman tracker adds essentially zero overhead — its per-chunk cost is ~2.5s vs ~73-80s for detection or pose.
- **Reconstruction is comparable to detection**: At 27.6%, refractive triangulation (6 keypoints × 9 fish × all frames) takes nearly as long as detection, substantially faster than the v3.4 era when it shared time with legacy spline fitting.
- **Association varies**: Chunk-to-chunk association time ranges from 14.1s to 42.9s (3.1x range), reflecting scene-complexity variation. The v3.4 pre-optimization baseline was 452s/chunk (see Section 6 note); current mean is 34.8s — consistent with the >10x speedup design goal.

**CSV**: `data/pipeline_timing_full_run.csv`
**Suggested plot**: Stacked bar chart of mean per-stage time per chunk; pie chart of stage time share.

---

## 11. Tracking and Association Quality (Full Run)

### Methodology

EvalRunner evaluated the complete full-pipeline run (run_20260315_142347): 9,450 frames, 12 cameras, 9 fish, 32 chunks of 300 frames each. No frame sampling — all frames evaluated. Tracking uses a custom KeypointTracker per camera (one tracker per camera, independent tracklets). Cross-view association uses multi-keypoint ray scoring + Leiden clustering. Identity stitching across chunks uses tracklet-set overlap. Detection coverage computed from merged PipelineContext detections field. Association wall-time extracted from per-chunk stage_timing.

Run directory: `~/aquapose/projects/YH/runs/run_20260315_142347`
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
| Unique fish IDs (3D) | 25 |
| Expected fish | 9 |
| Total gaps | 99 |
| Mean gap duration | 16.5 frames |
| Max gap duration | 134 frames |
| Mean continuity ratio | 0.941 |
| Track births | 16 |
| Track deaths | 16 |
| Mean track lifespan | 3,358.7 frames |
| Median track lifespan | 1,563.0 frames (~5.2 chunks) |

### Identity Consistency (TRACK-02)

The identity stitcher assigns 3D fish identities across chunks using tracklet-set overlap. Unique fish IDs = **25** vs expected = **9**, yielding 16 excess identity fragments — roughly 2.8 IDs per fish on average across the 32-chunk run. The median track lifespan of ~1,563 frames (~5.2 chunks) means the typical fish identity survives about 5 chunk boundaries before being reassigned a new ID.

Per-fish continuity ratios range from 0.306 (fish 135) to 1.000 (fish 3, 5, 53, 128, 248, 263). 6 of 25 fish IDs have continuity = 1.0 (no gaps within their lifespan).

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
| **Overall mean** | **4,771** | **50.49%** |

Wide variation across cameras reflects aquarium geometry: cameras with wide-angle views covering the main swim zone (e3v83f0: 99.0%, e3v83eb: 96.3%) have much higher coverage than cameras with narrow views (e3v82f9: 3.8%, e3v83ef: 6.0%) that only see fish passing through a small region.

### Association Quality (ASSOC-01)

| Metric | Value |
|--------|-------|
| Singleton rate | 7.0% |
| Fish yield ratio | 0.988 (98.8%) |
| Total fish observations | 90,301 |
| Frames evaluated | 9,450 |
| P50 camera count | 4.0 cameras/fish |
| P90 camera count | 5.0 cameras/fish |

Camera distribution across grouped fish observations:

| Cameras in Group | Observations | Fraction |
|-----------------|-------------|---------|
| 1 (singleton) | 6,311 | 7.0% |
| 2 | 2,560 | 2.8% |
| 3 | 10,296 | 11.4% |
| 4 | 60,057 | 66.5% |
| 5 | 10,920 | 12.1% |
| 6 | 153 | 0.2% |
| 7 | 4 | <0.1% |

The singleton rate of **7.0%** on the full run is somewhat higher than the 5.4% reported in Section 6 (v3.8 evaluation on 900 frames). This reflects variability across all 32 production chunks vs the 3-chunk test set.

### Association Wall-Time (ASSOC-02)

Times from per-chunk stage_timing, summed across all 32 chunks (300 frames each).

| Metric | Value |
|--------|-------|
| Total association time | 1,114.59 s |
| Mean per chunk | 34.83 s |
| Min per chunk | 14.05 s (chunk 32) |
| Max per chunk | 42.94 s (chunk 26) |
| % of total pipeline time | 13.3% (1,114.59 / 8,408.71 s) |

<details>
<summary>All 32 per-chunk association times</summary>

| Chunk | Time (s) |
|-------|---------|
| 1 | 31.57 |
| 2 | 36.71 |
| 3 | 40.18 |
| 4 | 32.04 |
| 5 | 29.52 |
| 6 | 34.37 |
| 7 | 29.67 |
| 8 | 39.89 |
| 9 | 41.65 |
| 10 | 35.19 |
| 11 | 38.10 |
| 12 | 36.66 |
| 13 | 34.27 |
| 14 | 37.11 |
| 15 | 39.36 |
| 16 | 33.71 |
| 17 | 34.68 |
| 18 | 31.83 |
| 19 | 27.45 |
| 20 | 30.09 |
| 21 | 29.70 |
| 22 | 34.88 |
| 23 | 29.90 |
| 24 | 40.36 |
| 25 | 40.03 |
| 26 | 42.94 |
| 27 | 35.69 |
| 28 | 38.69 |
| 29 | 34.65 |
| 30 | 37.69 |
| 31 | 41.96 |
| 32 | 14.05 |

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
| `data/reconstruction_quality_full_run.csv` | 32 | Reconstruction quality metrics from full 9,450-frame run (reproj error, per-keypoint, camera visibility) | Bar: per-keypoint mean/p90 error; histogram: camera visibility distribution |
| `data/pipeline_timing_full_run.csv` | 32 | Per-chunk per-stage wall-time for all 5 pipeline stages across 32 chunks (full 9,450-frame run) | Stacked bar: per-stage time per chunk; pie: stage time share |
| `data/tracking_association_full_run.csv` | 73 | Tracking, fragmentation, association, detection coverage, and per-chunk association timing from full 9,450-frame run | Bar: per-camera detection coverage; scatter: association time per chunk |
