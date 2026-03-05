# Pseudo-Labeling Milestone Seed

## Problem

The OBB detection and pose estimation models are trained on very limited manual annotations (~150 frames for OBB). The multi-camera pipeline produces 3D reconstructions with built-in quality metrics that can be reprojected into each camera view to generate training labels at scale.

## Goal

Add pseudo-label generation, training run management, and iterative retraining capabilities to AquaPose. Two model targets: YOLO-OBB (detection) and YOLO-pose (keypoint estimation).

## Pseudo-Label Sources

### Source A — Consensus Reprojections

High-confidence 3D reconstructions reprojected into views where the model *already detected the fish*. These reinforce existing model capability and add training volume across viewpoints.

- Input: `Midline3D` with `mean_residual < threshold`, `n_cameras >= 3`, `is_low_confidence == False`
- For each contributing camera: reproject 3D spline back to 2D, generate OBB and/or keypoint annotations
- Confidence score derived from reconstruction quality metrics (residual, camera count, per-camera residual for that specific view)

### Source B — Gap Fills

Fish reconstructed well from 3+ cameras, but missing or malformed in other cameras where the inverse LUT indicates the fish should be visible. These directly target the model's failure modes (low-contrast females, edge-of-frame, partial occlusion).

- Input: same high-quality `Midline3D`, cross-referenced with the inverse LUT visibility map and per-camera detection results
- A camera is a "gap" if: (a) the LUT says the fish's 3D position is visible in that camera, and (b) no detection/tracklet exists there for that fish at that frame
- Reproject the 3D reconstruction into the gap camera to generate the pseudo-label
- Stored separately from Source A labels — different folder, tagged metadata
- Potentially lower confidence than Source A (the model failed here for a reason — occlusion, lighting, etc.)

## Label Formats

### OBB Labels

Reproject the 3D spline (sampled at N body points) plus half-widths into a camera view. The projected body outline (midline +/- half-widths) defines an oriented region. Fit a minimum-area oriented bounding box to the projected contour.

Output: YOLO-OBB txt format (class, x1, y1, x2, y2, x3, y3, x4, y4 — normalized coordinates).

### Pose Labels

The 3D spline is parameterized on [0, 1] and the pipeline already has calibrated `keypoint_t_values` mapping each named anatomical landmark (nose, head, spine1, ..., tail) to its arc-length fraction along the body. These t-values are computed from manual annotations via `aquapose train prep calibrate-keypoints` and are not evenly spaced — they reflect actual anatomical positions.

To generate pose pseudo-labels:
1. Evaluate the B-spline at the configured `keypoint_t_values`: `spline(t_values)` -> 3D keypoint positions
2. Reproject each 3D keypoint into the camera view using `RefractiveProjectionModel.project()`
3. The OBB for the bounding box portion comes from the OBB label generation path

Output: YOLO-pose txt format (class, cx, cy, w, h, kp1_x, kp1_y, kp1_vis, ..., kpN_x, kpN_y, kpN_vis — normalized coordinates). Visibility flag set based on whether the projected point falls within the image bounds and the reconstruction quality at that body position.

**Known approximation:** The t-values are averages across manual annotations. Intermediate landmark positions (everything except nose=0.0 and tail=1.0) shift slightly with body proportions and posture. This is acceptable for pseudo-label quality — small positional errors are absorbed during training — but worth monitoring if per-keypoint reprojection error is ever evaluated.

### Confidence Scores

Every pseudo-label carries a confidence score (float in [0, 1]) stored in sidecar metadata. Derived from:

- Reconstruction `mean_residual` (lower is better)
- **Per-camera residual variance** — a reconstruction where all cameras agree (low variance) is more trustworthy than one with the same mean but high spread. Computed from `per_camera_residuals` on `Midline3D`. Inspired by the Mahalanobis distance approach in multi-view animal pose literature (arXiv 2510.09903).
- `n_cameras` contributing to the reconstruction
- Per-camera residual for the specific view being labeled
- Source type (A vs B — Source B inherently lower confidence)

Confidence scores enable threshold-based filtering at training time without regenerating labels.

## Frame Selection

Two-stage selection: temporal subsampling followed by pose-diversity filtering.

1. **Temporal subsampling**: select every Kth frame (configurable, e.g., every 5th or 10th)
2. **Filter**: drop frames with zero reconstructions
3. **Pose-diversity sampling**: cluster the reconstructed 3D midline configurations (k-means on flattened spline control points across all fish in the frame) and sample to maximize pose diversity — ensures the training set covers curved, straight, turning, and stationary poses rather than oversampling the most common body configuration

Rationale: research on multi-view animal pose estimation (arXiv 2510.09903) shows that diversity-aware sampling significantly outperforms uniform temporal sampling for pseudo-label selection. Fish in a tank spend most time in similar postures, so temporal subsampling alone overrepresents the dominant pose.

## Training Strategy

### Pooled Training with Source Tagging

1. **Manual annotations** always included as the anchor (highest trust)
2. **Source A pseudo-labels** added with confidence threshold filtering
3. **Source B pseudo-labels** kept in a separate directory, added with (typically stricter) confidence threshold
4. **Each round**: train from pretrained YOLO base weights (not previous round's weights) to limit error propagation
5. Per-round adjustable: confidence thresholds for Source A and B, temporal subsampling rate

### Iterative Loop

```
Round 0: Train on manual annotations only (current baseline)
Round N:
  1. Run pipeline with Round N-1 model
  2. Generate pseudo-labels (Sources A + B) from pipeline output
  3. Filter by confidence thresholds
  4. Pool: manual + filtered Source A + filtered Source B
  5. Train from pretrained YOLO base
  6. Evaluate against holdout set
  7. Compare to Round N-1 metrics
```

Each round's outputs go to a unique timestamped folder. No round overwrites another.

## Training Run Management

### Run Organization

```
projects/<project>/training/
  manual/                      # Manual annotations (immutable across rounds)
    obb/                       # YOLO-OBB txt+yaml format
    pose/                      # YOLO-pose txt+yaml format
  pseudo/
    round_001/                 # Per-round pseudo-label output
      source_a/
        obb/                   # YOLO-OBB txt+yaml format
        pose/                  # YOLO-pose txt+yaml format
        metadata.json          # Generation params, confidence stats, frame selection
      source_b/
        obb/
        pose/
        metadata.json
    round_002/
      ...
  runs/
    run_<timestamp>_<model>/   # Per-training-run output
      _ultralytics/            # Ultralytics native output (don't fight it)
      config.yaml              # Training config snapshot
      metrics_summary.json     # Extracted key metrics for cross-run comparison
      best_model.pt
      last_model.pt
  summaries/
    comparison.csv             # Cross-run metric comparison table
    latest_report.md           # Auto-generated summary of recent runs
```

### Cross-Run Comparison

- After each training run, extract key metrics from Ultralytics output (mAP, precision, recall, loss curves) into a standardized `metrics_summary.json`
- Append to `comparison.csv` for tabular cross-run comparison
- CLI command to generate/refresh the comparison report
- Track which pseudo-label round + confidence thresholds were used for each run

## Early Phase: `aquapose prep` CLI Group

Before pseudo-labeling work begins, consolidate pre-pipeline setup commands under a new `aquapose prep` CLI group. This phase resolves two existing gaps:

### 1. Wire up `calibrate-keypoints`

The `prep calibrate-keypoints` command exists in `training/prep.py` but is never registered with the CLI. Currently `keypoint_t_values` defaults to uniform spacing (`np.linspace(0, 1, n_keypoints)`) when not set in config, which is incorrect — anatomical landmarks are not evenly spaced. This must be fixed before generating pose pseudo-labels.

Actions:
- Register `prep_group` as a subcommand (either under `train` or as a top-level `aquapose prep` group)
- Document the workflow: run `calibrate-keypoints` -> paste t-values into project config
- Consider: should pseudo-label generation fail-fast if `keypoint_t_values` is None (uniform fallback)?

### 2. Move LUT generation to `aquapose prep generate-luts`

Per existing TODO (`.planning/todos/pending/2026-02-28-move-lut-generation-to-pre-pipeline.md`): LUT generation currently lives inside `AssociationStage.run()` as lazy initialization, violating the principle that stages are pure computation. LUTs are pre-pipeline input materialization.

Actions:
- Add `aquapose prep generate-luts --calibration <path> --output <path>` CLI command
- Move generation logic out of `AssociationStage.run()` into a callable function in `calibration/luts.py` (may already be factored this way)
- `AssociationStage` receives LUTs as required input, fails fast if missing
- Pipeline setup (CLI/ChunkOrchestrator) loads LUTs from cache or calls generation before starting the batch loop
- Existing lazy-generation behavior can remain as a fallback during transition, but the CLI command becomes the documented path

### Why this phase comes first

Both pseudo-label sources depend on these:
- **Pose labels** require correct `keypoint_t_values` to sample the 3D spline at anatomical positions
- **Gap-fill detection (Source B)** uses inverse LUTs to determine which cameras should see a fish — the LUTs need to be reliably pre-generated, not lazily created mid-pipeline

## Architecture

All new code lives in `src/aquapose/training/`:

- `pseudo_labels.py` — Core pseudo-label generation logic
  - `generate_obb_labels()` — 3D spline + half-widths -> OBB annotation
  - `generate_pose_labels()` — 3D spline at keypoint t-values -> keypoint annotation
  - `compute_label_confidence()` — Reconstruction metrics -> confidence score
- `gap_detection.py` — Source B gap-fill logic
  - `find_detection_gaps()` — Cross-reference reconstructions with LUT visibility and detection results
  - `generate_gap_fill_labels()` — Reproject into gap cameras
- `frame_selection.py` — Temporal subsampling and filtering
- `export.py` — Write YOLO txt+yaml format datasets from pseudo-labels
- `run_manager.py` — Training run organization, metric extraction, cross-run comparison
- Updates to `cli.py` — New subcommands for pseudo-label generation, dataset assembly, and run comparison

### CLI Surface

```
aquapose train pseudo-label --run-dir <pipeline_run> --round <N> --config <params.yaml>
aquapose train assemble --manual-dir <...> --pseudo-dir <...> --output <...> --confidence-threshold-a 0.8 --confidence-threshold-b 0.9
aquapose train compare --training-dir <...>
aquapose train yolo-obb --data-dir <assembled> --output-dir <...>   # existing
aquapose train yolo-pose --data-dir <assembled> --output-dir <...>  # existing
```

## Dependencies on Existing Code

- `RefractiveProjectionModel.project()` — 3D-to-pixel reprojection (exists)
- `Midline3D` — 3D reconstruction with quality metrics (exists)
- `scipy.interpolate.BSpline` — Evaluate spline at arbitrary t-values (exists in reconstruction)
- Inverse LUT — Voxel-to-camera visibility map (exists, used in association)
- Diagnostic caches — Per-chunk `cache.pkl` with full `PipelineContext` (exists)
- `train_yolo_obb()` / `train_yolo_pose()` — Training wrappers (exist, may need minor updates for output organization)

## Future Directions

These techniques are well-supported in literature but deferred from this milestone due to Ultralytics integration complexity:

### Confidence-Based Loss Reweighting

Multiple papers (Adaptive Self-Training for Object Detection, arXiv 2212.05911; Error Mitigation Teacher, etc.) show that soft per-sample loss weighting by pseudo-label confidence outperforms hard accept/reject filtering. Rather than discarding labels below a threshold, weight each sample's contribution to the loss by its confidence score. This reduces information loss from borderline samples while still down-weighting noisy ones. Deferred because Ultralytics does not expose per-sample loss weighting without forking the training loop.

### Multi-View Consistency Loss During Training

SelfPose3d (CVPR 2024) and related work add a reprojection consistency loss term: given a 2D pose prediction in camera A, triangulate with predictions from cameras B and C, then penalize reprojection error back into A. This enforces geometric consistency during training, not just during label generation. Powerful for closing the loop between 2D and 3D, but requires custom training infrastructure beyond Ultralytics. Worth revisiting if pseudo-labeling alone plateaus after several rounds.

### References

- [Rethinking Data Annotation for Multi-view 3D Pose (WACV 2023)](https://arxiv.org/abs/2112.13709) — active learning + self-training with multi-view geometry
- [Uncertainty-Aware Multi-View Animal Pose Estimation (2025)](https://arxiv.org/abs/2510.09903) — Mahalanobis distance filtering, diversity sampling, variance inflation
- [View-to-Label (CVPR 2025)](https://arxiv.org/abs/2305.17972) — iterative pseudo-label refinement via multi-view consistency
- [SelfPose3d (CVPR 2024)](https://arxiv.org/abs/2404.02041) — self-supervised multi-view pose with reprojection consistency loss
- [Adaptive Self-Training for Object Detection](https://arxiv.org/abs/2212.05911) — adaptive thresholds, error propagation mitigation
- [Curriculum Labeling (AAAI 2021)](https://cdn.aaai.org/ojs/16852/16852-13-20346-1-2-20210518.pdf) — progressive pseudo-label inclusion

## Open Questions

- What holdout strategy for evaluating pseudo-label rounds? Separate manually-annotated validation set? Cross-camera holdout?
- Should Source B labels include a "reason" field (e.g., "no detection", "detection but no tracklet", "tracklet but failed midline")?
- Minimum number of cameras for Source B gap-fill trust? (3 good cameras seems like a floor)
- Should the comparison system track eval-suite metrics (from `aquapose eval`) in addition to training metrics?
