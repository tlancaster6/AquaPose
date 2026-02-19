# Cross-View Holdout: Implementation Spec

## Purpose

Validate 3D pose reconstruction accuracy and tune loss weights by optimizing against a subset of cameras and measuring prediction quality in held-out cameras. This is the primary ongoing validation metric for Phase III.

## Method

For each fish in each frame:

1. Partition the cameras that observe the fish into a **fit set** (N-k cameras) and a **holdout set** (k cameras).
2. Run Phase III pose optimization using only the fit set in the silhouette loss.
3. Project the optimized mesh into the holdout cameras via the refractive projection.
4. Compare the projected silhouette against the observed mask in each holdout camera.

## Parameters

- **Holdout count (k)**: 2-3 cameras per evaluation. Holding out more gives a stronger test but degrades the fit. With 13 cameras and typical 5+ coverage per fish, holding out 2-3 still leaves a well-constrained optimization.
- **Holdout selection strategy**: Do not hold out randomly — hold out cameras that provide the most geometrically distinct views (maximize angular separation from the fit set). This makes the test maximally sensitive to 3D errors. In practice: for each fish, rank cameras by angular distance from the fit-set centroid viewpoint and hold out the top k.
- **Rotation**: For hyperparameter tuning, rotate which cameras are held out across multiple runs to average over geometric biases. For routine validation during processing, a single fixed holdout partition per fish per frame is sufficient.

## Metrics

**Primary — Holdout Silhouette IoU**:

$$\text{IoU}_{\text{holdout}} = \frac{1}{k} \sum_{i \in \text{holdout}} \text{IoU}(\text{Render}(\mathcal{M}(\mathbf{S}), \Pi_{\text{ref}}^{(i)}), \mathbf{M}_i)$$

This is the single number that drives loss weight tuning.

**Secondary — Holdout Boundary Error**:

Mean distance (in pixels) from predicted mask boundary to observed mask boundary in holdout views. More sensitive to subtle shape errors than IoU.

**Diagnostic — Per-Camera Residual Map**:

For each holdout camera, compute the pixel-wise XOR between projected and observed masks. Systematic patterns (e.g., consistently too wide, or offset in one direction) indicate specific failure modes — depth bias, scale error, orientation error.

## Loss Weight Tuning Procedure

1. Define a grid or Bayesian search over $(\lambda_{\text{grav}}, \lambda_{\text{shape}}, \lambda_{\text{temp}})$.
2. For each weight configuration, run Phase III on a validation subset (e.g., 200 frames spanning different fish positions and behaviors).
3. For each frame, run cross-view holdout with k=2, rotating holdout cameras across frames.
4. Compute mean holdout IoU across all frames and fish.
5. Select the weight configuration that maximizes mean holdout IoU.
6. Sanity check: inspect the per-camera residual maps for the winning configuration to confirm no systematic bias.

## Acceptance Criteria

- Mean holdout IoU ≥ 0.80. This is a stricter test than fit-set IoU because the holdout cameras were not used in optimization, and they are selected to be the most geometrically challenging views.
- No systematic directional bias in residual maps (would indicate calibration or refraction model error rather than pose estimation error).
- Holdout IoU should not degrade significantly when the fish is in geometric weak zones (tank edges, star-pattern low-triangulation regions from the coverage analysis).

## Integration

Cross-view holdout is a **validation tool**, not part of the production pipeline. In production, all available cameras contribute to the silhouette loss. Holdout runs offline on sampled frames for:

- Initial loss weight tuning (once, during Stage 2 development)
- Regression testing when pipeline components change
- Per-recording quality reports (sample 1-2% of frames, report holdout IoU distribution)
