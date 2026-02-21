# Phase 4: Single-Fish Reconstruction - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Fit the parametric fish mesh to multi-view silhouettes via differentiable refractive rendering and gradient-based optimization. Process each detected fish independently per frame. Validate via cross-view holdout IoU. Multi-fish interactions, tracking, and temporal smoothness activation belong to Phase 5.

</domain>

<decisions>
## Implementation Decisions

### Rendering pipeline
- Silhouettes only — no depth or normal maps
- Wrap existing `calibration/` refractive ray-casting code in a differentiable layer (thin PyTorch wrapper), do not rewrite Snell's law from scratch
- Per-camera angular-diversity weighting computed from camera extrinsics — down-weight clustered viewpoints
- Camera selection is input-driven: whatever videos are in the input folder get processed. Problematic cameras (e.g., e3v8250) are excluded by the user at the data level, not hardcoded

### Loss design
- Crop-space IoU: compute silhouette IoU within the bounding box crop region, not the full frame
- Gravity prior = soft roll regularization: penalize deviations of the fish's dorsal-ventral axis from upright orientation. Low weight — just enough to break ambiguities when silhouette alone can't distinguish rolled vs. unrolled pose
- Morphological constraints on both scale (s) and curvature (kappa): curvature bounds should be stricter than scale bounds
- Hand-tuned fixed loss weights (e.g., IoU=1.0, gravity=low, morph=moderate). Tuned empirically, not learned
- Temporal smoothness term is architecturally present but inactive until Phase 5 provides track associations

### Optimization strategy
- Process one fish at a time per frame — no joint multi-fish optimization
- 2-start initialization on first frame: forward + 180-degree flip. Early exit heuristic: run both ~50 iters, discard the clearly-worse one, finish the better one to completion
- Convergence criterion with hard iteration cap as safety net. Loss delta below threshold for ~3 consecutive steps triggers early stop. Cap prevents runaway optimization on pathological frames
- Warm-start between frames uses constant-velocity prediction: extrapolate position from last 2 frames' solutions

### Validation approach
- Leave-one-out cross-view holdout: hold out 1 camera, rotate across frames/fish to accumulate statistics
- Full diversity test clip (~500+ frames) covering occlusions, fast motion, edge cases
- Output: quantitative IoU metrics + visual overlays (rendered mesh on real camera frames)
- Target: global average holdout IoU >= 0.80, with no individual camera below 0.60 floor
- Phase 4 includes a diagnose-and-iterate cycle if the target isn't met initially

### Claude's Discretion
- Adam learning rate and hyperparameter defaults
- Exact iteration cap value and convergence threshold
- Angular-diversity weighting formula
- Crop padding strategy for IoU computation
- Visual overlay rendering style

</decisions>

<specifics>
## Specific Ideas

- Gravity prior specifically targets roll angle toward zero — fish almost always swim upright
- Curvature constraints should be stricter than scale constraints because unrealistic bending is more visually obvious than slight size errors
- Early-exit for 2-start: most of the time one initialization is clearly worse by ~50 iters, so don't waste compute finishing it
- Convergence after warm-start should be fast (previous frame is a good init) — the hard cap is mainly a safety net for pathological frames

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-single-fish-reconstruction*
*Context gathered: 2026-02-20*
