# Pitfalls Research

**Domain:** 3D fish pose estimation via direct triangulation (medial axis → arc-length → RANSAC triangulation → spline fitting)
**Researched:** 2026-02-19 (updated 2026-02-21 for pipeline pivot)
**Confidence:** MEDIUM — core optics/math pitfalls are HIGH confidence from literature; multi-fish extension pitfalls are MEDIUM from analogous animal tracking work; some implementation specifics are LOW and flagged

> **Pipeline pivot note:** AquaPose pivoted from analysis-by-synthesis (differentiable mesh rendering + Adam optimization) to a direct triangulation pipeline. Pitfalls marked **[SHELVED PIPELINE]** apply only to the original analysis-by-synthesis approach and are retained for reference. All other pitfalls apply to the primary (direct triangulation) pipeline or both pipelines.

---

## Critical Pitfalls

### Pitfall 1: Treating Refractive Distortion as Depth-Independent *(both pipelines)*

**What goes wrong:**
The refractive projection through a flat air-water port is depth-dependent — the 2D image coordinates of a 3D point depend on its Z value, not just its XY position. Systems that model refraction as a fixed pixel-wise correction (e.g., a static distortion map or polynomial warp applied once at calibration time) will produce systematic 3D reconstruction errors that grow with distance from calibration depth. The error is not random noise — it introduces a consistent bias in triangulated 3D positions.

**Why it happens:**
Developers import standard camera calibration pipelines (OpenCV, COLMAP) that use pinhole + radial/tangential distortion. These models absorb refractive distortion into the distortion coefficients, but those coefficients are fit at a fixed effective depth and will be wrong at other depths. The mistake is treating refraction like lens distortion when it is geometrically distinct.

**How to avoid:**
Implement a full per-ray refractive projection: trace each camera ray through the air-glass-water interface using Snell's law, applying n_air=1.0, n_glass≈1.5, n_water≈1.333, and the measured glass thickness and flat port position. This must happen inside the projection function itself, not as a pre/post warp. The Newton-Raphson solver approach (10 fixed iterations for differentiability) is correct — do not shortcut to a lookup table unless you verify it at multiple depths.

**Warning signs:**
- Reprojection error that varies systematically with fish depth (fish near bottom of tank reproject worse than fish near water surface)
- 3D reconstructions that look plausible from above but collapse in Z
- Calibration error that is low but reconstruction accuracy that is poor when verified with a known-3D reference object at multiple depths

**Phase to address:**
Camera model and calibration phase (before any pose optimization is attempted). Validate with a 3D target at 3+ known depths before proceeding. Applies to both the primary triangulation pipeline (refractive ray casting) and the shelved analysis-by-synthesis pipeline (refractive projection).

---

### Pitfall 2: All-Top-Down Camera Configuration Creates Pathologically Weak Z-Reconstruction *(both pipelines)*

**What goes wrong:**
13 cameras all looking straight down share nearly parallel optical axes. This is a degenerate multi-view geometry for reconstructing depth (Z). Triangulation uncertainty scales with baseline/depth ratio and with the sine of the angle between rays. With all cameras looking down, the angle between any two rays to a point is small, making Z reconstruction extremely noisy compared to XY. A 1px reprojection error can translate to centimeters of Z error at tank depth.

**Why it happens:**
The experimental setup was designed for overhead observation (biological constraints), not for optimal 3D geometry. The weakness is inherent, not a coding error. It becomes a critical pitfall when developers validate reprojection error in 2D (which looks fine) but never validate 3D reconstruction accuracy in Z directly.

**How to avoid:**
- Quantify the theoretical Z uncertainty bound given the actual camera baselines and operating depth before writing any optimization code. If Z uncertainty is 5x worse than XY, the system must be designed to account for this.
- Add Z-regularization in the pose optimizer — prior on fish Z-position from water depth constraints (fish cannot be above the water surface or below the tank floor).
- During evaluation, always report separate X, Y, Z errors on held-out ground truth, not just aggregate reprojection error.
- Consider whether the parametric fish model can provide implicit Z constraints via silhouette size (apparent size is a depth cue even from above).

**Warning signs:**
- Reprojection error looks good (< 2px) but 3D keypoint positions fluctuate by centimeters frame-to-frame in Z
- Optimizer converges easily on XY but oscillates on Z
- Z estimates correlate with fish buoyancy behavior only weakly

**Phase to address:**
Geometry validation phase (Phase 1) — already resolved. Z/XY anisotropy quantified at 132x. Z-accuracy budget established before committing to the triangulation approach.

---

### Pitfall 3: Silhouette-Only Fitting Converges to Wrong Local Minimum — **[SHELVED PIPELINE]**

> **Shelved pipeline only.** This pitfall applies to the original analysis-by-synthesis approach. In the primary triangulation pipeline, the equivalent risk is arc-length correspondence errors on curved fish (see Pitfall 12).

**What goes wrong:**
Analysis-by-synthesis with silhouette loss is highly non-convex. A rendered silhouette that matches the observed silhouette area and rough shape can correspond to many different 3D poses (front-back flips, yaw ambiguities, depth-scale confounds). The optimizer converges to whichever basin of attraction contains the initialization. For top-down cameras, a fish rotated 180° around its vertical axis produces nearly the same silhouette from above. Gradient-based optimization cannot escape once trapped.

**Why it happens:**
Silhouette loss is an area-overlap metric (IoU or binary cross-entropy on masks). It is smooth but nearly flat in regions where the silhouette topology does not change, providing weak gradient signal. The problem is compounded by the top-down camera geometry: head/tail orientation is ambiguous from above.

**How to avoid:**
- Never initialize pose from random or zero values. Use a detector (even a simple bounding-box center + tank-floor Z prior) to seed XY, and use optical flow or frame-to-frame tracking to seed orientation.
- Add keypoint loss alongside silhouette loss if even coarse keypoints (e.g., head tip, tail tip) can be detected. A single reliable keypoint breaks most head-tail ambiguities.
- Implement multi-start optimization for the first frame of each fish track, sampling 4-8 orientation initializations and selecting the one with lowest loss.
- Use temporal smoothness regularization across frames to resist single-frame escapes into wrong basins.

**Warning signs:**
- Fish pose estimates flip 180° between consecutive frames
- Loss converges to the same value from very different initializations (plateau, not a minimum)
- Visual overlay of rendered mesh on image looks like the wrong fish half

**Phase to address:**
Pose optimizer design phase. Multi-start must be built in from the start, not retrofitted.

---

### Pitfall 4: Newton-Raphson Fixed-Iteration Solver Fails Near Flat Port Edges (Grazing Angles) *(both pipelines)*

**What goes wrong:**
The Newton-Raphson solver for the refractive projection equation is initialized assuming the ray hits the flat port at a moderate angle. Near the edges of a wide-angle camera's field of view, rays approach the flat port at steep angles (approaching the critical angle for total internal reflection, ~48.6° from normal for water-glass). At these grazing angles:
1. The refracted angle changes rapidly for small changes in incident angle (Snell's law becomes steep)
2. Newton-Raphson convergence slows dramatically or oscillates
3. 10 fixed iterations may be insufficient to converge to required accuracy

Beyond the critical angle, total internal reflection occurs and no physical solution exists — the solver will return a nonsense result silently.

**Why it happens:**
A fixed iteration count was chosen for differentiability (to allow backpropagation through the solver). The iteration count was calibrated for well-behaved central rays. Edge cases are not checked at runtime.

**How to avoid:**
- During development, log the residual after 10 iterations for all projected points. If any residual exceeds a threshold (e.g., 0.1px equivalent), the iteration count or initialization needs adjustment.
- Mask out pixels near the theoretical field-of-view limit (calculate the critical angle for your specific n_water, n_glass, glass thickness). Flag projected points outside this boundary as invalid rather than using their result.
- Add a convergence check even if you use fixed iterations: if `|f(x_10)| > epsilon`, mark that reprojection as unreliable and exclude from loss computation.
- Test the solver explicitly with rays at 30°, 40°, 45°, 48° incidence angle and verify residuals.

**Warning signs:**
- Reprojection error is much higher for fish near tank edges than fish near center
- Optimizer gradients become NaN or extremely large for edge-region observations
- Rendered silhouettes are distorted or clipped near image boundaries

**Phase to address:**
Custom refractive projection layer implementation (the very first technical component). Validate with an analytical test suite before using in any optimization.

---

### Pitfall 5: MOG2 Background Subtraction Fails on Female Fish (Low-Contrast, Similar Coloring to Background) *(partially resolved)*

**What goes wrong:**
Female zebrafish (or similarly colored species) have lower visual contrast against the tank substrate than males. MOG2 models each pixel as a mixture of Gaussians for the background. When foreground fish have pixel intensities within 2-3 standard deviations of the background model, MOG2 incorrectly absorbs them into the background. The fish becomes invisible to detection. Additionally:
- Cast shadows from top-down lighting appear as separate foreground blobs (value=127 in MOG2), creating ghost detections
- Slow-moving or stationary fish are absorbed into the background model within seconds (MOG2 history parameter effect)
- Fish that school together create merged blobs that a simple instance-count heuristic misinterprets as one fish

**Why it happens:**
MOG2 was designed for pedestrian/vehicle detection in outdoor scenes with high contrast objects. Aquatic settings combine low contrast with dynamic backgrounds (water ripple, lighting flicker), which confuses the Gaussian mixture update.

**Mitigation status:**
YOLO has been added as an alternative detector (`make_detector("yolo", model_path=...)`), which is less susceptible to low-contrast and stationary-fish dropout than MOG2. MOG2 is retained as a fallback. The general concern about detection recall for low-contrast fish remains relevant regardless of detector choice.

**How to avoid:**
- Enable MOG2's shadow detection (`detectShadows=True`) and threshold at value>127 to exclude shadows before extracting contours.
- Tune `history` (frames in background model) explicitly — default 500 frames may be too long or too short depending on frame rate. For stationary fish, reduce history.
- Add frame-differencing as a fallback: when MOG2 foreground area drops below expected total fish area, cross-check against an inter-frame difference mask.
- For female fish specifically, consider running detection in a color space where the fish-background contrast is higher (e.g., saturation channel if females have color variance the background does not).
- Validate detection recall separately for male vs. female fish before deploying the full pipeline.
- Prefer YOLO-based detection for production use; MOG2 is a useful preprocessing step but not a reliable primary detector.

**Warning signs:**
- Detection count drops below expected fish count during certain tank lighting conditions
- Ghost blobs appear where fish shadows fall, not where fish bodies are
- "Stationary" fish disappear from detection when they stop moving for > N seconds

**Phase to address:**
Detection module phase (Phase 2). YOLO added as mitigation; remaining risk is general detection recall validation.

---

### Pitfall 6: Flat Refractive Port Normal Assumed Perfect — Tilt Creates Unmodeled Asymmetric Distortion *(both pipelines)*

**What goes wrong:**
The refractive projection model assumes the flat port glass is perfectly perpendicular to the camera optical axis. In practice, the port may be tilted by even 1-2°. A tilted flat port breaks the radial symmetry of refractive distortion: the distortion becomes asymmetric, with more bending on one side of the image than the other. A model that assumes perfect alignment will have systematic reprojection errors that are directionally biased — harder to diagnose because they look like calibration noise rather than a model error.

**Why it happens:**
Tank camera mounting hardware has finite precision. The port normal is typically assumed to be aligned during calibration, not measured. Published papers often assume alignment and show good results on controlled lab setups but this assumption is violated in production.

**How to avoid:**
- Treat the flat port normal as an additional calibration parameter (2 angles: tilt in X, tilt in Y). Estimate it during the calibration procedure using a calibration target at multiple depths.
- Use the open-source refractive calibration toolbox (arxiv 2405.18018) which explicitly handles port tilt estimation.
- After calibration, visually inspect the residual distortion map: if residuals are radially symmetric, alignment is good; if they have a systematic directional component, port tilt exists.

**Warning signs:**
- Reprojection errors are systematically higher on one side of the image than the other
- Adding higher-order polynomial distortion terms reduces error but still has a directional component
- Calibration accuracy varies depending on which half of the image the calibration target occupies

**Phase to address:**
Camera calibration phase. Port tilt must be estimated before pose optimization begins.

---

## Moderate Pitfalls

### Pitfall 7: Soft Rasterizer Hyperparameters Require Per-Setup Tuning — **[SHELVED PIPELINE]**

> **Shelved pipeline only.** The primary triangulation pipeline does not use differentiable rendering. Retained for reference if analysis-by-synthesis is revisited.

**What goes wrong:**
PyTorch3D's soft rasterizer has three critical hyperparameters: `sigma` (blur radius, controls silhouette softness), `gamma` (aggregation weight temperature), and `faces_per_pixel`. The optimal values depend on fish size in pixels, camera distance, and mesh resolution. Values that work for a 100px fish silhouette will produce over-blurred gradients for a 400px fish or under-blurred gradients that behave like hard rasterization (no gradient at silhouette boundary). Most implementations copy values from PyTorch3D tutorials without validating them for their specific geometry.

**How to avoid:**
- Before optimization, render test images with candidate sigma values and visualize the gradient magnitude map. Sigma should produce visible (non-zero) gradients within ~5% of the silhouette boundary.
- Set sigma proportional to expected fish apparent diameter (in pixels) divided by mesh face count, as a starting heuristic.
- Run a sweep over {sigma, gamma} on a single frame before committing to values for the full dataset.

**Warning signs:**
- Optimizer loss decreases but rendered silhouette does not visually improve
- Gradient norm is zero or near-zero for most of the optimization
- Results are sensitive to minor changes in initial pose (symptom of gradient being too narrow)

**Phase to address:**
Differentiable rendering integration phase.

---

### Pitfall 8: Cross-Section Profile Self-Calibration Overfits to Individual Fish Variance — **[SHELVED PIPELINE]**

> **Shelved pipeline only.** In the primary triangulation pipeline, width profiles come from the distance transform on masks, not from optimization. Retained for reference if analysis-by-synthesis is revisited.

**What goes wrong:**
Self-calibrating the parametric fish cross-section profile from data means the shape model will reflect the specific fish in the calibration set. If calibration fish are all from the same sex, size class, or developmental stage, the model will not generalize to other fish. More critically, if the optimization jointly estimates fish pose AND cross-section profile, gradient will flow partly into shape updates that "explain away" pose error — the shape adapts to compensate for pose mistakes, creating a coupled failure mode where neither shape nor pose is correctly estimated.

**How to avoid:**
- Separate shape calibration from pose estimation: freeze cross-section profile while optimizing pose, and vice versa. Only alternate between them (EM-style) rather than jointly optimizing.
- Include fish from multiple individuals, both sexes, and size ranges in the shape calibration set.
- Add a shape regularization loss penalizing deviation from prior cross-section measurements (biological prior on fish proportions).
- After calibration, test shape generalization by holding out fish and measuring fit quality on held-out individuals.

**Warning signs:**
- Shape parameters drift significantly over the course of a video sequence
- Individual fish within the same tank are assigned very different cross-section profiles
- Adding more shape flexibility improves training loss but degrades pose accuracy on held-out frames

**Phase to address:**
Shape model calibration phase, before multi-fish scaling.

---

### Pitfall 9: Single-Fish Architecture Does Not Isolate State Per Fish — Prevents Clean 9-Fish Extension *(both pipelines)*

**What goes wrong:**
V1 builds a single-fish pipeline with global state: one background model, one detector, one set of calibration results. When extending to 9 fish, this architecture forces a full rewrite rather than a scaled-out instantiation. Specific failure points in the primary pipeline:
- Identity association that assumes a fixed fish count or fixed ordering across cameras
- Triangulation functions that process one fish at a time instead of batching across N fish
- Midline extraction that returns a single skeleton rather than a list of per-fish midlines
- Detection/segmentation that returns one mask per camera rather than N per-fish masks per camera

**How to avoid:**
- Design the pipeline to pass lists of per-fish data structures from the start, even if the list always has length 1 in v1.
- Abstract the detection step to return a list of per-fish masks, and carry fish identity through midline extraction and triangulation.
- Vectorize triangulation across fish (batch dimension) rather than looping in Python.

**Warning signs:**
- Function signatures that take single mask/midline/pose rather than lists
- Global state (camera calibration, background model) mutated during processing
- "Quick" decisions to handle multi-fish "later" without specifying the extension interface

**Phase to address:**
Core architecture design. Interface contracts must support N fish before any implementation begins.

---

### Pitfall 10: Optimizer Applies Gradient Updates to Rotation Representation That Introduces Gimbal Lock or Discontinuities — **[SHELVED PIPELINE]**

> **Shelved pipeline only.** The primary triangulation pipeline does not use an explicit rotation representation — fish orientation is derived from the reconstructed midline spline. Retained for reference if analysis-by-synthesis or optional LM refinement with rotation parameters is revisited.

**What goes wrong:**
Representing fish 3D orientation as Euler angles in PyTorch and applying Adam updates will hit gimbal lock singularities at certain orientations (e.g., fish oriented directly at a camera). More subtly, if the optimizer drives an angle through 180°, the parameter space wraps but gradient descent does not know this — the loss appears to increase (optimizer sees a discontinuity) even as the physically correct orientation passes through.

**How to avoid:**
- Use 6D rotation representation (a pair of 3D vectors) or quaternions (with normalization after each step) rather than Euler angles. PyTorch3D provides `so3_exponential_map` and `axis_angle_to_matrix` utilities.
- For fish swimming in a tank (constrained environment), validate that the chosen rotation parameterization has no singularities within the physically reachable pose space.

**Warning signs:**
- Loss spikes at specific fish orientations that should be smooth
- Gradient norm becomes very large at certain frames
- Optimizer requires much smaller learning rate to be stable

**Phase to address:**
Pose optimizer design phase.

---

### Pitfall 11: Reprojection Error Used as Only Validation Metric — Masks 3D Reconstruction Failures *(both pipelines)*

**What goes wrong:**
Reprojection error (average 2D pixel distance between projected estimate and observed keypoint) is necessary but not sufficient. With 13 top-down cameras and weak Z reconstruction, a system can achieve < 2px reprojection error while having centimeter-scale errors in 3D. This is because the cameras collectively over-constrain XY (good average reprojection) while under-constraining Z — the Z errors are distributed across cameras and partially cancel in the reprojection average.

**How to avoid:**
- Place a physical object with known 3D coordinates in the tank and measure 3D reconstruction accuracy directly (not reprojection). Use this as the primary accuracy metric during calibration validation.
- Report reconstruction error in all three axes separately.
- Use temporal consistency as a proxy metric: frame-to-frame 3D pose velocity should be biologically plausible (fish do not teleport 5 cm between frames at 30fps).

**Warning signs:**
- Reprojection error looks excellent during development but outputs are unusable for biological analysis
- Z-coordinate estimates correlate with tank water level or camera mounting changes (spurious correlations)
- Users of the system report that fish body measurements from 3D poses are inconsistent with manual measurements

**Phase to address:**
Evaluation framework design (must be defined before any results are reported). Define the 3D accuracy metric and the measurement procedure in the geometry validation phase.

---

### Pitfall 12: Arc-Length Correspondence Errors on Curved Fish *(primary pipeline)*

**What goes wrong:**
Arc-length normalization assumes the midline projection preserves parameterization across views. For a significantly curved fish viewed from very different angles, foreshortening compresses the arc-length mapping unevenly. Body point t=0.5 in one camera may not correspond to t=0.5 in another. This creates systematic triangulation errors at body points away from the head/tail endpoints, producing a reconstructed midline with kinks or bulges at high-curvature regions.

**Why it happens:**
A 2D projection of a 3D curve does not preserve arc-length proportions. Two cameras at different azimuthal angles see different foreshortening of the fish body. The further the fish curves out of a flat plane, the worse the mismatch.

**How to avoid:**
- Use RANSAC per body point during triangulation to reject outlier camera contributions where foreshortening is severe.
- Implement view-angle weighting: downweight cameras whose optical axis is nearly parallel to the fish body axis (where foreshortening is worst).
- Consider iterative refinement: triangulate an initial midline, then re-project to update the arc-length correspondence before a second triangulation pass.

**Warning signs:**
- Triangulation residual correlates with fish curvature (straight fish triangulate well, curved fish have high residuals)
- Reconstructed midline has kinks or inflection points at high-curvature regions
- Per-camera reprojection error is systematically higher for cameras looking along the fish body axis

**Phase to address:**
Triangulation (Phase 7).

---

### Pitfall 13: Medial Axis Instability on Noisy Masks *(primary pipeline)*

**What goes wrong:**
`skeletonize` / `medial_axis` on masks with IoU ~0.62 produces unstable, branchy skeletons that wobble frame-to-frame. Mask boundary noise (from imperfect segmentation) creates spurious skeleton branches that survive basic pruning. The resulting midline is too noisy for reliable arc-length parameterization, and temporal jitter in the skeleton translates directly into jittery 3D reconstruction.

**Why it happens:**
Medial axis computation is topologically sensitive to boundary perturbations. A single pixel bump on the mask boundary can create a new skeleton branch. At IoU ~0.62, mask boundaries have significant noise relative to the fish width.

**How to avoid:**
- Apply morphological smoothing (closing then opening) before skeletonization, with adaptive kernel radius proportional to the mask minor-axis width.
- Prune skeleton branches by length threshold relative to the main axis length (discard branches shorter than ~15% of the longest path).
- Validate skeleton stability: for the same fish in consecutive frames, skeleton length should not vary by more than ~10-15%.
- Consider distance-transform-based midline extraction as an alternative to topological skeletonization.

**Warning signs:**
- Skeleton length varies >20% between frames for the same fish
- Many spurious branches remain after pruning
- Arc-length parameterization produces inconsistent body-point positions frame-to-frame

**Phase to address:**
Midline Extraction (Phase 6).

---

### Pitfall 14: Head-Tail Ambiguity in Arc-Length Parameterization *(primary pipeline)*

**What goes wrong:**
Arc-length normalization does not inherently know which end of the skeleton is the head. If head/tail assignment is inconsistent across cameras for the same fish, the correspondence mapping is reversed: body point t=0.2 (near head) in one camera corresponds to t=0.8 (near tail) in another. Triangulation of these mismatched points produces garbage 3D positions — the reconstructed midline collapses or crosses itself.

**Why it happens:**
A skeleton extracted from a 2D mask has two endpoints. Without additional information, either end could be the head. Different camera views may resolve this ambiguity differently (e.g., head is on the left in one camera, on the right in another).

**How to avoid:**
- Project the 3D centroid from the identity/tracking stage into each camera view and assign the skeleton endpoint closer to the centroid-adjacent region as a consistent anchor.
- Use a width heuristic as fallback: the wider end of the fish is typically the head (measure distance-transform width at each skeleton endpoint).
- Validate consistency: after head-tail assignment, verify that the head endpoint is on the same physical side of the fish across all cameras by checking reprojection of the assigned head point.

**Warning signs:**
- Triangulated midline crosses itself or has impossible geometry
- Reconstructed fish length is much shorter than expected (head and tail are being averaged together)
- Head-tail assignment flips between consecutive frames for the same camera

**Phase to address:**
Midline Extraction (Phase 6).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use standard OpenCV calibration without refractive model | Faster calibration implementation | Systematic depth-dependent 3D errors; requires full rework | Never for underwater setups |
| Fixed-iteration Newton-Raphson without convergence check | Differentiable backward pass, simpler code | Silent failures at edge rays produce bad gradients | Only if field of view is verified to exclude near-critical-angle rays |
| Skip mask morphological smoothing before skeletonization | Faster pipeline, fewer parameters | Unstable skeletons, spurious branches, jittery midlines | Only if mask IoU > 0.90 |
| Naive DLT triangulation without RANSAC | Simpler implementation | Outlier cameras corrupt 3D points; no robustness to mask errors | Only for initial prototyping with known-good masks |
| Skip head-tail consistency check across cameras | Fewer heuristics to tune | Reversed arc-length mapping produces garbage triangulation | Never — must validate head-tail consistency |
| Skip female-specific detection validation | Faster integration testing | Silently drops detections for half the experimental fish | Never if females are in the experimental population |
| Report reprojection error as primary metric | Easy to compute, familiar | Masks Z-axis failures completely | Only as a secondary metric alongside 3D reconstruction error |
| Python loop over fish/body-points for triangulation | Easier to debug | O(N*K) Python overhead; unusable at 9 fish × 20 body points × 30fps | Only for single-fish prototyping |
| ~~Euler angle rotation representation~~ | ~~Familiar, easy to debug~~ | ~~Gimbal lock~~ | *Shelved pipeline only* |
| ~~Single global optimizer for all 9 fish~~ | ~~Simpler code initially~~ | ~~One fish destabilizes others~~ | *Shelved pipeline only* |
| ~~Optimize shape and pose jointly from start~~ | ~~One optimization loop~~ | ~~Shape compensates for pose errors~~ | *Shelved pipeline only* |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| scikit-image `skeletonize` | Run on raw mask without smoothing | Apply morphological closing+opening before skeletonization; prune branches by length |
| scipy `splprep` / `splev` | Fit spline to noisy skeleton points without filtering | Pre-filter skeleton points; use smoothing parameter `s > 0`; validate spline length is biologically plausible |
| Crop-to-frame coordinate transforms | Forget to undo crop offset when projecting midline points back to full frame | Maintain crop origin (x0, y0) and apply inverse transform before any cross-camera correspondence |
| OpenCV MOG2 / YOLO detection | Use default parameters in production | Tune for tank-specific conditions; validate on worst-case fish (females, stationary); prefer YOLO for production |
| Snell's law solver (custom) | Test only with central-field rays | Include edge-field rays at 40-48° incidence in the unit test suite |
| Multi-view triangulation | Use linear least squares on all cameras equally | Weight contributions by reprojection confidence; use RANSAC; down-weight edge-of-frame observations |
| ~~PyTorch3D soft rasterizer~~ | ~~Copy default sigma/gamma from tutorials~~ | *Shelved pipeline only* |
| ~~PyTorch3D batched meshes~~ | ~~Use single Meshes object~~ | *Shelved pipeline only* |
| ~~Rotation gradients in PyTorch~~ | ~~Euler angles with direct Adam update~~ | *Shelved pipeline only* |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Python loops over fish/body-points for triangulation | N*K times slower than vectorized; frame rate unusable | Vectorize triangulation: batch across body points and fish using numpy/scipy broadcasting | At > 3 fish × 20 body points per frame |
| Recomputing camera visibility mask from scratch each frame | Redundant computation when camera set is static | Cache the valid-camera set per fish identity; only update when fish moves significantly | At 13 cameras × 9 fish × 30fps |
| Running `skeletonize` on full-resolution masks | Skeletonization is O(pixels); unnecessarily slow on 1080p crops | Downsample mask to working resolution before skeletonization; scale midline points back up | At > 512px crop dimension |
| Storing all camera views in memory simultaneously | OOM error at 13 cameras × 1080p | Downsample images before processing; process cameras in streaming fashion if memory-constrained | At 13 cameras × full resolution |
| Re-extracting masks every frame without caching | Duplicate detection+segmentation work when masks haven't changed | Cache detection/segmentation results; only re-run when frame changes | Any batch processing over video sequences |
| ~~Rendering 9 fish sequentially in Python loop~~ | ~~9x slower~~ | *Shelved pipeline only* |
| ~~Re-computing refractive projection Jacobian numerically~~ | ~~100x slower~~ | *Shelved pipeline only* |

---

## "Looks Done But Isn't" Checklist

- [ ] **Refractive projection:** Validate that reprojection error is consistent across tank depth (near-surface vs. near-floor) — if error varies by depth, the model is not truly depth-aware
- [ ] **Camera calibration:** Place a physical object at 3+ known depths and measure 3D reconstruction accuracy, not just reprojection error
- [ ] **Detection recall:** Measure detection recall separately for (a) males, (b) females, (c) stationary fish, (d) fish near tank edges
- [ ] **Newton-Raphson solver:** Log convergence residuals at iteration 10 for all rays; verify < 0.1px equivalent residual for rays within the valid field of view
- [ ] **Medial axis stability:** For the same fish in consecutive frames, skeleton length should not vary by more than ~10-15%; verify after morphological smoothing
- [ ] **Arc-length correspondence accuracy:** Triangulate a known straight object and verify body-point correspondence is correct; then test on a curved fish and check for kinks
- [ ] **Coordinate transform crop-to-frame:** Verify that midline points extracted from crops are correctly transformed back to full-frame coordinates before cross-camera triangulation
- [ ] **Head-tail consistency:** Verify that head endpoint is assigned to the same physical end of the fish across all cameras for the same frame
- [ ] **Multi-fish extension interface:** Verify that the v1 single-fish code accepts a list-of-length-1 and that no function signature assumes exactly-one fish
- [ ] **Z-axis accuracy:** Report X, Y, Z errors separately on a 3D ground truth target — do not report only aggregate reprojection error

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Depth-independent refraction model | HIGH | Rewrite projection layer, recalibrate cameras, re-run all experiments |
| Top-down Z-weakness not quantified early | MEDIUM | Add Z-regularization; add held-out 3D accuracy metric; may not fully recover without adding oblique cameras |
| Newton-Raphson failure at edge rays | LOW-MEDIUM | Add convergence check and invalid-ray mask; rerun affected frames |
| MOG2 female fish dropout | LOW-MEDIUM | Switch to YOLO detector (already available); validate recall |
| Port tilt unmodeled | MEDIUM | Re-estimate port tilt as calibration parameter; recalibrate; re-run |
| Single-fish architecture can't extend to 9 | HIGH | Requires refactoring core pipeline with batch-first interfaces |
| Arc-length correspondence errors on curved fish | MEDIUM | Add RANSAC per body point; implement view-angle weighting; may need iterative refinement |
| Medial axis instability on noisy masks | LOW-MEDIUM | Add morphological smoothing before skeletonization; tune kernel size; validate stability |
| Head-tail ambiguity | LOW | Add width heuristic + centroid projection for head-tail assignment; reprocess affected frames |
| ~~Silhouette local minima~~ | ~~MEDIUM~~ | *Shelved pipeline only* |
| ~~Euler angle gimbal lock~~ | ~~LOW~~ | *Shelved pipeline only* |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1: Depth-independent refraction model | Phase 1 — Camera model & calibration | Reprojection error flat across Z-depth range; 3D accuracy at 3+ depths |
| P2: Top-down Z-weakness | Phase 1 — Geometry validation (resolved) | Theoretical uncertainty bounds calculated (132x Z/XY anisotropy) |
| P4: Newton-Raphson edge instability | Phase 1 — Refractive projection layer | Residual-at-iteration-10 unit tests pass for all ray angles in FOV |
| P5: MOG2/detection dropout | Phase 2 — Detection module | Per-sex, per-behavior detection recall > 95%; YOLO as primary detector |
| P6: Port tilt unmodeled | Phase 1 — Camera calibration | Directional residual map shows radial (not directional) symmetry |
| P9: Single-fish architecture blocks N-fish | Core architecture design | All functions accept N-fish lists; batch-first reviewed before implementation |
| P11: Reprojection-only validation | Evaluation framework | 3D accuracy metric defined with physical ground truth measurement protocol |
| P12: Arc-length correspondence errors | Phase 7 — Triangulation | RANSAC per body point; residual does not correlate with fish curvature |
| P13: Medial axis instability | Phase 6 — Midline extraction | Skeleton length stable (< 15% variation) across consecutive frames |
| P14: Head-tail ambiguity | Phase 6 — Midline extraction | Head endpoint consistent across all cameras for same fish/frame |
| Identity swap in multi-fish | Multi-fish extension phase | Track continuity > 95% across simulated occlusion events |
| ~~P3: Silhouette local minima~~ | *Shelved pipeline* | — |
| ~~P7: Soft rasterizer hyperparameters~~ | *Shelved pipeline* | — |
| ~~P8: Shape/pose coupling~~ | *Shelved pipeline* | — |
| ~~P10: Rotation gimbal lock~~ | *Shelved pipeline* | — |

---

## Sources

- [A Calibration Tool for Refractive Underwater Vision (arXiv 2405.18018)](https://arxiv.org/html/2405.18018v1) — MEDIUM confidence; peer-reviewed, current (2024)
- [Refractive Two-View Reconstruction for Underwater 3D Vision (IJCV 2019)](https://link.springer.com/article/10.1007/s11263-019-01218-9) — HIGH confidence; established result
- [Analysis of Refraction-Parameter Deviation on Underwater Stereo-Vision (Remote Sensing 2024)](https://www.mdpi.com/2072-4292/16/17/3286) — MEDIUM confidence
- [Adventures with Differentiable Mesh Rendering (Andrew Chan blog)](https://andrewkchan.dev/posts/diff-render.html) — MEDIUM confidence; practical implementation experience
- [PyTorch3D differentiable rendering GitHub issues #1626, #905, #1855](https://github.com/facebookresearch/pytorch3d/issues/1626) — MEDIUM confidence; official issue tracker
- [Multi-animal pose estimation and tracking with DeepLabCut (Nature Methods 2022)](https://www.nature.com/articles/s41592-022-01443-0) — HIGH confidence; peer-reviewed
- [vmTracking enables highly accurate multi-animal pose tracking (PLOS Biology 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11845028/) — HIGH confidence; peer-reviewed, current
- [Expose Camouflage in the Water: Underwater Camouflaged Instance Segmentation (arXiv 2510.17585)](https://arxiv.org/abs/2510.17585) — MEDIUM confidence; preprint 2025
- [WaterMask: Instance Segmentation for Underwater Imagery (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.pdf) — HIGH confidence; peer-reviewed
- [Differentiable Rendering: A Survey (arXiv 2006.12057)](https://arxiv.org/pdf/2006.12057) — HIGH confidence; comprehensive survey
- [Flat port total internal reflection field of view limit (ResearchGate diagram)](https://www.researchgate.net/figure/In-a-flat-port-the-maximum-field-of-view-is-limited-by-the-total-internal-reflection_fig3_289496840) — MEDIUM confidence
- [OpenCV MOG2 background subtraction documentation](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html) — HIGH confidence; official docs

---
*Pitfalls research for: AquaPose — 3D fish pose estimation via direct triangulation (primary) / analysis-by-synthesis (shelved)*
*Researched: 2026-02-19 | Updated: 2026-02-21 (pipeline pivot)*
