# Pitfalls Research

**Domain:** 3D fish pose estimation via analysis-by-synthesis (multi-view, refractive, differentiable rendering)
**Researched:** 2026-02-19
**Confidence:** MEDIUM — core optics/math pitfalls are HIGH confidence from literature; multi-fish extension pitfalls are MEDIUM from analogous animal tracking work; some implementation specifics are LOW and flagged

---

## Critical Pitfalls

### Pitfall 1: Treating Refractive Distortion as Depth-Independent

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
Camera model and calibration phase (before any pose optimization is attempted). Validate with a 3D target at 3+ known depths before proceeding.

---

### Pitfall 2: All-Top-Down Camera Configuration Creates Pathologically Weak Z-Reconstruction

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
Geometry validation phase (immediately after camera calibration, before building the pose optimizer). Establish a Z-accuracy budget before committing to the optimization approach.

---

### Pitfall 3: Silhouette-Only Fitting Converges to Wrong Local Minimum

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

### Pitfall 4: Newton-Raphson Fixed-Iteration Solver Fails Near Flat Port Edges (Grazing Angles)

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

### Pitfall 5: MOG2 Background Subtraction Fails on Female Fish (Low-Contrast, Similar Coloring to Background)

**What goes wrong:**
Female zebrafish (or similarly colored species) have lower visual contrast against the tank substrate than males. MOG2 models each pixel as a mixture of Gaussians for the background. When foreground fish have pixel intensities within 2-3 standard deviations of the background model, MOG2 incorrectly absorbs them into the background. The fish becomes invisible to detection. Additionally:
- Cast shadows from top-down lighting appear as separate foreground blobs (value=127 in MOG2), creating ghost detections
- Slow-moving or stationary fish are absorbed into the background model within seconds (MOG2 history parameter effect)
- Fish that school together create merged blobs that a simple instance-count heuristic misinterprets as one fish

**Why it happens:**
MOG2 was designed for pedestrian/vehicle detection in outdoor scenes with high contrast objects. Aquatic settings combine low contrast with dynamic backgrounds (water ripple, lighting flicker), which confuses the Gaussian mixture update.

**How to avoid:**
- Enable MOG2's shadow detection (`detectShadows=True`) and threshold at value>127 to exclude shadows before extracting contours.
- Tune `history` (frames in background model) explicitly — default 500 frames may be too long or too short depending on frame rate. For stationary fish, reduce history.
- Add frame-differencing as a fallback: when MOG2 foreground area drops below expected total fish area, cross-check against an inter-frame difference mask.
- For female fish specifically, consider running detection in a color space where the fish-background contrast is higher (e.g., saturation channel if females have color variance the background does not).
- Validate detection recall separately for male vs. female fish before deploying the full pipeline.

**Warning signs:**
- Detection count drops below expected fish count during certain tank lighting conditions
- Ghost blobs appear where fish shadows fall, not where fish bodies are
- "Stationary" fish disappear from detection when they stop moving for > N seconds

**Phase to address:**
Detection module phase. Requires controlled experiments varying illumination and fish sex before integration testing.

---

### Pitfall 6: Flat Refractive Port Normal Assumed Perfect — Tilt Creates Unmodeled Asymmetric Distortion

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

### Pitfall 7: Soft Rasterizer Hyperparameters Require Per-Setup Tuning

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

### Pitfall 8: Cross-Section Profile Self-Calibration Overfits to Individual Fish Variance

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

### Pitfall 9: Single-Fish Architecture Does Not Isolate State Per Fish — Prevents Clean 9-Fish Extension

**What goes wrong:**
V1 builds a single-fish pipeline with global state: one background model, one optimizer, one set of calibration results. When extending to 9 fish, this architecture forces a full rewrite rather than a scaled-out instantiation. Specific failure points:
- Background subtraction that returns one mask cannot cleanly support 9 tracked instances
- A single Newton-Raphson projection layer with fixed camera parameters cannot handle per-fish optimization in parallel
- A single mesh object in the renderer cannot render 9 fish simultaneously with independent pose gradients

**How to avoid:**
- Design the single-fish pipeline as a parameterized `Fish` class from day one: each instance has its own pose parameters, optimizer state, and render buffer.
- Use PyTorch3D's batched mesh operations (`join_meshes_as_batch`) rather than single-mesh APIs — this supports N=1 and N=9 with the same code path.
- Abstract the detection step to return a list of per-fish masks from the start, even if the list always has length 1 in v1.

**Warning signs:**
- Function signatures that take single mask/pose rather than lists
- Global state (camera calibration, background model) mutated by pose optimization
- "Quick" decisions to handle multi-fish "later" without specifying the extension interface

**Phase to address:**
Core architecture design (phase 1). Interface contracts must support N fish before any implementation begins.

---

### Pitfall 10: Optimizer Applies Gradient Updates to Rotation Representation That Introduces Gimbal Lock or Discontinuities

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

### Pitfall 11: Reprojection Error Used as Only Validation Metric — Masks 3D Reconstruction Failures

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

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use standard OpenCV calibration without refractive model | Faster calibration implementation | Systematic depth-dependent 3D errors; requires full rework | Never for underwater setups |
| Fixed-iteration Newton-Raphson without convergence check | Differentiable backward pass, simpler code | Silent failures at edge rays produce bad gradients | Only if field of view is verified to exclude near-critical-angle rays |
| Euler angle rotation representation | Familiar, easy to debug | Gimbal lock at certain fish orientations; requires restart or workaround | Only for rapid prototyping if fish orientation is constrained |
| Single global optimizer for all 9 fish | Simpler code initially | Can't tune per-fish learning rates; one fish destabilizes others | Never — parameterize from day 1 |
| Skip female-specific detection validation | Faster integration testing | Silently drops detections for half the experimental fish | Never if females are in the experimental population |
| Report reprojection error as primary metric | Easy to compute, familiar | Masks Z-axis failures completely | Only as a secondary metric alongside 3D reconstruction error |
| Optimize shape and pose jointly from start | One optimization loop | Shape adapts to compensate for pose errors; both are wrong | Acceptable only for a fixed, known fish with no shape uncertainty |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PyTorch3D soft rasterizer | Copy default sigma/gamma from tutorials | Profile gradient magnitude for your specific fish pixel size and set sigma accordingly |
| PyTorch3D batched meshes | Use single Meshes object, extend to N fish by rewriting | Use `join_meshes_as_batch` from day one; indexing into a batch is clean extension |
| OpenCV MOG2 | Use default parameters in production | Tune `history`, `varThreshold`, `detectShadows` for tank-specific conditions; validate on worst-case fish (females, stationary) |
| Snell's law solver (custom) | Test only with central-field rays | Include edge-field rays at 40-48° incidence in the unit test suite |
| Multi-view triangulation | Use linear least squares on all cameras equally | Weight contributions by reprojection confidence; down-weight edge-of-frame observations where refractive solver is less reliable |
| Rotation gradients in PyTorch | `torch.Tensor` with Euler angles, direct Adam update | Use `so3_exponential_map` or normalized quaternion representation with custom update step |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Rendering 9 fish sequentially in Python loop | 9x slower than expected; frame rate drops | Use PyTorch3D batched rendering over all fish in one forward pass | At > 3 fish with any realtime requirement |
| Re-computing refractive projection Jacobian numerically | Optimization is 100x slower than expected | Implement analytic Jacobian or use autograd from the start | At every iteration for any mesh with > 100 faces |
| Storing all camera views in GPU memory simultaneously | OOM error at 13 cameras × 1080p | Downsample images before passing to renderer; use half-precision for render buffers | At 13 cameras × full resolution |
| Background model update during optimization | Model absorbs fish into background during long optimization runs | Freeze MOG2 model during pose optimization; update only between inference steps | Any continuous optimization over > 10 seconds of video |
| Using PyTorch autograd through 10 Newton-Raphson iterations without `torch.no_grad` on inner variables | Computation graph grows exponentially; memory explosion | Explicitly manage which variables in the solver carry gradients | At > 5 iterations and > 100 projected points |

---

## "Looks Done But Isn't" Checklist

- [ ] **Refractive projection:** Validate that reprojection error is consistent across tank depth (near-surface vs. near-floor) — if error varies by depth, the model is not truly depth-aware
- [ ] **Camera calibration:** Place a physical object at 3+ known depths and measure 3D reconstruction accuracy, not just reprojection error
- [ ] **Detection recall:** Measure detection recall separately for (a) males, (b) females, (c) stationary fish, (d) fish near tank edges
- [ ] **Newton-Raphson solver:** Log convergence residuals at iteration 10 for all rays; verify < 0.1px equivalent residual for rays within the valid field of view
- [ ] **Rotation parameterization:** Test optimizer stability with fish at 90°, 180°, 270° orientation angles (gimbal lock test)
- [ ] **Multi-fish extension interface:** Verify that the v1 single-fish code accepts a list-of-length-1 and that no function signature assumes exactly-one fish
- [ ] **Shape/pose coupling:** Hold out one fish from shape calibration and verify that the calibrated shape fits that fish without reoptimizing shape
- [ ] **Z-axis accuracy:** Report X, Y, Z errors separately on a 3D ground truth target — do not report only aggregate reprojection error

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Depth-independent refraction model | HIGH | Rewrite projection layer, recalibrate cameras, re-run all experiments |
| Top-down Z-weakness not quantified early | MEDIUM | Add Z-regularization to optimizer; add held-out 3D accuracy metric; may not fully recover without adding oblique cameras |
| Silhouette local minima without multi-start | MEDIUM | Add multi-start wrapper around optimizer; re-run affected sequences |
| Newton-Raphson failure at edge rays | LOW-MEDIUM | Add convergence check and invalid-ray mask; rerun affected frames |
| MOG2 female fish dropout | MEDIUM | Tune detection parameters; may need to add secondary detector for females specifically |
| Port tilt unmodeled | MEDIUM | Re-estimate port tilt as calibration parameter; recalibrate; re-run |
| Single-fish architecture can't extend to 9 | HIGH | Requires refactoring core pipeline with batch-first interfaces |
| Euler angle gimbal lock | LOW | Switch to 6D/quaternion representation; reset optimizer state for affected sequences |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Depth-independent refraction model | Camera model & calibration phase | Reprojection error flat across Z-depth range; 3D accuracy at 3+ depths |
| Top-down Z-weakness | Geometry validation phase (after calibration, before pose optimization) | Theoretical uncertainty bounds calculated; Z-regularization designed |
| Silhouette local minima | Pose optimizer design phase | Multi-start implemented; frame-flip metric defined |
| Newton-Raphson edge instability | Custom refractive projection layer (earliest phase) | Residual-at-iteration-10 unit tests pass for all ray angles in FOV |
| MOG2 female fish dropout | Detection module phase | Per-sex, per-behavior detection recall > 95% before integration |
| Port tilt unmodeled | Camera calibration phase | Directional residual map shows radial (not directional) symmetry |
| Single-fish architecture blocks v2 | Core architecture design (phase 1) | All functions accept N-fish lists; batch-first reviewed before implementation |
| Rotation gimbal lock | Pose optimizer design phase | Stress test at 4 cardinal orientations; no loss spikes |
| Shape/pose coupling | Shape model calibration phase | Held-out fish shape fit validates generalization |
| Reprojection-only validation | Evaluation framework phase | 3D accuracy metric defined with physical ground truth measurement protocol |
| Soft rasterizer hyperparameters | Differentiable rendering integration phase | Gradient magnitude map shows nonzero signal at silhouette boundary |
| Identity swap in multi-fish | Multi-fish extension phase (v2) | Track continuity > 95% across simulated occlusion events |

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
*Pitfalls research for: AquaPose — 3D fish pose estimation via analysis-by-synthesis*
*Researched: 2026-02-19*
