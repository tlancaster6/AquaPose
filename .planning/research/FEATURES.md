# Feature Research

**Domain:** Multi-view 3D fish pose estimation — direct triangulation research system (with shelved analysis-by-synthesis alternative)
**Researched:** 2026-02-21
**Confidence:** MEDIUM (ecosystem well-surveyed; AquaPose's specific combination of refractive ray model + 3D midline triangulation is novel and has no direct comparators)

---

## Context: Who Are the "Users"?

AquaPose is a research tool, not a product. "Users" are:

1. **The researchers themselves** — need to iterate on reconstruction quality, validate results, run full-day recordings
2. **The downstream behavioral biology pipeline** — needs reliable 3D trajectories, identity assignments, behavioral features
3. **Other researchers/reviewers** — need reproducibility, clear validation, publishable outputs

"Table stakes" in this context means: missing this makes the system scientifically invalid or operationally unusable. "Differentiators" are capabilities that separate AquaPose from existing tools (DLC+Anipose, DANNCE, SLEAP+3D) and constitute the novel research contribution.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that must work or the system is incomplete as a research tool.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Multi-view silhouette extraction | Core input to reconstruction; all multi-view systems require segmentation masks as primary signal | MEDIUM | YOLO detection + U-Net segmentation pipeline; must produce clean binary masks per camera per frame |
| Camera calibration (intrinsics + extrinsics) | Without calibrated geometry, 3D reconstruction is geometrically meaningless | MEDIUM | Already achieved sub-mm; refractive calibration (air-water interface, no glass) is non-standard and must be encoded in the ray model |
| 3D midline triangulation from multi-view medial axes | The core reconstruction mechanism — extract 2D medial axes from silhouettes, establish arc-length correspondence, triangulate matched points across views using refractive rays | HIGH | Must correctly model refraction; standard triangulation assumes pinhole cameras in air; refractive ray casting is a custom requirement |
| Parametric fish mesh model (shelved pipeline) / 3D midline spline (primary pipeline) | Primary: 3D midline spline with width profile defines the reconstructed body shape. Shelved: differentiable mesh for analysis-by-synthesis if triangulation proves insufficient | HIGH | Primary pipeline uses midline spline + cross-section width profile; encodes biological shape constraints (fineness ratio, cross-section tapering) |
| Per-fish 3D midline reconstruction via triangulation | The v1 deliverable — reconstructing one fish's 3D midline per frame | HIGH | Includes 3D position, orientation, body curvature, and width profile |
| Cross-view holdout validation | Scientifically required to demonstrate reconstruction is genuine, not overfit to available views | MEDIUM | Withhold N cameras during fitting; evaluate reprojection on held-out views |
| Per-frame pose parameters | Output must be time-series, not just per-clip aggregate | MEDIUM | Position (3D centroid), orientation (3 DOF), body curvature per frame at 30fps |
| Reprojection error metric | Standard quantitative measure for reconstruction quality | LOW | IoU of reprojected 3D midline silhouette vs. observed silhouette, per camera per frame |
| Video I/O and frame synchronization | 13 cameras at 30fps must be read in sync; frame-dropping or desync corrupts reconstruction | MEDIUM | Confirmed synchronized capture; need reliable multi-stream reader |
| Configurable pipeline (per-clip runs) | Researchers need to run different clips, adjust parameters, restart without manual intervention | LOW | Config file or CLI; not a GUI requirement |
| Output trajectory storage | 3D pose time-series must be storable and loadable for downstream analysis | LOW | HDF5 or CSV; standard for this domain (NWB, SLEAP, DLC all use HDF5) |

### Differentiators (Competitive Advantage)

Features that no existing tool provides and that constitute AquaPose's research contribution.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Refractive multi-view triangulation | Correctly models air-water refraction without a glass port — existing tools (Anipose, DLC, DANNCE) assume pinhole cameras in air; this is physically accurate for the rig | HIGH | Core novelty. Requires ray-bending at water surface integrated into the triangulation path. No off-the-shelf solution exists for this exact geometry |
| 3D midline spline + width profile (continuous body shape, not keypoints) | Recovers full 3D body shape including lateral bend, dorsal/ventral profile, and volume — keypoint-only systems (DLC, SLEAP, Anipose) cannot recover body shape | HIGH | Enables morphometric measurements (body length, girth, curvature) that are scientifically valuable for cichlid behavior and sexual dimorphism studies |
| Cross-view identity via RANSAC centroid clustering + Hungarian 3D tracking | Clusters 2D detections across views into per-fish identities using RANSAC on refractive-projected centroids, then tracks across frames via Hungarian assignment on 3D positions | HIGH | Enables multi-fish reconstruction without appearance-based Re-ID; leverages the calibrated refractive geometry directly |
| Full-day continuous tracking | Processing 5-30 min clips is standard; continuous tracking over hours with identity persistence is not | HIGH | Requires: efficient per-frame warm-starting from previous frame's solution, occlusion handling, identity recovery |
| Behavioral feature extraction from 3D body state | Computing ethologically meaningful features (tail-beat frequency, body curvature, approach angle, distance between fish) from the 3D midline — existing tools provide only keypoints | MEDIUM | Downstream of 3D reconstruction; enables: dominance behavior detection, spawning behavior, aggression bouts in cichlids |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem useful but should be deliberately excluded from AquaPose's scope.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| General-purpose animal pose estimation (SMAL-based) | SMAL works for quadrupeds; seems extensible | Fish are not quadrupeds — they have no limbs, bilateral symmetry, and the relevant DOF is lateral body curvature, not joint angles. SMAL fitting is the wrong model. Using SMAL would inherit quadruped topology and require extensive surgery with no benefit. | Keep the custom midline spline + cross-section mesh; it is better matched to fish biomechanics |
| Real-time processing | Useful for live behavioral monitoring | Batch offline only; real-time is not a research requirement. The new triangulation pipeline is much faster than analysis-by-synthesis, but real-time is still not a goal — correctness and validation come first. | Batch-process offline; if real-time is eventually needed, optimize the pipeline or train a regression network as a separate project |
| GUI annotation tool | Researchers may want to manually fix bad frames | Building a GUI is a major engineering investment orthogonal to the reconstruction problem. Annotation is better done in existing tools (CVAT, Labelbox). | Export frames with overlaid reconstruction; use external tools for any manual correction |
| Monocular (single-camera) reconstruction | Seems like a simpler starting point | Monocular fish reconstruction is fundamentally ill-posed: scale, depth, and lateral bend are ambiguous from one view. The 13-camera rig exists precisely to avoid this. Building monocular support first would bias the architecture away from multi-view. | Start multi-view; the geometry is better constrained and scientifically more defensible |
| Appearance-based Re-ID (texture features) | Standard approach in fish tracking literature | Cichlids are visually similar within sex; texture-based Re-ID fails under view changes, occlusion, and color variation. The research bet is that shape-based Re-ID (from the parametric mesh) is more robust. Building appearance Re-ID as a fallback would create two competing identity systems. | Commit to shape-signature identity; validate cross-view holdout before adding any appearance component |
| 2D behavioral analysis (DLC-style keypoints) | Simpler to implement; huge ecosystem of downstream tools | This would replicate what DLC+Anipose already does well. AquaPose's value is 3D body shape, not 2D keypoints. | Output 3D trajectories that can feed into SimBA/DeepOF if 2D-equivalent behavioral analysis is needed |
| Cloud/multi-user infrastructure | Useful for lab sharing | AquaPose is a single-lab research tool processing a fixed rig's data. Building authentication, storage abstraction, and multi-tenancy adds weeks of engineering with zero scientific value. | Single-machine deployment; share results as files |

---

## Feature Dependencies

```
[Camera Calibration (refractive)]
    └──required by──> [Refractive Ray Casting]
                          └──required by──> [Cross-View Identity (RANSAC centroid clustering)]
                          └──required by──> [3D Midline Triangulation]

[Multi-View Silhouette Extraction]
    └──required by──> [Medial Axis + Arc-Length Correspondence]
    └──required by──> [Cross-View Holdout Validation]
                          └──required by──> [Cross-View Identity]

[Medial Axis + Arc-Length Correspondence]
    └──required by──> [3D Midline Triangulation (RANSAC)]
                          └──required by──> [3D Spline Fitting + Width Profile]
                                                └──required by──> [Per-Frame 3D Trajectories]
                                                                      └──required by──> [Behavioral Feature Extraction]

[Cross-View Identity (RANSAC centroid clustering)]
    └──required by──> [3D Midline Triangulation] (assigns 2D detections to fish before triangulating)
    └──required by──> [Hungarian 3D Tracking]
                          └──required by──> [Full-Day Continuous Tracking]

[Cross-View Holdout Validation]
    └──gates──> [Multi-Fish Tracking] (must demonstrate v1 works before scaling)
```

### Dependency Notes

- **Refractive ray casting requires calibration:** The refraction model must encode the exact air-water interface geometry from calibration; calibration is not separable from ray casting.
- **Medial axis extraction requires silhouettes:** Clean binary masks are the input to skeletonization and arc-length parameterization.
- **Cross-view identity requires refractive rays:** Clustering 2D centroids into 3D fish identities uses refractive back-projection; without the ray model, centroid clustering is geometrically meaningless.
- **Triangulation requires both arc-length correspondence and cross-view identity:** You need to know which 2D medial axis belongs to which fish (identity) and which points along those axes correspond (arc-length) before triangulating.
- **Full-day tracking requires single-fish validation:** Attempting multi-fish tracking before single-fish reconstruction is reliable will compound errors. Holdout validation gates this transition.
- **Behavioral features require 3D trajectories:** No shortcut here — you need the 3D pose time-series before you can compute tail-beat frequency, curvature time-series, or inter-fish distances.

---

## MVP Definition

### Launch With (v1) — Single-Fish 3D Midline Reconstruction Validated

Minimum to demonstrate the direct triangulation approach works and is scientifically defensible.

- [ ] Refractive ray model (ray casting through air-water interface) — without this, the core novelty does not exist
- [ ] Medial axis extraction from binary silhouettes — required input to arc-length correspondence
- [ ] Arc-length parameterization of 2D medial axes — establishes cross-view point correspondence without keypoints
- [ ] RANSAC triangulation of matched medial axis points across views — the core 3D reconstruction step
- [ ] 3D midline spline fitting with width profile — produces the final per-frame body shape
- [ ] Multi-view silhouette extraction pipeline (YOLO detection + U-Net segmentation) — produces inputs
- [ ] Cross-view holdout validation with reprojection IoU metric — makes v1 scientifically publishable
- [ ] Per-frame 3D trajectory output (position, orientation, curvature) in HDF5 — enables downstream use

### Add After Validation (v1.x) — Multi-Fish, Identity, and Refinement

Add once v1 reconstruction quality is confirmed.

- [ ] Levenberg-Marquardt refinement of triangulated midline (if needed) — trigger: v1 holdout IoU below threshold
- [ ] Temporal smoothing across frames — trigger: per-frame reconstructions are noisy but correct on average
- [ ] Shape-signature identity (body length, width profile as biometric) — trigger: v1 reconstruction produces consistent shape measurements

### Future Consideration (v2+) — Full-Day and Behavioral Analysis

Defer until multi-fish identity is demonstrated to be reliable.

- [ ] Sex classification from reconstructed shape parameters — research enhancement; requires labeled morphometric data per sex
- [ ] Shape-pose decomposition (separate identity-linked shape from instantaneous pose) — enables richer identity modeling
- [ ] Full-day continuous tracking (hours-long recordings) — requires efficient warm-starting and robust identity; only worth building after identity is proven on short clips
- [ ] Behavioral feature extraction library (tail-beat, curvature, approach angle, inter-fish distance) — depends on reliable 3D trajectories; high scientific value but not required to validate the core method
- [ ] Batch processing infrastructure for full experimental dataset — engineering investment justified only when the method is stable

---

## Feature Prioritization Matrix

| Feature | Research Value | Implementation Cost | Priority |
|---------|---------------|---------------------|----------|
| Refractive ray model | HIGH (core novelty) | HIGH | P1 |
| Medial axis extraction + arc-length correspondence | HIGH (enables triangulation) | MEDIUM | P1 |
| Multi-view silhouette extraction | HIGH (required input) | MEDIUM | P1 |
| RANSAC triangulation of matched points | HIGH (v1 deliverable) | HIGH | P1 |
| 3D midline spline fitting + width profile | HIGH (body shape output) | MEDIUM | P1 |
| Cross-view holdout validation | HIGH (scientific credibility) | LOW | P1 |
| Per-frame trajectory output (HDF5) | HIGH (enables downstream) | LOW | P1 |
| Cross-view identity (RANSAC centroid clustering) | HIGH (enables multi-fish) | MEDIUM | P1 |
| Hungarian 3D tracking across frames | HIGH (multi-fish deliverable) | MEDIUM | P2 |
| LM refinement of triangulated midline | MEDIUM (accuracy boost) | MEDIUM | P2 |
| Temporal smoothing | MEDIUM (trajectory quality) | LOW | P2 |
| Shape-signature identity | HIGH (research novelty) | HIGH | P2 |
| Behavioral feature extraction | MEDIUM (biology value) | MEDIUM | P2 |
| Full-day continuous tracking | MEDIUM (practical value) | HIGH | P3 |
| Shape-pose decomposition | MEDIUM (richer identity) | HIGH | P3 |
| Sex classification from shape | MEDIUM (biological insight) | HIGH | P3 |
| Batch processing infrastructure | LOW (operational) | MEDIUM | P3 |

**Priority key:**
- P1: Must have for v1 launch (single-fish reconstruction paper)
- P2: Should have for v1.x (multi-fish tracking, refinement)
- P3: Future work / v2+

---

## Competitor Feature Analysis

| Feature | DLC + Anipose | DANNCE | SLEAP + 3D | AquaPose (planned) |
|---------|--------------|--------|-----------|-------------------|
| Multi-view 3D reconstruction | Yes (triangulation) | Yes (3D CNN) | Partial | Yes (direct triangulation) |
| Refractive optics modeling | No | No | No | Yes (core novelty) |
| Full 3D body shape (not just keypoints) | No | No | No | Yes (midline spline + width profile) |
| Cross-view identity (geometry-based) | No | No | No | Yes (RANSAC centroid clustering) |
| Fish-specific body model | No | No | No | Yes (midline + cross-sections) |
| Underwater environment support | Not natively | Not natively | Not natively | Yes |
| Multi-animal tracking | Yes | Yes | Yes | v1.x (Hungarian 3D tracking) |
| Behavioral analysis downstream | Via SimBA/DeepOF | Manual | Via SimBA | Direct from 3D midline |
| Open-source | Yes | Yes | Yes | Yes (planned) |
| Camera calibration | Standard pinhole | Standard pinhole | Standard pinhole | Refractive |
| Real-time | No (DLC can be) | No | Partial | No (by design) |

**Key insight from competitor analysis:** No existing tool handles refractive optics, full fish body shape, or geometry-based cross-view identity. AquaPose is genuinely novel on these three axes. The table stakes (calibration, segmentation, 3D output, validation) are well-understood from competitors; the differentiators are in the refractive ray model and continuous body reconstruction.

---

## Sources

- [Multi-animal pose estimation, identification and tracking with DeepLabCut — Nature Methods](https://www.nature.com/articles/s41592-022-01443-0) — MEDIUM confidence (2022, well-cited)
- [SLEAP: A deep learning system for multi-animal pose tracking — Nature Methods](https://www.nature.com/articles/s41592-022-01426-1) — MEDIUM confidence (2022)
- [Anipose: A toolkit for robust markerless 3D pose estimation — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8498918/) — MEDIUM confidence (multi-view triangulation approach)
- [Anipose documentation](https://anipose.readthedocs.io/) — MEDIUM confidence (official docs)
- [Current Opinion on Animal Pose Estimation Tools — Snawar Hussain](https://snawarhussain.com/Current-Opinion-on-Animal-Pose-Estimation-Tools-A-Review/) — LOW confidence (blog/review, not peer-reviewed)
- [Take good care of your fish: fish re-identification with synchronized multi-view camera system — Frontiers in Marine Science 2024](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1429459/full) — MEDIUM confidence (peer-reviewed, relevant domain)
- [Feature point based 3D tracking of multiple fish from multi-view images — PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180254) — MEDIUM confidence (established prior work on multi-view fish tracking)
- [A Multi-Fish Tracking and Behavior Modeling Framework — MDPI Sensors 2025](https://www.mdpi.com/1424-8220/26/1/256) — MEDIUM confidence (recent, relevant)
- [SMALify: 3D animal reconstruction from monocular image/video — GitHub](https://github.com/benjiebob/SMALify) — MEDIUM confidence (establishes parametric mesh fitting approach for animals)
- [VoGE: A Differentiable Volume Renderer using Gaussian Ellipsoids for Analysis-by-Synthesis — OpenReview](https://openreview.net/forum?id=AdPJb9cud_Y) — MEDIUM confidence (establishes analysis-by-synthesis with differentiable rendering)
- [A Calibration Tool for Refractive Underwater Vision — arXiv 2024](https://arxiv.org/abs/2405.18018) — MEDIUM confidence (establishes refractive calibration as non-trivial, custom problem)
- [Fish Tracking, Counting, and Behaviour Analysis in Digital Aquaculture — Reviews in Aquaculture 2025](https://onlinelibrary.wiley.com/doi/10.1111/raq.13001) — MEDIUM confidence (comprehensive survey of the domain)

---
*Feature research for: AquaPose — 3D fish pose estimation via direct triangulation (analysis-by-synthesis shelved)*
*Researched: 2026-02-21*
