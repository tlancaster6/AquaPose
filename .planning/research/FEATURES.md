# Feature Research

**Domain:** Multi-view 3D fish pose estimation — analysis-by-synthesis research system
**Researched:** 2026-02-19
**Confidence:** MEDIUM (ecosystem well-surveyed; AquaPose's specific combination of refractive rendering + parametric fish mesh is novel and has no direct comparators)

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
| Multi-view silhouette extraction | Core input to analysis-by-synthesis; all multi-view systems require segmentation masks as primary signal | MEDIUM | MOG2 + Mask R-CNN pipeline already planned; must produce clean binary masks per camera per frame |
| Camera calibration (intrinsics + extrinsics) | Without calibrated geometry, 3D reconstruction is geometrically meaningless | MEDIUM | Already achieved sub-mm; refractive calibration (air-water interface, no glass) is non-standard and must be encoded in the renderer |
| Differentiable silhouette renderer | The core analysis-by-synthesis mechanism — cannot fit mesh to silhouettes without differentiable gradients | HIGH | Must correctly model refraction; standard renderers (PyTorch3D, nvdiffrast) assume air; this is a custom requirement |
| Parametric fish mesh model | Defines the shape space being optimized; without a plausible mesh prior, optimization is unconstrained | HIGH | Midline spline + cross-sections; needs to encode biological shape constraints (fineness ratio, cross-section tapering) |
| Single-fish 3D pose/shape optimization | The v1 deliverable — fitting one fish per clip | HIGH | Includes position, orientation, body pose (bend), shape (size/morphology) |
| Cross-view holdout validation | Scientifically required to demonstrate reconstruction is genuine, not overfit to available views | MEDIUM | Withhold N cameras during fitting; evaluate reprojection on held-out views |
| Per-frame pose parameters | Output must be time-series, not just per-clip aggregate | MEDIUM | Position (3D centroid), orientation (3 DOF), body curvature per frame at 30fps |
| Reprojection error metric | Standard quantitative measure for any analysis-by-synthesis system | LOW | IoU of projected mesh silhouette vs. observed silhouette, per camera per frame |
| Video I/O and frame synchronization | 13 cameras at 30fps must be read in sync; frame-dropping or desync corrupts reconstruction | MEDIUM | Confirmed synchronized capture; need reliable multi-stream reader |
| Configurable pipeline (per-clip runs) | Researchers need to run different clips, adjust parameters, restart without manual intervention | LOW | Config file or CLI; not a GUI requirement |
| Output trajectory storage | 3D pose time-series must be storable and loadable for downstream analysis | LOW | HDF5 or CSV; standard for this domain (NWB, SLEAP, DLC all use HDF5) |

### Differentiators (Competitive Advantage)

Features that no existing tool provides and that constitute AquaPose's research contribution.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Refractive differentiable rendering | Correctly models air-water refraction without a glass port — existing tools (Anipose, DLC, DANNCE) assume pinhole cameras in air; this is physically accurate for the rig | HIGH | Core novelty. Requires ray-bending at water surface integrated into the render/gradient path. No off-the-shelf solution exists for this exact geometry |
| Parametric fish body mesh (not keypoints) | Recovers full 3D body shape including lateral bend, dorsal/ventral profile, and volume — keypoint-only systems (DLC, SLEAP, Anipose) cannot recover body shape | HIGH | Enables morphometric measurements (body length, girth, curvature) that are scientifically valuable for cichlid behavior and sexual dimorphism studies |
| Shape-pose decomposition for fish | Separates identity-linked shape (body plan, size) from instantaneous pose (position, orientation, bend) — critical for multi-fish identity: each fish has a characteristic shape | HIGH | Enables downstream: "this is fish #3 because it has these shape parameters" |
| Multi-fish identity via shape signatures | Using shape parameters as a biometric identity cue across full-day recordings — no existing fish tracking system does this | HIGH | Depends on shape-pose decomposition; enables re-identification after occlusion without appearance-based Re-ID |
| Full-day continuous tracking | Processing 5-30 min clips is standard; continuous tracking over hours with identity persistence is not | HIGH | Requires: efficient per-frame warm-starting from previous frame's solution, occlusion handling, identity recovery |
| Behavioral feature extraction from 3D body state | Computing ethologically meaningful features (tail-beat frequency, body curvature, approach angle, distance between fish) from the 3D mesh — existing tools provide only keypoints | MEDIUM | Downstream of 3D reconstruction; enables: dominance behavior detection, spawning behavior, aggression bouts in cichlids |
| Sex-differentiated shape model | Rig has 3 males + 6 females; fitting sex-specific shape priors improves reconstruction accuracy and enables sexing-by-shape | HIGH | Research novelty; requires labeled training data or manual annotation of sex-specific shape parameters |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem useful but should be deliberately excluded from AquaPose's scope.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| General-purpose animal pose estimation (SMAL-based) | SMAL works for quadrupeds; seems extensible | Fish are not quadrupeds — they have no limbs, bilateral symmetry, and the relevant DOF is lateral body curvature, not joint angles. SMAL fitting is the wrong model. Using SMAL would inherit quadruped topology and require extensive surgery with no benefit. | Keep the custom midline spline + cross-section mesh; it is better matched to fish biomechanics |
| Real-time processing | Useful for live behavioral monitoring | Analysis-by-synthesis optimization is iterative gradient descent — even with GPU, fitting one frame takes seconds to minutes. Building real-time constraints into v1 creates architectural pressure that conflicts with the optimization approach. | Batch-process offline; if real-time is eventually needed, train a regression network on the fitted results as a separate project |
| GUI annotation tool | Researchers may want to manually fix bad frames | Building a GUI is a major engineering investment orthogonal to the reconstruction problem. Annotation is better done in existing tools (CVAT, Labelbox, Label Studio). | Export frames with overlaid reconstruction; use external tools for any manual correction |
| Monocular (single-camera) reconstruction | Seems like a simpler starting point | Monocular fish reconstruction is fundamentally ill-posed: scale, depth, and lateral bend are ambiguous from one view. The 13-camera rig exists precisely to avoid this. Building monocular support first would bias the architecture away from multi-view. | Start multi-view; the geometry is better constrained and scientifically more defensible |
| Appearance-based Re-ID (texture features) | Standard approach in fish tracking literature | Cichlids are visually similar within sex; texture-based Re-ID fails under view changes, occlusion, and color variation. The research bet is that shape-based Re-ID (from the parametric mesh) is more robust. Building appearance Re-ID as a fallback would create two competing identity systems. | Commit to shape-signature identity; validate cross-view holdout before adding any appearance component |
| 2D behavioral analysis (DLC-style keypoints) | Simpler to implement; huge ecosystem of downstream tools | This would replicate what DLC+Anipose already does well. AquaPose's value is 3D body shape, not 2D keypoints. | Output 3D trajectories that can feed into SimBA/DeepOF if 2D-equivalent behavioral analysis is needed |
| Cloud/multi-user infrastructure | Useful for lab sharing | AquaPose is a single-lab research tool processing a fixed rig's data. Building authentication, storage abstraction, and multi-tenancy adds weeks of engineering with zero scientific value. | Single-machine deployment; share results as files |

---

## Feature Dependencies

```
[Camera Calibration (refractive)]
    └──required by──> [Differentiable Refractive Renderer]
                          └──required by──> [Single-Fish Optimization]
                                                └──required by──> [Per-Frame 3D Trajectories]
                                                                      └──required by──> [Behavioral Feature Extraction]

[Parametric Fish Mesh Model]
    └──required by──> [Single-Fish Optimization]
    └──required by──> [Shape-Pose Decomposition]
                          └──required by──> [Multi-Fish Identity via Shape]
                                                └──required by──> [Full-Day Continuous Tracking]

[Multi-View Silhouette Extraction]
    └──required by──> [Single-Fish Optimization]
    └──required by──> [Cross-View Holdout Validation]

[Single-Fish Optimization]
    └──required by──> [Cross-View Holdout Validation]
    └──required by──> [Multi-Fish Tracking]

[Cross-View Holdout Validation]
    └──gates──> [Multi-Fish Tracking] (must demonstrate v1 works before scaling)
```

### Dependency Notes

- **Refractive renderer requires calibration:** The refraction model must encode the exact air-water interface geometry from calibration; calibration is not separable from rendering.
- **Shape-pose decomposition requires mesh model:** You cannot decompose shape from pose without a parameterized mesh that factorizes these two things explicitly. Keypoint-only systems cannot support identity-by-shape.
- **Full-day tracking requires single-fish validation:** Attempting multi-fish tracking before single-fish reconstruction is reliable will compound errors. Holdout validation gates this transition.
- **Behavioral features require 3D trajectories:** No shortcut here — you need the 3D pose time-series before you can compute tail-beat frequency, curvature time-series, or inter-fish distances.
- **Sex-differentiated shape model enhances but does not block:** This is an enhancement to the mesh model; single-sex fitting works first, then sex-specific priors layer on top.

---

## MVP Definition

### Launch With (v1) — Single-Fish 3D Reconstruction Validated

Minimum to demonstrate the analysis-by-synthesis approach works and is scientifically defensible.

- [ ] Refractive differentiable renderer — without this, the core novelty does not exist
- [ ] Parametric fish mesh (midline spline + cross-sections) with differentiable parameters — required by optimizer
- [ ] Single-fish per-frame pose/shape optimization on 5-30 min clips — the v1 claim
- [ ] Multi-view silhouette extraction pipeline (MOG2 + Mask R-CNN) — produces inputs
- [ ] Cross-view holdout validation with reprojection IoU metric — makes v1 scientifically publishable
- [ ] Per-frame 3D trajectory output (position, orientation, curvature) in HDF5/CSV — enables downstream use

### Add After Validation (v1.x) — Multi-Fish and Identity

Add once v1 reconstruction quality is confirmed.

- [ ] Shape-pose decomposition and shape signature per fish — trigger: v1 holdout IoU meets threshold
- [ ] Multi-fish detection and separate optimization per fish — trigger: single-fish pipeline is stable
- [ ] Identity assignment via shape signature — trigger: shape decomposition demonstrated to be consistent across clips
- [ ] Occlusion handling (warm-starting, identity recovery after fish cross) — trigger: multi-fish baseline works

### Future Consideration (v2+) — Full-Day and Behavioral Analysis

Defer until multi-fish identity is demonstrated to be reliable.

- [ ] Full-day continuous tracking (hours-long recordings) — requires efficient warm-starting and robust identity; only worth building after identity is proven on short clips
- [ ] Behavioral feature extraction library (tail-beat, curvature, approach angle, inter-fish distance) — depends on reliable 3D trajectories; high scientific value but not required to validate the core method
- [ ] Sex-differentiated shape model — research enhancement; requires labeled morphometric data per sex
- [ ] Batch processing infrastructure for full experimental dataset — engineering investment justified only when the method is stable

---

## Feature Prioritization Matrix

| Feature | Research Value | Implementation Cost | Priority |
|---------|---------------|---------------------|----------|
| Refractive differentiable renderer | HIGH (core novelty) | HIGH | P1 |
| Parametric fish mesh model | HIGH (enables all shape features) | HIGH | P1 |
| Multi-view silhouette extraction | HIGH (required input) | MEDIUM | P1 |
| Single-fish optimization | HIGH (v1 deliverable) | HIGH | P1 |
| Cross-view holdout validation | HIGH (scientific credibility) | LOW | P1 |
| Per-frame trajectory output | HIGH (enables downstream) | LOW | P1 |
| Shape-pose decomposition | HIGH (enables identity) | HIGH | P2 |
| Multi-fish tracking | HIGH (v2 deliverable) | HIGH | P2 |
| Identity via shape signatures | HIGH (research novelty) | HIGH | P2 |
| Behavioral feature extraction | MEDIUM (biology value) | MEDIUM | P2 |
| Full-day continuous tracking | MEDIUM (practical value) | HIGH | P3 |
| Sex-differentiated shape model | MEDIUM (biological insight) | HIGH | P3 |
| Batch processing infrastructure | LOW (operational) | MEDIUM | P3 |

**Priority key:**
- P1: Must have for v1 launch (single-fish reconstruction paper)
- P2: Should have for v2 (multi-fish identity paper)
- P3: Future work / v3+

---

## Competitor Feature Analysis

| Feature | DLC + Anipose | DANNCE | SLEAP + 3D | AquaPose (planned) |
|---------|--------------|--------|-----------|-------------------|
| Multi-view 3D reconstruction | Yes (triangulation) | Yes (3D CNN) | Partial | Yes (analysis-by-synthesis) |
| Refractive optics modeling | No | No | No | Yes (core novelty) |
| Full 3D body shape (not just keypoints) | No | No | No | Yes |
| Shape-based identity | No | No | No | Yes (planned) |
| Fish-specific parametric model | No | No | No | Yes |
| Underwater environment support | Not natively | Not natively | Not natively | Yes |
| Multi-animal tracking | Yes | Yes | Yes | v2 |
| Behavioral analysis downstream | Via SimBA/DeepOF | Manual | Via SimBA | Direct from 3D mesh |
| Open-source | Yes | Yes | Yes | Yes (planned) |
| Camera calibration | Standard pinhole | Standard pinhole | Standard pinhole | Refractive |
| Real-time | No (DLC can be) | No | Partial | No (by design) |

**Key insight from competitor analysis:** No existing tool handles refractive optics, full fish body shape, or shape-based identity. AquaPose is genuinely novel on these three axes. The table stakes (calibration, segmentation, 3D output, validation) are well-understood from competitors; the differentiators are in the rendering and shape model.

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
*Feature research for: AquaPose — 3D fish pose estimation via analysis-by-synthesis*
*Researched: 2026-02-19*
