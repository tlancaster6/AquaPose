# Phase 1: Calibration and Refractive Geometry - Context

**Gathered:** 2026-02-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Differentiable refractive projection layer validated and ready for downstream phases. Loads calibration from AquaCal, adapts AquaMVS's existing PyTorch refractive projection, provides ray casting, and produces an analytical Z-uncertainty characterization. The 1px reprojection criterion is dropped — validation means numerical equivalence with AquaMVS.

</domain>

<decisions>
## Implementation Decisions

### Refractive model scope
- Single flat air-water interface only — no glass panels in the optical path
- All 13 cameras are top-down, looking through the water surface
- Water surface height comes from AquaCal's calibration output (already calibrated, not estimated at runtime)
- Single constant index of refraction (n=1.333) — temperature variation is negligible

### Validation approach
- Drop the "1px reprojection against ground truth" success criterion — that was already validated in AquaMVS
- Validation means: our adapted PyTorch code produces numerically equivalent output to AquaMVS's reference implementation for the same inputs
- AquaMVS is the primary reference (already PyTorch); AquaCal's numpy projection is not used for validation
- AquaMVS is in a separate repo — researcher should examine it to understand the code to adapt

### Z-uncertainty report
- Serves dual purpose: inform downstream optimizer weighting AND document system accuracy for the paper
- Uniform depth sampling across the full tank range (e.g., every 5-10cm)
- Purely analytical — geometric ray intersection calculations from camera geometry and Snell's law, no empirical data
- Output as markdown report with embedded matplotlib/seaborn plots (error vs. depth curves for X, Y, Z separately)

### AquaCal integration
- AquaCal is an importable Python package — use its API to load calibration data
- Provides full camera model: intrinsics (focal length, principal point, distortion), extrinsics (rotation, translation), and water surface plane
- AquaCal is a data loader for this phase — all differentiable projection math comes from adapting AquaMVS's PyTorch code

### Claude's Discretion
- Exact numerical tolerance for "equivalence" with AquaMVS (e.g., 1e-5 or 1e-6)
- Depth sampling resolution for Z-uncertainty report
- Plot styling and report structure details
- How to handle AquaCal's distortion model in the differentiable path

</decisions>

<specifics>
## Specific Ideas

- AquaMVS already has a working PyTorch refractive projection that "just needs light modification" — this is an adapt-and-validate task, not a from-scratch reimplementation
- AquaCal has numpy projection functions too, but we only validate against AquaMVS's PyTorch version

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-calibration-and-refractive-geometry*
*Context gathered: 2026-02-19*
