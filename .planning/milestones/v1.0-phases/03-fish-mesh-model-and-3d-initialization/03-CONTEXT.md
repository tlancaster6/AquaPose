# Phase 3: Fish Mesh Model and 3D Initialization - Context

**Gathered:** 2026-02-19
**Status:** Ready for planning

<domain>
## Phase Boundary

A fully differentiable parametric fish mesh model that can be posed via a state vector, plus a cold-start initializer that triangulates coarse keypoints (derived from binary masks) into a 3D state estimate. The mesh must be compatible with PyTorch3D for rasterization in Phase 4. PyTorch3D should be installed early in this phase to validate mesh output.

</domain>

<decisions>
## Implementation Decisions

### Mesh geometry
- Swept cross-sections along a spine curve (not deformable template)
- Body trunk only — no fins modeled; fin silhouette contribution treated as noise in Phase 4 loss
- 5-8 cross-sections along the spine (coarse resolution)
- Cross-sections are left-right symmetric about their local normal plane, but overall mesh is asymmetric when the spine is curved

### Cross-section profiles
- Elliptical cross-sections — each defined by height and width (2 parameters per section)
- Free cross-section mode: both height and width are optimizable per section
- Hardcoded default profile ships with the mesh (cichlid-like: tapered head, wider mid-body, narrow tail) as starting point for optimization
- 8-12 vertices around each ellipse to generate mesh surface

### State vector & posing
- State vector: {p, ψ/θ, κ, s} — position (3D), heading as yaw + pitch (2 angles), curvature (single constant arc), scale (uniform multiplier)
- Curvature (κ): single constant arc — entire spine bends as a circular arc (1 parameter)
- Heading: yaw (ψ) + pitch (θ) — allows fish to tilt nose-up/down, not just rotate in XY plane
- Scale (s): uniform scale factor on a unit-length template — s=0.15 means a 15cm fish; cross-section positions defined as fractions along [0, 1]

### Keypoint initialization
- Keypoints derived from binary mask pixels (not manual clicks, not a learned detector):
  1. Get binary mask pixel coordinates as Nx2 array
  2. Centroid (mean) = center keypoint
  3. PCA on coordinates → first component = major axis direction
  4. Project all pixels onto first component → min/max = two endpoint keypoints
- Head vs tail disambiguation deferred to downstream (Phase 4's 2-start forward/180° flip)
- Minimum 3 cameras required for triangulation (matching success criterion)
- Triangulation uses refractive ray casting from Phase 1 (not pinhole approximation)
- Testing uses synthetic masks at known positions (Phase 2 not required)

### Claude's Discretion
- Exact cross-section positions along the spine (spacing strategy)
- Default cichlid profile dimensions (height/width ratios at each section)
- Triangulation algorithm (DLT, midpoint, etc.)
- PyTorch3D mesh format specifics and conversion details
- Watertight mesh closure at head and tail tips

</decisions>

<specifics>
## Specific Ideas

- Z-uncertainty is ~15x over most of the tank (corrected from earlier 132x figure — bug was found)
- Phase 4 optimizer weight for Z loss should reflect the ~15x anisotropy (updated from earlier guidance)
- PyTorch3D must be installed early in this phase to validate mesh output against rasterizer
- The PCA-based keypoint extraction is specifically designed to work with any binary mask shape — no assumption about mask quality

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-fish-mesh-model-and-3d-initialization*
*Context gathered: 2026-02-19*
