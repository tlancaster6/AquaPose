# Phase 23: Refractive Lookup Tables - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Build and persist forward (pixel->ray) and inverse (voxel->pixel) lookup tables for all cameras, eliminating per-frame refraction math during association. Forward LUT maps every pixel coordinate to a 3D refracted ray via bilinear interpolation. Inverse LUT discretizes the tank volume into a voxel grid recording per-voxel camera visibility masks and projected pixel coordinates, producing a camera overlap graph and ghost-point lookup.

LUT generation and CLI subcommands are separate from the association algorithm (Phase 25) and pipeline wiring (Phase 26).

</domain>

<decisions>
## Implementation Decisions

### Generation workflow
- Auto-generate on first pipeline run when LUTs are missing (no separate CLI subcommand needed)
- Generation happens during pre-pipeline materialization (alongside calibration and frame source loading, before any stage runs) — consistent with GUIDEBOOK section 5
- Forward and inverse LUTs are always generated together as a single operation
- Per-camera progress logging during generation (e.g., "Generating LUT for camera e3v8122... done (1.2s)")

### Tank volume definition
- Cylindrical tank geometry only (rectangular-prism deferred — see Deferred Ideas)
- Tank dimensions (diameter, height) specified in YAML pipeline config
- Tank center derived from camera position centroid (cx, cy of all camera positions), NOT assumed to be at calibration origin (reference camera is off-center)
- Water surface height (water_z) read from AquaCal calibration data — single source of truth
- Voxel resolution configurable in config (default 2 cm, e.g., `lut.voxel_resolution_m: 0.02`)
- LUT volume extends 10% beyond configured tank dimensions in all directions (configurable margin) to handle calibration drift and measurement error
- TODO: greater flexibility for tank center specification in future (config override, manual XYZ)

### Storage and caching
- LUT files stored alongside the calibration file (e.g., `calibration_dir/luts/`)
- Forward LUT: one file per camera (e.g., `luts/e3v8122_forward.npz`)
- Inverse LUT: one shared file (single voxel grid for the whole tank)
- Hash-based cache invalidation: store hash of calibration file + LUT config parameters in LUT metadata; if hash mismatches on load, regenerate automatically

### Validation and diagnostics
- Auto-validate accuracy after generation: sample random points and compare LUT lookups vs on-the-fly AquaCal refractive projection; report max/mean error
- Abort with error if validation exceeds floating-point tolerance — do not save or use bad LUTs
- Print camera overlap coverage histogram after inverse LUT generation: percent of volume covered by 1+, 2+, 3+, 4+, etc. cameras
- Report total memory footprint of loaded LUTs (forward + inverse) in generation summary

### Claude's Discretion
- LUT serialization format (numpy .npz, HDF5, etc.)
- Forward LUT grid density per camera
- Exact validation tolerance and sample count
- Ghost-point lookup data structure design
- Bilinear interpolation implementation details

</decisions>

<specifics>
## Specific Ideas

- Tank center is NOT at (0,0,water_z) because the reference camera is off-center from the tank; using camera position centroid is a practical approximation
- The 10% margin is a safety net — extra voxels outside the physical tank are harmless (valid visibility data, no fish there), but missing voxels at the edge would break association
- Overlap coverage histogram (1+, 2+, 3+ cameras) gives a quick sanity check of rig geometry quality

</specifics>

<deferred>
## Deferred Ideas

- Rectangular-prism tank geometry support — future TODO
- Configurable tank center override (manual XYZ specification) — future TODO
- CLI subcommand for explicit LUT generation (`aquapose lut generate`) — not needed now since auto-generation covers the use case

</deferred>

---

*Phase: 23-refractive-lookup-tables*
*Context gathered: 2026-02-27*
