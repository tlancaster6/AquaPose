# AquaPose

3D fish pose estimation via refractive multi-view triangulation. Reconstructs fish 3D midlines from multi-view silhouettes using a 13-camera aquarium rig with refractive calibration.

## Quick Start

```bash
pip install hatch
hatch env create
hatch run pre-commit install
hatch run pre-commit install --hook-type pre-push
```

pytorch3d must be installed separately (not on PyPI) — see pyproject.toml comment.

## Commands

```bash
hatch run test              # run unit tests (excludes @slow)
hatch run test-all          # run all tests including slow
hatch run lint              # ruff check
hatch run format            # ruff format
hatch run typecheck         # basedpyright
hatch run check             # lint + typecheck
hatch run docs:build        # sphinx docs
hatch run pre-commit run --all-files  # all pre-commit hooks
```

## Architecture

```
src/aquapose/
├── calibration/    # AquaCal loading, refractive projection, ray casting
├── core/           # Core domain logic (pose state, loss functions)
├── engine/         # PosePipeline, Stage protocol, PipelineContext, config
├── reconstruction/ # RANSAC triangulation, B-spline midline fitting
├── segmentation/   # MOG2/YOLO detection, SAM pseudo-labels, U-Net inference
├── tracking/       # FishTracker, Hungarian matching, cross-view association
├── io/             # HDF5 output, data loaders
└── visualization/  # Reprojection overlays, diagnostic video writers
tests/
├── unit/           # Per-module unit tests
├── integration/    # Cross-module tests
└── e2e/            # End-to-end pipeline tests
```

## Common Pitfalls

- **CUDA tensors → numpy**: Always use `.cpu().numpy()`, never bare `.numpy()`. Projection and model functions may return CUDA tensors.
- **Running tests**: Always use `hatch run test` (which excludes `@slow` and `@e2e`). Never use `pytest` directly with `-k` for quick confirmation — it bypasses the slow-test exclusion and can pick up GPU/data-dependent tests that take minutes.

## Domain Conventions

- **Refractive projection**: 3D-to-pixel through Snell's law at air-water interface (flat surface, no glass)
- **Fish state vector**: `{p, ψ, κ, s}` — position, heading, curvature, scale
- **Direct triangulation**: Medial axis extraction → arc-length sampling → RANSAC triangulation → spline fitting
- **Cross-view identity**: RANSAC centroid clustering to associate fish across cameras before reconstruction
- AquaCal is the calibration dependency; AquaMVS is reference code only (not imported)

## Agent-Specific Instructions

### discuss-phase
When running `gsd:discuss-phase`, before doing anything else:
1. Inform the user: "I noticed project instructions to read refactoring context documents — reading them now."
2. Read this documents and incorporate its content as context for the discussion: .planning/GUIDEBOOK.md
3. Then proceed with the normal discuss-phase workflow.
