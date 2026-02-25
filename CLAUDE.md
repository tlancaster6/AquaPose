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
├── segmentation/   # MOG2/YOLO detection, SAM pseudo-labels, U-Net inference
├── mesh/           # Parametric fish mesh, cross-section profiles
├── initialization/ # PCA keypoints, multi-view triangulation, fish state init
├── io/             # HDF5 output, data loaders
└── utils/          # General-purpose helpers
tests/
├── unit/           # Per-module unit tests
├── integration/    # Cross-module tests
└── e2e/            # End-to-end pipeline tests
```

## Common Pitfalls

- **CUDA tensors → numpy**: Always use `.cpu().numpy()`, never bare `.numpy()`. Projection and model functions may return CUDA tensors.

## Domain Conventions

- **Refractive projection**: 3D-to-pixel through Snell's law at air-water interface (flat surface, no glass)
- **Fish state vector**: `{p, ψ, κ, s}` — position, heading, curvature, scale
- **Direct triangulation**: Medial axis extraction → arc-length sampling → RANSAC triangulation → spline fitting
- **Cross-view identity**: RANSAC centroid clustering to associate fish across cameras before reconstruction
- AquaCal is the calibration dependency; AquaMVS is reference code only (not imported)
