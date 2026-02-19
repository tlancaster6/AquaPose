# AquaPose

3D fish pose estimation via analysis-by-synthesis. Fits a parametric fish mesh to multi-view silhouettes from a 13-camera aquarium rig using differentiable refractive rendering.

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
├── segmentation/   # MOG2 detection, SAM pseudo-labels, Mask R-CNN inference
├── mesh/           # Parametric fish mesh, cross-section profiles
├── optimization/   # Analysis-by-synthesis pose optimizer
├── io/             # HDF5 output, data loaders
└── utils/          # General-purpose helpers
tests/
├── unit/           # Per-module unit tests
├── integration/    # Cross-module tests
└── e2e/            # End-to-end pipeline tests
```

## Domain Conventions

- **Refractive projection**: 3D-to-pixel through Snell's law at air-water interface (flat surface, no glass)
- **Fish state vector**: `{p, ψ, κ, s}` — position, heading, curvature, scale
- **Analysis-by-synthesis**: Render silhouette from mesh, compare to observed mask, optimize via gradient descent
- **Cross-view holdout**: Validate by fitting on N-1 cameras, measuring IoU on held-out camera
- AquaCal is the calibration dependency; AquaMVS is reference code only (not imported)
