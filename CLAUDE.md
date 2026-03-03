# AquaPose

3D fish pose estimation via refractive multi-view triangulation. Reconstructs fish 3D midlines from multi-view silhouettes using a 13-camera aquarium rig with refractive calibration.

## Quick Start

```bash
pip install hatch
hatch env create
hatch run pre-commit install
hatch run pre-commit install --hook-type pre-push
```

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
├── core/           # Core domain logic (types, stages, backends)
│   ├── types/      # Cross-stage shared types (Detection, CropRegion, Midline2D, etc.)
│   ├── detection/  # Detection stage + YOLO/OBB backends
│   ├── midline/    # Midline extraction, crop utilities, backends (segmentation, pose_estimation)
│   ├── reconstruction/ # Triangulation, curve optimizer, backends
│   ├── tracking/   # 2D tracking stage + OC-SORT
│   └── association/ # Cross-view association
├── engine/         # PosePipeline, Stage protocol, PipelineContext, config
├── io/             # HDF5 output, data loaders
├── training/       # YOLO training wrappers
└── visualization/  # Reprojection overlays, diagnostic video writers
tests/
├── unit/           # Per-module unit tests
├── integration/    # Cross-module tests
└── e2e/            # End-to-end pipeline tests
```

## Common Pitfalls

- **CUDA tensors → numpy**: Always use `.cpu().numpy()`, never bare `.numpy()`. Projection and model functions may return CUDA tensors.
- **Running tests**: Always use `hatch run test` (which excludes `@slow` and `@e2e`). Never use `pytest` directly with `-k` for quick confirmation — it bypasses the slow-test exclusion and can pick up GPU/data-dependent tests that take minutes.
- **Ultralytics OBB corner order**: `obb.xyxyxyxy` returns corners as `[right-bottom, right-top, left-top, left-bottom]` — **not** `[TL, TR, BR, BL]`. The true top-left is `pts[2]`, not `pts[0]`. Training data uses `pca_obb` which returns `[TL, TR, BR, BL]`. When building affine crops from Ultralytics OBB corners, use `src = [pts[2], pts[1], pts[3]]` (TL, TR, BL).
- **Training-inference crop matching**: Both seg and pose models are trained on stretch-filled crops (3-point affine mapping OBB corners to canvas corners). Inference must use the same stretch-fill — never letterbox/scale-to-fit. Letterboxing was removed from the codebase.

## Domain Conventions

- **Refractive projection**: 3D-to-pixel through Snell's law at air-water interface (flat surface, no glass)
- **Fish state vector**: `{p, ψ, κ, s}` — position, heading, curvature, scale
- **Direct triangulation**: Medial axis extraction → arc-length sampling → RANSAC triangulation → spline fitting
- **Cross-view identity**: RANSAC centroid clustering to associate fish across cameras before reconstruction
- AquaCal is the calibration dependency; AquaMVS is reference code only (not imported)

## Agent-Specific Instructions

### discuss-phase
When running `gsd:discuss-phase`, before doing anything else:
1. Inform the user: "I noticed project instructions to read the guidebook — reading it now."
2. Read this document and incorporate its content as context for the discussion: .planning/GUIDEBOOK.md
3. Then proceed with the normal discuss-phase workflow.
