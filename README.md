# AquaPose

3D fish pose estimation via refractive multi-view triangulation. AquaPose reconstructs fish 3D midlines from multi-view video using a 13-camera aquarium rig with refractive calibration, producing dense 3D trajectories and midline kinematics for behavioral research on cichlids.

## Pipeline

AquaPose processes multi-view video through a 5-stage pipeline:

1. **Detection** — YOLO-based fish detection (standard or oriented bounding boxes)
2. **Tracking** — Per-camera 2D temporal tracking via OC-SORT
3. **Association** — Cross-camera tracklet association using ray-ray geometry and Leiden clustering
4. **Midline** — 2D midline extraction via YOLO-seg or YOLO-pose backends
5. **Reconstruction** — DLT triangulation of 2D midlines into 3D B-spline midlines

Long videos are processed in fixed-size temporal chunks with identity continuity across chunk boundaries.

## Quick Start

```bash
# Initialize a project
aquapose init-config my_project

# Run the pipeline
aquapose run --config path/to/config.yaml
```

## Development

```bash
# Set up the development environment
pip install hatch
hatch env create
hatch run pre-commit install
hatch run pre-commit install --hook-type pre-push

# Run tests, lint, and type check
hatch run test
hatch run lint
hatch run typecheck
```

### GPU Support

Hatch installs the CPU-only PyTorch by default. For GPU support, manually install
the CUDA build after creating the environment:

```bash
hatch run pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu124
```

> **Tip:** If you see `nvrtc: error: failed to open libnvrtc-builtins.so` during
> training, your PyTorch CUDA version may not match the bundled NVIDIA library
> layout. Reinstalling with an explicit CUDA 12 index URL (as shown above)
> typically resolves this.

See [Contributing](docs/contributing.md) for full development guidelines.

## Documentation

<!-- TODO: Uncomment once docs are deployed -->
<!-- Full documentation is available at [aquapose.readthedocs.io](https://aquapose.readthedocs.io). -->

## License

[MIT](LICENSE)
