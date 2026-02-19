# AquaPose

3D fish pose estimation via differentiable refractive rendering. AquaPose fits a parametric fish mesh to multi-view silhouettes from a 13-camera aquarium rig, producing dense 3D trajectories and midline kinematics for behavioral research on cichlids.

## Installation

```bash
pip install aquapose
```

## Quick Start

```python
from aquapose.calibration import load_calibration
from aquapose.segmentation import segment_frame
from aquapose.optimization import optimize_pose

# Load multi-camera calibration (from AquaCal)
cameras = load_calibration("calibration.json")

# Segment fish in a multi-view frame
masks = segment_frame(frame, cameras)

# Reconstruct 3D pose via analysis-by-synthesis
pose = optimize_pose(masks, cameras)
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

See [Contributing](docs/contributing.md) for full development guidelines.

## Documentation

<!-- TODO: Uncomment once docs are deployed -->
<!-- Full documentation is available at [aquapose.readthedocs.io](https://aquapose.readthedocs.io). -->

## License

[MIT](LICENSE)
