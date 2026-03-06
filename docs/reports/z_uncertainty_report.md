# Z-Uncertainty Characterization Report

**Rig:** 13-camera top-down aquarium rig (real calibration, test point at origin under reference camera)
**Method:** Ray simulation with 0.5px pixel noise perturbation
**Point:** Tank center (0, 0, Z) projected through all cameras, triangulated via SVD

## Error vs. Depth Table

| Depth (m) | X error (mm) | Y error (mm) | Z error (mm) | Z/XY ratio | Cameras visible |
|-----------|-------------|-------------|-------------|------------|-----------------|
| 1.081 | 0.000 | 0.000 | 0.002 | 8.314 | 4 |
| 1.131 | 0.000 | 0.000 | 0.002 | 7.402 | 4 |
| 1.181 | 0.000 | 0.000 | 0.002 | 9.244 | 4 |
| 1.231 | 0.000 | 0.000 | 0.003 | 9.268 | 4 |
| 1.281 | 0.000 | 0.000 | 0.003 | 10.589 | 4 |
| 1.331 | 0.000 | 0.000 | 0.003 | 9.881 | 4 |
| 1.381 | 0.000 | 0.000 | 0.004 | 10.848 | 4 |
| 1.431 | 0.000 | 0.000 | 0.002 | 9.348 | 5 |
| 1.481 | 0.000 | 0.000 | 0.003 | 12.136 | 5 |
| 1.531 | 0.000 | 0.000 | 0.003 | 12.029 | 5 |
| 1.581 | 0.000 | 0.000 | 0.003 | 12.008 | 5 |
| 1.631 | 0.000 | 0.000 | 0.003 | 11.202 | 5 |
| 1.681 | 0.000 | 0.000 | 0.003 | 12.115 | 5 |
| 1.731 | 0.000 | 0.001 | 0.004 | 13.067 | 5 |
| 1.781 | 0.000 | 0.001 | 0.004 | 13.099 | 5 |
| 1.831 | 0.000 | 0.001 | 0.004 | 13.107 | 5 |
| 1.881 | 0.000 | 0.001 | 0.004 | 14.211 | 5 |
| 1.931 | 0.000 | 0.001 | 0.005 | 14.909 | 5 |
| 1.981 | 0.000 | 0.001 | 0.004 | 12.190 | 6 |
| 2.031 | 0.000 | 0.001 | 0.005 | 12.699 | 6 |

## Summary Statistics

| Metric | X error | Y error | Z error |
|--------|---------|---------|---------|
| Mean   | 0.000 mm | 0.000 mm | 0.003 mm |
| Min    | 0.000 mm | 0.000 mm | 0.002 mm |
| Max    | 0.000 mm | 0.001 mm | 0.005 mm |

**Z/XY anisotropy ratio:**
- Mean: 11.4x
- Max: 14.9x
- Best depth (lowest Z error): 1.081 m
- Worst depth (highest Z error): 1.931 m

## Interpretation

Z uncertainty is approximately **11x** worse than XY uncertainty for this
top-down camera geometry (range: 7.4x to 14.9x across the
depth range). This confirms that top-down cameras have substantially poorer Z-axis
observability than X/Y observability: rays from top-down cameras converge nearly
parallel in the Z direction, so small pixel errors produce large depth errors.

**Implication for optimizer (Phase 4):** The loss function should weight X and Y
reprojection errors more aggressively than Z, or equivalently, apply a
prior/regularizer on Z that is approximately 11x stronger than the
equivalent X/Y constraint. This anisotropy is a fundamental geometric property of
the rig, not a calibration deficiency.

## Plots

### X/Y/Z Error vs. Depth
![XYZ error vs depth](z_uncertainty_xyz_error.png)

### Z/XY Anisotropy Ratio vs. Depth
![Z/XY ratio vs depth](z_uncertainty_ratio.png)

### Camera Visibility vs. Depth
![Camera visibility vs depth](z_uncertainty_cameras.png)
