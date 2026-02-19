"""Z-uncertainty analytical characterization via ray-simulation for top-down multi-camera rigs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .projection import RefractiveProjectionModel


@dataclass
class UncertaintyResult:
    """Results from triangulation uncertainty simulation.

    Attributes:
        depths: Evaluated Z depths (world frame), shape (D,).
        x_errors: Mean absolute X error at each depth (meters), shape (D,).
        y_errors: Mean absolute Y error at each depth (meters), shape (D,).
        z_errors: Mean absolute Z error at each depth (meters), shape (D,).
        n_cameras_visible: Number of cameras that saw each depth point, shape (D,).
    """

    depths: torch.Tensor
    x_errors: torch.Tensor
    y_errors: torch.Tensor
    z_errors: torch.Tensor
    n_cameras_visible: torch.Tensor


def triangulate_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Triangulate 3D point from multiple rays using least-squares (SVD).

    Solves: sum_i (I - d_i @ d_i^T) @ p = sum_i (I - d_i @ d_i^T) @ o_i

    Args:
        origins: Ray origin points, shape (N, 3), float32.
        directions: Unit ray direction vectors, shape (N, 3), float32. Must
            be unit vectors.

    Returns:
        Estimated 3D point, shape (3,), float32.

    Raises:
        ValueError: If fewer than 2 rays are provided.
    """
    if origins.shape[0] < 2:
        raise ValueError(f"Need at least 2 rays, got {origins.shape[0]}")

    device = origins.device
    dtype = origins.dtype

    # Build normal equations: A @ p = b where A = sum_i (I - d_i d_i^T)
    A = torch.zeros(3, 3, device=device, dtype=dtype)
    b = torch.zeros(3, device=device, dtype=dtype)

    eye3 = torch.eye(3, device=device, dtype=dtype)
    for i in range(origins.shape[0]):
        d = directions[i]  # (3,)
        o = origins[i]  # (3,)
        M = eye3 - d.unsqueeze(1) @ d.unsqueeze(0)  # (3, 3)
        A = A + M
        b = b + M @ o

    # Solve via least-squares (handles degenerate cases like parallel rays)
    result = torch.linalg.lstsq(A, b.unsqueeze(1))
    return result.solution.squeeze(1)


def compute_triangulation_uncertainty(
    models: list[RefractiveProjectionModel],
    depths: torch.Tensor,
    pixel_noise: float = 0.5,
) -> UncertaintyResult:
    """Compute X/Y/Z reconstruction error as a function of tank depth.

    For each depth, projects a ground-truth point at tank center (0, 0, Z)
    into all cameras, perturbs the pixels by ±pixel_noise, casts rays from
    the perturbed pixels, and triangulates to measure reconstruction error.

    Args:
        models: List of RefractiveProjectionModel instances, one per camera.
        depths: 1D tensor of Z values to evaluate (underwater, Z > water_z
            for each camera), shape (D,), float32.
        pixel_noise: Half-pixel perturbation magnitude in pixels (default 0.5).

    Returns:
        UncertaintyResult with per-depth error statistics.
    """
    n_depths = depths.shape[0]
    x_errors = torch.zeros(n_depths)
    y_errors = torch.zeros(n_depths)
    z_errors = torch.zeros(n_depths)
    n_visible = torch.zeros(n_depths, dtype=torch.int32)

    # 4 perturbation directions per camera: ±x, ±y in pixel space
    perturbations = torch.tensor(
        [
            [pixel_noise, 0.0],
            [-pixel_noise, 0.0],
            [0.0, pixel_noise],
            [0.0, -pixel_noise],
        ]
    )  # (4, 2)

    for i, depth in enumerate(depths):
        gt_point = torch.tensor([[0.0, 0.0, depth.item()]])  # (1, 3)

        all_origins: list[torch.Tensor] = []
        all_directions: list[torch.Tensor] = []
        n_cams_visible = 0

        for model in models:
            pixels, valid = model.project(gt_point)
            if not valid[0]:
                continue

            n_cams_visible += 1
            center_px = pixels[0]  # (2,)

            # 4 perturbed pixel positions for this camera
            perturbed_px = center_px.unsqueeze(0) + perturbations  # (4, 2)
            origins_cam, dirs_cam = model.cast_ray(perturbed_px)  # (4, 3) each
            all_origins.append(origins_cam)
            all_directions.append(dirs_cam)

        n_visible[i] = n_cams_visible

        if n_cams_visible < 2:
            # Cannot triangulate with fewer than 2 cameras
            x_errors[i] = float("nan")
            y_errors[i] = float("nan")
            z_errors[i] = float("nan")
            continue

        origins = torch.cat(all_origins, dim=0)  # (N_rays, 3)
        directions = torch.cat(all_directions, dim=0)  # (N_rays, 3)

        with torch.no_grad():
            est_point = triangulate_rays(origins, directions)  # (3,)

        gt = gt_point[0]  # (3,)
        x_errors[i] = torch.abs(est_point[0] - gt[0])
        y_errors[i] = torch.abs(est_point[1] - gt[1])
        z_errors[i] = torch.abs(est_point[2] - gt[2])

    return UncertaintyResult(
        depths=depths,
        x_errors=x_errors,
        y_errors=y_errors,
        z_errors=z_errors,
        n_cameras_visible=n_visible,
    )


def build_synthetic_rig() -> list[RefractiveProjectionModel]:
    """Build the 13-camera synthetic rig matching the real aquarium hardware.

    Uses AquaCal's generate_real_rig_array() for realistic camera placement
    (1 center + 6 inner ring + 6 outer ring at 300mm/600mm radius, 1600x1200
    resolution, 56 deg FOV, ~750mm above water surface).

    Returns:
        List of 13 RefractiveProjectionModel instances ready for simulation.
    """
    from aquacal.datasets.synthetic import generate_real_rig_array

    intrinsics, extrinsics, distances = generate_real_rig_array(
        height_above_water=0.75,
        height_variation=0.002,
        seed=42,
    )

    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    models: list[RefractiveProjectionModel] = []

    for cam_name in sorted(intrinsics.keys()):
        intr = intrinsics[cam_name]
        extr = extrinsics[cam_name]
        height = distances[cam_name]  # meters above water surface

        K = torch.from_numpy(intr.K).to(torch.float32)
        R = torch.from_numpy(extr.R).to(torch.float32)
        t_np = extr.t
        if t_np.ndim == 2:
            t_np = t_np.squeeze()
        t = torch.from_numpy(t_np).to(torch.float32)

        # Camera center is at Z=0; water surface is 'height' below in +Z direction
        water_z = float(height)

        model = RefractiveProjectionModel(
            K=K,
            R=R,
            t=t,
            water_z=water_z,
            normal=normal,
            n_air=1.0,
            n_water=1.333,
        )
        models.append(model)

    return models


def generate_uncertainty_report(
    result: UncertaintyResult,
    output_dir: Path | str,
) -> Path:
    """Generate a markdown report with plots from UncertaintyResult.

    Creates three PNG plots and a markdown file summarizing X/Y/Z
    reconstruction error vs. depth for the top-down camera rig.

    Args:
        result: Uncertainty simulation results from compute_triangulation_uncertainty.
        output_dir: Directory where report and plots will be written.
            Created if it does not exist.

    Returns:
        Path to the generated markdown report file.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("Agg")  # Non-interactive backend for script use

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depths_m = result.depths.numpy()
    x_mm = result.x_errors.numpy() * 1000.0
    y_mm = result.y_errors.numpy() * 1000.0
    z_mm = result.z_errors.numpy() * 1000.0
    n_cams = result.n_cameras_visible.numpy()

    # Mask NaN values (depths with < 2 cameras)
    valid = np.isfinite(x_mm) & np.isfinite(y_mm) & np.isfinite(z_mm)

    # --- Plot 1: X/Y/Z error vs depth ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths_m[valid], x_mm[valid], "b-o", label="X error", markersize=5)
    ax.plot(depths_m[valid], y_mm[valid], "g-s", label="Y error", markersize=5)
    ax.plot(depths_m[valid], z_mm[valid], "r-^", label="Z error", markersize=5)
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Reconstruction Error (mm)")
    ax.set_title(
        "Triangulation Error vs. Tank Depth\n(0.5px pixel noise, 13-camera rig)"
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plot1_path = output_dir / "z_uncertainty_xyz_error.png"
    fig.savefig(plot1_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Z/X error ratio vs depth ---
    xy_mean_mm = (x_mm + y_mm) / 2.0
    ratio = np.where(xy_mean_mm > 0, z_mm / xy_mean_mm, np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        depths_m[valid],
        ratio[valid],
        "m-D",
        label="Z / mean(X,Y) ratio",
        markersize=5,
    )
    ax.axhline(
        y=1.0, color="gray", linestyle="--", linewidth=1, label="Equal (ratio=1)"
    )
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Z Error / XY Error Ratio")
    ax.set_title("Z/XY Anisotropy vs. Tank Depth")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plot2_path = output_dir / "z_uncertainty_ratio.png"
    fig.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Camera visibility vs depth ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        depths_m,
        n_cams,
        width=(depths_m[1] - depths_m[0]) * 0.8 if len(depths_m) > 1 else 0.04,
        color="steelblue",
        alpha=0.8,
    )
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Cameras Visible")
    ax.set_title("Camera Visibility vs. Tank Depth")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plot3_path = output_dir / "z_uncertainty_cameras.png"
    fig.savefig(plot3_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Compute summary statistics ---
    x_mean = float(np.nanmean(x_mm[valid]))
    y_mean = float(np.nanmean(y_mm[valid]))
    z_mean = float(np.nanmean(z_mm[valid]))
    x_max = float(np.nanmax(x_mm[valid]))
    y_max = float(np.nanmax(y_mm[valid]))
    z_max = float(np.nanmax(z_mm[valid]))
    x_min = float(np.nanmin(x_mm[valid]))
    y_min = float(np.nanmin(y_mm[valid]))
    z_min = float(np.nanmin(z_mm[valid]))

    ratio_valid = ratio[valid]
    mean_ratio = float(np.nanmean(ratio_valid))
    max_ratio = float(np.nanmax(ratio_valid))

    worst_z_idx = int(np.nanargmax(z_mm[valid]))
    best_z_idx = int(np.nanargmin(z_mm[valid]))
    depths_valid = depths_m[valid]
    worst_depth = float(depths_valid[worst_z_idx])
    best_depth = float(depths_valid[best_z_idx])

    # --- Build table rows ---
    table_rows = []
    for j in range(len(depths_m)):
        depth_val = depths_m[j]
        x_val = x_mm[j]
        y_val = y_mm[j]
        z_val = z_mm[j]
        xy_val = (
            (x_val + y_val) / 2.0
            if math.isfinite(x_val) and math.isfinite(y_val)
            else float("nan")
        )
        r_val = z_val / xy_val if math.isfinite(xy_val) and xy_val > 0 else float("nan")
        n_val = int(n_cams[j])

        def _fmt(v: float) -> str:
            return f"{v:.3f}" if math.isfinite(v) else "N/A"

        table_rows.append(
            f"| {depth_val:.3f} | {_fmt(x_val)} | {_fmt(y_val)} | {_fmt(z_val)} "
            f"| {_fmt(r_val)} | {n_val} |"
        )

    # --- Write markdown report ---
    report_path = output_dir / "z_uncertainty_report.md"
    report = f"""# Z-Uncertainty Characterization Report

**Rig:** 13-camera top-down aquarium rig (1 center + 6 inner ring + 6 outer ring)
**Method:** Ray simulation with {0.5}px pixel noise perturbation
**Point:** Tank center (0, 0, Z) projected through all cameras, triangulated via SVD

## Error vs. Depth Table

| Depth (m) | X error (mm) | Y error (mm) | Z error (mm) | Z/XY ratio | Cameras visible |
|-----------|-------------|-------------|-------------|------------|-----------------|
{chr(10).join(table_rows)}

## Summary Statistics

| Metric | X error | Y error | Z error |
|--------|---------|---------|---------|
| Mean   | {x_mean:.3f} mm | {y_mean:.3f} mm | {z_mean:.3f} mm |
| Min    | {x_min:.3f} mm | {y_min:.3f} mm | {z_min:.3f} mm |
| Max    | {x_max:.3f} mm | {y_max:.3f} mm | {z_max:.3f} mm |

**Z/XY anisotropy ratio:**
- Mean: {mean_ratio:.1f}x
- Max: {max_ratio:.1f}x
- Best depth (lowest Z error): {best_depth:.3f} m
- Worst depth (highest Z error): {worst_depth:.3f} m

## Interpretation

Z uncertainty is approximately **{mean_ratio:.0f}x** worse than XY uncertainty for this
top-down camera geometry (range: {min(ratio_valid):.1f}x to {max_ratio:.1f}x across the
depth range). This confirms that top-down cameras have substantially poorer Z-axis
observability than X/Y observability: rays from top-down cameras converge nearly
parallel in the Z direction, so small pixel errors produce large depth errors.

**Implication for optimizer (Phase 4):** The loss function should weight X and Y
reprojection errors more aggressively than Z — or equivalently, apply a
prior/regularizer on Z that is approximately {mean_ratio:.0f}x stronger than the
equivalent X/Y constraint. This anisotropy is a fundamental geometric property of
the rig, not a calibration deficiency.

## Plots

### X/Y/Z Error vs. Depth
![XYZ error vs depth](z_uncertainty_xyz_error.png)

### Z/XY Anisotropy Ratio vs. Depth
![Z/XY ratio vs depth](z_uncertainty_ratio.png)

### Camera Visibility vs. Depth
![Camera visibility vs depth](z_uncertainty_cameras.png)
"""

    report_path.write_text(report)
    return report_path
