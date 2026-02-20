"""Z-uncertainty analytical characterization via ray-simulation for top-down multi-camera rigs.

Example::

    import torch
    from aquapose.calibration.uncertainty import (
        build_rig_from_calibration,
        compute_triangulation_uncertainty,
        generate_uncertainty_report,
    )

    models = build_rig_from_calibration("path/to/calibration.json")
    depths = torch.linspace(1.0, 2.0, 20)
    result = compute_triangulation_uncertainty(models, depths)
    generate_uncertainty_report(result, "reports/")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .projection import RefractiveProjectionModel, triangulate_rays


@dataclass
class UncertaintyResult:
    """Results from triangulation uncertainty simulation.

    Attributes:
        depths: Evaluated Z depths (world frame), shape (D,).
        x_errors: Mean absolute X error at each depth (meters), shape (D,).
        y_errors: Mean absolute Y error at each depth (meters), shape (D,).
        z_errors: Mean absolute Z error at each depth (meters), shape (D,).
        n_cameras_visible: Number of cameras that saw each depth point, shape (D,).
        xy_position: (X, Y) world coordinates of the test point.
    """

    depths: torch.Tensor
    x_errors: torch.Tensor
    y_errors: torch.Tensor
    z_errors: torch.Tensor
    n_cameras_visible: torch.Tensor
    xy_position: tuple[float, float]


def compute_triangulation_uncertainty(
    models: list[RefractiveProjectionModel],
    depths: torch.Tensor,
    pixel_noise: float = 0.5,
    xy_position: tuple[float, float] | None = None,
    image_size: tuple[int, int] = (1600, 1200),
) -> UncertaintyResult:
    """Compute X/Y/Z reconstruction error as a function of tank depth.

    For each depth, projects a ground-truth point into all cameras, perturbs
    the pixels by ±pixel_noise, casts rays from the perturbed pixels, and
    triangulates to measure reconstruction error.

    Args:
        models: List of RefractiveProjectionModel instances, one per camera.
        depths: 1D tensor of Z values to evaluate (underwater, Z > water_z
            for each camera), shape (D,), float32.
        pixel_noise: Half-pixel perturbation magnitude in pixels (default 0.5).
        xy_position: (X, Y) world coordinates of the test point. If None,
            uses the origin (0, 0) — directly below the reference camera.
        image_size: (width, height) in pixels for bounds checking. Projections
            outside this frame are treated as invisible.

    Returns:
        UncertaintyResult with per-depth error statistics.
    """
    # Determine XY test position (default: origin, i.e. under reference camera)
    if xy_position is None:
        test_x, test_y = 0.0, 0.0
    else:
        test_x, test_y = xy_position

    img_w, img_h = image_size

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
        gt_point = torch.tensor([[test_x, test_y, depth.item()]])  # (1, 3)

        all_origins: list[torch.Tensor] = []
        all_directions: list[torch.Tensor] = []
        n_cams_visible = 0

        for model in models:
            pixels, valid = model.project(gt_point)
            if not valid[0]:
                continue

            # Check image bounds
            u, v = pixels[0, 0].item(), pixels[0, 1].item()
            if u < 0 or u >= img_w or v < 0 or v >= img_h:
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
        xy_position=(test_x, test_y),
    )


def build_rig_from_calibration(
    calibration_path: str | Path,
) -> list[RefractiveProjectionModel]:
    """Build projection models from a real AquaCal calibration file.

    Loads the calibration JSON and creates a RefractiveProjectionModel for
    each camera, using the actual intrinsics, extrinsics, and per-camera
    water_z values.

    Args:
        calibration_path: Path to AquaCal calibration JSON file.

    Returns:
        List of RefractiveProjectionModel instances, one per camera,
        sorted by camera name.
    """
    from .loader import load_calibration_data

    cal = load_calibration_data(calibration_path)
    models: list[RefractiveProjectionModel] = []

    for name in sorted(cal.cameras.keys()):
        cam = cal.cameras[name]
        model = RefractiveProjectionModel(
            K=cam.K,
            R=cam.R,
            t=cam.t,
            water_z=cal.water_z,
            normal=cal.interface_normal,
            n_air=cal.n_air,
            n_water=cal.n_water,
        )
        models.append(model)

    return models


def build_synthetic_rig() -> list[RefractiveProjectionModel]:
    """Build a 13-camera synthetic rig approximating the real aquarium hardware.

    Rig geometry (from real calibration):
    - 1 center camera with wide-angle lens (fx~746, ~94 deg FOV)
    - 12 cameras in a single ring at ~0.65m radius, evenly spaced at 30 deg
    - Ring cameras have ~54 deg roll relative to radial direction
    - All cameras oriented straight down through flat water surface
    - Mounted ~1m above water surface
    - 1600x1200 resolution, ring cameras ~54 deg horizontal FOV

    For accurate results, prefer ``build_rig_from_calibration()`` with the
    real calibration JSON. This synthetic version is for tests and fallback.

    Returns:
        List of 13 RefractiveProjectionModel instances ready for simulation.
    """
    import numpy as np
    from aquacal.datasets.synthetic import generate_camera_intrinsics

    RING_IMAGE_SIZE = (1600, 1200)
    RING_FOV_DEG = 54.0
    CENTER_IMAGE_SIZE = (1600, 1200)
    CENTER_FOV_DEG = 94.0  # Wide-angle center camera
    RING_RADIUS = 0.650  # ~650mm from calibration
    N_RING_CAMERAS = 12
    ROLL_OFFSET_RAD = np.deg2rad(54.0)  # ~54 deg from radial direction
    HEIGHT_ABOVE_WATER = 1.0  # ~1m above water surface
    HEIGHT_VARIATION_STD = 0.002  # Small per-camera variation

    rng = np.random.default_rng(42)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    models: list[RefractiveProjectionModel] = []

    def _rotation_z(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    def _make_model(
        K_np: np.ndarray, R_np: np.ndarray, t_np: np.ndarray, water_z: float
    ) -> RefractiveProjectionModel:
        K = torch.from_numpy(K_np).to(torch.float32)
        R = torch.from_numpy(R_np).to(torch.float32)
        t_vec = t_np.squeeze() if t_np.ndim == 2 else t_np
        t = torch.from_numpy(t_vec).to(torch.float32)
        return RefractiveProjectionModel(
            K=K, R=R, t=t, water_z=water_z, normal=normal, n_air=1.0, n_water=1.333
        )

    # Camera 0: wide-angle center camera at origin
    intr0 = generate_camera_intrinsics(
        image_size=CENTER_IMAGE_SIZE, fov_horizontal_deg=CENTER_FOV_DEG
    )
    R0 = np.eye(3, dtype=np.float64)
    t0 = np.zeros(3, dtype=np.float64)
    models.append(_make_model(intr0.K, R0, t0, HEIGHT_ABOVE_WATER))

    # Cameras 1-12: single ring at ~650mm radius, 30 deg spacing, ~54 deg roll
    angular_spacing = 2 * np.pi / N_RING_CAMERAS
    for i in range(N_RING_CAMERAS):
        theta = i * angular_spacing
        x = RING_RADIUS * np.cos(theta)
        y = RING_RADIUS * np.sin(theta)
        C = np.array([x, y, 0.0], dtype=np.float64)

        roll = theta + ROLL_OFFSET_RAD
        R = _rotation_z(roll)
        t = -R @ C

        intr = generate_camera_intrinsics(
            image_size=RING_IMAGE_SIZE, fov_horizontal_deg=RING_FOV_DEG
        )
        water_z = HEIGHT_ABOVE_WATER + rng.normal(0, HEIGHT_VARIATION_STD)
        models.append(_make_model(intr.K, R, t, water_z))

    return models


def generate_uncertainty_report(
    result: UncertaintyResult,
    output_dir: Path | str,
    rig_description: str = "13-camera top-down aquarium rig",
    pixel_noise: float = 0.5,
) -> Path:
    """Generate a markdown report with plots from UncertaintyResult.

    Creates three PNG plots and a markdown file summarizing X/Y/Z
    reconstruction error vs. depth for the top-down camera rig.

    Args:
        result: Uncertainty simulation results from compute_triangulation_uncertainty.
        output_dir: Directory where report and plots will be written.
            Created if it does not exist.
        rig_description: Human-readable rig description for the report header.
        pixel_noise: Pixel noise value used in the simulation, for the report header.

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
    def _fmt(v: float) -> str:
        return f"{v:.3f}" if math.isfinite(v) else "N/A"

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

        table_rows.append(
            f"| {depth_val:.3f} | {_fmt(x_val)} | {_fmt(y_val)} | {_fmt(z_val)} "
            f"| {_fmt(r_val)} | {n_val} |"
        )

    # --- Write markdown report ---
    report_path = output_dir / "z_uncertainty_report.md"
    report = f"""# Z-Uncertainty Characterization Report

**Rig:** {rig_description}
**Method:** Ray simulation with {pixel_noise}px pixel noise perturbation
**Point:** ({result.xy_position[0]}, {result.xy_position[1]}, Z) projected through all cameras, triangulated via SVD

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
reprojection errors more aggressively than Z, or equivalently, apply a
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
