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
        xy_errors: XY RMSE at each depth (meters), computed as
            sqrt(mean(dx² + dy²)) — Euclidean norm, matching AquaCal's
            convention. Shape (D,).
        z_errors: Z RMSE at each depth (meters), shape (D,).
        n_cameras_visible: Number of cameras that saw each depth point, shape (D,).
        xy_position: (X, Y) world coordinates of the test point.
    """

    depths: torch.Tensor
    xy_errors: torch.Tensor
    z_errors: torch.Tensor
    n_cameras_visible: torch.Tensor
    xy_position: tuple[float, float]


def compute_triangulation_uncertainty(
    models: list[RefractiveProjectionModel],
    depths: torch.Tensor,
    pixel_noise: float = 0.5,
    xy_position: tuple[float, float] | None = None,
    image_size: tuple[int, int] = (1600, 1200),
    n_trials: int = 1000,
    seed: int = 42,
) -> UncertaintyResult:
    """Compute X/Y/Z reconstruction error as a function of tank depth.

    For each depth, projects a ground-truth point into all visible cameras,
    then runs Monte Carlo trials: each trial adds independent Gaussian noise
    (std = ``pixel_noise``) to each camera's pixel, casts one refracted ray
    per camera, and triangulates. Per-axis RMSE across trials gives the
    reconstruction uncertainty.

    Args:
        models: List of RefractiveProjectionModel instances, one per camera.
        depths: 1D tensor of Z values to evaluate (underwater, Z > water_z
            for each camera), shape (D,), float32.
        pixel_noise: Standard deviation of Gaussian pixel noise (default 0.5).
        xy_position: (X, Y) world coordinates of the test point. If None,
            uses the origin (0, 0) — directly below the reference camera.
        image_size: (width, height) in pixels for bounds checking. Projections
            outside this frame are treated as invisible.
        n_trials: Number of Monte Carlo noise trials per depth (default 1000).
        seed: Random seed for reproducibility.

    Returns:
        UncertaintyResult with per-depth RMSE error statistics.
    """
    import numpy as np

    if xy_position is None:
        test_x, test_y = 0.0, 0.0
    else:
        test_x, test_y = xy_position

    img_w, img_h = image_size
    rng = np.random.default_rng(seed)

    n_depths = depths.shape[0]
    xy_errors = torch.zeros(n_depths)
    z_errors = torch.zeros(n_depths)
    n_visible = torch.zeros(n_depths, dtype=torch.int32)

    for i, depth in enumerate(depths):
        gt_point = torch.tensor([[test_x, test_y, depth.item()]])  # (1, 3)

        # Find visible cameras and their clean pixel projections
        visible_models: list[RefractiveProjectionModel] = []
        clean_pixels: list[torch.Tensor] = []

        for model in models:
            pixels, valid = model.project(gt_point)
            if not valid[0]:
                continue

            u, v = pixels[0, 0].item(), pixels[0, 1].item()
            if u < 0 or u >= img_w or v < 0 or v >= img_h:
                continue

            visible_models.append(model)
            clean_pixels.append(pixels[0])  # (2,)

        n_cams_visible = len(visible_models)
        n_visible[i] = n_cams_visible

        if n_cams_visible < 2:
            xy_errors[i] = float("nan")
            z_errors[i] = float("nan")
            continue

        # Monte Carlo: independent Gaussian noise per camera per trial
        gt_vec = gt_point[0]  # (3,)
        sq_errs_xy = torch.zeros(n_trials)
        sq_errs_z = torch.zeros(n_trials)

        for t in range(n_trials):
            trial_origins: list[torch.Tensor] = []
            trial_dirs: list[torch.Tensor] = []

            for ci in range(n_cams_visible):
                noise = torch.tensor(
                    rng.normal(0.0, pixel_noise, 2), dtype=torch.float32
                )
                noisy_px = (clean_pixels[ci] + noise).unsqueeze(0)  # (1, 2)
                o, d = visible_models[ci].cast_ray(noisy_px)
                trial_origins.append(o)
                trial_dirs.append(d)

            origins = torch.cat(trial_origins, dim=0)
            directions = torch.cat(trial_dirs, dim=0)

            with torch.no_grad():
                est = triangulate_rays(origins, directions)

            sq_errs_xy[t] = (est[0] - gt_vec[0]) ** 2 + (est[1] - gt_vec[1]) ** 2
            sq_errs_z[t] = (est[2] - gt_vec[2]) ** 2

        xy_errors[i] = sq_errs_xy.mean().sqrt()
        z_errors[i] = sq_errs_z.mean().sqrt()

    return UncertaintyResult(
        depths=depths,
        xy_errors=xy_errors,
        z_errors=z_errors,
        n_cameras_visible=n_visible,
        xy_position=(test_x, test_y),
    )


def build_rig_from_calibration(
    calibration_path: str | Path,
    exclude_cameras: set[str] | None = None,
) -> list[RefractiveProjectionModel]:
    """Build projection models from a real AquaCal calibration file.

    Loads the calibration JSON and creates a RefractiveProjectionModel for
    each camera, using the actual intrinsics, extrinsics, and per-camera
    water_z values.

    Args:
        calibration_path: Path to AquaCal calibration JSON file.
        exclude_cameras: Optional set of camera names to skip (e.g.
            ``{"e3v8250"}`` to exclude the wide-angle center camera).

    Returns:
        List of RefractiveProjectionModel instances, one per camera,
        sorted by camera name.
    """
    from .loader import compute_undistortion_maps, load_calibration_data

    cal = load_calibration_data(calibration_path)
    models: list[RefractiveProjectionModel] = []
    skip = exclude_cameras or set()

    for name in sorted(cal.cameras.keys()):
        if name in skip:
            continue
        cam = cal.cameras[name]
        # Use K_new (undistorted intrinsics) so that image-bounds visibility
        # matches the pipeline's working space.  Raw cam.K has a narrower
        # pinhole FOV that ignores barrel distortion, under-counting visible
        # cameras.
        undist = compute_undistortion_maps(cam)
        model = RefractiveProjectionModel(
            K=undist.K_new,
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
    xy_mm = result.xy_errors.numpy() * 1000.0
    z_mm = result.z_errors.numpy() * 1000.0
    n_cams = result.n_cameras_visible.numpy()

    # Mask NaN values (depths with < 2 cameras)
    valid = np.isfinite(xy_mm) & np.isfinite(z_mm)

    # --- Plot 1: XY/Z error vs depth ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths_m[valid], xy_mm[valid], "b-o", label="XY error", markersize=5)
    ax.plot(depths_m[valid], z_mm[valid], "r-^", label="Z error", markersize=5)
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Reconstruction RMSE (mm)")
    ax.set_title("Triangulation Error vs. Tank Depth")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plot1_path = output_dir / "z_uncertainty_xyz_error.png"
    fig.savefig(plot1_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Z/XY error ratio vs depth ---
    ratio = np.where(xy_mm > 0, z_mm / xy_mm, np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        depths_m[valid],
        ratio[valid],
        "m-D",
        label="Z / XY ratio",
        markersize=5,
    )
    ax.axhline(
        y=1.0, color="gray", linestyle="--", linewidth=1, label="Equal (ratio=1)"
    )
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Z RMSE / XY RMSE Ratio")
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
    xy_mean = float(np.nanmean(xy_mm[valid]))
    z_mean = float(np.nanmean(z_mm[valid]))
    xy_max = float(np.nanmax(xy_mm[valid]))
    z_max_val = float(np.nanmax(z_mm[valid]))
    xy_min = float(np.nanmin(xy_mm[valid]))
    z_min = float(np.nanmin(z_mm[valid]))

    ratio_valid = ratio[valid]
    mean_ratio = float(np.nanmean(ratio_valid))
    max_ratio = float(np.nanmax(ratio_valid))
    min_ratio = float(np.nanmin(ratio_valid))

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
        xy_val = xy_mm[j]
        z_val = z_mm[j]
        r_val = z_val / xy_val if math.isfinite(xy_val) and xy_val > 0 else float("nan")
        n_val = int(n_cams[j])

        table_rows.append(
            f"| {depth_val:.3f} | {_fmt(xy_val)} | {_fmt(z_val)} "
            f"| {_fmt(r_val)} | {n_val} |"
        )

    # --- Write markdown report ---
    report_path = output_dir / "z_uncertainty_report.md"
    report = f"""# Z-Uncertainty Characterization Report

**Rig:** {rig_description}
**Method:** Monte Carlo ray simulation with {pixel_noise}px pixel noise (sigma)
**Point:** ({result.xy_position[0]}, {result.xy_position[1]}, Z) projected through all cameras, triangulated via least-squares

## Error vs. Depth Table

| Depth (m) | XY RMSE (mm) | Z RMSE (mm) | Z/XY ratio | Cameras visible |
|-----------|-------------|-------------|------------|-----------------|
{chr(10).join(table_rows)}

## Summary Statistics

| Metric | XY RMSE | Z RMSE |
|--------|---------|--------|
| Mean   | {xy_mean:.3f} mm | {z_mean:.3f} mm |
| Min    | {xy_min:.3f} mm | {z_min:.3f} mm |
| Max    | {xy_max:.3f} mm | {z_max_val:.3f} mm |

**Z/XY anisotropy ratio:**
- Mean: {mean_ratio:.1f}x
- Range: {min_ratio:.1f}x - {max_ratio:.1f}x
- Best depth (lowest Z error): {best_depth:.3f} m
- Worst depth (highest Z error): {worst_depth:.3f} m

## Interpretation

Z uncertainty is approximately **{mean_ratio:.1f}x** worse than XY uncertainty for this
top-down camera geometry (range: {min_ratio:.1f}x to {max_ratio:.1f}x across the
depth range). This confirms that top-down cameras have substantially poorer Z-axis
observability than X/Y observability: rays from top-down cameras converge nearly
parallel in the Z direction, so small pixel errors produce large depth errors.

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
