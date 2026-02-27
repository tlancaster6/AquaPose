"""Forward and inverse lookup tables for fast refraction-based ray casting and voxel projection."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from aquapose.calibration.loader import CalibrationData, UndistortionMaps
from aquapose.calibration.projection import RefractiveProjectionModel

logger = logging.getLogger(__name__)


@runtime_checkable
class LutConfigLike(Protocol):
    """Structural protocol for LUT configuration objects.

    Any object with these attributes can be passed to LUT generation and
    serialization functions. Satisfied by ``LutConfig`` from ``engine.config``
    without an explicit import, preserving the core→engine import boundary.

    Attributes:
        tank_diameter: Cylindrical tank diameter in metres.
        tank_height: Tank depth (water column height) in metres.
        voxel_resolution_m: Voxel grid spacing in metres.
        margin_fraction: Fractional margin beyond tank dimensions.
        forward_grid_step: Pixel step size for forward LUT grid.
    """

    tank_diameter: float
    tank_height: float
    voxel_resolution_m: float
    margin_fraction: float
    forward_grid_step: int


@dataclass
class ForwardLUT:
    """Forward lookup table mapping pixel coordinates to refracted 3D rays.

    Precomputes a regular grid of rays via RefractiveProjectionModel.cast_ray(),
    then serves arbitrary pixel queries via bilinear interpolation over the grid.

    Attributes:
        camera_id: Camera identifier.
        grid_origins: Ray origins on water surface, shape (H, W, 3), float32.
        grid_directions: Unit ray directions into water, shape (H, W, 3), float32.
        grid_step: Pixel spacing of the precomputed grid.
        image_size: (width, height) of the camera image.
    """

    camera_id: str
    grid_origins: np.ndarray  # shape (H, W, 3), float32
    grid_directions: np.ndarray  # shape (H, W, 3), float32
    grid_step: int
    image_size: tuple[int, int]  # (width, height)

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast rays for arbitrary pixel coordinates via bilinear interpolation.

        Args:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.

        Returns:
            origins: Ray origin points on water surface, shape (N, 3), float32.
            directions: Unit ray direction vectors into water, shape (N, 3), float32.
        """
        device = pixels.device
        dtype = pixels.dtype

        u = pixels[:, 0]  # (N,)
        v = pixels[:, 1]  # (N,)

        # Convert pixel coords to fractional grid indices
        gx = u / self.grid_step  # (N,) fractional column index
        gy = v / self.grid_step  # (N,) fractional row index

        grid_h, grid_w = self.grid_origins.shape[:2]

        # Floor and ceil, clamped to valid grid bounds
        gx0 = torch.clamp(gx.floor().long(), 0, grid_w - 1)
        gx1 = torch.clamp(gx.ceil().long(), 0, grid_w - 1)
        gy0 = torch.clamp(gy.floor().long(), 0, grid_h - 1)
        gy1 = torch.clamp(gy.ceil().long(), 0, grid_h - 1)

        # Fractional part (weights for interpolation)
        fx = (gx - gx.floor()).to(dtype).unsqueeze(-1)  # (N, 1)
        fy = (gy - gy.floor()).to(dtype).unsqueeze(-1)  # (N, 1)

        # Load grid as torch tensors
        origins_grid = torch.from_numpy(self.grid_origins).to(
            device=device, dtype=dtype
        )
        dirs_grid = torch.from_numpy(self.grid_directions).to(
            device=device, dtype=dtype
        )

        # Four corner values: (gy0,gx0), (gy0,gx1), (gy1,gx0), (gy1,gx1)
        o00 = origins_grid[gy0, gx0]  # (N, 3)
        o10 = origins_grid[gy0, gx1]  # (N, 3)
        o01 = origins_grid[gy1, gx0]  # (N, 3)
        o11 = origins_grid[gy1, gx1]  # (N, 3)

        d00 = dirs_grid[gy0, gx0]  # (N, 3)
        d10 = dirs_grid[gy0, gx1]  # (N, 3)
        d01 = dirs_grid[gy1, gx0]  # (N, 3)
        d11 = dirs_grid[gy1, gx1]  # (N, 3)

        # Bilinear interpolation weights
        w00 = (1 - fx) * (1 - fy)  # (N, 1)
        w10 = fx * (1 - fy)  # (N, 1)
        w01 = (1 - fx) * fy  # (N, 1)
        w11 = fx * fy  # (N, 1)

        origins = w00 * o00 + w10 * o10 + w01 * o01 + w11 * o11  # (N, 3)
        directions = w00 * d00 + w10 * d10 + w01 * d01 + w11 * d11  # (N, 3)

        # Re-normalize directions to unit vectors
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)

        return origins, directions


def generate_forward_lut(
    camera_id: str,
    model: RefractiveProjectionModel,
    image_size: tuple[int, int],
    grid_step: int = 1,
) -> ForwardLUT:
    """Generate a ForwardLUT for a single camera by evaluating cast_ray on a grid.

    Args:
        camera_id: Camera identifier string.
        model: RefractiveProjectionModel for this camera.
        image_size: (width, height) of the camera image.
        grid_step: Pixel step size for the precomputed grid (1 = every pixel).

    Returns:
        ForwardLUT with precomputed origins and directions arrays.
    """
    width, height = image_size

    # Build regular pixel grid: u in [0, width), v in [0, height), step=grid_step
    us = torch.arange(0, width, grid_step, dtype=torch.float32)
    vs = torch.arange(0, height, grid_step, dtype=torch.float32)

    grid_w = len(us)
    grid_h = len(vs)

    # Create meshgrid of (u, v) pairs
    vv, uu = torch.meshgrid(vs, us, indexing="ij")  # each (grid_h, grid_w)
    grid_pixels = torch.stack(
        [uu.flatten(), vv.flatten()], dim=-1
    )  # (grid_h*grid_w, 2)

    # Cast rays for all grid points
    with torch.no_grad():
        grid_origins_flat, grid_directions_flat = model.cast_ray(grid_pixels)

    # Reshape to (grid_h, grid_w, 3)
    grid_origins = (
        grid_origins_flat.reshape(grid_h, grid_w, 3).cpu().numpy().astype(np.float32)
    )
    grid_directions = (
        grid_directions_flat.reshape(grid_h, grid_w, 3).cpu().numpy().astype(np.float32)
    )

    return ForwardLUT(
        camera_id=camera_id,
        grid_origins=grid_origins,
        grid_directions=grid_directions,
        grid_step=grid_step,
        image_size=image_size,
    )


def generate_forward_luts(
    calibration: CalibrationData,
    lut_config: LutConfigLike,
    undistortion_maps: dict[str, UndistortionMaps] | None = None,
) -> dict[str, ForwardLUT]:
    """Generate forward LUTs for all ring cameras from CalibrationData.

    Args:
        calibration: Complete calibration data with camera intrinsics/extrinsics.
        lut_config: LUT generation configuration including grid step size.
        undistortion_maps: Optional per-camera undistortion maps. When provided,
            the post-undistortion K_new is used instead of the raw K for each
            camera.

    Returns:
        Dictionary mapping camera IDs to their forward LUTs.
    """
    luts: dict[str, ForwardLUT] = {}

    for cam_id in calibration.ring_cameras:
        cam = calibration.cameras[cam_id]

        # Use post-undistortion K if undistortion maps are available
        if undistortion_maps is not None and cam_id in undistortion_maps:
            K = undistortion_maps[cam_id].K_new
        else:
            K = cam.K

        model = RefractiveProjectionModel(
            K=K,
            R=cam.R,
            t=cam.t,
            water_z=calibration.water_z,
            normal=calibration.interface_normal,
            n_air=calibration.n_air,
            n_water=calibration.n_water,
        )

        t0 = time.perf_counter()
        lut = generate_forward_lut(
            camera_id=cam_id,
            model=model,
            image_size=cam.image_size,
            grid_step=lut_config.forward_grid_step,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Generating LUT for camera %s... done (%.1fs)", cam_id, elapsed)

        luts[cam_id] = lut

    return luts


def compute_lut_hash(calibration_path: str | Path, lut_config: LutConfigLike) -> str:
    """Compute a deterministic hash of calibration file contents and LUT config.

    The hash changes whenever the calibration file changes or LUT config fields
    that affect generation change (tank geometry, grid step, etc.).

    Args:
        calibration_path: Path to the calibration JSON file.
        lut_config: LUT configuration dataclass.

    Returns:
        First 16 hex characters of the SHA-256 hash.
    """
    hasher = hashlib.sha256()

    # Hash calibration file contents
    path = Path(calibration_path)
    with path.open("rb") as fh:
        hasher.update(fh.read())

    # Hash all LutConfig fields
    config_str = (
        f"{lut_config.tank_diameter}|"
        f"{lut_config.tank_height}|"
        f"{lut_config.voxel_resolution_m}|"
        f"{lut_config.margin_fraction}|"
        f"{lut_config.forward_grid_step}"
    )
    hasher.update(config_str.encode())

    return hasher.hexdigest()[:16]


def save_forward_lut(
    lut: ForwardLUT,
    path: str | Path,
    config_hash: str = "",
) -> None:
    """Save a ForwardLUT to a .npz file.

    Args:
        lut: The ForwardLUT to save.
        path: Destination file path (should end in .npz).
        config_hash: Optional hash string for cache invalidation.
    """
    np.savez(
        str(path),
        grid_origins=lut.grid_origins,
        grid_directions=lut.grid_directions,
        grid_step=np.array(lut.grid_step, dtype=np.int32),
        image_size=np.array(lut.image_size, dtype=np.int32),
        camera_id=np.array(lut.camera_id),
        config_hash=np.array(config_hash),
    )


def load_forward_lut(path: str | Path) -> tuple[ForwardLUT, str]:
    """Load a ForwardLUT from a .npz file.

    Args:
        path: Path to the .npz file written by save_forward_lut().

    Returns:
        Tuple of (ForwardLUT, config_hash) where config_hash is the stored
        hash string for cache invalidation.
    """
    data = np.load(str(path), allow_pickle=False)

    camera_id = str(data["camera_id"])
    grid_step = int(data["grid_step"])
    image_size = tuple(int(x) for x in data["image_size"])
    config_hash = str(data["config_hash"])

    lut = ForwardLUT(
        camera_id=camera_id,
        grid_origins=data["grid_origins"],
        grid_directions=data["grid_directions"],
        grid_step=grid_step,
        image_size=image_size,  # type: ignore[arg-type]
    )

    return lut, config_hash


def save_forward_luts(
    luts: dict[str, ForwardLUT],
    calibration_path: str | Path,
    lut_config: LutConfigLike,
) -> None:
    """Save all forward LUTs to the luts/ directory next to the calibration file.

    Args:
        luts: Dictionary mapping camera IDs to forward LUTs.
        calibration_path: Path to the calibration file (determines output directory).
        lut_config: LUT config used to compute the cache-invalidation hash.
    """
    lut_dir = Path(calibration_path).parent / "luts"
    lut_dir.mkdir(parents=True, exist_ok=True)

    config_hash = compute_lut_hash(calibration_path, lut_config)

    for cam_id, lut in luts.items():
        out_path = lut_dir / f"{cam_id}_forward.npz"
        save_forward_lut(lut, out_path, config_hash=config_hash)
        logger.debug("Saved forward LUT for %s → %s", cam_id, out_path)


def load_forward_luts(
    calibration_path: str | Path,
    lut_config: LutConfigLike,
) -> dict[str, ForwardLUT] | None:
    """Load all forward LUTs from disk, returning None if cache is stale or missing.

    Args:
        calibration_path: Path to the calibration file (determines LUT directory).
        lut_config: LUT config used to check the cache-invalidation hash.

    Returns:
        Dictionary mapping camera IDs to forward LUTs, or None if LUTs are
        missing or the stored hash doesn't match the current calibration/config.
    """
    lut_dir = Path(calibration_path).parent / "luts"

    if not lut_dir.exists():
        return None

    npz_files = sorted(lut_dir.glob("*_forward.npz"))
    if not npz_files:
        return None

    expected_hash = compute_lut_hash(calibration_path, lut_config)

    luts: dict[str, ForwardLUT] = {}
    for npz_path in npz_files:
        lut, stored_hash = load_forward_lut(npz_path)
        if stored_hash != expected_hash:
            logger.info(
                "LUT cache miss for %s (hash mismatch: stored=%s, expected=%s)",
                lut.camera_id,
                stored_hash,
                expected_hash,
            )
            return None
        luts[lut.camera_id] = lut

    return luts if luts else None


def validate_forward_lut(
    lut: ForwardLUT,
    model: RefractiveProjectionModel,
    n_samples: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """Validate ForwardLUT accuracy against RefractiveProjectionModel.cast_ray().

    Samples random pixel coordinates and compares LUT interpolation against
    the ground-truth model output.

    Args:
        lut: Forward LUT to validate.
        model: The RefractiveProjectionModel used to generate the LUT.
        n_samples: Number of random pixel samples for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:
            max_angular_error_deg: Maximum angular error between directions (degrees).
            mean_angular_error_deg: Mean angular error between directions (degrees).
            max_origin_error_m: Maximum origin distance (metres).
            mean_origin_error_m: Mean origin distance (metres).

    Raises:
        ValueError: If max angular error exceeds 0.1 degrees (indicates a bug).
    """
    rng = np.random.default_rng(seed)

    width, height = lut.image_size
    us = rng.uniform(0, width - 1, size=n_samples).astype(np.float32)
    vs = rng.uniform(0, height - 1, size=n_samples).astype(np.float32)
    samples = torch.from_numpy(np.stack([us, vs], axis=-1))  # (N, 2)

    with torch.no_grad():
        lut_origins, lut_dirs = lut.cast_ray(samples)
        model_origins, model_dirs = model.cast_ray(samples)

    lut_origins = lut_origins.cpu()
    lut_dirs = lut_dirs.cpu()
    model_origins = model_origins.cpu()
    model_dirs = model_dirs.cpu()

    # Angular error (degrees)
    dot = (lut_dirs * model_dirs).sum(dim=-1).clamp(-1.0, 1.0)
    angular_errors_deg = torch.acos(dot).abs() * (180.0 / torch.pi)

    # Origin distance (metres)
    origin_errors_m = torch.linalg.norm(lut_origins - model_origins, dim=-1)

    max_angular_error_deg = float(angular_errors_deg.max())
    mean_angular_error_deg = float(angular_errors_deg.mean())
    max_origin_error_m = float(origin_errors_m.max())
    mean_origin_error_m = float(origin_errors_m.mean())

    if max_angular_error_deg > 0.1:
        raise ValueError(
            f"ForwardLUT validation failed: max angular error {max_angular_error_deg:.4f}° "
            f"exceeds 0.1° threshold — possible bug in interpolation or grid generation."
        )

    return {
        "max_angular_error_deg": max_angular_error_deg,
        "mean_angular_error_deg": mean_angular_error_deg,
        "max_origin_error_m": max_origin_error_m,
        "mean_origin_error_m": mean_origin_error_m,
    }


# ---------------------------------------------------------------------------
# Inverse LUT — voxel grid to per-camera pixel projections
# ---------------------------------------------------------------------------


def _build_cylindrical_voxel_grid(
    tank_center_xy: tuple[float, float],
    tank_diameter: float,
    tank_height: float,
    water_z: float,
    voxel_resolution: float,
    margin_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a regular 3D voxel grid clipped to a cylindrical volume.

    Args:
        tank_center_xy: (cx, cy) of the tank centre in world XY coordinates.
        tank_diameter: Tank diameter in metres.
        tank_height: Water column height in metres.
        water_z: Z-coordinate of the water surface in world frame.
        voxel_resolution: Grid spacing in metres.
        margin_fraction: Fractional margin beyond tank dimensions (e.g., 0.1 → 10%).

    Returns:
        voxel_centers: Array of shape (N_voxels, 3), float32, world-frame XYZ.
        grid_bounds: Dict with keys 'x_min', 'x_max', 'y_min', 'y_max',
            'z_min', 'z_max' representing the extents of the candidate grid.
    """
    cx, cy = tank_center_xy
    radius = (tank_diameter / 2.0) * (1.0 + margin_fraction)
    z_min = water_z
    z_max = water_z + tank_height * (1.0 + margin_fraction)

    x_min = cx - radius
    x_max = cx + radius
    y_min = cy - radius
    y_max = cy + radius

    # Generate a regular grid of candidate voxel centres.
    # Use a tiny epsilon (1e-6 * resolution) as the stop tolerance so that
    # voxels exactly on the boundary are included but no extra step beyond
    # the boundary is added (avoids floating-point overshoot from 0.5*step).
    _eps = voxel_resolution * 1e-6
    xs = np.arange(x_min, x_max + _eps, voxel_resolution, dtype=np.float32)
    ys = np.arange(y_min, y_max + _eps, voxel_resolution, dtype=np.float32)
    zs = np.arange(z_min, z_max + _eps, voxel_resolution, dtype=np.float32)

    # Build full 3D meshgrid (ij indexing: x, y, z order)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")  # each (Nx, Ny, Nz)

    # Flatten to candidate list
    all_x = gx.ravel()
    all_y = gy.ravel()
    all_z = gz.ravel()

    # Filter to cylindrical volume
    dist_sq = (all_x - cx) ** 2 + (all_y - cy) ** 2
    inside_cylinder = dist_sq <= radius**2

    voxel_centers = np.stack(
        [all_x[inside_cylinder], all_y[inside_cylinder], all_z[inside_cylinder]],
        axis=-1,
    ).astype(np.float32)

    grid_bounds = {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
    }

    return voxel_centers, grid_bounds


def _build_grid_index(
    voxel_centers: np.ndarray,
    grid_bounds: dict[str, float],
    voxel_resolution: float,
) -> dict[tuple[int, int, int], int]:
    """Build a mapping from integer grid coordinates to voxel array indices.

    Args:
        voxel_centers: Array of shape (N, 3), float32.
        grid_bounds: Dict with x_min, y_min, z_min keys.
        voxel_resolution: Grid spacing in metres.

    Returns:
        Dictionary mapping (ix, iy, iz) integer grid coordinates to voxel
        array indices. Enables O(1) nearest-voxel lookup for arbitrary points.
    """
    x_min = grid_bounds["x_min"]
    y_min = grid_bounds["y_min"]
    z_min = grid_bounds["z_min"]

    grid_to_idx: dict[tuple[int, int, int], int] = {}
    for idx, (x, y, z) in enumerate(voxel_centers):
        ix = round((float(x) - x_min) / voxel_resolution)
        iy = round((float(y) - y_min) / voxel_resolution)
        iz = round((float(z) - z_min) / voxel_resolution)
        grid_to_idx[(ix, iy, iz)] = idx

    return grid_to_idx


@dataclass
class InverseLUT:
    """Inverse lookup table mapping 3D voxel centres to per-camera pixel projections.

    Discretizes the cylindrical tank volume and records which cameras can see
    each voxel and where it projects. Provides camera overlap graph and
    ghost-point lookups without running the refraction model at query time.

    Attributes:
        voxel_centers: 3D coordinates of all voxel centres, shape (N, 3), float32.
        visibility_mask: Boolean array, shape (N, C), True if camera c sees voxel n.
        projected_pixels: Pixel coordinates, shape (N, C, 2), float32. NaN where not visible.
        camera_ids: Ordered list of camera IDs (index into C dimension).
        voxel_resolution: Grid spacing in metres.
        grid_bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'.
    """

    voxel_centers: np.ndarray  # shape (N, 3), float32
    visibility_mask: np.ndarray  # shape (N, C), bool
    projected_pixels: np.ndarray  # shape (N, C, 2), float32; NaN where not visible
    camera_ids: list[str]
    voxel_resolution: float
    grid_bounds: dict[str, float]
    _grid_to_voxel_idx: dict[tuple[int, int, int], int] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        """Build the grid index mapping after initialisation."""
        if not self._grid_to_voxel_idx:
            self._grid_to_voxel_idx = _build_grid_index(
                self.voxel_centers, self.grid_bounds, self.voxel_resolution
            )


def generate_inverse_lut(
    calibration: CalibrationData,  # type: ignore[name-defined]
    lut_config: LutConfigLike,
    undistortion_maps: dict[str, UndistortionMaps] | None = None,  # type: ignore[name-defined]
) -> InverseLUT:
    """Generate an InverseLUT for the full cylindrical tank volume.

    Projects all voxel centres into each ring camera using
    RefractiveProjectionModel.project(), recording visibility and pixel
    coordinates. Prints a coverage histogram and memory report after generation.

    Args:
        calibration: Complete calibration data.
        lut_config: LUT generation configuration (tank geometry, voxel resolution).
        undistortion_maps: Optional per-camera undistortion maps. When provided,
            K_new is used instead of the raw K.

    Returns:
        InverseLUT with voxel centres, visibility masks, projected pixels, and
        the camera overlap graph index.
    """
    from aquapose.calibration.loader import (  # noqa: F401
        CalibrationData,
        UndistortionMaps,
    )

    # Derive tank centre from ring camera position centroid (XY only)
    cam_positions = calibration.camera_positions()
    ring_cams = calibration.ring_cameras
    xs = [float(cam_positions[c][0]) for c in ring_cams]
    ys = [float(cam_positions[c][1]) for c in ring_cams]
    tank_cx = float(np.mean(xs))
    tank_cy = float(np.mean(ys))

    # Build cylindrical voxel grid
    voxel_centers, grid_bounds = _build_cylindrical_voxel_grid(
        tank_center_xy=(tank_cx, tank_cy),
        tank_diameter=lut_config.tank_diameter,
        tank_height=lut_config.tank_height,
        water_z=calibration.water_z,
        voxel_resolution=lut_config.voxel_resolution_m,
        margin_fraction=lut_config.margin_fraction,
    )

    n_voxels = len(voxel_centers)
    n_cameras = len(ring_cams)

    visibility_mask = np.zeros((n_voxels, n_cameras), dtype=bool)
    projected_pixels = np.full((n_voxels, n_cameras, 2), np.nan, dtype=np.float32)

    voxel_tensor = torch.from_numpy(voxel_centers)  # (N, 3), float32

    for cam_idx, cam_id in enumerate(ring_cams):
        cam = calibration.cameras[cam_id]

        # Use post-undistortion K if available
        if undistortion_maps is not None and cam_id in undistortion_maps:
            K = undistortion_maps[cam_id].K_new
        else:
            K = cam.K

        model = RefractiveProjectionModel(
            K=K,
            R=cam.R,
            t=cam.t,
            water_z=calibration.water_z,
            normal=calibration.interface_normal,
            n_air=calibration.n_air,
            n_water=calibration.n_water,
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            pixels, valid = model.project(voxel_tensor)  # (N, 2), (N,)
        elapsed = time.perf_counter() - t0

        pixels_np = pixels.cpu().numpy()  # (N, 2)
        valid_np = valid.cpu().numpy()  # (N,)

        # Check within image bounds
        width, height = cam.image_size
        in_bounds = (
            valid_np
            & (pixels_np[:, 0] >= 0)
            & (pixels_np[:, 0] < width)
            & (pixels_np[:, 1] >= 0)
            & (pixels_np[:, 1] < height)
        )

        visibility_mask[:, cam_idx] = in_bounds
        projected_pixels[in_bounds, cam_idx, :] = pixels_np[in_bounds]

        logger.info("Projecting voxels into camera %s... done (%.1fs)", cam_id, elapsed)

    # Coverage histogram
    n_visible_cameras = visibility_mask.sum(axis=1)  # (N,)
    print("Camera coverage histogram:")
    for min_cams in range(1, n_cameras + 1):
        pct = float((n_visible_cameras >= min_cams).sum()) / n_voxels * 100.0
        print(f"  {min_cams}+ cameras: {pct:.1f}% of voxels")

    # Memory footprint (the forward LUT memory is not available here, so just report inverse)
    inv_mb = (
        voxel_centers.nbytes + visibility_mask.nbytes + projected_pixels.nbytes
    ) / (1024 * 1024)
    print(f"LUT memory: inverse {inv_mb:.1f} MB")

    return InverseLUT(
        voxel_centers=voxel_centers,
        visibility_mask=visibility_mask,
        projected_pixels=projected_pixels,
        camera_ids=ring_cams,
        voxel_resolution=lut_config.voxel_resolution_m,
        grid_bounds=grid_bounds,
    )


def camera_overlap_graph(
    inverse_lut: InverseLUT,
    min_shared_voxels: int = 100,
) -> dict[tuple[str, str], int]:
    """Compute a camera overlap graph from shared voxel visibility counts.

    For each camera pair, counts the number of voxels visible to both cameras.
    Only pairs exceeding min_shared_voxels are included.

    Args:
        inverse_lut: The InverseLUT containing visibility_mask.
        min_shared_voxels: Minimum number of shared voxels to include a pair.

    Returns:
        Dictionary mapping sorted camera ID pairs to shared voxel counts.
        Keys are (min(cam_a, cam_b), max(cam_a, cam_b)).
    """
    cam_ids = inverse_lut.camera_ids
    n_cameras = len(cam_ids)
    mask = inverse_lut.visibility_mask  # (N, C), bool

    graph: dict[tuple[str, str], int] = {}

    for i in range(n_cameras):
        for j in range(i + 1, n_cameras):
            shared = int((mask[:, i] & mask[:, j]).sum())
            if shared >= min_shared_voxels:
                pair = (min(cam_ids[i], cam_ids[j]), max(cam_ids[i], cam_ids[j]))
                graph[pair] = shared

    return graph


def ghost_point_lookup(
    inverse_lut: InverseLUT,
    points_3d: torch.Tensor,
) -> list[list[tuple[str, float, float]]]:
    """Look up which cameras can see each 3D point and return pixel coordinates.

    Snaps each point to the nearest voxel using integer grid coordinates.
    Points outside the cylindrical volume return an empty list.

    Args:
        inverse_lut: The InverseLUT to query.
        points_3d: 3D world points, shape (N, 3), float32.

    Returns:
        List of length N. Each element is a list of (camera_id, pixel_u, pixel_v)
        for cameras that can see the nearest voxel. Empty list for out-of-volume points.
    """
    x_min = inverse_lut.grid_bounds["x_min"]
    y_min = inverse_lut.grid_bounds["y_min"]
    z_min = inverse_lut.grid_bounds["z_min"]
    res = inverse_lut.voxel_resolution
    grid_map = inverse_lut._grid_to_voxel_idx
    cam_ids = inverse_lut.camera_ids

    pts = points_3d.cpu().numpy() if isinstance(points_3d, torch.Tensor) else points_3d
    results: list[list[tuple[str, float, float]]] = []

    for pt in pts:
        ix = round((float(pt[0]) - x_min) / res)
        iy = round((float(pt[1]) - y_min) / res)
        iz = round((float(pt[2]) - z_min) / res)

        voxel_idx = grid_map.get((ix, iy, iz), None)
        if voxel_idx is None:
            results.append([])
            continue

        visible_cams: list[tuple[str, float, float]] = []
        for cam_idx, cam_id in enumerate(cam_ids):
            if inverse_lut.visibility_mask[voxel_idx, cam_idx]:
                pix = inverse_lut.projected_pixels[voxel_idx, cam_idx]
                visible_cams.append((cam_id, float(pix[0]), float(pix[1])))

        results.append(visible_cams)

    return results


def validate_inverse_lut(
    inverse_lut: InverseLUT,
    calibration: CalibrationData,  # type: ignore[name-defined]
    n_samples: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """Validate InverseLUT projected_pixels against RefractiveProjectionModel.project().

    Samples random visible voxels and compares stored pixel coordinates with
    on-the-fly model projections.

    Args:
        inverse_lut: The InverseLUT to validate.
        calibration: CalibrationData used to reconstruct projection models.
        n_samples: Number of voxels to sample per camera.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'max_pixel_error' and 'mean_pixel_error' keys (pixels).

    Raises:
        ValueError: If max pixel error exceeds 1.0 px.
    """
    rng = np.random.default_rng(seed)
    max_error = 0.0
    errors: list[float] = []

    for cam_idx, cam_id in enumerate(inverse_lut.camera_ids):
        cam = calibration.cameras[cam_id]
        model = RefractiveProjectionModel(
            K=cam.K,
            R=cam.R,
            t=cam.t,
            water_z=calibration.water_z,
            normal=calibration.interface_normal,
            n_air=calibration.n_air,
            n_water=calibration.n_water,
        )

        visible_indices = np.where(inverse_lut.visibility_mask[:, cam_idx])[0]
        if len(visible_indices) == 0:
            continue

        sample_size = min(n_samples, len(visible_indices))
        sampled = rng.choice(visible_indices, size=sample_size, replace=False)

        centers = torch.from_numpy(inverse_lut.voxel_centers[sampled])  # (K, 3)
        stored_pix = inverse_lut.projected_pixels[sampled, cam_idx, :]  # (K, 2)

        with torch.no_grad():
            model_pix, _ = model.project(centers)

        model_pix_np = model_pix.cpu().numpy()  # (K, 2)

        # NaN in model output means invalid — skip those
        valid = ~np.isnan(model_pix_np[:, 0])
        if not valid.any():
            continue

        diffs = np.linalg.norm(stored_pix[valid] - model_pix_np[valid], axis=-1)
        errors.extend(diffs.tolist())
        max_error = max(max_error, float(diffs.max()))

    mean_error = float(np.mean(errors)) if errors else 0.0

    if max_error > 1.0:
        raise ValueError(
            f"InverseLUT validation failed: max pixel error {max_error:.4f} px "
            f"exceeds 1.0 px threshold."
        )

    return {
        "max_pixel_error": max_error,
        "mean_pixel_error": mean_error,
    }


def save_inverse_lut(
    lut: InverseLUT,
    path: str | Path,
    config_hash: str = "",
) -> None:
    """Save an InverseLUT to a .npz file.

    Args:
        lut: The InverseLUT to save.
        path: Destination file path (should end in .npz).
        config_hash: Optional hash string for cache invalidation.
    """
    # Serialise grid_to_voxel_idx as parallel coordinate arrays
    grid_keys = np.array(list(lut._grid_to_voxel_idx.keys()), dtype=np.int32)  # (M, 3)
    grid_vals = np.array(list(lut._grid_to_voxel_idx.values()), dtype=np.int32)  # (M,)

    # grid_bounds as a flat float64 array: x_min, x_max, y_min, y_max, z_min, z_max
    # Use float64 to preserve full precision of Python floats for voxel_resolution
    # and grid_bounds (float32 would lose ~1e-7 relative precision).
    bounds_arr = np.array(
        [
            lut.grid_bounds["x_min"],
            lut.grid_bounds["x_max"],
            lut.grid_bounds["y_min"],
            lut.grid_bounds["y_max"],
            lut.grid_bounds["z_min"],
            lut.grid_bounds["z_max"],
        ],
        dtype=np.float64,
    )

    np.savez(
        str(path),
        voxel_centers=lut.voxel_centers,
        visibility_mask=lut.visibility_mask,
        projected_pixels=lut.projected_pixels,
        camera_ids=np.array(lut.camera_ids),
        voxel_resolution=np.array(lut.voxel_resolution, dtype=np.float64),
        grid_bounds=bounds_arr,
        grid_keys=grid_keys,
        grid_vals=grid_vals,
        config_hash=np.array(config_hash),
    )


def load_inverse_lut(path: str | Path) -> tuple[InverseLUT, str]:
    """Load an InverseLUT from a .npz file.

    Args:
        path: Path to the .npz file written by save_inverse_lut().

    Returns:
        Tuple of (InverseLUT, config_hash) where config_hash is the stored
        hash string for cache invalidation.
    """
    data = np.load(str(path), allow_pickle=False)

    camera_ids: list[str] = [str(s) for s in data["camera_ids"]]
    voxel_resolution = float(data["voxel_resolution"])
    config_hash = str(data["config_hash"])

    bounds_arr = data["grid_bounds"]
    grid_bounds = {
        "x_min": float(bounds_arr[0]),
        "x_max": float(bounds_arr[1]),
        "y_min": float(bounds_arr[2]),
        "y_max": float(bounds_arr[3]),
        "z_min": float(bounds_arr[4]),
        "z_max": float(bounds_arr[5]),
    }

    # Reconstruct the grid index mapping
    grid_keys = data["grid_keys"]  # (M, 3)
    grid_vals = data["grid_vals"]  # (M,)
    grid_to_voxel_idx: dict[tuple[int, int, int], int] = {
        (int(k[0]), int(k[1]), int(k[2])): int(v)
        for k, v in zip(grid_keys, grid_vals, strict=False)
    }

    lut = InverseLUT(
        voxel_centers=data["voxel_centers"],
        visibility_mask=data["visibility_mask"],
        projected_pixels=data["projected_pixels"],
        camera_ids=camera_ids,
        voxel_resolution=voxel_resolution,
        grid_bounds=grid_bounds,
        _grid_to_voxel_idx=grid_to_voxel_idx,
    )

    return lut, config_hash


def save_inverse_luts(
    lut: InverseLUT,
    calibration_path: str | Path,
    lut_config: LutConfigLike,
) -> None:
    """Save an InverseLUT to the luts/ directory next to the calibration file.

    Args:
        lut: The InverseLUT to save.
        calibration_path: Path to the calibration file (determines output directory).
        lut_config: LUT config used to compute the cache-invalidation hash.
    """
    lut_dir = Path(calibration_path).parent / "luts"
    lut_dir.mkdir(parents=True, exist_ok=True)

    config_hash = compute_lut_hash(calibration_path, lut_config)
    out_path = lut_dir / "inverse.npz"
    save_inverse_lut(lut, out_path, config_hash=config_hash)
    logger.debug("Saved inverse LUT → %s", out_path)


def load_inverse_luts(
    calibration_path: str | Path,
    lut_config: LutConfigLike,
) -> InverseLUT | None:
    """Load InverseLUT from disk, returning None if cache is stale or missing.

    Args:
        calibration_path: Path to the calibration file (determines LUT directory).
        lut_config: LUT config used to check the cache-invalidation hash.

    Returns:
        InverseLUT if the cache is valid, or None if LUT is missing or stale.
    """
    lut_dir = Path(calibration_path).parent / "luts"
    inv_path = lut_dir / "inverse.npz"

    if not inv_path.exists():
        return None

    expected_hash = compute_lut_hash(calibration_path, lut_config)
    lut, stored_hash = load_inverse_lut(inv_path)

    if stored_hash != expected_hash:
        logger.info(
            "Inverse LUT cache miss (hash mismatch: stored=%s, expected=%s)",
            stored_hash,
            expected_hash,
        )
        return None

    return lut
