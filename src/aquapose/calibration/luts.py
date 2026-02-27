"""Forward lookup table (pixel to refracted ray) for fast ray casting."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
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
