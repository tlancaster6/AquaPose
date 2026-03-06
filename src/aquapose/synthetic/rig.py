"""Fabricated camera rig builder for synthetic data generation.

Creates rectangular grids of synthetic cameras suitable for controlled
testing of the triangulation and curve optimization pipelines without
real calibration data.
"""

from __future__ import annotations

import torch

from aquapose.calibration.projection import RefractiveProjectionModel


def build_fabricated_rig(
    n_cameras_x: int = 3,
    n_cameras_y: int = 3,
    spacing_x: float = 0.4,
    spacing_y: float = 0.4,
    water_z: float = 0.75,
    n_air: float = 1.0,
    n_water: float = 1.333,
    fx: float = 1400.0,
    cx: float = 800.0,
    cy: float = 600.0,
) -> dict[str, RefractiveProjectionModel]:
    """Build a rectangular grid of synthetic downward-looking cameras.

    Creates ``n_cameras_x * n_cameras_y`` cameras arranged in a grid at
    Z=0 (above the water surface). Each camera uses identity rotation
    (looks straight down) and is positioned symmetrically around the
    origin. Camera IDs are assigned as ``"syn_00"``, ``"syn_01"``, etc.

    Args:
        n_cameras_x: Number of cameras along the X axis.
        n_cameras_y: Number of cameras along the Y axis.
        spacing_x: Inter-camera spacing in metres along X.
        spacing_y: Inter-camera spacing in metres along Y.
        water_z: Z coordinate of the water surface in world frame (metres).
        n_air: Refractive index of air.
        n_water: Refractive index of water.
        fx: Focal length in pixels (used for both fx and fy).
        cx: Principal point x coordinate in pixels.
        cy: Principal point y coordinate in pixels.

    Returns:
        Dict mapping camera ID strings to RefractiveProjectionModel instances.
        Camera IDs are zero-padded two-digit strings prefixed with ``"syn_"``.
    """
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)

    # Compute grid offsets so cameras are centred around origin
    x_offsets = [(i - (n_cameras_x - 1) / 2.0) * spacing_x for i in range(n_cameras_x)]
    y_offsets = [(j - (n_cameras_y - 1) / 2.0) * spacing_y for j in range(n_cameras_y)]

    models: dict[str, RefractiveProjectionModel] = {}
    idx = 0
    for cam_y in y_offsets:
        for cam_x in x_offsets:
            cam_id = f"syn_{idx:02d}"
            t = torch.tensor([-cam_x, -cam_y, 0.0], dtype=torch.float32)
            models[cam_id] = RefractiveProjectionModel(
                K=K.clone(),
                R=R.clone(),
                t=t,
                water_z=water_z,
                normal=normal.clone(),
                n_air=n_air,
                n_water=n_water,
            )
            idx += 1

    return models
