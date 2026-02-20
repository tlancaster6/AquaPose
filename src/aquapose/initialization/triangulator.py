"""Multi-camera keypoint triangulation and FishState estimation."""

from __future__ import annotations

import numpy as np
import torch

from aquapose.calibration.projection import RefractiveProjectionModel, triangulate_rays
from aquapose.initialization.keypoints import extract_keypoints
from aquapose.mesh.state import FishState


def triangulate_keypoint(
    pixel_coords: list[tuple[float, float]],
    models: list[RefractiveProjectionModel],
) -> torch.Tensor:
    """Triangulate a single 2D keypoint from multiple camera views.

    Uses refractive ray casting (not pinhole) from Phase 1.

    Args:
        pixel_coords: (u, v) pixel coords, one per camera. Length >= 3.
        models: RefractiveProjectionModel instances, one per camera.

    Returns:
        point_3d: Triangulated 3D point, shape (3,), float32.

    Raises:
        ValueError: If fewer than 3 camera observations are provided.
    """
    if len(pixel_coords) < 3:
        raise ValueError(
            f"Need at least 3 cameras for triangulation, got {len(pixel_coords)}"
        )

    all_origins = []
    all_directions = []

    for (u, v), model in zip(pixel_coords, models, strict=True):
        pixel_tensor = torch.tensor([[u, v]], dtype=torch.float32)
        origins, directions = model.cast_ray(pixel_tensor)
        all_origins.append(origins)  # (1, 3)
        all_directions.append(directions)  # (1, 3)

    origins_stacked = torch.cat(all_origins, dim=0)  # (N, 3)
    directions_stacked = torch.cat(all_directions, dim=0)  # (N, 3)

    return triangulate_rays(origins_stacked, directions_stacked)


def init_fish_state(
    center_3d: torch.Tensor,
    endpoint_a_3d: torch.Tensor,
    endpoint_b_3d: torch.Tensor,
) -> FishState:
    """Estimate a FishState from 3 triangulated keypoints.

    Heading is taken from endpoint_b toward endpoint_a (canonical direction
    from extract_keypoints). Head/tail disambiguation deferred to Phase 4
    (2-start forward + 180 flip).

    Args:
        center_3d: Fish center in world frame, shape (3,).
        endpoint_a_3d: One endpoint, shape (3,).
        endpoint_b_3d: Other endpoint, shape (3,).

    Returns:
        FishState with p=center, psi/theta from heading direction,
        kappa=0 (straight), s=distance between endpoints.
    """
    # Axis vector from endpoint_b to endpoint_a (canonical direction)
    axis = endpoint_a_3d - endpoint_b_3d  # (3,)
    length = torch.linalg.norm(axis)
    unit_heading = axis / (length + 1e-8)

    # Yaw from XY components
    psi = torch.atan2(unit_heading[1], unit_heading[0])

    # Pitch from Z component (positive Z = down in AquaPose)
    theta = torch.asin(unit_heading[2].clamp(-1.0, 1.0))

    # Initialize as straight fish
    kappa = torch.zeros((), dtype=torch.float32)

    return FishState(
        p=center_3d.float(),
        psi=psi.float(),
        theta=theta.float(),
        kappa=kappa,
        s=length.float(),
    )


def init_fish_states_from_masks(
    masks_per_camera: list[list[np.ndarray | None]],
    models: list[RefractiveProjectionModel],
) -> list[FishState]:
    """Full pipeline: masks from N cameras → list of FishStates.

    Args:
        masks_per_camera: For each camera, a list of masks (one per fish).
            None entries mean that fish is not visible in that camera.
            Shape: [n_cameras][n_fish] where each entry is (H, W) or None.
        models: RefractiveProjectionModel, one per camera.

    Returns:
        List of FishState, one per fish.

    Raises:
        ValueError: If fewer than 3 cameras have valid masks for any fish.
    """
    n_cameras = len(masks_per_camera)
    if n_cameras == 0:
        return []

    n_fish = len(masks_per_camera[0])
    states: list[FishState] = []

    for fish_idx in range(n_fish):
        # Collect cameras with valid masks for this fish
        valid_cameras: list[tuple[int, np.ndarray]] = []
        for cam_idx in range(n_cameras):
            mask = masks_per_camera[cam_idx][fish_idx]
            if mask is not None:
                valid_cameras.append((cam_idx, mask))

        if len(valid_cameras) < 3:
            raise ValueError(
                f"Fish {fish_idx} has only {len(valid_cameras)} valid cameras; "
                "need at least 3 for triangulation."
            )

        # Extract keypoints from each valid camera
        center_pixels: list[tuple[float, float]] = []
        ep_a_pixels: list[tuple[float, float]] = []
        ep_b_pixels: list[tuple[float, float]] = []
        cam_models: list[RefractiveProjectionModel] = []

        for cam_idx, mask in valid_cameras:
            center, ep_a, ep_b = extract_keypoints(mask)
            center_pixels.append((float(center[0]), float(center[1])))
            ep_a_pixels.append((float(ep_a[0]), float(ep_a[1])))
            ep_b_pixels.append((float(ep_b[0]), float(ep_b[1])))
            cam_models.append(models[cam_idx])

        # Triangulate each keypoint separately
        center_3d = triangulate_keypoint(center_pixels, cam_models)
        ep_a_3d = triangulate_keypoint(ep_a_pixels, cam_models)
        ep_b_3d = triangulate_keypoint(ep_b_pixels, cam_models)

        state = init_fish_state(center_3d, ep_a_3d, ep_b_3d)
        states.append(state)

    return states
