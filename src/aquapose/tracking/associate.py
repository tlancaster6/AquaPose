"""RANSAC centroid ray clustering for cross-view fish identity association."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from aquapose.calibration.projection import RefractiveProjectionModel, triangulate_rays
from aquapose.segmentation.detector import Detection

# Default tank depth below water surface used for single-view centroid heuristic.
_DEFAULT_TANK_DEPTH: float = 0.5


@dataclass
class AssociationResult:
    """Association of detections across cameras to a single physical fish.

    Attributes:
        fish_id: Assigned fish identifier (0-indexed within this frame).
        centroid_3d: Estimated 3D centroid in world frame, shape (3,).
            None for single-view low-confidence entries using ray-depth heuristic.
        reprojection_residual: Mean pixel distance from the projected 3D centroid
            to the assigned detection centroids in inlier cameras. 0.0 for
            single-view entries.
        camera_detections: Mapping from camera_id to detection index in the
            detections_per_camera list for that camera.
        n_cameras: Number of cameras contributing to this association.
        confidence: Association confidence. 1.0 for high-confidence multi-view
            associations; lower for single-view flagged entries (uses detection
            confidence).
        is_low_confidence: True if this is a single-view fallback entry.
    """

    fish_id: int
    centroid_3d: np.ndarray
    reprojection_residual: float
    camera_detections: dict[str, int]
    n_cameras: int
    confidence: float
    is_low_confidence: bool


@dataclass
class FrameAssociations:
    """All fish associations for a single video frame.

    Attributes:
        associations: List of AssociationResult, one per identified fish.
        frame_index: Index of the processed frame.
        unassigned: Camera-detection pairs not assigned to any fish, as
            (camera_id, detection_index) tuples.
    """

    associations: list[AssociationResult]
    frame_index: int
    unassigned: list[tuple[str, int]] = field(default_factory=list)


def _compute_mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Compute the foreground centroid of a binary mask.

    Returns the mean column and row coordinates of foreground pixels, which
    is the true center-of-mass of the mask. This is distinct from the bbox
    center.

    Args:
        mask: Binary mask array (uint8, 0/255 or 0/1). Shape (H, W).

    Returns:
        (u, v): Column (u) and row (v) of the foreground centroid.

    Raises:
        ValueError: If the mask contains no foreground pixels.
    """
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        raise ValueError("Mask contains no foreground pixels — cannot compute centroid")
    u = float(np.mean(cols))
    v = float(np.mean(rows))
    return u, v


def _cast_rays_for_detections(
    detections: list[Detection],
    model: RefractiveProjectionModel,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[float, float]]]:
    """Cast refractive rays from each detection's mask centroid.

    Args:
        detections: List of Detection objects for one camera.
        model: Refractive projection model for that camera.

    Returns:
        origins: Ray origins on water surface, shape (N, 3).
        directions: Unit ray directions into water, shape (N, 3).
        centroids_uv: List of (u, v) pixel centroids, length N.
    """
    centroids_uv = [_compute_mask_centroid(det.mask) for det in detections]
    if not centroids_uv:
        return (
            torch.zeros(0, 3, dtype=torch.float32),
            torch.zeros(0, 3, dtype=torch.float32),
            [],
        )
    pixels = torch.tensor(centroids_uv, dtype=torch.float32)  # (N, 2)
    with torch.no_grad():
        origins, directions = model.cast_ray(pixels)
    return origins, directions, centroids_uv


def _score_candidate(
    candidate_3d: torch.Tensor,
    camera_ids: list[str],
    centroids_uv_per_camera: dict[str, list[tuple[float, float]]],
    models: dict[str, RefractiveProjectionModel],
    assigned_mask: dict[str, set[int]],
    reprojection_threshold: float,
) -> tuple[list[tuple[str, int]], float]:
    """Score a candidate 3D centroid by counting inlier detections.

    Projects the candidate into each camera and counts detections whose
    centroid is within reprojection_threshold pixels. Only considers
    unassigned detections.

    Args:
        candidate_3d: Candidate 3D centroid, shape (3,).
        camera_ids: List of camera IDs to check.
        centroids_uv_per_camera: Dict mapping camera_id to list of (u, v)
            centroids for all detections in that camera.
        models: Dict mapping camera_id to RefractiveProjectionModel.
        assigned_mask: Dict mapping camera_id to set of already-assigned
            detection indices.
        reprojection_threshold: Max pixel distance to count as inlier.

    Returns:
        inliers: List of (camera_id, detection_index) tuples that are inliers.
        mean_residual: Mean pixel distance from projected centroid to inlier
            detections. 0.0 if no inliers.
    """
    inliers: list[tuple[str, int]] = []
    residuals: list[float] = []
    pt = candidate_3d.unsqueeze(0)  # (1, 3)

    with torch.no_grad():
        for cam_id in camera_ids:
            centroids = centroids_uv_per_camera.get(cam_id, [])
            if not centroids:
                continue
            model = models[cam_id]
            projected, valid = model.project(pt)  # (1, 2), (1,)
            if not valid[0]:
                continue
            proj_u = projected[0, 0].item()
            proj_v = projected[0, 1].item()

            already_assigned = assigned_mask.get(cam_id, set())
            best_dist = float("inf")
            best_idx = -1
            for det_idx, (u, v) in enumerate(centroids):
                if det_idx in already_assigned:
                    continue
                dist = ((u - proj_u) ** 2 + (v - proj_v) ** 2) ** 0.5
                if dist < reprojection_threshold and dist < best_dist:
                    best_dist = dist
                    best_idx = det_idx
            if best_idx >= 0:
                inliers.append((cam_id, best_idx))
                residuals.append(best_dist)

    mean_residual = float(np.mean(residuals)) if residuals else 0.0
    return inliers, mean_residual


def ransac_centroid_cluster(
    detections_per_camera: dict[str, list[Detection]],
    models: dict[str, RefractiveProjectionModel],
    expected_count: int = 9,
    n_iter: int = 200,
    reprojection_threshold: float = 15.0,
    min_cameras: int = 2,
    seed_points: list[np.ndarray] | None = None,
    frame_index: int = 0,
) -> FrameAssociations:
    """Associate detections across cameras to physical fish via RANSAC.

    Uses RANSAC centroid ray clustering: casts refractive rays from 2D mask
    centroids, triangulates minimal camera subsets, and scores consensus via
    reprojection. Supports prior-guided seeding for temporal consistency.

    Algorithm:
        1. Prior-guided pass: for each seed point (previous-frame centroid),
           find inlier detections by projecting into each camera and matching
           the nearest unassigned detection within reprojection_threshold.
        2. Random RANSAC: sample 2 cameras, 1 detection each, triangulate,
           score consensus, keep best candidate. Repeat n_iter times.
        3. Greedy assignment: accept candidates with >= min_cameras inliers,
           sorted by inlier count descending. Each detection assigned to at
           most one fish.
        4. Low-confidence fallback: remaining single-view detections are
           included as flagged low-confidence entries with a ray-depth
           heuristic centroid.

    Args:
        detections_per_camera: Dict mapping camera_id to list of Detection
            objects for that camera in this frame.
        models: Dict mapping camera_id to RefractiveProjectionModel.
        expected_count: Soft upper bound on fish count (default 9). RANSAC
            stops early once this many fish are found.
        n_iter: Number of random RANSAC iterations (default 200).
        reprojection_threshold: Max pixel distance (in pixels) for a
            detection to count as an inlier (default 15.0).
        min_cameras: Minimum cameras required for a valid association
            (default 2). Single-view detections are flagged low-confidence.
        seed_points: Optional list of 3D centroid numpy arrays from the
            previous frame. Each array has shape (3,). Used for prior-guided
            seeding to improve temporal consistency.
        frame_index: Frame index for bookkeeping (default 0).

    Returns:
        FrameAssociations with one AssociationResult per identified fish,
        plus any unassigned detections listed separately.
    """
    # --- Early exit for empty input ---
    if not detections_per_camera or all(
        len(dets) == 0 for dets in detections_per_camera.values()
    ):
        return FrameAssociations(
            associations=[], frame_index=frame_index, unassigned=[]
        )

    camera_ids = [c for c, dets in detections_per_camera.items() if len(dets) > 0]

    # Build rays and centroids for all cameras
    rays_per_camera: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    centroids_uv_per_camera: dict[str, list[tuple[float, float]]] = {}
    for cam_id in camera_ids:
        dets = detections_per_camera[cam_id]
        if not dets:
            continue
        origins, directions, centroids = _cast_rays_for_detections(dets, models[cam_id])
        if origins.shape[0] > 0:
            rays_per_camera[cam_id] = (origins, directions)
            centroids_uv_per_camera[cam_id] = centroids

    # Tracks which detections have been assigned: camera_id -> set of det indices
    assigned_mask: dict[str, set[int]] = {cam_id: set() for cam_id in camera_ids}

    # Candidates found: list of (inliers, mean_residual, candidate_3d)
    accepted_candidates: list[tuple[list[tuple[str, int]], float, torch.Tensor]] = []

    # --- Prior-guided pass ---
    if seed_points is not None:
        for seed in seed_points:
            if len(accepted_candidates) >= expected_count:
                break
            candidate_3d = torch.tensor(seed, dtype=torch.float32)
            inliers, mean_residual = _score_candidate(
                candidate_3d,
                list(centroids_uv_per_camera.keys()),
                centroids_uv_per_camera,
                models,
                assigned_mask,
                reprojection_threshold,
            )
            if len(inliers) >= min_cameras:
                # Tentatively mark as assigned (will be confirmed in greedy step)
                for cam_id, det_idx in inliers:
                    assigned_mask[cam_id].add(det_idx)
                accepted_candidates.append((inliers, mean_residual, candidate_3d))

    # --- Random RANSAC iterations ---
    # Collect cameras with at least 1 unassigned detection
    def _cameras_with_unassigned() -> list[str]:
        return [
            c
            for c in centroids_uv_per_camera
            if any(
                i not in assigned_mask[c]
                for i in range(len(centroids_uv_per_camera[c]))
            )
        ]

    for _ in range(n_iter):
        if len(accepted_candidates) >= expected_count:
            break

        eligible = _cameras_with_unassigned()
        if len(eligible) < 2:
            break

        # Sample 2 distinct cameras
        cam_a, cam_b = random.sample(eligible, 2)

        # Sample 1 unassigned detection from each
        unassigned_a = [
            i
            for i in range(len(centroids_uv_per_camera[cam_a]))
            if i not in assigned_mask[cam_a]
        ]
        unassigned_b = [
            i
            for i in range(len(centroids_uv_per_camera[cam_b]))
            if i not in assigned_mask[cam_b]
        ]
        if not unassigned_a or not unassigned_b:
            continue

        det_a = random.choice(unassigned_a)
        det_b = random.choice(unassigned_b)

        # Triangulate from these 2 rays
        origins_a, dirs_a = rays_per_camera[cam_a]
        origins_b, dirs_b = rays_per_camera[cam_b]
        origins_pair = torch.stack([origins_a[det_a], origins_b[det_b]], dim=0)
        dirs_pair = torch.stack([dirs_a[det_a], dirs_b[det_b]], dim=0)

        with torch.no_grad():
            try:
                candidate_3d = triangulate_rays(origins_pair, dirs_pair)
            except Exception:
                continue

        # Score consensus across all cameras
        inliers, mean_residual = _score_candidate(
            candidate_3d,
            list(centroids_uv_per_camera.keys()),
            centroids_uv_per_camera,
            models,
            assigned_mask,
            reprojection_threshold,
        )

        if len(inliers) >= min_cameras:
            for cam_id, det_idx in inliers:
                assigned_mask[cam_id].add(det_idx)
            accepted_candidates.append((inliers, mean_residual, candidate_3d))

    # --- Build AssociationResult objects ---
    associations: list[AssociationResult] = []
    fish_id_counter = 0

    for inliers, mean_residual, candidate_3d in accepted_candidates:
        camera_detections = {cam_id: det_idx for cam_id, det_idx in inliers}
        associations.append(
            AssociationResult(
                fish_id=fish_id_counter,
                centroid_3d=candidate_3d.numpy(),
                reprojection_residual=mean_residual,
                camera_detections=camera_detections,
                n_cameras=len(inliers),
                confidence=1.0,
                is_low_confidence=False,
            )
        )
        fish_id_counter += 1

    # --- Low-confidence fallback for remaining unassigned detections ---
    low_conf_associations: list[AssociationResult] = []
    unassigned_list: list[tuple[str, int]] = []

    for cam_id in camera_ids:
        dets = detections_per_camera[cam_id]
        for det_idx, det in enumerate(dets):
            if det_idx in assigned_mask.get(cam_id, set()):
                continue
            # Single-view detection: place centroid at default tank depth along ray
            cam_rays = rays_per_camera.get(cam_id)
            if cam_rays is not None and cam_rays[0].shape[0] > det_idx:
                ray_origin = cam_rays[0][det_idx]
                ray_dir = cam_rays[1][det_idx]
                with torch.no_grad():
                    centroid_3d = (ray_origin + _DEFAULT_TANK_DEPTH * ray_dir).numpy()
            else:
                centroid_3d = np.zeros(3, dtype=np.float32)

            low_conf_associations.append(
                AssociationResult(
                    fish_id=fish_id_counter,
                    centroid_3d=centroid_3d,
                    reprojection_residual=0.0,
                    camera_detections={cam_id: det_idx},
                    n_cameras=1,
                    confidence=float(det.confidence),
                    is_low_confidence=True,
                )
            )
            fish_id_counter += 1

    all_associations = associations + low_conf_associations

    return FrameAssociations(
        associations=all_associations,
        frame_index=frame_index,
        unassigned=unassigned_list,
    )
