"""RANSAC centroid ray clustering and track-driven claiming for cross-view fish association."""

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


@dataclass
class ClaimResult:
    """Result of a track claiming detections across cameras.

    Attributes:
        track_id: Fish ID of the claiming track.
        camera_detections: Mapping from camera_id to detection index.
        centroid_3d: Triangulated 3D centroid from claimed detections.
        reprojection_residual: Mean reprojection residual (pixels).
        n_cameras: Number of cameras contributing.
    """

    track_id: int
    camera_detections: dict[str, int]
    centroid_3d: np.ndarray
    reprojection_residual: float
    n_cameras: int


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


def _merge_nearby_candidates(
    candidates: list[tuple[list[tuple[str, int]], float, torch.Tensor]],
    rays_per_camera: dict[str, tuple[torch.Tensor, torch.Tensor]],
    centroids_uv_per_camera: dict[str, list[tuple[float, float]]],
    models: dict[str, RefractiveProjectionModel],
    reprojection_threshold: float,
    merge_distance: float = 0.05,
) -> list[tuple[list[tuple[str, int]], float, torch.Tensor]]:
    """Merge candidates whose 3D centroids are within merge_distance (XY).

    When two candidates are close enough to be the same fish seen by different
    camera subsets, their inlier sets are combined and the 3D centroid is
    re-triangulated from all contributing rays. This prevents the same fish
    from producing multiple 2-camera associations.

    Args:
        candidates: List of (inliers, mean_residual, centroid_3d) tuples.
        rays_per_camera: Per-camera ray origins and directions.
        centroids_uv_per_camera: Per-camera pixel centroids.
        models: Per-camera refractive projection models.
        reprojection_threshold: Pixel threshold for re-scoring after merge.
        merge_distance: Maximum XY distance (metres) between 3D centroids
            to consider them the same fish. Default 0.05 (5 cm).

    Returns:
        Merged list of candidates with combined inlier sets.
    """
    if len(candidates) <= 1:
        return candidates

    # Union-Find for merging
    parent = list(range(len(candidates)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Find pairs to merge based on XY proximity
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            ci = candidates[i][2]  # centroid_3d
            cj = candidates[j][2]
            xy_dist = float(torch.norm(ci[:2] - cj[:2]))
            if xy_dist < merge_distance:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(len(candidates)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Build merged candidates
    merged: list[tuple[list[tuple[str, int]], float, torch.Tensor]] = []
    for member_indices in groups.values():
        if len(member_indices) == 1:
            merged.append(candidates[member_indices[0]])
            continue

        # Combine inlier sets — one detection per camera (keep the one with
        # the lowest reprojection residual if there are conflicts).
        combined_inliers: dict[str, int] = {}
        for idx in member_indices:
            for cam_id, det_idx in candidates[idx][0]:
                if cam_id not in combined_inliers:
                    combined_inliers[cam_id] = det_idx

        # Re-triangulate from all contributing rays
        ray_origins = []
        ray_dirs = []
        for cam_id, det_idx in combined_inliers.items():
            if cam_id in rays_per_camera:
                origins, directions = rays_per_camera[cam_id]
                if det_idx < origins.shape[0]:
                    ray_origins.append(origins[det_idx])
                    ray_dirs.append(directions[det_idx])

        if len(ray_origins) >= 2:
            origins_stack = torch.stack(ray_origins, dim=0)
            dirs_stack = torch.stack(ray_dirs, dim=0)
            with torch.no_grad():
                try:
                    new_centroid = triangulate_rays(origins_stack, dirs_stack)
                except Exception:
                    # Fallback: use the candidate with the most inliers
                    best = max(member_indices, key=lambda i: len(candidates[i][0]))
                    merged.append(candidates[best])
                    continue
        else:
            best = max(member_indices, key=lambda i: len(candidates[i][0]))
            merged.append(candidates[best])
            continue

        # Re-score: drop inliers that exceed reprojection_threshold after
        # re-triangulation (prevents bad detections contaminating a merge).
        validated_inliers: list[tuple[str, int]] = []
        residuals: list[float] = []
        with torch.no_grad():
            pt = new_centroid.unsqueeze(0)
            for cam_id, det_idx in combined_inliers.items():
                if cam_id not in models or cam_id not in centroids_uv_per_camera:
                    continue
                projected, valid = models[cam_id].project(pt)
                if valid[0]:
                    pu = projected[0, 0].item()
                    pv = projected[0, 1].item()
                    cu, cv = centroids_uv_per_camera[cam_id][det_idx]
                    dist = ((pu - cu) ** 2 + (pv - cv) ** 2) ** 0.5
                    if dist <= reprojection_threshold:
                        validated_inliers.append((cam_id, det_idx))
                        residuals.append(dist)

        if len(validated_inliers) >= 2:
            mean_residual = float(np.mean(residuals))
            merged.append((validated_inliers, mean_residual, new_centroid))
        else:
            # Merge made it worse — keep the best original candidate
            best = max(member_indices, key=lambda i: len(candidates[i][0]))
            merged.append(candidates[best])

    return merged


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

        # Score consensus across all cameras — include already-assigned
        # detections so the merge pass can combine overlapping candidates.
        _empty_mask: dict[str, set[int]] = {c: set() for c in centroids_uv_per_camera}
        inliers, mean_residual = _score_candidate(
            candidate_3d,
            list(centroids_uv_per_camera.keys()),
            centroids_uv_per_camera,
            models,
            _empty_mask,
            reprojection_threshold,
        )

        if len(inliers) >= min_cameras:
            # Only mark the *sampled* detections as assigned (not all inliers)
            # so future iterations can still sample from other cameras for
            # the same fish — the merge pass will unify them.
            assigned_mask[cam_a].add(det_a)
            assigned_mask[cam_b].add(det_b)
            accepted_candidates.append((inliers, mean_residual, candidate_3d))

    # --- Merge pass: combine candidates whose 3D centroids are close ---
    accepted_candidates = _merge_nearby_candidates(
        accepted_candidates,
        rays_per_camera,
        centroids_uv_per_camera,
        models,
        reprojection_threshold,
        merge_distance=0.05,
    )

    # Rebuild assigned_mask from merged candidates so low-confidence fallback
    # correctly identifies truly unassigned detections.
    assigned_mask = {cam_id: set() for cam_id in camera_ids}
    for inliers, _res, _pt in accepted_candidates:
        for cam_id, det_idx in inliers:
            assigned_mask[cam_id].add(det_idx)

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


# ---------------------------------------------------------------------------
# Track-driven claiming
# ---------------------------------------------------------------------------


def claim_detections_for_tracks(
    predicted_positions: dict[int, np.ndarray],
    track_priorities: dict[int, int],
    detections_per_camera: dict[str, list[Detection]],
    models: dict[str, RefractiveProjectionModel],
    reprojection_threshold: float = 15.0,
) -> tuple[list[ClaimResult], dict[str, list[int]]]:
    """Let existing tracks claim detections via reprojection proximity.

    For each track, projects its predicted 3D position into each camera and
    finds the nearest detection centroid within ``reprojection_threshold``.
    Uses greedy assignment sorted by (pixel_distance, priority) to resolve
    conflicts — confirmed tracks get priority over probationary.

    Args:
        predicted_positions: Mapping from track fish_id to predicted 3D
            position, shape (3,).
        track_priorities: Mapping from track fish_id to priority (0=high,
            1=low). Confirmed/coasting tracks should be 0.
        detections_per_camera: Per-camera detection lists.
        models: Per-camera refractive projection models.
        reprojection_threshold: Max pixel distance for a claim.

    Returns:
        Tuple of (claims, unclaimed_indices):
        - claims: List of ClaimResult, one per track that claimed detections.
        - unclaimed_indices: Dict mapping camera_id to list of unclaimed
          detection indices.
    """
    if not predicted_positions or not detections_per_camera:
        unclaimed = {
            cam: list(range(len(dets)))
            for cam, dets in detections_per_camera.items()
            if len(dets) > 0
        }
        return [], unclaimed

    # Precompute mask centroids per camera
    centroids_uv_per_camera: dict[str, list[tuple[float, float]]] = {}
    for cam_id, dets in detections_per_camera.items():
        if not dets:
            continue
        centroids = []
        for det in dets:
            try:
                centroids.append(_compute_mask_centroid(det.mask))
            except ValueError:
                centroids.append((0.0, 0.0))
        centroids_uv_per_camera[cam_id] = centroids

    # Build candidate assignments: (pixel_distance, priority, track_id, cam_id, det_idx)
    candidates: list[tuple[float, int, int, str, int]] = []

    for track_id, pred_3d in predicted_positions.items():
        pt = torch.tensor(pred_3d, dtype=torch.float32).unsqueeze(0)  # (1, 3)
        priority = track_priorities.get(track_id, 1)

        with torch.no_grad():
            for cam_id, centroids in centroids_uv_per_camera.items():
                if cam_id not in models:
                    continue
                model = models[cam_id]
                projected, valid = model.project(pt)
                if not valid[0]:
                    continue
                proj_u = projected[0, 0].item()
                proj_v = projected[0, 1].item()

                for det_idx, (u, v) in enumerate(centroids):
                    dist = ((u - proj_u) ** 2 + (v - proj_v) ** 2) ** 0.5
                    if dist < reprojection_threshold:
                        candidates.append((dist, priority, track_id, cam_id, det_idx))

    # Sort by (distance, priority) ascending — closest first, high-priority first
    candidates.sort(key=lambda x: (x[0], x[1]))

    # Greedy assign: skip if detection already taken or track already has that camera
    taken_detections: dict[str, set[int]] = {}  # cam_id -> set of det_idx
    track_claims: dict[int, dict[str, int]] = {}  # track_id -> {cam_id: det_idx}

    for _dist, _priority, track_id, cam_id, det_idx in candidates:
        if det_idx in taken_detections.get(cam_id, set()):
            continue
        if cam_id in track_claims.get(track_id, {}):
            continue

        taken_detections.setdefault(cam_id, set()).add(det_idx)
        track_claims.setdefault(track_id, {})[cam_id] = det_idx

    # Cast rays and triangulate for each track's claimed detections
    # Pre-build rays per camera (only for cameras with claimed detections)
    rays_per_camera: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for cam_id in taken_detections:
        if cam_id not in models or cam_id not in detections_per_camera:
            continue
        dets = detections_per_camera[cam_id]
        if dets and cam_id not in rays_per_camera:
            origins, directions, _ = _cast_rays_for_detections(dets, models[cam_id])
            if origins.shape[0] > 0:
                rays_per_camera[cam_id] = (origins, directions)

    claims: list[ClaimResult] = []
    for track_id, cam_dets in track_claims.items():
        n_cameras = len(cam_dets)

        if n_cameras >= 2:
            # Triangulate from claimed rays
            ray_origins = []
            ray_dirs = []
            for cam_id, det_idx in cam_dets.items():
                if cam_id in rays_per_camera:
                    origins, directions = rays_per_camera[cam_id]
                    if det_idx < origins.shape[0]:
                        ray_origins.append(origins[det_idx])
                        ray_dirs.append(directions[det_idx])

            if len(ray_origins) >= 2:
                origins_stack = torch.stack(ray_origins, dim=0)
                dirs_stack = torch.stack(ray_dirs, dim=0)
                with torch.no_grad():
                    try:
                        centroid_3d = triangulate_rays(origins_stack, dirs_stack)
                    except Exception:
                        # Fallback: use predicted position
                        centroid_3d_np = predicted_positions[track_id].copy()
                        claims.append(
                            ClaimResult(
                                track_id=track_id,
                                camera_detections=cam_dets,
                                centroid_3d=centroid_3d_np,
                                reprojection_residual=0.0,
                                n_cameras=n_cameras,
                            )
                        )
                        continue

                # Compute reprojection residual
                residuals: list[float] = []
                pt = centroid_3d.unsqueeze(0)
                with torch.no_grad():
                    for cam_id, det_idx in cam_dets.items():
                        if (
                            cam_id not in models
                            or cam_id not in centroids_uv_per_camera
                        ):
                            continue
                        projected, valid = models[cam_id].project(pt)
                        if valid[0]:
                            pu = projected[0, 0].item()
                            pv = projected[0, 1].item()
                            cu, cv = centroids_uv_per_camera[cam_id][det_idx]
                            dist = ((pu - cu) ** 2 + (pv - cv) ** 2) ** 0.5
                            residuals.append(dist)

                mean_residual = float(np.mean(residuals)) if residuals else 0.0
                claims.append(
                    ClaimResult(
                        track_id=track_id,
                        camera_detections=cam_dets,
                        centroid_3d=centroid_3d.numpy(),
                        reprojection_residual=mean_residual,
                        n_cameras=n_cameras,
                    )
                )
            else:
                # Not enough rays — use predicted position
                claims.append(
                    ClaimResult(
                        track_id=track_id,
                        camera_detections=cam_dets,
                        centroid_3d=predicted_positions[track_id].copy(),
                        reprojection_residual=0.0,
                        n_cameras=n_cameras,
                    )
                )
        else:
            # Single camera — ray-depth heuristic
            cam_id = next(iter(cam_dets))
            det_idx = cam_dets[cam_id]
            if cam_id in rays_per_camera:
                origins, directions = rays_per_camera[cam_id]
                if det_idx < origins.shape[0]:
                    with torch.no_grad():
                        centroid_3d_np = (
                            origins[det_idx] + _DEFAULT_TANK_DEPTH * directions[det_idx]
                        ).numpy()
                else:
                    centroid_3d_np = predicted_positions[track_id].copy()
            else:
                centroid_3d_np = predicted_positions[track_id].copy()

            claims.append(
                ClaimResult(
                    track_id=track_id,
                    camera_detections=cam_dets,
                    centroid_3d=centroid_3d_np,
                    reprojection_residual=0.0,
                    n_cameras=1,
                )
            )

    # Build unclaimed indices
    unclaimed_indices: dict[str, list[int]] = {}
    for cam_id, dets in detections_per_camera.items():
        if not dets:
            continue
        taken = taken_detections.get(cam_id, set())
        unclaimed = [i for i in range(len(dets)) if i not in taken]
        if unclaimed:
            unclaimed_indices[cam_id] = unclaimed

    return claims, unclaimed_indices


def discover_births(
    unclaimed_indices: dict[str, list[int]],
    detections_per_camera: dict[str, list[Detection]],
    models: dict[str, RefractiveProjectionModel],
    expected_count: int = 9,
    reprojection_threshold: float = 15.0,
    min_cameras: int = 2,
    n_iter: int = 200,
) -> list[AssociationResult]:
    """Run RANSAC on unclaimed detections to discover new fish.

    Filters ``detections_per_camera`` to only unclaimed indices, then calls
    ``ransac_centroid_cluster`` to find new associations.

    Args:
        unclaimed_indices: Dict mapping camera_id to list of unclaimed
            detection indices.
        detections_per_camera: Full per-camera detection lists.
        models: Per-camera refractive projection models.
        expected_count: Expected fish count for RANSAC.
        reprojection_threshold: Pixel threshold for RANSAC.
        min_cameras: Minimum cameras for a valid birth.
        n_iter: RANSAC iterations.

    Returns:
        List of AssociationResult for newly discovered fish.
    """
    if not unclaimed_indices:
        return []

    # Build filtered detection lists containing only unclaimed detections.
    # We need to track the original indices for correct ray casting.
    filtered_dets: dict[str, list[Detection]] = {}
    for cam_id, indices in unclaimed_indices.items():
        if cam_id not in detections_per_camera:
            continue
        all_dets = detections_per_camera[cam_id]
        filtered_dets[cam_id] = [all_dets[i] for i in indices]

    if not filtered_dets:
        return []

    result = ransac_centroid_cluster(
        detections_per_camera=filtered_dets,
        models=models,
        expected_count=expected_count,
        n_iter=n_iter,
        reprojection_threshold=reprojection_threshold,
        min_cameras=min_cameras,
    )

    # Filter to high-confidence associations only (births require multi-view)
    return [a for a in result.associations if not a.is_low_confidence]
