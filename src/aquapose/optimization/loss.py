"""Soft IoU loss, angular diversity weighting, and multi-objective loss for pose optimization."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from aquapose.mesh.state import FishState


def soft_iou_loss(
    pred_alpha: torch.Tensor,
    target_mask: torch.Tensor,
    crop_region: tuple[int, int, int, int] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute 1 - soft IoU between a predicted silhouette and a binary target mask.

    Computes the differentiable soft IoU within an optional bounding-box crop region.
    Restricting to the crop region focuses gradients on the fish and avoids the
    background dominating the union denominator.

    Args:
        pred_alpha: Predicted silhouette alpha map, shape (H, W), float32, values
            in [0, 1].
        target_mask: Observed binary mask, shape (H, W), float32, values in {0, 1}.
        crop_region: Optional ``(y1, x1, y2, x2)`` pixel coordinates to restrict IoU
            computation to the fish bounding box. If None, the full frame is used.
        eps: Numerical stability constant added to the union denominator.

    Returns:
        Scalar tensor: ``1 - soft_IoU`` in range [0, 1]. Lower is better.
            Differentiable with respect to ``pred_alpha``.
    """
    if crop_region is not None:
        y1, x1, y2, x2 = crop_region
        pred_alpha = pred_alpha[y1:y2, x1:x2]
        target_mask = target_mask[y1:y2, x1:x2]

    intersection = (pred_alpha * target_mask).sum()
    union = (pred_alpha + target_mask - pred_alpha * target_mask).sum()
    return 1.0 - intersection / (union + eps)


def compute_angular_diversity_weights(
    models: list,
    camera_ids: list[str],
    temperature: float = 0.5,
) -> dict[str, float]:
    """Compute per-camera weights based on angular separation in view direction.

    Cameras whose viewing directions are clustered together (e.g., ring cameras
    pointing inward at similar angles) receive lower weights to prevent them from
    dominating the multi-view IoU loss over cameras with more unique viewpoints.

    Each camera's view direction is the world-space direction it looks along,
    computed as ``R.T @ [0, 0, -1]`` where R is the world-to-camera rotation.

    For camera i the weight is proportional to its minimum angular separation from
    all other cameras, raised to the power ``temperature``::

        min_angle_i = min_{j != i} arccos(clip(dot(v_i, v_j), -1, 1))
        weight_i = (min_angle_i / max_j(min_angle_j)) ** temperature

    The camera with the most unique viewpoint receives weight 1.0.  Weights are
    computed once (not per optimiser iteration) using numpy.

    Args:
        models: List of ``RefractiveProjectionModel`` instances (one per camera).
            Each model must have a ``.R`` attribute, shape (3, 3) float32 tensor.
        camera_ids: String identifier for each camera.
        temperature: Exponent applied after normalisation.  Values in (0, 1) compress
            the spread (clustered cameras get relatively higher weight); values > 1
            exaggerate differences (clustered cameras get lower relative weight).

    Returns:
        Dict mapping camera_id -> weight in (0, 1].

    Raises:
        ValueError: If ``models`` and ``camera_ids`` have different lengths or if
            fewer than 2 cameras are provided.
    """
    if len(models) != len(camera_ids):
        raise ValueError(
            f"models ({len(models)}) and camera_ids ({len(camera_ids)}) "
            "must have the same length"
        )
    if len(models) < 2:
        # Single camera: give it full weight.
        return {camera_ids[0]: 1.0}

    # Extract world-space view directions: camera looks along -Z in camera frame.
    view_dirs: list[np.ndarray] = []
    for model in models:
        r_mat = model.R.detach().cpu().numpy().astype(np.float64)  # (3, 3)
        view_dir = r_mat.T @ np.array([0.0, 0.0, -1.0])  # (3,) world-space
        norm = np.linalg.norm(view_dir)
        view_dirs.append(view_dir / (norm + 1e-12))
    view_mat = np.stack(view_dirs, axis=0)  # (N, 3)

    # Pairwise dot products -> angles.
    dots = view_mat @ view_mat.T  # (N, N)
    dots = np.clip(dots, -1.0, 1.0)

    # For each camera find the minimum angle to any OTHER camera.
    min_angles = np.zeros(len(models))
    for i in range(len(models)):
        row = dots[i].copy()
        row[i] = -2.0  # sentinel: exclude self (impossible dot product value)
        most_similar_dot = row.max()
        min_angles[i] = np.arccos(most_similar_dot)

    # Normalise: most isolated camera -> 1.0; apply temperature.
    max_min_angle = min_angles.max()
    if max_min_angle < 1e-8:
        # All cameras point in nearly identical directions; give uniform weights.
        weights = np.ones(len(models))
    else:
        weights = np.power(min_angles / max_min_angle, temperature)

    return {cam_id: float(w) for cam_id, w in zip(camera_ids, weights, strict=True)}


def multi_objective_loss(
    state: FishState,
    pred_alphas: dict[str, torch.Tensor],
    target_masks: dict[str, torch.Tensor],
    crop_regions: dict[str, tuple[int, int, int, int] | None],
    camera_weights: dict[str, float],
    loss_weights: dict[str, float],
    temporal_state: FishState | None = None,
    temporal_weight: float = 0.1,
    kappa_max: float = 10.0,
    s_min: float = 0.05,
    s_max: float = 0.30,
) -> dict[str, torch.Tensor]:
    """Compute multi-objective loss for analysis-by-synthesis pose optimisation.

    Combines four loss terms:

    1. **Silhouette IoU** — weighted average of per-camera soft IoU losses
       (normalised by the sum of camera weights).
    2. **Gravity prior** — penalises pitch deviation from horizontal as a proxy for
       roll regularisation. ``loss = state.theta ** 2``.  Low-weight soft constraint
       to break pose ambiguities.
    3. **Morphological constraints** — keeps curvature and scale within biologically
       plausible ranges via soft L2 penalties outside the bounds.
    4. **Temporal smoothness** — penalises positional and heading change relative to
       the previous-frame state.  Architecturally present but inactive in Phase 4
       (pass ``temporal_state=None``).

    Args:
        state: Current optimisable ``FishState``.
        pred_alphas: Dict camera_id -> rendered alpha map, shape (H, W), float32.
        target_masks: Dict camera_id -> binary mask, shape (H, W), float32.
        crop_regions: Dict camera_id -> ``(y1, x1, y2, x2)`` or None.  Restricts
            IoU computation to the fish bounding box.
        camera_weights: Dict camera_id -> float weight from
            ``compute_angular_diversity_weights``.
        loss_weights: Scalar multipliers for each term, e.g.
            ``{"iou": 1.0, "gravity": 0.05, "morph": 0.2}``.  Missing keys default to
            the values shown (except "iou" which defaults to 1.0).
        temporal_state: Optional ``FishState`` from the previous frame.  If None,
            the temporal loss is zero.
        temporal_weight: Multiplier for the temporal smoothness term.
        kappa_max: Maximum absolute curvature before the morphological penalty
            activates.
        s_min: Minimum scale before penalty activates.
        s_max: Maximum scale before penalty activates.

    Returns:
        Dict with keys ``"total"``, ``"iou"``, ``"gravity"``, ``"morph"``,
        ``"temporal"`` — all scalar tensors.  ``"total"`` is the weighted sum used
        for ``backward()``.
    """
    device = state.p.device

    # -----------------------------------------------------------------------
    # 1. Silhouette IoU loss
    # -----------------------------------------------------------------------
    iou_loss_terms: list[torch.Tensor] = []
    weight_sum = sum(camera_weights.get(cam_id, 1.0) for cam_id in pred_alphas)

    for cam_id, alpha in pred_alphas.items():
        w = camera_weights.get(cam_id, 1.0)
        target = target_masks[cam_id]
        crop = crop_regions.get(cam_id)
        iou_l = soft_iou_loss(alpha, target, crop_region=crop)
        iou_loss_terms.append(w * iou_l)

    if iou_loss_terms:
        iou_loss = torch.stack(iou_loss_terms).sum() / max(weight_sum, 1e-8)
    else:
        iou_loss = torch.tensor(0.0, device=device)

    # -----------------------------------------------------------------------
    # 2. Gravity prior: penalise pitch deviation (proxy for roll)
    # -----------------------------------------------------------------------
    gravity_loss = state.theta**2

    # -----------------------------------------------------------------------
    # 3. Morphological constraints
    # -----------------------------------------------------------------------
    # Curvature: soft L2 beyond |kappa| > kappa_max.
    kappa_excess = F.relu(state.kappa.abs() - kappa_max)
    kappa_loss = kappa_excess**2

    # Scale: soft L2 below s_min and above s_max.
    s_below = F.relu(s_min - state.s)
    s_above = F.relu(state.s - s_max)
    s_loss = (s_below + s_above) ** 2

    morph_loss = kappa_loss + s_loss

    # -----------------------------------------------------------------------
    # 4. Temporal smoothness (inactive in Phase 4; hook for Phase 5)
    # -----------------------------------------------------------------------
    temporal_loss = torch.tensor(0.0, device=device)
    if temporal_state is not None:
        pos_delta = (state.p - temporal_state.p).norm() ** 2
        psi_delta = (state.psi - temporal_state.psi) ** 2
        temporal_loss = pos_delta + psi_delta

    # -----------------------------------------------------------------------
    # Weighted total
    # -----------------------------------------------------------------------
    w_iou = loss_weights.get("iou", 1.0)
    w_gravity = loss_weights.get("gravity", 0.05)
    w_morph = loss_weights.get("morph", 0.2)

    total = (
        w_iou * iou_loss
        + w_gravity * gravity_loss
        + w_morph * morph_loss
        + temporal_weight * temporal_loss
    )

    return {
        "total": total,
        "iou": iou_loss,
        "gravity": gravity_loss,
        "morph": morph_loss,
        "temporal": temporal_loss,
    }
