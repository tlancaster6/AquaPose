"""Correspondence-free 3D midline optimizer using B-spline curve fitting.

Replaces point-wise triangulation with direct optimization of 3D B-spline control
points against 2D skeleton observations via chamfer distance and refractive
reprojection. All fish in a frame are batched into a single (N_fish, K, 3) tensor
and optimized in parallel on GPU.

Coarse-to-fine strategy: K=4 control points for coarse stage, K=7 for fine stage.
Warm-start from previous frame's solution with cold-start fallback.
Adaptive early stopping freezes converged fish while others continue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.interpolate
import torch
import torch.nn.functional as F

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.reconstruction.triangulation import (
    N_SAMPLE_POINTS,
    SPLINE_K,
    SPLINE_KNOTS,
    SPLINE_N_CTRL,
    Midline3D,
    MidlineSet,
    _pixel_half_width_to_metres,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Fish biomechanics: max bend angle between adjacent body segments.
# Literature: adult zebrafish/small fish max lateral body angle ~30-35 deg.
_MAX_BEND_ANGLE_DEG_DEFAULT: float = 30.0

# Coarse and fine control point counts
_N_COARSE_CTRL: int = 4
_N_FINE_CTRL: int = SPLINE_N_CTRL  # 7, matches Midline3D contract

# Huber delta in pixel units (per proposal Pitfall 7)
_HUBER_DELTA_PX: float = 17.5  # midpoint of 15-20px range

# ---------------------------------------------------------------------------
# B-spline basis cache
# ---------------------------------------------------------------------------

_BASIS_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _build_basis_matrix(n_eval: int, n_ctrl: int) -> torch.Tensor:
    """Build a B-spline basis matrix for evaluation at uniform parameter values.

    Constructs a clamped uniform cubic B-spline basis using scipy.interpolate.BSpline
    with the same knot formula as SPLINE_KNOTS for n_ctrl=7, extended to arbitrary
    n_ctrl. The resulting matrix B has shape (n_eval, n_ctrl) such that evaluating
    the spline at n_eval uniform parameter values in [0, 1] is: P = B @ C, where C
    is the (n_ctrl, 3) control point tensor.

    Args:
        n_eval: Number of evaluation parameter values in [0, 1].
        n_ctrl: Number of B-spline control points.

    Returns:
        Basis matrix of shape (n_eval, n_ctrl), float32.
    """
    degree = 3
    # Clamped uniform knots: degree+1 copies at each end, uniform interior
    n_interior = n_ctrl - degree - 1
    if n_interior > 0:
        interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    else:
        interior = np.array([], dtype=np.float64)

    knots = np.concatenate(
        [
            np.zeros(degree + 1),
            interior,
            np.ones(degree + 1),
        ]
    )

    t_eval = np.linspace(0.0, 1.0, n_eval)
    # Clamp endpoints to avoid extrapolation issues
    t_eval = np.clip(t_eval, knots[degree], knots[-(degree + 1)])

    # Build basis row by row: one-hot control point activations
    rows = []
    for i in range(n_ctrl):
        c = np.zeros(n_ctrl)
        c[i] = 1.0
        spl = scipy.interpolate.BSpline(knots, c, degree, extrapolate=False)
        col = spl(t_eval)
        # Replace NaN (outside knot range) with 0
        col = np.where(np.isnan(col), 0.0, col)
        rows.append(col)

    # rows[i] is the i-th basis function evaluated at t_eval
    # Stack into (n_eval, n_ctrl)
    basis = np.stack(rows, axis=1).astype(np.float32)
    return torch.from_numpy(basis)


def get_basis(n_eval: int, n_ctrl: int, device: torch.device | str) -> torch.Tensor:
    """Return the cached B-spline basis matrix, computing it if not cached.

    Args:
        n_eval: Number of evaluation points.
        n_ctrl: Number of control points.
        device: Target device for the returned tensor.

    Returns:
        Basis matrix of shape (n_eval, n_ctrl), float32, on ``device``.
    """
    key = (n_eval, n_ctrl)
    if key not in _BASIS_CACHE:
        _BASIS_CACHE[key] = _build_basis_matrix(n_eval, n_ctrl)
    return _BASIS_CACHE[key].to(device)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CurveOptimizerConfig:
    """Hyperparameters for the curve-based 3D midline optimizer.

    All regularization weights are exposed here so they can be tuned without
    modifying implementation code.

    Attributes:
        nominal_length_m: Nominal fish body length in metres (midpoint of
            70-100mm range per species prior). Default 0.085.
        length_tolerance: Fractional tolerance on nominal length (±30% = 0.30).
            The length penalty is zero within [nominal*(1-tol), nominal*(1+tol)].
        lambda_length: Weight for arc-length penalty.
        lambda_curvature: Weight for per-joint curvature penalty.
        lambda_smoothness: Weight for second-difference smoothness penalty.
        max_bend_angle_deg: Maximum allowable bend angle (degrees) between
            consecutive control-point triplets. Default 30.0 from fish biomechanics.
        n_coarse_ctrl: Number of control points in coarse optimization stage.
        n_fine_ctrl: Number of control points in fine optimization stage.
        n_eval_points: Number of spline evaluation points used during loss.
        lbfgs_lr: L-BFGS learning rate (step size). Default 1.0.
        lbfgs_max_iter_coarse: Maximum L-BFGS iterations for coarse stage.
        lbfgs_max_iter_fine: Maximum L-BFGS iterations for fine stage.
        lbfgs_history_size: L-BFGS history size. Default 10.
        convergence_delta: Loss delta below which a fish is considered converged.
        convergence_patience: Steps with delta < convergence_delta before freeze.
        warm_start_loss_ratio: If warm-start loss / prev_loss > this ratio, fall
            back to cold start for that fish. Default 2.0.
    """

    nominal_length_m: float = 0.085
    length_tolerance: float = 0.30
    lambda_length: float = 10.0
    lambda_curvature: float = 5.0
    lambda_smoothness: float = 1.0
    max_bend_angle_deg: float = _MAX_BEND_ANGLE_DEG_DEFAULT
    n_coarse_ctrl: int = _N_COARSE_CTRL
    n_fine_ctrl: int = _N_FINE_CTRL
    n_eval_points: int = 20
    lbfgs_lr: float = 1.0
    lbfgs_max_iter_coarse: int = 50
    lbfgs_max_iter_fine: int = 100
    lbfgs_history_size: int = 10
    convergence_delta: float = 1e-4
    convergence_patience: int = 3
    warm_start_loss_ratio: float = 2.0
    # Not directly used in loss but provided for downstream consumers
    _extra: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------------


def _chamfer_distance_2d(
    proj: torch.Tensor,
    obs: torch.Tensor,
) -> torch.Tensor:
    """Symmetric 2D chamfer distance between two point sets.

    Uses torch.cdist for efficient pairwise distance computation.
    Returns mean of both directed chamfer distances (proj→obs and obs→proj).
    Returns zero scalar if either set is empty.

    Args:
        proj: Projected spline points, shape (M, 2), float32.
        obs: Observed skeleton points, shape (N, 2), float32.

    Returns:
        Scalar chamfer distance tensor.
    """
    if proj.numel() == 0 or obs.numel() == 0:
        return torch.zeros(1, device=proj.device if proj.numel() > 0 else obs.device)

    # (M, N) pairwise distances
    dists = torch.cdist(proj.unsqueeze(0), obs.unsqueeze(0)).squeeze(0)  # (M, N)

    # proj→obs: for each projected point, nearest obs
    min_proj_to_obs = dists.min(dim=1).values  # (M,)
    # obs→proj: for each observed point, nearest projected
    min_obs_to_proj = dists.min(dim=0).values  # (N,)

    return (min_proj_to_obs.mean() + min_obs_to_proj.mean()) / 2.0


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _data_loss(
    ctrl_pts: torch.Tensor,
    basis: torch.Tensor,
    midlines_per_fish: list[dict[str, torch.Tensor]],
    models: dict[str, RefractiveProjectionModel],
    config: CurveOptimizerConfig,
) -> torch.Tensor:
    """Compute the data loss: chamfer distance between reprojected spline and observed skeleton.

    For each fish, evaluates the spline via basis matrix multiply, reprojects
    through each camera's refractive model, computes chamfer distance to the
    observed 2D skeleton points, and aggregates per-camera with Huber loss.
    NaN/invalid projections are filtered before chamfer computation.

    Args:
        ctrl_pts: Control points tensor, shape (N_fish, K, 3), float32.
        basis: B-spline basis matrix, shape (n_eval, K), float32.
        midlines_per_fish: Per-fish dict mapping camera_id to observed skeleton
            points tensor of shape (M, 2), float32.
        models: Per-camera refractive projection models.
        config: Optimizer configuration.

    Returns:
        Scalar total data loss.
    """
    n_fish = ctrl_pts.shape[0]
    total_loss = torch.zeros(1, device=ctrl_pts.device)

    for i in range(n_fish):
        c = ctrl_pts[i]  # (K, 3)
        # Evaluate spline: (n_eval, K) @ (K, 3) -> (n_eval, 3)
        spline_pts = basis @ c  # (n_eval, 3)

        for cam_id, obs_pts in midlines_per_fish[i].items():
            if cam_id not in models:
                continue
            model = models[cam_id]

            # Project spline points into camera
            proj_px, valid = model.project(spline_pts)  # (n_eval, 2), (n_eval,)

            # Filter out NaN and invalid projections
            valid_mask = valid & ~torch.isnan(proj_px).any(dim=1)
            if not valid_mask.any():
                continue

            proj_valid = proj_px[valid_mask]  # (M_valid, 2)

            # Chamfer distance between projected and observed
            chamfer = _chamfer_distance_2d(proj_valid, obs_pts)

            # Huber loss for robust per-camera aggregation
            cam_loss = F.huber_loss(
                chamfer,
                torch.zeros_like(chamfer),
                delta=_HUBER_DELTA_PX,
                reduction="mean",
            )
            total_loss = total_loss + cam_loss

    return total_loss


def _length_penalty(
    ctrl_pts: torch.Tensor,
    basis: torch.Tensor,
    config: CurveOptimizerConfig,
) -> torch.Tensor:
    """Penalize deviations of arc length from the nominal species length.

    Penalty is zero within [nominal*(1-tolerance), nominal*(1+tolerance)].
    Outside the tolerance band, penalty grows quadratically with deviation.

    Args:
        ctrl_pts: Control points, shape (N_fish, K, 3).
        basis: B-spline basis matrix, shape (n_eval, K).
        config: Optimizer configuration.

    Returns:
        Scalar length penalty.
    """
    # Evaluate splines for all fish at once: (N_fish, n_eval, 3)
    spline_pts = torch.einsum("ek,nkd->ned", basis, ctrl_pts)

    # Arc length via segment sums
    diffs = spline_pts[:, 1:, :] - spline_pts[:, :-1, :]  # (N_fish, n_eval-1, 3)
    seg_lengths = torch.linalg.norm(diffs, dim=2)  # (N_fish, n_eval-1)
    arc_lengths = seg_lengths.sum(dim=1)  # (N_fish,)

    nominal = config.nominal_length_m
    tol = config.length_tolerance
    lower = nominal * (1.0 - tol)
    upper = nominal * (1.0 + tol)

    # Penalize below lower bound and above upper bound
    below = torch.clamp(lower - arc_lengths, min=0.0)
    above = torch.clamp(arc_lengths - upper, min=0.0)
    deviation = below + above  # (N_fish,)

    return (deviation**2).mean()


def _curvature_penalty(
    ctrl_pts: torch.Tensor,
    config: CurveOptimizerConfig,
) -> torch.Tensor:
    """Penalize bend angles exceeding the maximum allowable joint angle.

    For each consecutive triplet of control points (C_{i-1}, C_i, C_{i+1}),
    computes the bend angle at C_i and applies a quadratic penalty for angles
    exceeding max_bend_angle_deg. This prevents physically implausible contortions.

    Args:
        ctrl_pts: Control points, shape (N_fish, K, 3).
        config: Optimizer configuration.

    Returns:
        Scalar curvature penalty.
    """
    if ctrl_pts.shape[1] < 3:
        return torch.zeros(1, device=ctrl_pts.device)

    # Vector from C_{i-1} to C_i and from C_i to C_{i+1}
    v1 = ctrl_pts[:, 1:-1, :] - ctrl_pts[:, :-2, :]  # (N_fish, K-2, 3)
    v2 = ctrl_pts[:, 2:, :] - ctrl_pts[:, 1:-1, :]  # (N_fish, K-2, 3)

    # Cosine of bend angle at each interior control point
    norm1 = torch.linalg.norm(v1, dim=2, keepdim=True).clamp(min=1e-8)
    norm2 = torch.linalg.norm(v2, dim=2, keepdim=True).clamp(min=1e-8)
    v1_unit = v1 / norm1
    v2_unit = v2 / norm2

    cos_angle = (v1_unit * v2_unit).sum(dim=2)  # (N_fish, K-2)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    # Bend angle at C_i is pi - acos(cos_angle) (exterior angle between segments)
    # Or equivalently the supplement: angle between v1 and -v2
    cos_bend = -(v1_unit * v2_unit).sum(dim=2)  # (N_fish, K-2), bend from straight
    cos_bend = torch.clamp(cos_bend, -1.0, 1.0)

    # Bend angle in degrees
    bend_angles_rad = torch.acos(cos_bend)  # (N_fish, K-2)

    max_rad = torch.tensor(
        config.max_bend_angle_deg * np.pi / 180.0,
        device=ctrl_pts.device,
        dtype=ctrl_pts.dtype,
    )

    # Penalize angles exceeding the limit
    excess = torch.clamp(bend_angles_rad - max_rad, min=0.0)
    return (excess**2).mean()


def _smoothness_penalty(ctrl_pts: torch.Tensor) -> torch.Tensor:
    """Compute second-difference (curvature) smoothness penalty on control points.

    Penalizes the squared magnitude of second-order finite differences of
    control points. This prevents high-frequency oscillations.

    Args:
        ctrl_pts: Control points, shape (N_fish, K, 3).

    Returns:
        Scalar smoothness penalty.
    """
    if ctrl_pts.shape[1] < 3:
        return torch.zeros(1, device=ctrl_pts.device)

    # Second differences: C_{i+1} - 2*C_i + C_{i-1}
    second_diff = ctrl_pts[:, 2:, :] - 2.0 * ctrl_pts[:, 1:-1, :] + ctrl_pts[:, :-2, :]
    return (second_diff**2).sum(dim=2).mean()


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------


def _cold_start(
    centroid: torch.Tensor,
    orientation: torch.Tensor | None,
    K: int,
    nominal_length: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Initialize a straight-line spline centered at centroid.

    Creates K control points evenly spaced along a straight line of length
    `nominal_length`, centered at `centroid` and oriented along `orientation`.

    Args:
        centroid: 3D centroid position, shape (3,), float32.
        orientation: Unit direction vector, shape (3,), float32. If None,
            defaults to (1, 0, 0).
        K: Number of control points.
        nominal_length: Total length of the initial straight spline (metres).
        device: Target device.

    Returns:
        Control points, shape (K, 3), float32.
    """
    if orientation is None or torch.linalg.norm(orientation) < 1e-8:
        orient: torch.Tensor = torch.tensor(
            [1.0, 0.0, 0.0], device=device, dtype=torch.float32
        )
    else:
        orient = orientation.to(device=device, dtype=torch.float32)
        orient = orient / torch.linalg.norm(orient)

    centroid = centroid.to(device=device, dtype=torch.float32)

    # Offsets along orientation: evenly spaced from -length/2 to +length/2
    offsets = torch.linspace(
        -nominal_length / 2.0, nominal_length / 2.0, K, device=device
    )
    ctrl_pts = centroid.unsqueeze(0) + offsets.unsqueeze(1) * orient.unsqueeze(0)
    return ctrl_pts.float()


def _estimate_orientation_from_skeleton(
    obs_pts: torch.Tensor,
) -> torch.Tensor:
    """Estimate PCA principal axis of 2D skeleton points as orientation hint.

    Args:
        obs_pts: Observed skeleton points, shape (N, 2), float32.

    Returns:
        3D unit direction vector, shape (3,), float32 (z=0).
    """
    if obs_pts.shape[0] < 2:
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    centered = obs_pts - obs_pts.mean(dim=0, keepdim=True)  # (N, 2)
    cov = centered.T @ centered  # (2, 2)
    try:
        _, _, Vt = torch.linalg.svd(cov)
        principal = Vt[0]  # (2,) — first principal component
    except RuntimeError:
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    return torch.tensor(
        [principal[0].item(), principal[1].item(), 0.0], dtype=torch.float32
    )


def _init_ctrl_pts(
    fish_ids: list[int],
    centroids: dict[int, torch.Tensor],
    warm_starts: dict[int, torch.Tensor],
    K: int,
    config: CurveOptimizerConfig,
    ref_orientations: dict[int, torch.Tensor | None],
    device: torch.device | str,
) -> torch.Tensor:
    """Initialize control points for all fish.

    For each fish, uses warm-start if available (detach().clone()), else cold-start.
    If warm-start has different K, re-evaluates via basis matrix to match target K.

    Args:
        fish_ids: List of fish IDs to initialize.
        centroids: Per-fish 3D centroid positions.
        warm_starts: Per-fish warm-start control points from previous frame.
        K: Target number of control points.
        config: Optimizer configuration.
        ref_orientations: Per-fish reference orientation vectors (from reference camera PCA).
        device: Target device.

    Returns:
        Control points tensor, shape (N_fish, K, 3), float32.
    """
    n_fish = len(fish_ids)
    all_ctrl: list[torch.Tensor] = []

    for fid in fish_ids:
        if fid in warm_starts:
            ws = warm_starts[fid]
            if ws.shape[0] == K:
                ctrl = ws.detach().clone().to(device)
            else:
                # Upsample warm-start from its K to target K via basis evaluation
                ws_k = ws.shape[0]
                B_ws = get_basis(K, ws_k, device)
                ws_dev = ws.detach().to(device)
                ctrl = (B_ws @ ws_dev).clone()
        else:
            centroid = centroids.get(fid, torch.zeros(3, device=device))
            orientation = ref_orientations.get(fid)
            ctrl = _cold_start(
                centroid, orientation, K, config.nominal_length_m, device
            )

        all_ctrl.append(ctrl)

    if n_fish == 0:
        return torch.zeros(0, K, 3, device=device, dtype=torch.float32)

    return torch.stack(all_ctrl, dim=0)  # (N_fish, K, 3)


def _upsample_ctrl_pts(
    coarse_ctrl: torch.Tensor,
    n_coarse: int,
    n_fine: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Upsample control points from coarse to fine resolution via basis evaluation.

    Evaluates the coarse spline at n_fine parameter positions to initialize
    fine control points. This preserves the curve shape from the coarse stage.

    Args:
        coarse_ctrl: Coarse control points, shape (N_fish, n_coarse, 3).
        n_coarse: Number of coarse control points.
        n_fine: Number of fine control points.
        device: Target device.

    Returns:
        Fine control points, shape (N_fish, n_fine, 3).
    """
    # Evaluate coarse spline at n_fine parameter positions
    B_up = get_basis(n_fine, n_coarse, device)  # (n_fine, n_coarse)
    # (N_fish, n_fine, 3) = (n_fine, n_coarse) @ (N_fish, n_coarse, 3)
    return torch.einsum("fk,nkd->nfd", B_up, coarse_ctrl)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CurveOptimizer:
    """Correspondence-free 3D midline optimizer using B-spline curve fitting.

    Optimizes 3D B-spline control points directly against 2D skeleton observations
    via chamfer distance and refractive reprojection. All fish in a frame are
    batched and optimized in parallel on GPU.

    Args:
        config: Optimizer configuration. Uses defaults if not provided.
    """

    def __init__(self, config: CurveOptimizerConfig | None = None) -> None:
        self.config = config or CurveOptimizerConfig()
        self._warm_starts: dict[int, torch.Tensor] = {}
        self._prev_losses: dict[int, float] = {}

    def optimize_midlines(
        self,
        midline_set: MidlineSet,
        models: dict[str, RefractiveProjectionModel],
        frame_index: int = 0,
        fish_centroids: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, Midline3D]:
        """Optimize 3D B-spline midlines for all fish in a frame.

        Runs coarse-to-fine L-BFGS optimization on batched GPU tensors.
        Warm-starts from previous frame when available, with cold-start fallback
        if warm-start loss is anomalously high. Adaptive early stopping freezes
        converged fish during optimization.

        Args:
            midline_set: Nested dict mapping fish_id → camera_id → Midline2D.
            models: Mapping from camera_id to RefractiveProjectionModel.
            frame_index: Frame index embedded in output Midline3D structs.
            fish_centroids: Optional per-fish 3D centroids for cold-start. If None,
                estimated from the mean of triangulated points in the reference camera.

        Returns:
            Dict mapping fish_id to Midline3D. Fish with no valid camera
            observations are skipped.
        """
        cfg = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move all models to device
        for model in models.values():
            model.to(device)

        fish_ids = sorted(midline_set.keys())
        if not fish_ids:
            return {}

        # Gather per-fish observed skeleton points as tensors on device
        # midlines_per_fish[i]: dict[cam_id -> tensor (N, 2)]
        midlines_per_fish: list[dict[str, torch.Tensor]] = []
        valid_fish_ids: list[int] = []

        for fid in fish_ids:
            cam_obs: dict[str, torch.Tensor] = {}
            for cam_id, midline2d in midline_set[fid].items():
                if cam_id not in models:
                    continue
                pts = midline2d.points  # (N, 2) numpy
                # Filter NaN rows
                valid_mask = ~np.isnan(pts).any(axis=1)
                pts_clean = pts[valid_mask]
                if len(pts_clean) == 0:
                    continue
                cam_obs[cam_id] = torch.from_numpy(pts_clean).float().to(device)

            if cam_obs:
                midlines_per_fish.append(cam_obs)
                valid_fish_ids.append(fid)
            else:
                logger.debug("Fish %d has no valid camera observations — skipping", fid)

        if not valid_fish_ids:
            return {}

        n_fish = len(valid_fish_ids)

        # Estimate reference orientations from reference camera (longest arc)
        ref_orientations: dict[int, torch.Tensor | None] = {}
        for idx, fid in enumerate(valid_fish_ids):
            cam_obs = midlines_per_fish[idx]
            # Pick camera with most points as reference
            best_cam = max(cam_obs, key=lambda c: cam_obs[c].shape[0])
            obs = cam_obs[best_cam]  # (N, 2) on device
            ref_orientations[fid] = _estimate_orientation_from_skeleton(obs).to(device)

        # Centroids: use provided or estimate from observation means
        centroids: dict[int, torch.Tensor] = {}
        if fish_centroids is not None:
            centroids = {
                fid: fish_centroids[fid].to(device)
                for fid in valid_fish_ids
                if fid in fish_centroids
            }

        for idx, fid in enumerate(valid_fish_ids):
            if fid not in centroids:
                # Estimate centroid from 2D observations: use water depth mid-range
                cam_obs = midlines_per_fish[idx]
                best_cam = max(cam_obs, key=lambda c: cam_obs[c].shape[0])
                obs_2d = cam_obs[best_cam]  # (N, 2)
                mean_2d = obs_2d.mean(dim=0)  # (2,)
                model = models[best_cam]
                # Approximate centroid by casting ray at estimated depth
                water_z = model.water_z
                depth_est = 0.5  # metres below water surface
                pixel_batch = mean_2d.unsqueeze(0)  # (1, 2)
                origin, direction = model.cast_ray(pixel_batch)
                # 3D point at estimated depth
                centroid_3d = origin[0] + depth_est * direction[0]
                centroids[fid] = centroid_3d.detach()

        # ---- Stage 1: Coarse optimization (K=n_coarse_ctrl) ----
        n_coarse = cfg.n_coarse_ctrl
        coarse_basis = get_basis(cfg.n_eval_points, n_coarse, device)

        coarse_init = _init_ctrl_pts(
            valid_fish_ids,
            centroids,
            self._warm_starts,
            n_coarse,
            cfg,
            ref_orientations,
            device,
        )
        coarse_ctrl = coarse_init.clone().requires_grad_(True)

        # Per-fish convergence tracking (frozen_coarse used in closure below)
        frozen_coarse = torch.zeros(n_fish, dtype=torch.bool, device=device)

        optimizer_coarse = torch.optim.LBFGS(
            [coarse_ctrl],
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter_coarse,
            history_size=cfg.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

        def closure_coarse() -> torch.Tensor:
            optimizer_coarse.zero_grad()

            data_l = _data_loss(
                coarse_ctrl, coarse_basis, midlines_per_fish, models, cfg
            )
            len_l = _length_penalty(coarse_ctrl, coarse_basis, cfg)
            curv_l = _curvature_penalty(coarse_ctrl, cfg)
            smooth_l = _smoothness_penalty(coarse_ctrl)

            total = (
                data_l
                + cfg.lambda_length * len_l
                + cfg.lambda_curvature * curv_l
                + cfg.lambda_smoothness * smooth_l
            )
            total.backward()

            # Zero gradients for frozen fish
            if coarse_ctrl.grad is not None and frozen_coarse.any():
                coarse_ctrl.grad[frozen_coarse] = 0.0

            return total

        optimizer_coarse.step(closure_coarse)

        # Update convergence tracking post-coarse (simplified: one step)
        with torch.no_grad():
            data_l = _data_loss(
                coarse_ctrl, coarse_basis, midlines_per_fish, models, cfg
            )

        coarse_loss_val = data_l.item()
        logger.debug("Frame %d coarse stage loss: %.4f", frame_index, coarse_loss_val)

        # ---- Stage 2: Fine optimization (K=n_fine_ctrl) ----
        n_fine = cfg.n_fine_ctrl
        fine_basis = get_basis(cfg.n_eval_points, n_fine, device)

        # Upsample coarse control points to fine resolution
        with torch.no_grad():
            fine_init = _upsample_ctrl_pts(
                coarse_ctrl.detach(), n_coarse, n_fine, device
            )

        # Warm-start fallback check: if warm-start available and warm-loss too high,
        # revert that fish to cold start
        if self._warm_starts:
            with torch.no_grad():
                # Evaluate fine loss at current initialization
                fine_check = fine_init.clone().requires_grad_(False)
                for idx, fid in enumerate(valid_fish_ids):
                    if fid in self._prev_losses:
                        prev_loss = self._prev_losses[fid]
                        # Quick per-fish loss estimate
                        single_ctrl = fine_check[idx : idx + 1]  # (1, K, 3)
                        single_obs = [midlines_per_fish[idx]]
                        curr_loss = _data_loss(
                            single_ctrl, fine_basis, single_obs, models, cfg
                        ).item()
                        if (
                            prev_loss > 0
                            and curr_loss > cfg.warm_start_loss_ratio * prev_loss
                        ):
                            logger.debug(
                                "Fish %d warm-start loss (%.4f) > %.1fx prev (%.4f), cold start",
                                fid,
                                curr_loss,
                                cfg.warm_start_loss_ratio,
                                prev_loss,
                            )
                            cold = _cold_start(
                                centroids[fid],
                                ref_orientations.get(fid),
                                n_fine,
                                cfg.nominal_length_m,
                                device,
                            )
                            fine_init[idx] = cold

        fine_ctrl = fine_init.clone().requires_grad_(True)

        # Per-fish convergence tracking for fine stage
        prev_losses_fine: list[float] = [float("inf")] * n_fish
        patience_counters_fine: list[int] = [0] * n_fish
        final_per_fish_losses: list[float] = [float("inf")] * n_fish

        optimizer_fine = torch.optim.LBFGS(
            [fine_ctrl],
            lr=cfg.lbfgs_lr,
            max_iter=1,  # Manual loop for adaptive early stopping
            history_size=cfg.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

        # Manual optimization loop for per-fish adaptive stopping
        fine_frozen_mask = torch.zeros(n_fish, dtype=torch.bool, device=device)

        def closure_fine() -> torch.Tensor:
            optimizer_fine.zero_grad()

            data_l = _data_loss(fine_ctrl, fine_basis, midlines_per_fish, models, cfg)
            len_l = _length_penalty(fine_ctrl, fine_basis, cfg)
            curv_l = _curvature_penalty(fine_ctrl, cfg)
            smooth_l = _smoothness_penalty(fine_ctrl)

            total = (
                data_l
                + cfg.lambda_length * len_l
                + cfg.lambda_curvature * curv_l
                + cfg.lambda_smoothness * smooth_l
            )
            total.backward()

            if fine_ctrl.grad is not None and fine_frozen_mask.any():
                fine_ctrl.grad[fine_frozen_mask] = 0.0

            return total

        # Run fine optimization with adaptive per-fish stopping
        for step in range(cfg.lbfgs_max_iter_fine):
            if fine_frozen_mask.all():
                logger.debug(
                    "Frame %d: all fish converged at step %d", frame_index, step
                )
                break

            optimizer_fine.step(closure_fine)

            # Monitor per-fish losses for adaptive stopping
            with torch.no_grad():
                for idx in range(n_fish):
                    if fine_frozen_mask[idx]:
                        continue
                    single_ctrl = fine_ctrl[idx : idx + 1]
                    single_obs = [midlines_per_fish[idx]]
                    fish_loss = _data_loss(
                        single_ctrl, fine_basis, single_obs, models, cfg
                    ).item()
                    final_per_fish_losses[idx] = fish_loss

                    delta = abs(prev_losses_fine[idx] - fish_loss)
                    if delta < cfg.convergence_delta:
                        patience_counters_fine[idx] += 1
                        if patience_counters_fine[idx] >= cfg.convergence_patience:
                            fine_frozen_mask[idx] = True
                            logger.debug(
                                "Frame %d fish %d converged at step %d (loss=%.4f)",
                                frame_index,
                                valid_fish_ids[idx],
                                step,
                                fish_loss,
                            )
                    else:
                        patience_counters_fine[idx] = 0

                    prev_losses_fine[idx] = fish_loss

        # ---- Build output Midline3D structs ----
        results: dict[int, Midline3D] = {}

        with torch.no_grad():
            final_ctrl_np = (
                fine_ctrl.detach().cpu().numpy().astype(np.float32)
            )  # (N_fish, K, 3)

        for idx, fid in enumerate(valid_fish_ids):
            ctrl_np = final_ctrl_np[idx]  # (K, 3)

            # Compute arc length via 1000-point numerical integration
            spl_obj = scipy.interpolate.BSpline(
                SPLINE_KNOTS, ctrl_np.astype(np.float64), SPLINE_K
            )
            u_fine = np.linspace(0.0, 1.0, 1000)
            curve_pts_3d = spl_obj(u_fine)  # (1000, 3)
            diffs_np = np.diff(curve_pts_3d, axis=0)
            arc_length = float(np.sum(np.linalg.norm(diffs_np, axis=1)))

            # Compute reprojection residuals
            spline_pts_torch = torch.from_numpy(
                spl_obj(np.linspace(0.0, 1.0, N_SAMPLE_POINTS)).astype(np.float32)
            ).to(device)

            all_residuals: list[float] = []
            per_cam_residuals: dict[str, float] = {}
            cam_obs = midlines_per_fish[idx]

            for cam_id, _obs_pts_gpu in cam_obs.items():
                if cam_id not in models:
                    continue
                proj_px, valid_proj = models[cam_id].project(spline_pts_torch)
                proj_np = proj_px.cpu().numpy()
                valid_np = valid_proj.cpu().numpy()

                obs_np = midline_set[fid][cam_id].points  # (N, 2)

                cam_errs: list[float] = []
                n_pts = min(proj_np.shape[0], obs_np.shape[0])
                for j in range(n_pts):
                    if (
                        valid_np[j]
                        and not np.any(np.isnan(proj_np[j]))
                        and not np.any(np.isnan(obs_np[j]))
                    ):
                        err = float(np.linalg.norm(proj_np[j] - obs_np[j]))
                        cam_errs.append(err)
                        all_residuals.append(err)

                if cam_errs:
                    per_cam_residuals[cam_id] = float(np.mean(cam_errs))

            mean_residual = float(np.mean(all_residuals)) if all_residuals else 0.0
            max_residual = float(np.max(all_residuals)) if all_residuals else 0.0

            n_cameras = len(cam_obs)
            is_low_confidence = n_cameras < 2 or mean_residual > 50.0

            # Compute half-widths: average per-camera half-widths from observed midlines
            # Using _pixel_half_width_to_metres for consistency with triangulation.py
            hw_metres_list: list[float] = []
            for cam_id in cam_obs:
                if cam_id not in models:
                    continue
                model = models[cam_id]
                midline2d = midline_set[fid][cam_id]
                focal_px = float((model.K[0, 0].item() + model.K[1, 1].item()) / 2.0)
                water_z = model.water_z

                for j in range(len(midline2d.half_widths)):
                    hw_px = float(midline2d.half_widths[j])
                    if np.isnan(hw_px):
                        continue
                    # Use arc-length param j/(N-1) to get depth at that position
                    t_j = j / max(len(midline2d.half_widths) - 1, 1)
                    pt_3d_j = spl_obj(t_j)  # (3,)
                    depth_m = max(0.0, float(pt_3d_j[2]) - water_z)
                    hw_m = _pixel_half_width_to_metres(hw_px, depth_m, focal_px)
                    hw_metres_list.append(hw_m)

            if hw_metres_list:
                # Assign uniform half-width array (one value per sample point)
                hw_metres_all = np.full(
                    N_SAMPLE_POINTS, float(np.mean(hw_metres_list)), dtype=np.float32
                )
            else:
                hw_metres_all = np.zeros(N_SAMPLE_POINTS, dtype=np.float32)

            midline_3d = Midline3D(
                fish_id=fid,
                frame_index=frame_index,
                control_points=ctrl_np,
                knots=SPLINE_KNOTS.astype(np.float32),
                degree=SPLINE_K,
                arc_length=arc_length,
                half_widths=hw_metres_all,
                n_cameras=n_cameras,
                mean_residual=mean_residual,
                max_residual=max_residual,
                is_low_confidence=is_low_confidence,
                per_camera_residuals=per_cam_residuals if per_cam_residuals else None,
            )
            results[fid] = midline_3d

            # Store warm-start for next frame
            self._warm_starts[fid] = fine_ctrl[idx].detach().clone().cpu()
            self._prev_losses[fid] = final_per_fish_losses[idx]

        return results


def optimize_midlines(
    midline_set: MidlineSet,
    models: dict[str, RefractiveProjectionModel],
    frame_index: int = 0,
    config: CurveOptimizerConfig | None = None,
    fish_centroids: dict[int, torch.Tensor] | None = None,
) -> dict[int, Midline3D]:
    """Convenience wrapper: create a CurveOptimizer and optimize a single frame.

    For multi-frame use, prefer instantiating CurveOptimizer directly to preserve
    warm-start state across frames.

    Args:
        midline_set: Nested dict mapping fish_id → camera_id → Midline2D.
        models: Mapping from camera_id to RefractiveProjectionModel.
        frame_index: Frame index embedded in output Midline3D structs.
        config: Optimizer configuration. Uses defaults if not provided.
        fish_centroids: Optional per-fish 3D centroids for cold-start.

    Returns:
        Dict mapping fish_id to Midline3D.
    """
    optimizer = CurveOptimizer(config=config)
    return optimizer.optimize_midlines(
        midline_set, models, frame_index=frame_index, fish_centroids=fish_centroids
    )
