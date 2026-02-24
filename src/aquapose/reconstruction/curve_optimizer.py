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
import scipy.spatial.distance
import torch

from aquapose.calibration.projection import RefractiveProjectionModel, triangulate_rays
from aquapose.reconstruction.triangulation import (
    N_SAMPLE_POINTS,
    SPLINE_K,
    SPLINE_KNOTS,
    SPLINE_N_CTRL,
    Midline3D,
    MidlineSet,
    _pixel_half_width_to_metres,
    triangulate_midlines,
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


@dataclass
class OptimizerSnapshot:
    """Snapshot of optimizer state for a single fish at frame 0.

    Captured during ``optimize_midlines`` for the first frame with valid fish to
    enable progression visualizations (cold-start → coarse → fine).

    Attributes:
        fish_id: Globally unique fish identifier.
        best_cam_id: Camera ID with the most observation points for this fish.
        obs_2d: Observed 2D skeleton points in the best camera, shape (N, 2).
        cold_start_ctrl: Cold-start control points, shape (K_coarse, 3).
        cold_start_loss: Per-fish data loss at cold-start initialization (pixels).
        post_coarse_ctrl: Control points after coarse optimization, shape (K_coarse, 3).
        coarse_loss: Per-fish data loss after coarse stage (pixels).
        post_fine_ctrl: Control points after fine optimization, shape (K_fine, 3).
        fine_loss: Per-fish data loss after fine stage (pixels).
    """

    fish_id: int
    best_cam_id: str
    obs_2d: np.ndarray
    cold_start_ctrl: np.ndarray
    cold_start_loss: float
    post_coarse_ctrl: np.ndarray
    coarse_loss: float
    post_fine_ctrl: np.ndarray
    fine_loss: float


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
        lambda_length: Weight for arc-length penalty in pixel-equivalent units.
            Each penalty is internally normalized by a reference scale so that
            lambda=10 means "trade up to 10px of chamfer to correct a
            reference-sized violation." Default 10.0.
        lambda_curvature: Weight for per-joint curvature penalty (pixel-equiv).
        lambda_chord_arc: Weight for chord-to-arc-length ratio penalty
            (pixel-equiv). Penalizes splines that fold back on themselves.
            Only activates when chord/arc drops below ``chord_arc_threshold``.
        chord_arc_threshold: Chord-to-arc ratio below which the penalty activates.
            Default 0.75 (~90-degree total bend). Ratios above this are penalty-free.
        lambda_z_variance: Weight for Z-variance penalty (pixel-equiv).
            Penalizes depth spread of the spline around its per-fish mean Z.
            Fish bodies are approximately planar in depth; without this the
            optimizer wobbles in Z (chamfer is Z-insensitive).
        lambda_smoothness: Weight for second-difference smoothness penalty
            (pixel-equiv).
        max_bend_angle_deg: Maximum allowable bend angle (degrees) between
            consecutive control-point triplets. Default 30.0 from fish biomechanics.
        n_coarse_ctrl: Number of control points in coarse optimization stage.
        n_fine_ctrl: Number of control points in fine optimization stage.
        n_eval_points: Number of spline evaluation points used during loss.
        lbfgs_lr: L-BFGS learning rate (step size). Default 1.0.
        lbfgs_max_iter_coarse: Maximum L-BFGS iterations for coarse stage.
        lbfgs_max_iter_fine: Maximum L-BFGS iterations for fine stage.
        lbfgs_history_size: L-BFGS history size. Default 10.
        convergence_delta: Absolute per-fish loss delta (in pixels) below which
            a fish is considered converged. Uses absolute not relative comparison
            to avoid premature freeze at high loss. Default 0.5 (half a pixel).
        convergence_patience: Steps with delta < convergence_delta before freeze.
        warm_start_loss_ratio: If warm-start loss / prev_loss > this ratio, fall
            back to cold start for that fish. Default 2.0.
        alm_outer_iters: Number of outer augmented Lagrangian iterations.
            Each outer iteration runs L-BFGS to convergence then updates
            Lagrange multipliers and increases rho. Default 5.
        alm_rho_init: Initial ALM penalty parameter. Controls how aggressively
            constraints are enforced from the start. Default 10.0.
        alm_rho_factor: Multiplicative increase of rho per outer iteration.
            Default 2.0.
        alm_rho_max: Upper bound on rho to prevent numerical issues. Default
            1000.0.
        alm_tol: Constraint satisfaction tolerance (metres/radians). If no
            constraint exceeds this, the outer loop terminates early.
            Default 0.001.
        max_depth: Maximum depth below water surface (metres) for triangulation
            seed validation. When set, triangulated points deeper than
            water_z + max_depth are rejected. None means no upper bound.
        tri_seed: Whether to attempt triangulation-seeded cold start. When
            True, runs ``triangulate_midlines`` for cold-start fish and uses
            the result (if valid) as the initial control points. Disabled by
            default because the triangulation pipeline rarely succeeds and a
            single bad seed can poison the shared L-BFGS Hessian, corrupting
            all fish in the coarse stage. The simple cold-start (straight
            fish at estimated centroid) is more reliable in practice.
    """

    nominal_length_m: float = 0.085
    length_tolerance: float = 0.30
    lambda_length: float = 10.0
    lambda_curvature: float = 10.0
    lambda_chord_arc: float = 10.0
    chord_arc_threshold: float = 0.75
    lambda_z_variance: float = 10.0
    lambda_smoothness: float = 1.0
    max_bend_angle_deg: float = _MAX_BEND_ANGLE_DEG_DEFAULT
    n_coarse_ctrl: int = _N_COARSE_CTRL
    n_fine_ctrl: int = _N_FINE_CTRL
    n_eval_points: int = 20
    lbfgs_lr: float = 1.0
    lbfgs_max_iter_coarse: int = 20
    lbfgs_max_iter_fine: int = 40
    lbfgs_history_size: int = 10
    convergence_delta: float = 0.5
    convergence_patience: int = 3
    warm_start_loss_ratio: float = 2.0
    alm_outer_iters: int = 5
    alm_rho_init: float = 10.0
    alm_rho_factor: float = 2.0
    alm_rho_max: float = 1000.0
    alm_tol: float = 0.001
    max_depth: float | None = None
    tri_seed: bool = False
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
    """Compute the data loss: mean chamfer distance per fish averaged over cameras.

    For each fish, evaluates the spline via basis matrix multiply, reprojects
    through each camera's refractive model, and computes the symmetric chamfer
    distance to the observed 2D skeleton points. The per-camera chamfer distances
    are averaged to give a per-fish loss, then averaged across all fish. This
    ensures the loss magnitude is always in units of pixels (mean chamfer distance)
    regardless of camera count or fish count.

    NaN/invalid projections are filtered before chamfer computation. Cameras with
    no valid projections are skipped and do not contribute to the per-fish average.

    If a fish has NO valid projections in ANY camera (all spline points are above
    the water surface), a depth penalty is used instead of chamfer distance. The
    depth penalty is proportional to how far above the water surface the spline
    lies, scaled by 100 to match chamfer pixel units. This provides gradients that
    push the spline back underwater, preventing silent convergence at loss=0 when
    the initialization is physically invalid.

    Args:
        ctrl_pts: Control points tensor, shape (N_fish, K, 3), float32.
        basis: B-spline basis matrix, shape (n_eval, K), float32.
        midlines_per_fish: Per-fish dict mapping camera_id to observed skeleton
            points tensor of shape (M, 2), float32.
        models: Per-camera refractive projection models.
        config: Optimizer configuration.

    Returns:
        Scalar mean loss in pixel units, averaged over fish and cameras.
    """
    n_fish = ctrl_pts.shape[0]
    fish_losses: list[torch.Tensor] = []

    # Use the first available model to get water_z for the depth penalty below
    any_model = next(iter(models.values()))
    water_z = any_model.water_z

    for i in range(n_fish):
        c = ctrl_pts[i]  # (K, 3)
        # Evaluate spline: (n_eval, K) @ (K, 3) -> (n_eval, 3)
        spline_pts = basis @ c  # (n_eval, 3)

        cam_losses: list[torch.Tensor] = []
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

            # Chamfer distance in pixels between projected spline and observed skeleton
            chamfer = _chamfer_distance_2d(proj_valid, obs_pts)
            cam_losses.append(chamfer)

        if cam_losses:
            # Average chamfer over cameras for this fish
            fish_loss = torch.stack(cam_losses).mean()
            fish_losses.append(fish_loss)
        else:
            # No valid projections: all control points are above the water surface.
            # Add a penalty proportional to how far above water the spline is, so
            # gradients push the spline back underwater. The penalty is zero when all
            # points are below water (h_q >= 0) and grows linearly with violation.
            # Scale factor (~100) puts this in the same pixel-unit ballpark as chamfer.
            h_q = spline_pts[:, 2] - water_z  # (n_eval,): positive = underwater
            above_water = torch.clamp(-h_q, min=0.0)  # (n_eval,): > 0 when above water
            depth_penalty = above_water.mean() * 100.0
            fish_losses.append(depth_penalty)
            logger.debug(
                "Fish %d in batch: no valid projections, depth_penalty=%.4f "
                "(all control points above water; Z range [%.4f, %.4f], water_z=%.4f)",
                i,
                float(depth_penalty.item()),
                float(spline_pts[:, 2].min().item()),
                float(spline_pts[:, 2].max().item()),
                water_z,
            )

    if not fish_losses:
        return torch.zeros(1, device=ctrl_pts.device)

    # Average over fish — loss is always in pixel units
    return torch.stack(fish_losses).mean()


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

    # Bend angle at C_i: angle between incoming segment (v1) and outgoing segment (v2).
    # For a straight spine, v1 and v2 are parallel (same direction), bend_angle = 0.
    # For a sharp 90° bend, v1 and v2 are orthogonal, bend_angle = 90°.
    norm1 = torch.linalg.norm(v1, dim=2, keepdim=True).clamp(min=1e-8)
    norm2 = torch.linalg.norm(v2, dim=2, keepdim=True).clamp(min=1e-8)
    v1_unit = v1 / norm1
    v2_unit = v2 / norm2

    # cos(bend_angle) = v1 · v2 — 1 when straight, <1 when bent
    cos_bend = (v1_unit * v2_unit).sum(dim=2)  # (N_fish, K-2)
    # Clamp away from ±1 to keep acos gradient finite (gradient of acos is
    # -1/sqrt(1 - x^2), which blows up at x = ±1 for collinear/antiparallel pts)
    _EPS = 1e-6
    cos_bend = torch.clamp(cos_bend, -1.0 + _EPS, 1.0 - _EPS)

    # Bend angle in radians (0 = straight, pi = U-turn)
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


def _z_variance_penalty(
    ctrl_pts: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Penalize depth (Z) spread of the evaluated spline around its mean.

    Fish bodies are approximately planar in Z — even during diving/surfacing,
    depth variation along the body is small relative to body length.  The
    optimizer's chamfer loss is insensitive to Z, so without this penalty
    the spline wobbles in depth, causing apparent hooks/curls when projected
    through the nonlinear refractive camera model.

    Penalizes the variance of Z across evaluated spline points for each fish.
    The mean Z is free to float (no penalty on depth itself), only the spread
    around the mean is penalized.

    Args:
        ctrl_pts: Control points, shape (N_fish, K, 3).
        basis: B-spline basis matrix, shape (n_eval, K).

    Returns:
        Scalar Z-variance penalty (in metres squared).
    """
    # Evaluate spline: (N_fish, n_eval, 3)
    spline_pts = torch.einsum("ek,nkd->ned", basis, ctrl_pts)

    z_vals = spline_pts[:, :, 2]  # (N_fish, n_eval)
    z_var = z_vals.var(dim=1)  # (N_fish,)

    return z_var.mean()


def _chord_arc_penalty(
    ctrl_pts: torch.Tensor,
    basis: torch.Tensor,
    config: CurveOptimizerConfig,
) -> torch.Tensor:
    """Penalize folded splines via chord-to-arc-length ratio.

    Only activates when chord/arc drops below ``config.chord_arc_threshold``
    (dead zone for normal swimming curvature). Penalty is quadratic in the
    deficit below the threshold. Unlike the per-joint curvature penalty, this
    metric cannot be gamed by distributing a fold across control points.

    Args:
        ctrl_pts: Control points, shape (N_fish, K, 3).
        basis: B-spline basis matrix, shape (n_eval, K).
        config: Optimizer configuration (for threshold).

    Returns:
        Scalar chord-arc penalty.
    """
    # Evaluate spline: (N_fish, n_eval, 3)
    spline_pts = torch.einsum("ek,nkd->ned", basis, ctrl_pts)

    # Chord: straight-line distance from head to tail
    chord = torch.linalg.norm(
        spline_pts[:, -1, :] - spline_pts[:, 0, :], dim=1
    )  # (N_fish,)

    # Arc: sum of segment lengths along the curve
    diffs = spline_pts[:, 1:, :] - spline_pts[:, :-1, :]  # (N_fish, n_eval-1, 3)
    arc = torch.linalg.norm(diffs, dim=2).sum(dim=1)  # (N_fish,)

    ratio = chord / (arc + 1e-8)
    # Dead zone: no penalty when ratio >= threshold
    deficit = torch.clamp(config.chord_arc_threshold - ratio, min=0.0)
    return (deficit**2).mean()


# ---------------------------------------------------------------------------
# Augmented Lagrangian constraint enforcement
# ---------------------------------------------------------------------------


def _constraint_violations(
    ctrl_pts: torch.Tensor,
    basis: torch.Tensor,
    config: CurveOptimizerConfig,
) -> torch.Tensor:
    """Compute per-fish constraint violations (positive = violated).

    Returns a tensor of shape (N_fish, 2 + K-2 + 1) where columns are:
    [arc_lower, arc_upper, curvature_0, ..., curvature_{K-3}, chord_arc].

    Convention: g_i <= 0 is feasible; g_i > 0 is violated.

    Args:
        ctrl_pts: Control points, shape (N_fish, K, 3).
        basis: B-spline basis matrix, shape (n_eval, K).
        config: Optimizer configuration.

    Returns:
        Violation tensor, shape (N_fish, n_constraints).
    """
    # Evaluate splines: (N_fish, n_eval, 3)
    spline_pts = torch.einsum("ek,nkd->ned", basis, ctrl_pts)

    # Arc lengths
    diffs = spline_pts[:, 1:, :] - spline_pts[:, :-1, :]
    arc_lengths = torch.linalg.norm(diffs, dim=2).sum(dim=1)  # (N_fish,)

    nominal = config.nominal_length_m
    tol = config.length_tolerance
    lower = nominal * (1.0 - tol)
    upper = nominal * (1.0 + tol)

    arc_lower_viol = lower - arc_lengths  # positive when too short
    arc_upper_viol = arc_lengths - upper  # positive when too long

    # Per-joint curvature
    v1 = ctrl_pts[:, 1:-1, :] - ctrl_pts[:, :-2, :]  # (N_fish, K-2, 3)
    v2 = ctrl_pts[:, 2:, :] - ctrl_pts[:, 1:-1, :]  # (N_fish, K-2, 3)
    norm1 = torch.linalg.norm(v1, dim=2, keepdim=True).clamp(min=1e-8)
    norm2 = torch.linalg.norm(v2, dim=2, keepdim=True).clamp(min=1e-8)
    cos_bend = (v1 / norm1 * v2 / norm2).sum(dim=2)  # (N_fish, K-2)
    cos_bend = torch.clamp(cos_bend, -1.0 + 1e-6, 1.0 - 1e-6)
    bend_angles = torch.acos(cos_bend)
    max_rad = config.max_bend_angle_deg * np.pi / 180.0
    curv_viol = bend_angles - max_rad  # positive when exceeding max

    # Chord-arc ratio
    chord = torch.linalg.norm(spline_pts[:, -1, :] - spline_pts[:, 0, :], dim=1)
    ratio = chord / (arc_lengths + 1e-8)
    chord_arc_viol = config.chord_arc_threshold - ratio  # positive when folded

    return torch.cat(
        [
            arc_lower_viol.unsqueeze(1),
            arc_upper_viol.unsqueeze(1),
            curv_viol,
            chord_arc_viol.unsqueeze(1),
        ],
        dim=1,
    )


def _alm_penalty(
    violations: torch.Tensor,
    lambdas: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """Augmented Lagrangian penalty for inequality constraints g_i <= 0.

    For each constraint: psi = (1/(2*rho)) * (max(0, lambda + rho*g)^2 - lambda^2).
    This is differentiable (piecewise quadratic) and provides gradients that push
    toward feasibility proportional to both the multiplier and penalty weight.

    Args:
        violations: Constraint values, shape (N_fish, n_c). Positive = violated.
        lambdas: Lagrange multipliers, shape (N_fish, n_c). Non-negative.
        rho: Penalty parameter (positive scalar).

    Returns:
        Scalar penalty averaged over fish.
    """
    shifted = lambdas + rho * violations
    active = torch.clamp(shifted, min=0.0)
    penalty = (active**2 - lambdas**2) / (2.0 * rho)
    return penalty.sum() / violations.shape[0]


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
    return torch.einsum("fk,nkd->nfd", B_up, coarse_ctrl).contiguous()


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
        self.snapshots: list[OptimizerSnapshot] = []
        self._snapshot_frame: int | None = None  # first frame with valid fish

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

        After optimization, a consistency check ensures the spline direction
        matches the previous frame's warm-start. If the dot product of the
        current and previous spline directions is negative, the control points
        are flipped to maintain frame-to-frame consistency.

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

        # Collect snapshots for the first frame with valid fish
        if self._snapshot_frame is None:
            self._snapshot_frame = frame_index
            self.snapshots = []

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
                cam_obs = midlines_per_fish[idx]
                # Gather rays from all cameras observing this fish
                all_origins = []
                all_directions = []
                for cam_id, obs_2d in cam_obs.items():
                    mean_2d = obs_2d.mean(dim=0)  # (2,)
                    origin, direction = models[cam_id].cast_ray(mean_2d.unsqueeze(0))
                    all_origins.append(origin[0])
                    all_directions.append(direction[0])

                if len(all_origins) >= 2:
                    # Multi-camera ray intersection
                    origins_t = torch.stack(all_origins)  # (N_cams, 3)
                    dirs_t = torch.stack(all_directions)  # (N_cams, 3)
                    centroid_3d = triangulate_rays(origins_t, dirs_t)
                else:
                    # Fallback: single camera, cast ray at estimated depth
                    origin, direction = all_origins[0], all_directions[0]
                    depth_est = 0.5  # metres below water surface
                    centroid_3d = origin + depth_est * direction
                centroids[fid] = centroid_3d.detach()

        # ---- Triangulation-seeded cold start ----
        # Track init type per fish for diagnostics
        init_type: dict[int, str] = {}
        for fid in valid_fish_ids:
            init_type[fid] = "warm" if fid in self._warm_starts else "cold"

        fish_needing_cold_start = [
            fid for fid in valid_fish_ids if fid not in self._warm_starts
        ]
        if fish_needing_cold_start and cfg.tri_seed:
            try:
                # triangulate_midlines is numpy/scipy-based — move models to CPU
                for model in models.values():
                    model.to("cpu")
                tri_results = triangulate_midlines(
                    midline_set,
                    models,
                    frame_index=frame_index,
                    max_depth=cfg.max_depth,
                )
                for model in models.values():
                    model.to(device)
                seeded = []
                skipped_bad_seed = []
                first_model = next(iter(models.values()))
                seed_water_z = first_model.water_z
                for fid in fish_needing_cold_start:
                    if fid in tri_results:
                        ctrl = torch.as_tensor(
                            tri_results[fid].control_points,
                            dtype=torch.float32,
                            device=device,
                        )
                        # Validate: require that the majority of control points are
                        # below the water surface (Z > water_z). Bad triangulation
                        # (e.g. RANSAC failure, degenerate geometry) can produce
                        # control points far above water, which prevents the optimizer
                        # from computing any valid projections and causes loss=0.
                        n_underwater = int((ctrl[:, 2] > seed_water_z).sum().item())
                        if n_underwater < ctrl.shape[0] // 2:
                            logger.warning(
                                "Frame %d fish %d: triangulation seed rejected "
                                "(%d/%d ctrl pts below water_z=%.4f); "
                                "falling back to cold start",
                                frame_index,
                                fid,
                                n_underwater,
                                ctrl.shape[0],
                                seed_water_z,
                            )
                            skipped_bad_seed.append(fid)
                            continue
                        self._warm_starts[fid] = ctrl
                        init_type[fid] = "tri-seed"
                        seeded.append(fid)
                if seeded:
                    logger.info(
                        "Frame %d: triangulation-seeded %d/%d cold-start fish: %s",
                        frame_index,
                        len(seeded),
                        len(fish_needing_cold_start),
                        seeded,
                    )
                if skipped_bad_seed:
                    logger.info(
                        "Frame %d: rejected %d bad triangulation seeds (cold-start "
                        "fallback): %s",
                        frame_index,
                        len(skipped_bad_seed),
                        skipped_bad_seed,
                    )
            except Exception:
                # Ensure models are back on GPU even if triangulation failed
                for model in models.values():
                    model.to(device)
                logger.warning(
                    "Frame %d: triangulation seeding failed, falling back to cold start",
                    frame_index,
                    exc_info=True,
                )

        # ---- Validate tri-seeds by initial loss ----
        # A bad tri-seed can poison the shared L-BFGS Hessian in the coarse
        # stage, corrupting ALL fish.  Evaluate each tri-seeded fish's initial
        # data loss and reject seeds that are significantly worse than a
        # cold-start initialization would produce.
        tri_seeded_fids = [
            fid for fid in valid_fish_ids if init_type.get(fid) == "tri-seed"
        ]
        if tri_seeded_fids:
            _MAX_SEED_LOSS = 100.0  # max acceptable initial chamfer (pixels)
            n_coarse_check = cfg.n_coarse_ctrl
            basis_check = get_basis(cfg.n_eval_points, n_coarse_check, device)
            reverted = []
            with torch.no_grad():
                for fid in tri_seeded_fids:
                    idx = valid_fish_ids.index(fid)
                    # Build seed control points
                    seed_ctrl = _init_ctrl_pts(
                        [fid],
                        centroids,
                        self._warm_starts,
                        n_coarse_check,
                        cfg,
                        ref_orientations,
                        device,
                    )  # (1, K, 3)
                    seed_loss = _data_loss(
                        seed_ctrl, basis_check, [midlines_per_fish[idx]], models, cfg
                    ).item()
                    if seed_loss > _MAX_SEED_LOSS:
                        logger.info(
                            "Frame %d fish %d: tri-seed rejected by loss check "
                            "(loss=%.1f > %.1f); falling back to cold start",
                            frame_index,
                            fid,
                            seed_loss,
                            _MAX_SEED_LOSS,
                        )
                        del self._warm_starts[fid]
                        init_type[fid] = "cold"
                        reverted.append(fid)
            if reverted:
                logger.info(
                    "Frame %d: reverted %d tri-seeds with high initial loss: %s",
                    frame_index,
                    len(reverted),
                    reverted,
                )

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

        # Precompute normalization denominators so each penalty is ~O(1) for a
        # "reference-sized" violation.  Lambdas then express pixel-equivalent
        # importance rather than compensating for unit mismatches.
        _nom2 = cfg.nominal_length_m**2  # m² — smoothness, z-var normalization

        # Snapshot: capture cold-start control points and per-fish losses
        if frame_index == self._snapshot_frame:
            _snap_cold_start = coarse_init.detach().cpu().numpy().copy()
            _snap_cold_losses: list[float] = []
            with torch.no_grad():
                for idx in range(n_fish):
                    single_ctrl = coarse_init[idx : idx + 1]
                    single_obs = [midlines_per_fish[idx]]
                    fish_loss = _data_loss(
                        single_ctrl, coarse_basis, single_obs, models, cfg
                    ).item()
                    _snap_cold_losses.append(fish_loss)

        # ---- Stage 1: Coarse optimization with ALM constraint enforcement ----
        n_coarse_constraints = 2 + (n_coarse - 2) + 1
        coarse_lambdas = torch.zeros(n_fish, n_coarse_constraints, device=device)
        coarse_rho = cfg.alm_rho_init
        frozen_coarse = torch.zeros(n_fish, dtype=torch.bool, device=device)

        logger.info(
            "Frame %d: coarse stage (%d fish, ALM outer=%d, L-BFGS inner=%d)...",
            frame_index,
            n_fish,
            cfg.alm_outer_iters,
            cfg.lbfgs_max_iter_coarse,
        )

        for alm_iter in range(cfg.alm_outer_iters):
            optimizer_coarse = torch.optim.LBFGS(
                [coarse_ctrl],
                lr=cfg.lbfgs_lr,
                max_iter=cfg.lbfgs_max_iter_coarse,
                history_size=cfg.lbfgs_history_size,
                line_search_fn="strong_wolfe",
            )

            # Snapshot lambdas/rho for closure (avoid stale captures)
            _alm_lambdas_c = coarse_lambdas.detach()
            _alm_rho_c = coarse_rho

            def closure_coarse(
                _opt: torch.optim.LBFGS = optimizer_coarse,
                _lam: torch.Tensor = _alm_lambdas_c,
                _rho: float = _alm_rho_c,
            ) -> torch.Tensor:
                _opt.zero_grad()

                data_l = _data_loss(
                    coarse_ctrl, coarse_basis, midlines_per_fish, models, cfg
                )
                smooth_l = _smoothness_penalty(coarse_ctrl)
                z_var_l = _z_variance_penalty(coarse_ctrl, coarse_basis)

                violations = _constraint_violations(coarse_ctrl, coarse_basis, cfg)
                alm_l = _alm_penalty(violations, _lam, _rho)

                total = (
                    data_l
                    + cfg.lambda_smoothness * (smooth_l / _nom2)
                    + cfg.lambda_z_variance * (z_var_l / _nom2)
                    + alm_l
                )
                total.backward()

                if coarse_ctrl.grad is not None and frozen_coarse.any():
                    coarse_ctrl.grad[frozen_coarse] = 0.0

                return total

            optimizer_coarse.step(closure_coarse)

            # Update multipliers
            with torch.no_grad():
                violations = _constraint_violations(coarse_ctrl, coarse_basis, cfg)
                coarse_lambdas = torch.clamp(
                    coarse_lambdas + coarse_rho * violations, min=0.0
                )
                max_viol = violations.clamp(min=0.0).max().item()

            if max_viol < cfg.alm_tol:
                logger.debug(
                    "Frame %d: coarse ALM converged at outer iter %d "
                    "(max violation=%.4f)",
                    frame_index,
                    alm_iter,
                    max_viol,
                )
                break

            coarse_rho = min(coarse_rho * cfg.alm_rho_factor, cfg.alm_rho_max)

        with torch.no_grad():
            data_l = _data_loss(
                coarse_ctrl, coarse_basis, midlines_per_fish, models, cfg
            )
        coarse_loss_val = data_l.item()
        logger.info(
            "Frame %d: coarse stage done, loss=%.4f", frame_index, coarse_loss_val
        )

        # Snapshot: capture post-coarse control points and per-fish losses
        if frame_index == self._snapshot_frame:
            _snap_post_coarse = coarse_ctrl.detach().cpu().numpy().copy()
            _snap_coarse_losses: list[float] = []
            with torch.no_grad():
                for idx in range(n_fish):
                    single_ctrl = coarse_ctrl[idx : idx + 1]
                    single_obs = [midlines_per_fish[idx]]
                    fish_loss = _data_loss(
                        single_ctrl, coarse_basis, single_obs, models, cfg
                    ).item()
                    _snap_coarse_losses.append(fish_loss)

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

        # ---- Stage 2: Fine optimization with ALM constraint enforcement ----
        fine_ctrl = fine_init.clone().requires_grad_(True)

        n_fine_constraints = 2 + (n_fine - 2) + 1
        fine_lambdas = torch.zeros(n_fish, n_fine_constraints, device=device)
        fine_rho = cfg.alm_rho_init

        final_per_fish_losses: list[float] = [float("inf")] * n_fish
        prev_losses_fine: list[float] = [float("inf")] * n_fish
        patience_counters_fine: list[int] = [0] * n_fish
        fine_frozen_mask = torch.zeros(n_fish, dtype=torch.bool, device=device)

        logger.info(
            "Frame %d: fine stage (ALM outer=%d, L-BFGS inner=%d)...",
            frame_index,
            cfg.alm_outer_iters,
            cfg.lbfgs_max_iter_fine,
        )

        for alm_iter in range(cfg.alm_outer_iters):
            # Fresh L-BFGS per outer iteration (Hessian reset)
            optimizer_fine = torch.optim.LBFGS(
                [fine_ctrl],
                lr=cfg.lbfgs_lr,
                max_iter=1,
                history_size=cfg.lbfgs_history_size,
                line_search_fn="strong_wolfe",
            )

            _alm_lambdas_f = fine_lambdas.detach()
            _alm_rho_f = fine_rho

            def closure_fine(
                _opt: torch.optim.LBFGS = optimizer_fine,
                _lam: torch.Tensor = _alm_lambdas_f,
                _rho: float = _alm_rho_f,
            ) -> torch.Tensor:
                _opt.zero_grad()

                data_l = _data_loss(
                    fine_ctrl, fine_basis, midlines_per_fish, models, cfg
                )
                smooth_l = _smoothness_penalty(fine_ctrl)
                z_var_l = _z_variance_penalty(fine_ctrl, fine_basis)

                violations = _constraint_violations(fine_ctrl, fine_basis, cfg)
                alm_l = _alm_penalty(violations, _lam, _rho)

                total = (
                    data_l
                    + cfg.lambda_smoothness * (smooth_l / _nom2)
                    + cfg.lambda_z_variance * (z_var_l / _nom2)
                    + alm_l
                )
                total.backward()

                if fine_ctrl.grad is not None and fine_frozen_mask.any():
                    fine_ctrl.grad[fine_frozen_mask] = 0.0

                return total

            # Inner L-BFGS loop with per-fish adaptive stopping
            for step in range(cfg.lbfgs_max_iter_fine):
                if fine_frozen_mask.all():
                    break

                optimizer_fine.step(closure_fine)

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
                                    "Frame %d fish %d converged at step %d "
                                    "(ALM iter %d, loss=%.4f)",
                                    frame_index,
                                    valid_fish_ids[idx],
                                    step,
                                    alm_iter,
                                    fish_loss,
                                )
                        else:
                            patience_counters_fine[idx] = 0

                        prev_losses_fine[idx] = fish_loss

                if alm_iter == 0 and step == 0:
                    for idx, fid in enumerate(valid_fish_ids):
                        logger.info(
                            "Frame %d fish %d [%s]: fine step 0 loss=%.1f",
                            frame_index,
                            fid,
                            init_type[fid],
                            final_per_fish_losses[idx],
                        )

            # Update ALM multipliers
            with torch.no_grad():
                violations = _constraint_violations(fine_ctrl, fine_basis, cfg)
                fine_lambdas = torch.clamp(
                    fine_lambdas + fine_rho * violations, min=0.0
                )
                max_viol = violations.clamp(min=0.0).max().item()

            logger.debug(
                "Frame %d: fine ALM iter %d, max violation=%.4f, rho=%.1f",
                frame_index,
                alm_iter,
                max_viol,
                fine_rho,
            )

            if max_viol < cfg.alm_tol:
                logger.debug(
                    "Frame %d: fine ALM converged at outer iter %d",
                    frame_index,
                    alm_iter,
                )
                break

            # Increase penalty and unfreeze fish for next outer iteration
            fine_rho = min(fine_rho * cfg.alm_rho_factor, cfg.alm_rho_max)
            fine_frozen_mask.zero_()
            prev_losses_fine = [float("inf")] * n_fish
            patience_counters_fine = [0] * n_fish

        n_converged = int(fine_frozen_mask.sum().item())
        logger.info(
            "Frame %d: fine stage done, %d/%d fish converged",
            frame_index,
            n_converged,
            n_fish,
        )

        # ---- Snapshot: assemble per-fish OptimizerSnapshot (first valid frame) ----
        if frame_index == self._snapshot_frame:
            _snap_post_fine = fine_ctrl.detach().cpu().numpy().copy()
            for idx, fid in enumerate(valid_fish_ids):
                cam_obs = midlines_per_fish[idx]
                best_cam = max(cam_obs, key=lambda c: cam_obs[c].shape[0])
                self.snapshots.append(
                    OptimizerSnapshot(
                        fish_id=fid,
                        best_cam_id=best_cam,
                        obs_2d=cam_obs[best_cam].cpu().numpy().copy(),
                        cold_start_ctrl=_snap_cold_start[idx],
                        cold_start_loss=_snap_cold_losses[idx],
                        post_coarse_ctrl=_snap_post_coarse[idx],
                        coarse_loss=_snap_coarse_losses[idx],
                        post_fine_ctrl=_snap_post_fine[idx],
                        fine_loss=final_per_fish_losses[idx],
                    )
                )

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

                # Filter to valid projected points
                proj_valid = proj_np[valid_np & ~np.any(np.isnan(proj_np), axis=1)]
                obs_valid = obs_np[~np.any(np.isnan(obs_np), axis=1)]

                if len(proj_valid) == 0 or len(obs_valid) == 0:
                    continue

                # Chamfer distance (same metric as optimizer loss)
                dists = scipy.spatial.distance.cdist(proj_valid, obs_valid)  # (M, N)
                proj_to_obs = dists.min(axis=1).mean()  # mean nearest obs
                obs_to_proj = dists.min(axis=0).mean()  # mean nearest proj
                chamfer = float((proj_to_obs + obs_to_proj) / 2.0)

                per_cam_residuals[cam_id] = chamfer
                all_residuals.append(chamfer)

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

            # Store warm-start for next frame, with consistency flip check
            ws_ctrl = fine_ctrl[idx].detach().clone().cpu()
            if fid in self._warm_starts:
                prev_dir = self._warm_starts[fid][-1] - self._warm_starts[fid][0]
                curr_dir = ws_ctrl[-1] - ws_ctrl[0]
                if torch.dot(curr_dir, prev_dir) < 0:
                    ws_ctrl = ws_ctrl.flip(0)
                    logger.debug(
                        "Frame %d fish %d: flipped warm-start "
                        "(spline direction reversed vs previous frame)",
                        frame_index,
                        fid,
                    )
            self._warm_starts[fid] = ws_ctrl
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
