"""Multi-view triangulation of 2D midline points into 3D B-spline midlines.

Implements exhaustive pairwise triangulation (<=7 cameras) and residual-based
rejection (>7 cameras) to produce Midline3D structs with fixed 7-control-point
cubic B-splines fit via scipy.interpolate.make_lsq_spline.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import scipy.interpolate
import torch

from aquapose.calibration.projection import RefractiveProjectionModel, triangulate_rays
from aquapose.reconstruction.midline import Midline2D

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SPLINE_K: int = 3
SPLINE_N_CTRL: int = 7
SPLINE_KNOTS: np.ndarray = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
)
MIN_BODY_POINTS: int = 9  # SPLINE_N_CTRL + 2
DEFAULT_INLIER_THRESHOLD: float = 15.0  # pixels
N_SAMPLE_POINTS: int = 15  # matches Phase 6 output

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# fish_id -> camera_id -> Midline2D
MidlineSet = dict[int, dict[str, Midline2D]]

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class Midline3D:
    """Continuous 3D midline for a single fish in a single frame.

    Attributes:
        fish_id: Globally unique fish identifier.
        frame_index: Frame index within the video.
        control_points: B-spline control points, shape (7, 3), float32.
        knots: B-spline knot vector, shape (11,), float32.
        degree: B-spline degree (always 3).
        arc_length: Total arc length of the spline in world metres.
        half_widths: Half-width of the fish at each of the 15 body positions
            in world metres, shape (N_SAMPLE_POINTS,), float32.
        n_cameras: Minimum number of camera observations across body points.
        mean_residual: Mean reprojection residual in pixels across all body
            points and inlier cameras.
        max_residual: Maximum reprojection residual in pixels across all body
            points and inlier cameras.
        is_low_confidence: True when any body point was triangulated from
            only 2 cameras.
    """

    fish_id: int
    frame_index: int
    control_points: np.ndarray  # shape (7, 3), float32
    knots: np.ndarray  # shape (11,), float32
    degree: int
    arc_length: float
    half_widths: np.ndarray  # shape (N,), float32 in world metres
    n_cameras: int
    mean_residual: float
    max_residual: float
    is_low_confidence: bool = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _triangulate_body_point(
    pixels: dict[str, torch.Tensor],
    models: dict[str, RefractiveProjectionModel],
    inlier_threshold: float,
) -> tuple[torch.Tensor, list[str], float] | None:
    """Triangulate a single 3D body point from multi-camera 2D observations.

    Dispatches to different strategies based on camera count:
    - 1 camera: returns None (dropped).
    - 2 cameras: triangulates the single pair directly.
    - 3-7 cameras: exhaustive pairwise search with held-out scoring, then
      re-triangulation using all inliers.
    - 8+ cameras: all-camera triangulation with residual-based outlier
      rejection (drop cameras with residual > median + 2*sigma).

    Args:
        pixels: Mapping from camera_id to pixel coordinate tensor, shape (2,),
            float32. Each tensor is a single (u, v) observation.
        models: Mapping from camera_id to RefractiveProjectionModel.
        inlier_threshold: Maximum reprojection error (pixels) to classify a
            camera as an inlier during re-triangulation.

    Returns:
        Tuple of (point_3d, inlier_cam_ids, max_residual) where point_3d has
        shape (3,) float32, or None if fewer than 2 cameras are available.
    """
    cam_ids = [cid for cid in pixels if cid in models]
    n_cams = len(cam_ids)

    if n_cams < 2:
        return None

    # Pre-cast rays for all cameras upfront
    origins: dict[str, torch.Tensor] = {}
    directions: dict[str, torch.Tensor] = {}
    for cid in cam_ids:
        px = pixels[cid].unsqueeze(0)  # (1, 2)
        o, d = models[cid].cast_ray(px)
        origins[cid] = o[0]  # (3,)
        directions[cid] = d[0]  # (3,)

    if n_cams == 2:
        # Single pair — no held-out scoring possible
        origs = torch.stack([origins[cid] for cid in cam_ids])  # (2, 3)
        dirs = torch.stack([directions[cid] for cid in cam_ids])  # (2, 3)
        pt3d = triangulate_rays(origs, dirs)
        return pt3d, cam_ids, 0.0

    if n_cams <= 7:
        # Exhaustive pairwise search
        best_pt3d: torch.Tensor | None = None
        best_cam_ids: list[str] = []
        best_max_error = float("inf")

        for pair in itertools.combinations(cam_ids, 2):
            pa, pb = pair
            origs = torch.stack([origins[pa], origins[pb]])  # (2, 3)
            dirs = torch.stack([directions[pa], directions[pb]])  # (2, 3)
            pt3d_candidate = triangulate_rays(origs, dirs)

            # Score by max reprojection error across held-out cameras
            held_out = [cid for cid in cam_ids if cid not in pair]
            if not held_out:
                max_held_out_error = 0.0
            else:
                errors = []
                pt3d_batch = pt3d_candidate.unsqueeze(0)  # (1, 3)
                for cid in held_out:
                    proj_px, valid = models[cid].project(pt3d_batch)
                    if valid[0]:
                        err = float(torch.linalg.norm(proj_px[0] - pixels[cid]).item())
                    else:
                        err = float("inf")
                    errors.append(err)
                max_held_out_error = max(errors)

            if max_held_out_error < best_max_error:
                best_max_error = max_held_out_error
                best_pt3d = pt3d_candidate
                best_cam_ids = list(pair)

        # Re-triangulate using all inlier cameras
        if best_pt3d is None:
            return None

        inlier_ids: list[str] = []
        pt3d_batch = best_pt3d.unsqueeze(0)  # (1, 3)
        for cid in cam_ids:
            proj_px, valid = models[cid].project(pt3d_batch)
            if valid[0]:
                err = float(torch.linalg.norm(proj_px[0] - pixels[cid]).item())
                if err < inlier_threshold:
                    inlier_ids.append(cid)

        # Ensure at least 2 cameras (fall back to seed pair if needed)
        if len(inlier_ids) < 2:
            inlier_ids = list(best_cam_ids)

        if len(inlier_ids) >= 2:
            origs = torch.stack([origins[cid] for cid in inlier_ids])
            dirs = torch.stack([directions[cid] for cid in inlier_ids])
            final_pt3d = triangulate_rays(origs, dirs)
        else:
            final_pt3d = best_pt3d
            inlier_ids = best_cam_ids

        # Compute max residual among inliers
        final_residuals = []
        pt3d_batch = final_pt3d.unsqueeze(0)  # (1, 3)
        for cid in inlier_ids:
            proj_px, valid = models[cid].project(pt3d_batch)
            if valid[0]:
                err = float(torch.linalg.norm(proj_px[0] - pixels[cid]).item())
                final_residuals.append(err)
        max_residual = max(final_residuals) if final_residuals else 0.0

        return final_pt3d, inlier_ids, max_residual

    # n_cams > 7: Residual rejection
    origs = torch.stack([origins[cid] for cid in cam_ids])
    dirs = torch.stack([directions[cid] for cid in cam_ids])
    pt3d_all = triangulate_rays(origs, dirs)

    # Compute per-camera reprojection residuals
    residuals_dict: dict[str, float] = {}
    pt3d_batch = pt3d_all.unsqueeze(0)  # (1, 3)
    for cid in cam_ids:
        proj_px, valid = models[cid].project(pt3d_batch)
        if valid[0]:
            err = float(torch.linalg.norm(proj_px[0] - pixels[cid]).item())
            residuals_dict[cid] = err
        else:
            residuals_dict[cid] = float("inf")

    residuals_arr = np.array([residuals_dict[cid] for cid in cam_ids])
    median = float(np.median(residuals_arr))
    sigma = float(np.std(residuals_arr))
    threshold = median + 2.0 * sigma

    inlier_ids = [cid for cid in cam_ids if residuals_dict[cid] <= threshold]
    if len(inlier_ids) < 2:
        # Fall back to best 2 by residual
        sorted_ids = sorted(cam_ids, key=lambda c: residuals_dict[c])
        inlier_ids = sorted_ids[:2]

    origs = torch.stack([origins[cid] for cid in inlier_ids])
    dirs = torch.stack([directions[cid] for cid in inlier_ids])
    final_pt3d = triangulate_rays(origs, dirs)

    pt3d_batch = final_pt3d.unsqueeze(0)
    final_residuals = []
    for cid in inlier_ids:
        proj_px, valid = models[cid].project(pt3d_batch)
        if valid[0]:
            err = float(torch.linalg.norm(proj_px[0] - pixels[cid]).item())
            final_residuals.append(err)
    max_residual = max(final_residuals) if final_residuals else 0.0

    return final_pt3d, inlier_ids, max_residual


def _fit_spline(
    u_param: np.ndarray,
    pts_3d: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Fit a fixed 7-control-point cubic B-spline to 3D body positions.

    Uses scipy.interpolate.make_lsq_spline with the fixed SPLINE_KNOTS
    interior knot vector. Requires at least MIN_BODY_POINTS observations.

    Args:
        u_param: Arc-length parameter values in [0, 1], shape (M,), float64.
            Must be strictly increasing.
        pts_3d: 3D point positions, shape (M, 3), float64.

    Returns:
        Tuple of (control_points, arc_length) where control_points has shape
        (7, 3), float32 and arc_length is the numerical integral of the spline
        curve length. Returns None if too few points or spline fitting fails.
    """
    if len(u_param) < MIN_BODY_POINTS:
        return None

    try:
        spl = scipy.interpolate.make_lsq_spline(
            u_param, pts_3d, SPLINE_KNOTS, k=SPLINE_K
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        # Schoenberg-Whitney condition violation or singular matrix
        logger.debug("Spline fitting failed: %s", exc)
        return None

    control_points = spl.c.astype(np.float32)  # shape (7, 3)

    # Compute arc length via 1000-point numerical integration
    u_fine = np.linspace(0.0, 1.0, 1000)
    curve_pts = spl(u_fine)  # shape (1000, 3)
    diffs = np.diff(curve_pts, axis=0)  # shape (999, 3)
    seg_lengths = np.linalg.norm(diffs, axis=1)  # shape (999,)
    arc_length = float(np.sum(seg_lengths))

    return control_points, arc_length


def _pixel_half_width_to_metres(
    hw_px: float,
    depth_m: float,
    focal_px: float,
) -> float:
    """Convert pixel half-width to world metres using pinhole approximation.

    Uses the formula: hw_m = hw_px * depth_m / focal_px

    This is an approximation valid near the optical axis. Sufficient for
    width profile estimation (not used in triangulation geometry).

    Args:
        hw_px: Half-width in pixels.
        depth_m: Depth of the body point below the water surface in metres.
        focal_px: Camera focal length in pixels (mean of fx and fy).

    Returns:
        Half-width in world metres.
    """
    return hw_px * depth_m / focal_px


def _flip_midline(midline: Midline2D) -> Midline2D:
    """Return a copy of the midline with points and half_widths reversed.

    Args:
        midline: Input 2D midline.

    Returns:
        New Midline2D with reversed point and half-width order.
    """
    return Midline2D(
        points=midline.points[::-1].copy(),
        half_widths=midline.half_widths[::-1].copy(),
        fish_id=midline.fish_id,
        camera_id=midline.camera_id,
        frame_index=midline.frame_index,
        is_head_to_tail=midline.is_head_to_tail,
    )


def _pairwise_chord_length(
    ref_ml: Midline2D,
    cand_ml: Midline2D,
    ref_model: RefractiveProjectionModel,
    cand_model: RefractiveProjectionModel,
    sample_indices: list[int],
) -> float:
    """Compute chord length of triangulated sample points from two cameras.

    Triangulates sampled body points from a reference and candidate camera,
    then sums the segment lengths between consecutive 3D points. A correct
    orientation produces a smooth curve with short chord; a flipped orientation
    produces a zigzag with much longer chord.

    Args:
        ref_ml: Reference camera midline (orientation fixed).
        cand_ml: Candidate camera midline (may be flipped).
        ref_model: Projection model for the reference camera.
        cand_model: Projection model for the candidate camera.
        sample_indices: Body point indices to triangulate.

    Returns:
        Total chord length in metres. Returns inf if triangulation fails.
    """
    pair_models = {"ref": ref_model, "cand": cand_model}
    pts_3d: list[torch.Tensor] = []

    for bp_idx in sample_indices:
        pixels = {
            "ref": torch.from_numpy(ref_ml.points[bp_idx]).float(),
            "cand": torch.from_numpy(cand_ml.points[bp_idx]).float(),
        }
        result = _triangulate_body_point(pixels, pair_models, inlier_threshold=50.0)
        if result is not None:
            pts_3d.append(result[0])

    if len(pts_3d) < 2:
        return float("inf")

    total = 0.0
    for j in range(len(pts_3d) - 1):
        total += float(torch.linalg.norm(pts_3d[j + 1] - pts_3d[j]).item())
    return total


def _align_midline_orientations(
    cam_midlines: dict[str, Midline2D],
    models: dict[str, RefractiveProjectionModel],
    inlier_threshold: float = DEFAULT_INLIER_THRESHOLD,
) -> dict[str, Midline2D]:
    """Align midline orientations across cameras via greedy pairwise alignment.

    Ensures that body point i from camera A corresponds to the same physical
    point as body point i from camera B. Without this, arbitrary BFS traversal
    order causes head/tail mismatch and zigzag 3D reconstructions.

    Algorithm: fixes the first camera (sorted order) as reference, then for
    each remaining camera independently tries both orientations and picks the
    one producing shorter chord length when triangulated against the reference.
    This is O(N) in camera count rather than O(2^N) brute-force.

    For N < 2 cameras: returns unchanged (triangulation will skip anyway).

    Args:
        cam_midlines: Per-camera midlines for a single fish.
        models: Per-camera refractive projection models.
        inlier_threshold: Maximum reprojection error for inlier classification.

    Returns:
        Dict of (possibly flipped) midlines with consistent orientation.
    """
    cam_ids = sorted(cid for cid in cam_midlines if cid in models)
    n_cams = len(cam_ids)

    if n_cams < 2:
        return cam_midlines

    # Sample head, mid, tail for chord-length comparison
    sample_indices = [0, N_SAMPLE_POINTS // 2, N_SAMPLE_POINTS - 1]

    ref_id = cam_ids[0]
    ref_ml = cam_midlines[ref_id]
    ref_model = models[ref_id]

    result = dict(cam_midlines)

    for cid in cam_ids[1:]:
        cand_ml = cam_midlines[cid]
        cand_model = models[cid]

        # Score original orientation
        chord_orig = _pairwise_chord_length(
            ref_ml, cand_ml, ref_model, cand_model, sample_indices
        )

        # Score flipped orientation
        flipped_ml = _flip_midline(cand_ml)
        chord_flip = _pairwise_chord_length(
            ref_ml, flipped_ml, ref_model, cand_model, sample_indices
        )

        if chord_flip < chord_orig:
            result[cid] = flipped_ml

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def triangulate_midlines(
    midline_set: MidlineSet,
    models: dict[str, RefractiveProjectionModel],
    frame_index: int = 0,
    inlier_threshold: float = DEFAULT_INLIER_THRESHOLD,
) -> dict[int, Midline3D]:
    """Triangulate 2D midlines from multiple cameras into 3D B-spline midlines.

    For each fish in midline_set, gathers per-body-point observations from all
    cameras, triangulates each body point, fits a cubic B-spline, converts
    half-widths to world metres, and returns a Midline3D per fish.

    Fish with fewer than MIN_BODY_POINTS valid triangulated points are skipped
    (spline cannot be fit reliably). Fish observed from only 2 cameras (at any
    body point) are flagged as is_low_confidence=True.

    Args:
        midline_set: Nested dict mapping fish_id to camera_id to Midline2D.
        models: Mapping from camera_id to RefractiveProjectionModel.
        frame_index: Frame index to embed in output Midline3D structs.
        inlier_threshold: Maximum reprojection error (pixels) for inlier
            classification during pairwise re-triangulation.

    Returns:
        Dict mapping fish_id to Midline3D. Only includes fish with sufficient
        valid body points for spline fitting.
    """
    results: dict[int, Midline3D] = {}

    for fish_id, cam_midlines in midline_set.items():
        cam_midlines = _align_midline_orientations(
            cam_midlines, models, inlier_threshold
        )
        valid_indices: list[int] = []
        pts_3d_list: list[np.ndarray] = []
        per_point_residuals: list[float] = []
        per_point_n_cams: list[int] = []
        per_point_inlier_ids: list[list[str]] = []
        per_point_hw_px: list[float] = []  # avg half-width in pixels across inliers
        per_point_depths: list[float] = []  # depth in metres below water

        for i in range(N_SAMPLE_POINTS):
            # Gather pixel observations and half-widths for body point i
            pixels: dict[str, torch.Tensor] = {}
            hw_px_list: list[float] = []
            for cam_id, midline in cam_midlines.items():
                if cam_id not in models:
                    continue
                pixels[cam_id] = torch.from_numpy(midline.points[i]).float()
                hw_px_list.append(float(midline.half_widths[i]))

            result = _triangulate_body_point(pixels, models, inlier_threshold)
            if result is None:
                continue

            pt3d, inlier_ids, max_res = result
            pt3d_np = pt3d.detach().cpu().numpy().astype(np.float64)

            valid_indices.append(i)
            pts_3d_list.append(pt3d_np)
            per_point_residuals.append(max_res)
            per_point_n_cams.append(len(inlier_ids))
            per_point_inlier_ids.append(inlier_ids)

            # Average half-width across cameras that observed this point
            avg_hw_px = float(np.mean(hw_px_list)) if hw_px_list else 0.0
            per_point_hw_px.append(avg_hw_px)

            # Depth below water surface: use first available model's water_z
            water_z = next(iter(models.values())).water_z
            depth_m = max(0.0, float(pt3d_np[2]) - water_z)
            per_point_depths.append(depth_m)

        if len(valid_indices) < MIN_BODY_POINTS:
            logger.debug(
                "Fish %d skipped: only %d valid body points (need %d)",
                fish_id,
                len(valid_indices),
                MIN_BODY_POINTS,
            )
            continue

        # Build arc-length parameter: preserve original positions, do NOT re-normalize
        u_param = np.array(
            [i / (N_SAMPLE_POINTS - 1) for i in valid_indices], dtype=np.float64
        )
        pts_3d_arr = np.stack(pts_3d_list, axis=0)  # shape (M, 3)

        spline_result = _fit_spline(u_param, pts_3d_arr)
        if spline_result is None:
            logger.debug("Fish %d skipped: spline fitting failed", fish_id)
            continue

        control_points, arc_length = spline_result

        # Convert half-widths to world metres
        # Compute mean focal length from first inlier camera model
        hw_metres_valid: list[float] = []
        for _idx, (hw_px, depth_m, inlier_ids) in enumerate(
            zip(per_point_hw_px, per_point_depths, per_point_inlier_ids, strict=True)
        ):
            if inlier_ids:
                cam_model = models[inlier_ids[0]]
                focal_px = float(
                    (cam_model.K[0, 0].item() + cam_model.K[1, 1].item()) / 2.0
                )
            else:
                focal_px = next(iter(models.values())).K[0, 0].item()
            hw_m = _pixel_half_width_to_metres(hw_px, depth_m, focal_px)
            hw_metres_valid.append(hw_m)

        # Interpolate half-widths for all 15 body positions (including invalid ones)
        hw_metres_all = np.zeros(N_SAMPLE_POINTS, dtype=np.float32)
        if len(valid_indices) >= 2:
            fill_bounds: tuple[float, float] = (hw_metres_valid[0], hw_metres_valid[-1])
            interp = scipy.interpolate.interp1d(
                u_param,
                np.array(hw_metres_valid, dtype=np.float64),
                kind="linear",
                bounds_error=False,
                fill_value=fill_bounds,  # type: ignore[arg-type]
            )
            u_all = np.linspace(0.0, 1.0, N_SAMPLE_POINTS)
            hw_metres_all = interp(u_all).astype(np.float32)
        elif len(valid_indices) == 1:
            hw_metres_all[:] = hw_metres_valid[0]

        # Compute summary stats
        mean_residual = (
            float(np.mean(per_point_residuals)) if per_point_residuals else 0.0
        )
        max_residual_val = (
            float(np.max(per_point_residuals)) if per_point_residuals else 0.0
        )
        min_n_cams = min(per_point_n_cams) if per_point_n_cams else 0
        is_low_confidence = min_n_cams <= 2

        midline_3d = Midline3D(
            fish_id=fish_id,
            frame_index=frame_index,
            control_points=control_points,
            knots=SPLINE_KNOTS.astype(np.float32),
            degree=SPLINE_K,
            arc_length=arc_length,
            half_widths=hw_metres_all,
            n_cameras=min_n_cams,
            mean_residual=mean_residual,
            max_residual=max_residual_val,
            is_low_confidence=is_low_confidence,
        )
        results[fish_id] = midline_3d

    return results


def refine_midline_lm(
    midline_3d: Midline3D,
    midline_set: MidlineSet,
    models: dict[str, RefractiveProjectionModel],
) -> Midline3D:
    """Refine 3D spline via Levenberg-Marquardt reprojection minimization.

    STUB: Returns midline_3d unchanged. Full implementation deferred.

    Args:
        midline_3d: Initial 3D midline estimate to refine.
        midline_set: Multi-camera 2D midline observations for this fish.
        models: Per-camera refractive projection models.

    Returns:
        Refined Midline3D (currently identical to input).
    """
    return midline_3d
