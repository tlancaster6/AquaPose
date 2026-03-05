"""DLT reconstruction backend — confidence-weighted triangulation with outlier rejection.

Implements a stripped-down reconstruction backend that replaces the complex
multi-strategy triangulation logic (pairwise exhaustive search, epipolar
refinement, orientation alignment) with a single uniform algorithm:

    triangulate all cameras → reject outliers → re-triangulate inliers → fit spline

This eliminates camera-count branching and upstream correspondence assumptions.
Upstream (pose estimation backend) provides ordered keypoints, so orientation
alignment and epipolar refinement are not needed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.interpolate
import torch

from aquapose.calibration.projection import triangulate_rays
from aquapose.core.reconstruction.utils import (
    SPLINE_K,
    build_spline_knots,
    fit_spline,
    pixel_half_width_to_metres,
    weighted_triangulate_rays,
)
from aquapose.core.types.midline import Midline2D
from aquapose.core.types.reconstruction import Midline3D, MidlineSet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_OUTLIER_THRESHOLD: float = 10.0
"""Default maximum reprojection residual (pixels) for inlier classification.

Tuned empirically via ``aquapose tune --stage reconstruction`` on YH dataset (2026-03-03).
Balances yield (74/403 = 18%) against mean error (2.91 px).
"""

DEFAULT_N_CONTROL_POINTS: int = 7
"""Default number of B-spline control points."""

DEFAULT_LOW_CONFIDENCE_FRACTION: float = 0.2
"""Fraction of body points with <3 inlier cameras above which the reconstruction
is flagged as is_low_confidence=True."""

_MIN_RAY_ANGLE_DEG: float = 5.0
"""Minimum ray angle (degrees) between two cameras below which the 2-camera
triangulation is considered ill-conditioned and skipped."""

_COS_MIN_RAY_ANGLE: float = math.cos(math.radians(_MIN_RAY_ANGLE_DEG))
"""Cosine of the minimum ray angle threshold. When |cos(angle)| > this value
the rays are too nearly parallel for reliable DLT."""

_MAX_ENDPOINT_GAP: float = 0.2
"""Maximum allowed gap at either endpoint of the valid body-point parameter range.

If the first valid body point has u > _MAX_ENDPOINT_GAP or the last has
u < 1.0 - _MAX_ENDPOINT_GAP, the reconstruction is rejected rather than
relying on spline extrapolation into unsupported regions."""

__all__ = ["DltBackend"]


# ---------------------------------------------------------------------------
# _TriangulationResult
# ---------------------------------------------------------------------------


@dataclass
class _TriangulationResult:
    """Structured result from _triangulate_fish_vectorized().

    Holds per-body-point triangulation results for all N body points.

    Attributes:
        pts_3d: Triangulated 3D positions, shape (N, 3). NaN for invalid points.
        valid_mask: Boolean mask of valid body points, shape (N,).
        inlier_masks: Per-body-point inlier camera mask, shape (N, C). True = inlier.
        mean_residuals: Mean inlier reprojection error per body point, shape (N,).
        inlier_cam_ids: Per-body-point list of inlier camera ID strings, length N.
    """

    pts_3d: torch.Tensor  # (N, 3)
    valid_mask: torch.Tensor  # (N,) bool
    inlier_masks: torch.Tensor  # (N, C) bool
    mean_residuals: torch.Tensor  # (N,) float
    inlier_cam_ids: list[list[str]]  # length N


# ---------------------------------------------------------------------------
# DltBackend
# ---------------------------------------------------------------------------


class DltBackend:
    """Confidence-weighted DLT triangulation backend for 3D midline reconstruction.

    Uses a single-strategy algorithm regardless of camera count:
    1. Triangulate all available cameras together.
    2. Compute per-camera reprojection residuals.
    3. Reject cameras whose residual exceeds outlier_threshold.
    4. Re-triangulate with inlier cameras only.
    5. Fit a B-spline to valid body points.

    Key simplifications over TriangulationBackend:
    - No orientation alignment (upstream pose backend provides ordered keypoints).
    - No epipolar refinement (ordered keypoints eliminate correspondence ambiguity).
    - No camera-count branching (single path: triangulate → reject → re-triangulate).

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        outlier_threshold: Maximum reprojection error (pixels) for inlier
            classification after initial all-camera triangulation.
        n_control_points: Number of B-spline control points per midline.
        low_confidence_fraction: Fraction of body points with <3 inlier cameras
            above which is_low_confidence is set True.

    Raises:
        FileNotFoundError: If calibration_path does not exist.
    """

    def __init__(
        self,
        calibration_path: str | Path | None = None,
        outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
        n_control_points: int = DEFAULT_N_CONTROL_POINTS,
        low_confidence_fraction: float = DEFAULT_LOW_CONFIDENCE_FRACTION,
        *,
        models: dict[str, Any] | None = None,
        z_flattening_enabled: bool = True,
    ) -> None:
        self._outlier_threshold = outlier_threshold
        self._n_control_points = n_control_points
        self._low_confidence_fraction = low_confidence_fraction
        self._z_flattening_enabled = z_flattening_enabled

        if models is not None:
            self._models = models
        elif calibration_path is not None:
            self._models = self._load_models(Path(calibration_path))
        else:
            raise TypeError("DltBackend requires either calibration_path or models.")

        # Precompute spline parameters
        self._spline_knots = build_spline_knots(n_control_points)
        self._min_body_points = n_control_points + 2

    @classmethod
    def from_models(
        cls,
        models: dict[str, Any],
        outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
        n_control_points: int = DEFAULT_N_CONTROL_POINTS,
        low_confidence_fraction: float = DEFAULT_LOW_CONFIDENCE_FRACTION,
        z_flattening_enabled: bool = True,
    ) -> DltBackend:
        """Create a DltBackend from pre-built projection models.

        Bypasses calibration file loading — useful when models are already
        constructed (e.g. from a pre-built projection model dict).

        Args:
            models: Dict mapping camera_id to RefractiveProjectionModel.
            outlier_threshold: Maximum reprojection error (pixels) for inlier
                classification.
            n_control_points: Number of B-spline control points per midline.
            low_confidence_fraction: Fraction of body points with <3 inlier
                cameras above which is_low_confidence is set True.
            z_flattening_enabled: Whether to flatten body points to centroid z
                before spline fitting.

        Returns:
            Configured DltBackend instance.
        """
        return cls(
            outlier_threshold=outlier_threshold,
            n_control_points=n_control_points,
            low_confidence_fraction=low_confidence_fraction,
            models=models,
            z_flattening_enabled=z_flattening_enabled,
        )

    def reconstruct_frame(
        self,
        frame_idx: int,
        midline_set: MidlineSet,
    ) -> dict[int, Midline3D]:
        """Triangulate all fish midlines for a single frame.

        For each fish in midline_set, applies confidence-weighted DLT
        triangulation with outlier rejection and B-spline fitting.

        Args:
            frame_idx: Frame index to embed in output Midline3D structs.
            midline_set: Nested dict mapping fish_id to camera_id to Midline2D.

        Returns:
            Dict mapping fish_id to Midline3D. Only includes fish with
            sufficient valid body points for spline fitting.
        """
        results: dict[int, Midline3D] = {}

        # Derive water_z from models (all cameras share the same value)
        if not self._models:
            return results
        water_z: float = next(iter(self._models.values())).water_z

        for fish_id, cam_midlines in midline_set.items():
            midline_3d = self._reconstruct_fish(
                fish_id=fish_id,
                frame_idx=frame_idx,
                cam_midlines=cam_midlines,
                water_z=water_z,
            )
            if midline_3d is not None:
                results[fish_id] = midline_3d

        return results

    def _reconstruct_fish(
        self,
        fish_id: int,
        frame_idx: int,
        cam_midlines: dict[str, Midline2D],
        water_z: float,
    ) -> Midline3D | None:
        """Reconstruct a single fish's 3D midline from multi-camera observations.

        Args:
            fish_id: Fish identifier.
            frame_idx: Frame index.
            cam_midlines: Per-camera 2D midlines for this fish.
            water_z: Z coordinate of the water surface in world frame.

        Returns:
            Midline3D if reconstruction succeeds, None if fish should be skipped.
        """
        # Derive body point count from first available midline
        first_midline = next(iter(cam_midlines.values()))
        n_body_points = len(first_midline.points)

        valid_indices: list[int] = []
        pts_3d_list: list[np.ndarray] = []
        per_point_residuals: list[float] = []
        per_point_n_cams: list[int] = []
        per_point_inlier_ids: list[list[str]] = []
        per_point_hw_px: list[float] = []
        per_point_depths: list[float] = []

        tri_result = self._triangulate_fish_vectorized(cam_midlines, water_z)

        for i in range(n_body_points):
            if not tri_result.valid_mask[i].item():
                continue
            pt3d_np = tri_result.pts_3d[i].detach().cpu().numpy().astype(np.float64)
            inlier_ids = tri_result.inlier_cam_ids[i]

            valid_indices.append(i)
            pts_3d_list.append(pt3d_np)
            per_point_residuals.append(float(tri_result.mean_residuals[i].item()))
            per_point_n_cams.append(len(inlier_ids))
            per_point_inlier_ids.append(inlier_ids)

            # Average half-width across cameras that observed this body point
            hw_px_list: list[float] = []
            for cam_id, midline in cam_midlines.items():
                if cam_id not in self._models:
                    continue
                pt = midline.points[i]
                if not np.any(np.isnan(pt)):
                    hw_px_list.append(float(midline.half_widths[i]))
            avg_hw_px = float(np.mean(hw_px_list)) if hw_px_list else 0.0
            per_point_hw_px.append(avg_hw_px)

            depth_m = max(0.0, float(pt3d_np[2]) - water_z)
            per_point_depths.append(depth_m)

        if len(valid_indices) < self._min_body_points:
            logger.debug(
                "Fish %d skipped: only %d valid body points (need %d)",
                fish_id,
                len(valid_indices),
                self._min_body_points,
            )
            return None

        # Build arc-length parameter preserving original body positions
        u_param = np.array(
            [i / (n_body_points - 1) for i in valid_indices], dtype=np.float64
        )

        # Reject if valid body points don't cover the endpoints — spline
        # extrapolation beyond the observed range is unreliable and can
        # produce degenerate control points far outside the aquarium.
        if u_param[0] > _MAX_ENDPOINT_GAP or u_param[-1] < 1.0 - _MAX_ENDPOINT_GAP:
            logger.debug(
                "Fish %d skipped: valid body points span [%.2f, %.2f], "
                "extrapolation gap exceeds %.0f%% threshold",
                fish_id,
                u_param[0],
                u_param[-1],
                _MAX_ENDPOINT_GAP * 100,
            )
            return None

        pts_3d_arr = np.stack(pts_3d_list, axis=0)  # shape (M, 3)

        # --- Z-flattening: set all body points to centroid z ---
        centroid_z: float | None = None
        z_offsets_valid: np.ndarray | None = None

        if self._z_flattening_enabled:
            centroid_z = float(pts_3d_arr[:, 2].mean())
            z_offsets_valid = (pts_3d_arr[:, 2] - centroid_z).astype(np.float32)
            pts_3d_arr[:, 2] = centroid_z

        spline_result = fit_spline(
            u_param,
            pts_3d_arr,
            knots=self._spline_knots,
            min_body_points=self._min_body_points,
        )
        if spline_result is None:
            logger.debug("Fish %d skipped: spline fitting failed", fish_id)
            return None

        control_points, arc_length = spline_result

        # Convert half-widths to world metres via pinhole approximation
        hw_metres_all = self._convert_half_widths(
            per_point_hw_px=per_point_hw_px,
            per_point_depths=per_point_depths,
            per_point_inlier_ids=per_point_inlier_ids,
            u_param=u_param,
            n_body_points=n_body_points,
        )

        # Compute spline-based per-camera residuals
        spline_obj = scipy.interpolate.BSpline(
            self._spline_knots, control_points.astype(np.float64), SPLINE_K
        )
        u_sample = np.linspace(0.0, 1.0, n_body_points)
        spline_pts_3d = torch.from_numpy(
            spline_obj(u_sample).astype(np.float32)
        )  # (n_body_points, 3)

        all_residuals: list[float] = []
        cam_residuals: dict[str, float] = {}
        cam_ids_active = [cid for cid in cam_midlines if cid in self._models]
        for cid in cam_ids_active:
            proj_px, valid = self._models[cid].project(spline_pts_3d)
            proj_np = proj_px.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            obs_pts = cam_midlines[cid].points  # (n_body_points, 2)
            cam_errs: list[float] = []
            for j in range(n_body_points):
                if (
                    valid_np[j]
                    and not np.any(np.isnan(proj_np[j]))
                    and not np.any(np.isnan(obs_pts[j]))
                ):
                    err = float(np.linalg.norm(proj_np[j] - obs_pts[j]))
                    cam_errs.append(err)
                    all_residuals.append(err)
            if cam_errs:
                cam_residuals[cid] = float(np.mean(cam_errs))

        mean_residual = float(np.mean(all_residuals)) if all_residuals else 0.0
        max_residual_val = float(np.max(all_residuals)) if all_residuals else 0.0

        min_n_cams = min(per_point_n_cams) if per_point_n_cams else 0
        n_weak = sum(1 for nc in per_point_n_cams if nc < 3)
        is_low_confidence = n_weak > self._low_confidence_fraction * len(
            per_point_n_cams
        )

        # Build z_offsets array: NaN for non-triangulated points,
        # actual offsets for valid body points.
        z_off: np.ndarray | None = None
        if self._z_flattening_enabled and z_offsets_valid is not None:
            z_off = np.full(n_body_points, np.nan, dtype=np.float32)
            for vi, body_idx in enumerate(valid_indices):
                z_off[body_idx] = float(z_offsets_valid[vi])

        return Midline3D(
            fish_id=fish_id,
            frame_index=frame_idx,
            control_points=control_points,
            knots=self._spline_knots.astype(np.float32),
            degree=SPLINE_K,
            arc_length=arc_length,
            half_widths=hw_metres_all,
            n_cameras=min_n_cams,
            mean_residual=mean_residual,
            max_residual=max_residual_val,
            is_low_confidence=is_low_confidence,
            per_camera_residuals=cam_residuals,
            centroid_z=centroid_z,
            z_offsets=z_off,
        )

    def _triangulate_fish_vectorized(
        self,
        cam_midlines: dict[str, Midline2D],
        water_z: float,
    ) -> _TriangulationResult:
        """Vectorized triangulation for all N body points simultaneously.

        Replaces the per-body-point Python loop in _reconstruct_fish(). Processes
        all N body points in batched torch operations: ray casting (C calls instead
        of N*C), normal-equation assembly (C-camera loop over (N,3,3) tensors), and
        a single batched torch.linalg.lstsq call.

        Outlier rejection is iterative: on each pass the single worst camera is
        removed per body point, then that body point is re-triangulated.  This
        prevents self-poisoning where one degenerate 2D midline corrupts the
        initial triangulation enough to make ALL cameras appear as outliers.
        The loop runs at most C-2 times and exits early when no outliers remain.

        Note: The 2-camera ray-angle filter (_MIN_RAY_ANGLE_DEG / _COS_MIN_RAY_ANGLE)
        is deliberately omitted from this vectorized path. 2-camera body points are
        uncommon (usually ≥3 cameras observe each point), and near-parallel rays
        within those are rarer still. The scalar _triangulate_body_point() retains
        the filter for reference.

        Also omits the first-pass water-surface check (only applied after
        re-triangulation). Above-water initial triangulations virtually always remain
        above-water after re-triangulation; the check is redundant in practice.

        Args:
            cam_midlines: Per-camera 2D midlines for this fish, keyed by camera ID.
            water_z: Z coordinate of the water surface in world frame.

        Returns:
            _TriangulationResult with per-body-point results.
        """
        # Determine device/dtype from the first model
        first_model = next(iter(self._models.values()))
        device = first_model.K.device
        dtype = torch.float32

        # Build ordered list of camera IDs that appear in both cam_midlines and _models
        cam_ids = [cid for cid in cam_midlines if cid in self._models]
        C = len(cam_ids)

        # Derive N from the first midline
        first_midline = next(iter(cam_midlines.values()))
        N = len(first_midline.points)

        if C < 2 or N == 0:
            # Not enough cameras — all body points are invalid
            empty_pts = torch.full((N, 3), float("nan"), device=device, dtype=dtype)
            empty_mask = torch.zeros(N, dtype=torch.bool, device=device)
            empty_inlier = torch.zeros(
                N, C if C > 0 else 0, dtype=torch.bool, device=device
            )
            empty_resid = torch.zeros(N, dtype=dtype, device=device)
            return _TriangulationResult(
                pts_3d=empty_pts,
                valid_mask=empty_mask,
                inlier_masks=empty_inlier,
                mean_residuals=empty_resid,
                inlier_cam_ids=[[] for _ in range(N)],
            )

        # ------------------------------------------------------------------
        # Step 1 — Build (N, C, 2) pixel tensor and (N, C) validity/weight
        # ------------------------------------------------------------------
        pixels_nc = torch.zeros(N, C, 2, device=device, dtype=dtype)
        valid_nc = torch.zeros(N, C, dtype=torch.bool, device=device)
        weights_nc = torch.zeros(N, C, device=device, dtype=dtype)

        for c, cam_id in enumerate(cam_ids):
            midline = cam_midlines[cam_id]
            pts_np = midline.points  # (N, 2) float32 numpy
            pts_t = torch.from_numpy(pts_np).to(device=device, dtype=dtype)
            is_nan = torch.isnan(pts_t).any(dim=-1)  # (N,)
            valid_nc[:, c] = ~is_nan
            # Replace NaN with 0.0 (zero-weight handles them)
            pts_t = torch.where(is_nan.unsqueeze(-1), torch.zeros_like(pts_t), pts_t)
            pixels_nc[:, c, :] = pts_t

            # Confidence weights: sqrt(confidence) where valid, 0.0 where invalid
            if midline.point_confidence is not None:
                conf_t = torch.from_numpy(midline.point_confidence).to(
                    device=device, dtype=dtype
                )
                w = torch.sqrt(conf_t.clamp(min=0.0))
            else:
                w = torch.ones(N, device=device, dtype=dtype)
            weights_nc[:, c] = torch.where(valid_nc[:, c], w, torch.zeros_like(w))

        # Pre-filter: body points with fewer than 2 valid cameras
        valid_cam_count = valid_nc.sum(dim=1)  # (N,)
        has_enough = valid_cam_count >= 2  # (N,)

        # ------------------------------------------------------------------
        # Step 2 — Cast rays: C calls, each processing N body points
        # ------------------------------------------------------------------
        origins_nc = torch.zeros(N, C, 3, device=device, dtype=dtype)
        dirs_nc = torch.zeros(N, C, 3, device=device, dtype=dtype)
        for c, cam_id in enumerate(cam_ids):
            o, d = self._models[cam_id].cast_ray(pixels_nc[:, c, :])  # (N, 3) each
            origins_nc[:, c, :] = o.to(device=device, dtype=dtype)
            dirs_nc[:, c, :] = d.to(device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # Step 3 — First-pass normal-equation assembly and lstsq solve
        # ------------------------------------------------------------------
        eye3 = torch.eye(3, device=device, dtype=dtype)
        A = torch.zeros(N, 3, 3, device=device, dtype=dtype)
        b = torch.zeros(N, 3, device=device, dtype=dtype)
        for c in range(C):
            d_c = dirs_nc[:, c, :]  # (N, 3)
            o_c = origins_nc[:, c, :]  # (N, 3)
            w_c = weights_nc[:, c]  # (N,)
            ddt = torch.einsum("ni,nj->nij", d_c, d_c)  # (N, 3, 3)
            M = eye3.unsqueeze(0) - ddt  # (N, 3, 3)
            wM = w_c[:, None, None] * M  # (N, 3, 3)
            A = A + wM
            b = b + torch.einsum("nij,nj->ni", wM, o_c)

        # Solve: pts_3d_pass1[n] = lstsq(A[n], b[n])
        # Invalid points will have A≈0, but we mask them out afterward
        pts_3d_pass1 = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(
            -1
        )  # (N, 3)

        # ------------------------------------------------------------------
        # Steps 4-6 — Iterative outlier rejection
        #
        # Single-round rejection is vulnerable to self-poisoning: one camera
        # with a degenerate 2D midline corrupts the initial triangulation,
        # inflating residuals for ALL cameras and causing valid cameras to be
        # rejected.  Iterative removal of the single worst camera per body
        # point avoids this — after removing the bad camera, good cameras'
        # residuals recover and they survive the threshold check.
        # ------------------------------------------------------------------
        active_nc = valid_nc.clone()  # (N, C) — cameras still active per point
        pts_3d_current = pts_3d_pass1
        max_reject_iters = max(C - 2, 0)
        inf_val = torch.tensor(float("inf"), device=device, dtype=dtype)

        for _reject_iter in range(max_reject_iters):
            # Compute per-camera residuals from current solution
            iter_resid = torch.full((N, C), float("inf"), device=device, dtype=dtype)
            for c, cam_id in enumerate(cam_ids):
                proj_px, proj_valid = self._models[cam_id].project(pts_3d_current)
                proj_px = proj_px.to(device=device, dtype=dtype)
                proj_valid = proj_valid.to(device=device)
                err = torch.linalg.norm(proj_px - pixels_nc[:, c, :], dim=-1)  # (N,)
                keep = proj_valid & active_nc[:, c]
                iter_resid[:, c] = torch.where(keep, err, inf_val)

            # Find worst active camera per body point (ignore inactive)
            masked_resid = torch.where(
                active_nc,
                iter_resid,
                torch.full_like(iter_resid, -float("inf")),
            )
            max_resid, worst_cam = masked_resid.max(dim=1)  # (N,)
            has_outlier = has_enough & (max_resid > self._outlier_threshold)

            if not has_outlier.any():
                break

            # Only remove if body point retains >= 2 cameras afterward
            active_count = active_nc.sum(dim=1)  # (N,)
            can_remove = has_outlier & (active_count > 2)

            if not can_remove.any():
                break

            # Deactivate worst camera for affected body points
            remove_idx = can_remove.nonzero(as_tuple=True)[0]
            active_nc[remove_idx, worst_cam[remove_idx]] = False

            # Re-triangulate with updated camera set
            w_active = weights_nc * active_nc.to(dtype=dtype)
            A_iter = torch.zeros(N, 3, 3, device=device, dtype=dtype)
            b_iter = torch.zeros(N, 3, device=device, dtype=dtype)
            for c in range(C):
                d_c = dirs_nc[:, c, :]
                o_c = origins_nc[:, c, :]
                w_c = w_active[:, c]
                ddt = torch.einsum("ni,nj->nij", d_c, d_c)
                M = eye3.unsqueeze(0) - ddt
                wM = w_c[:, None, None] * M
                A_iter = A_iter + wM
                b_iter = b_iter + torch.einsum("nij,nj->ni", wM, o_c)

            pts_3d_current = torch.linalg.lstsq(
                A_iter, b_iter.unsqueeze(-1)
            ).solution.squeeze(-1)

        # Final inlier state from iterative rejection
        inlier_nc = active_nc
        inlier_count = inlier_nc.sum(dim=1)  # (N,)
        has_enough_inliers = has_enough & (inlier_count >= 2)  # (N,)
        pts_3d_pass2 = pts_3d_current

        # ------------------------------------------------------------------
        # Step 7 — Water surface rejection (post re-triangulation only)
        # ------------------------------------------------------------------
        above_water = pts_3d_pass2[:, 2] <= water_z  # (N,)
        valid_mask = has_enough_inliers & ~above_water  # (N,)

        # ------------------------------------------------------------------
        # Step 8 — Compute mean inlier residuals for pass-2 points
        # ------------------------------------------------------------------
        residuals2_nc = torch.full((N, C), float("inf"), device=device, dtype=dtype)
        for c, cam_id in enumerate(cam_ids):
            proj_px2, proj_valid2 = self._models[cam_id].project(pts_3d_pass2)
            proj_px2 = proj_px2.to(device=device, dtype=dtype)
            proj_valid2 = proj_valid2.to(device=device)
            obs_px = pixels_nc[:, c, :]
            err2 = torch.linalg.norm(proj_px2 - obs_px, dim=-1)
            keep2 = proj_valid2 & inlier_nc[:, c]
            residuals2_nc[:, c] = torch.where(
                keep2,
                err2,
                torch.tensor(float("inf"), device=device, dtype=dtype),
            )

        # Mean over inlier cameras per body point (ignore inf)
        finite_mask2 = residuals2_nc.isfinite()  # (N, C)
        inlier_sum = (residuals2_nc * finite_mask2.to(dtype=dtype)).sum(dim=1)  # (N,)
        inlier_cnt = finite_mask2.sum(dim=1).clamp(min=1).to(dtype=dtype)  # (N,)
        mean_residuals = inlier_sum / inlier_cnt  # (N,)

        # ------------------------------------------------------------------
        # Step 9 — Build inlier_cam_ids from inlier_nc and cam_ids
        # ------------------------------------------------------------------
        inlier_cam_ids: list[list[str]] = [
            [cam_ids[c] for c in range(C) if inlier_nc[n, c].item()] for n in range(N)
        ]

        # NaN-fill invalid points in pts_3d
        pts_3d_out = pts_3d_pass2.clone()
        pts_3d_out[~valid_mask] = float("nan")

        return _TriangulationResult(
            pts_3d=pts_3d_out,
            valid_mask=valid_mask,
            inlier_masks=inlier_nc,
            mean_residuals=mean_residuals,
            inlier_cam_ids=inlier_cam_ids,
        )

    def _triangulate_body_point(
        self,
        point_idx: int,
        cam_midlines: dict[str, Midline2D],
        water_z: float,
    ) -> tuple[torch.Tensor, list[str], float] | None:
        """Triangulate a single body point from all available cameras.

        Algorithm: single-strategy, no camera-count branching.
        1. Gather valid pixel observations from all cameras (skip NaN).
        2. Apply ray-angle filter for 2-camera case.
        3. Triangulate all cameras together (weighted or unweighted).
        4. Reject points at or above the water surface (Z <= water_z).
        5. Compute per-camera reprojection residuals.
        6. Reject cameras with residual > outlier_threshold.
        7. If <2 inlier cameras remain, drop the point.
        8. Re-triangulate with inlier cameras only.
        9. Apply water surface rejection again.

        Args:
            point_idx: Body point index.
            cam_midlines: Per-camera 2D midlines for this fish.
            water_z: Z coordinate of the water surface.

        Returns:
            Tuple of (point_3d, inlier_cam_ids, mean_residual) or None if dropped.
        """
        # Gather valid observations and cast rays
        cam_ids: list[str] = []
        origins: dict[str, torch.Tensor] = {}
        directions: dict[str, torch.Tensor] = {}
        pixels: dict[str, torch.Tensor] = {}
        weights: dict[str, float] = {}

        for cam_id, midline in cam_midlines.items():
            if cam_id not in self._models:
                continue
            pt = midline.points[point_idx]
            if np.any(np.isnan(pt)):
                continue  # NaN pixel — skip this camera for this body point

            px_tensor = torch.from_numpy(pt).float()  # (2,)
            pixels[cam_id] = px_tensor

            o, d = self._models[cam_id].cast_ray(px_tensor.unsqueeze(0))
            origins[cam_id] = o[0]  # (3,)
            directions[cam_id] = d[0]  # (3,)

            # Collect confidence weight: sqrt(confidence) per plan spec
            if midline.point_confidence is not None:
                weights[cam_id] = float(np.sqrt(midline.point_confidence[point_idx]))
            else:
                weights[cam_id] = 1.0

            cam_ids.append(cam_id)

        if len(cam_ids) < 2:
            return None

        # Ray-angle filter for 2-camera case only
        if len(cam_ids) == 2:
            pa, pb = cam_ids[0], cam_ids[1]
            cos_angle = float(torch.dot(directions[pa], directions[pb]).abs().item())
            if cos_angle > _COS_MIN_RAY_ANGLE:
                logger.debug(
                    "2-cam pair (%s, %s) skipped: ray angle too small (cos=%.4f > %.4f)",
                    pa,
                    pb,
                    cos_angle,
                    _COS_MIN_RAY_ANGLE,
                )
                return None

        # Initial triangulation: all cameras together (single strategy)
        pt3d = self._tri_rays(cam_ids, origins, directions, weights)

        # Water surface rejection: drop if Z <= water_z
        if float(pt3d[2].item()) <= water_z:
            logger.debug(
                "Body point %d rejected (initial): Z=%.3f <= water_z=%.3f",
                point_idx,
                float(pt3d[2].item()),
                water_z,
            )
            return None

        # Compute per-camera reprojection residuals
        pt3d_batch = pt3d.unsqueeze(0)  # (1, 3)
        residuals: dict[str, float] = {}
        for cam_id in cam_ids:
            proj_px, valid = self._models[cam_id].project(pt3d_batch)
            if valid[0]:
                err = float(torch.linalg.norm(proj_px[0] - pixels[cam_id]).item())
                residuals[cam_id] = err
            else:
                residuals[cam_id] = float("inf")

        # Outlier rejection: keep cameras within threshold
        inlier_ids = [
            cid
            for cid in cam_ids
            if residuals.get(cid, float("inf")) <= self._outlier_threshold
        ]

        if len(inlier_ids) < 2:
            logger.debug(
                "Body point %d dropped: only %d inlier cameras after rejection",
                point_idx,
                len(inlier_ids),
            )
            return None

        # Re-triangulate with inlier cameras only
        pt3d = self._tri_rays(inlier_ids, origins, directions, weights)

        # Water surface rejection again on re-triangulated point
        if float(pt3d[2].item()) <= water_z:
            logger.debug(
                "Body point %d rejected (re-triangulated): Z=%.3f <= water_z=%.3f",
                point_idx,
                float(pt3d[2].item()),
                water_z,
            )
            return None

        # Compute mean residual among inlier cameras for the re-triangulated point
        pt3d_batch = pt3d.unsqueeze(0)
        inlier_residuals: list[float] = []
        for cam_id in inlier_ids:
            proj_px, valid = self._models[cam_id].project(pt3d_batch)
            if valid[0]:
                err = float(torch.linalg.norm(proj_px[0] - pixels[cam_id]).item())
                inlier_residuals.append(err)
        mean_res = float(np.mean(inlier_residuals)) if inlier_residuals else 0.0

        return pt3d, inlier_ids, mean_res

    def _tri_rays(
        self,
        cam_ids: list[str],
        origins: dict[str, torch.Tensor],
        directions: dict[str, torch.Tensor],
        weights: dict[str, float],
    ) -> torch.Tensor:
        """Triangulate from the given camera IDs using weighted or unweighted DLT.

        Weighted path is used when any weight differs from 1.0; otherwise falls
        back to the unweighted triangulate_rays for backward compatibility.

        Args:
            cam_ids: Camera IDs to triangulate from.
            origins: Ray origin tensors per camera, shape (3,).
            directions: Unit ray direction tensors per camera, shape (3,).
            weights: Per-camera scalar weights (sqrt of confidence).

        Returns:
            Triangulated 3D point, shape (3,).
        """
        origs = torch.stack([origins[cid] for cid in cam_ids])
        dirs = torch.stack([directions[cid] for cid in cam_ids])

        use_weights = any(weights.get(cid, 1.0) != 1.0 for cid in cam_ids)

        if use_weights:
            w = torch.tensor(
                [weights.get(cid, 1.0) for cid in cam_ids],
                dtype=origs.dtype,
                device=origs.device,
            )
            return weighted_triangulate_rays(origs, dirs, w)

        return triangulate_rays(origs, dirs)

    def _convert_half_widths(
        self,
        per_point_hw_px: list[float],
        per_point_depths: list[float],
        per_point_inlier_ids: list[list[str]],
        u_param: np.ndarray,
        n_body_points: int,
    ) -> np.ndarray:
        """Convert pixel half-widths to world metres and interpolate to full body.

        Args:
            per_point_hw_px: Average half-width in pixels per valid body point.
            per_point_depths: Depth below water surface per valid body point.
            per_point_inlier_ids: Inlier camera IDs per valid body point.
            u_param: Arc-length parameter values for valid body points.
            n_body_points: Total number of body points (including invalid).

        Returns:
            Half-widths in world metres for all n_body_points positions,
            shape (n_body_points,), float32. Invalid positions are filled
            via linear interpolation / boundary extension.
        """
        hw_metres_valid: list[float] = []
        for hw_px, depth_m, inlier_ids in zip(
            per_point_hw_px, per_point_depths, per_point_inlier_ids, strict=True
        ):
            if inlier_ids:
                cam_model = self._models[inlier_ids[0]]
                focal_px = float(
                    (cam_model.K[0, 0].item() + cam_model.K[1, 1].item()) / 2.0
                )
            else:
                focal_px = float(next(iter(self._models.values())).K[0, 0].item())
            hw_m = pixel_half_width_to_metres(hw_px, depth_m, focal_px)
            hw_metres_valid.append(hw_m)

        hw_metres_all = np.zeros(n_body_points, dtype=np.float32)
        if len(u_param) >= 2:
            fill_bounds: tuple[float, float] = (hw_metres_valid[0], hw_metres_valid[-1])
            interp = scipy.interpolate.interp1d(
                u_param,
                np.array(hw_metres_valid, dtype=np.float64),
                kind="linear",
                bounds_error=False,
                fill_value=fill_bounds,  # type: ignore[arg-type]
            )
            u_all = np.linspace(0.0, 1.0, n_body_points)
            hw_metres_all = interp(u_all).astype(np.float32)
        elif len(u_param) == 1:
            hw_metres_all[:] = hw_metres_valid[0]

        return hw_metres_all

    @staticmethod
    def _load_models(
        calibration_path: Path,
    ) -> dict[str, Any]:
        """Load calibration and build per-camera RefractiveProjectionModel dict.

        Args:
            calibration_path: Path to AquaCal calibration JSON.

        Returns:
            Dict mapping camera_id to RefractiveProjectionModel.

        Raises:
            FileNotFoundError: If calibration_path does not exist.
        """
        from aquapose.calibration.loader import (
            compute_undistortion_maps,
            load_calibration_data,
        )
        from aquapose.calibration.projection import RefractiveProjectionModel

        calib = load_calibration_data(str(calibration_path))
        models: dict[str, Any] = {}
        for cam_id, cam_data in calib.cameras.items():
            maps = compute_undistortion_maps(cam_data)
            models[cam_id] = RefractiveProjectionModel(
                K=maps.K_new,
                R=cam_data.R,
                t=cam_data.t,
                water_z=calib.water_z,
                normal=calib.interface_normal,
                n_air=calib.n_air,
                n_water=calib.n_water,
            )
        return models
