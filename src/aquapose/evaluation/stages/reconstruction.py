"""Pure-function evaluator for the reconstruction stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.interpolate
import torch

from aquapose.core.reconstruction.utils import SPLINE_K
from aquapose.core.types.midline import Midline2D
from aquapose.core.types.reconstruction import Midline3D, MidlineSet
from aquapose.evaluation.metrics import Tier2Result, compute_tier1

DEFAULT_GRID: dict[str, list[float]] = {
    "outlier_threshold": [float(t) for t in range(10, 105, 5)],
    "n_points": [7.0, 11.0, 15.0, 21.0],
}


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Metrics for the reconstruction stage.

    Attributes:
        mean_reprojection_error: Overall mean pixel reprojection error across
            all fish and cameras.
        max_reprojection_error: Overall maximum pixel reprojection error across
            all fish and cameras.
        fish_reconstructed: Total fish-frame pairs successfully reconstructed.
        fish_available: Total fish-frame pairs available for reconstruction.
        inlier_ratio: Fraction of fish-frame pairs NOT flagged as low-confidence.
        low_confidence_flag_rate: Fraction of fish-frame pairs flagged as
            low-confidence by the reconstruction backend.
        tier2_stability: Overall maximum leave-one-out displacement in metres.
            None when Tier 2 stability data is not provided.
        per_camera_error: Per-camera mean and max reprojection error aggregates.
            Maps camera_id to dict with keys ``"mean_px"`` and ``"max_px"``.
        per_fish_error: Per-fish mean and max reprojection error aggregates.
            Maps fish_id to dict with keys ``"mean_px"`` and ``"max_px"``.
    """

    mean_reprojection_error: float
    max_reprojection_error: float
    fish_reconstructed: int
    fish_available: int
    inlier_ratio: float
    low_confidence_flag_rate: float
    tier2_stability: float | None
    per_camera_error: dict[str, dict[str, float]]
    per_fish_error: dict[int, dict[str, float]]
    z_denoising: ZDenoisingMetrics | None = None
    p50_reprojection_error: float | None = None
    p90_reprojection_error: float | None = None
    p95_reprojection_error: float | None = None
    per_point_error: dict[int, dict[str, float]] | None = None
    curvature_stratified: dict[str, dict[str, float | int | str]] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict representation.

        Returns:
            Dict with all fields. Integer fish keys in per_fish_error are
            converted to strings for JSON compatibility. tier2_stability is
            serialized as float or None.
        """
        return {
            "mean_reprojection_error": float(self.mean_reprojection_error),
            "max_reprojection_error": float(self.max_reprojection_error),
            "fish_reconstructed": int(self.fish_reconstructed),
            "fish_available": int(self.fish_available),
            "inlier_ratio": float(self.inlier_ratio),
            "low_confidence_flag_rate": float(self.low_confidence_flag_rate),
            "tier2_stability": float(self.tier2_stability)
            if self.tier2_stability is not None
            else None,
            "per_camera_error": {
                cam_id: {k: float(v) for k, v in stats.items()}
                for cam_id, stats in self.per_camera_error.items()
            },
            "per_fish_error": {
                str(fish_id): {k: float(v) for k, v in stats.items()}
                for fish_id, stats in self.per_fish_error.items()
            },
            "z_denoising": self.z_denoising.to_dict()
            if self.z_denoising is not None
            else None,
            "p50_reprojection_error": float(self.p50_reprojection_error)
            if self.p50_reprojection_error is not None
            else None,
            "p90_reprojection_error": float(self.p90_reprojection_error)
            if self.p90_reprojection_error is not None
            else None,
            "p95_reprojection_error": float(self.p95_reprojection_error)
            if self.p95_reprojection_error is not None
            else None,
            "per_point_error": {
                str(k): {sk: float(sv) for sk, sv in v.items()}
                for k, v in self.per_point_error.items()
            }
            if self.per_point_error is not None
            else None,
            "curvature_stratified": self.curvature_stratified,
        }


def evaluate_reconstruction(
    frame_results: list[tuple[int, dict[int, Midline3D]]],
    fish_available: int = 0,
    *,
    tier2_result: Tier2Result | None = None,
    per_point_error: dict[int, dict[str, float]] | None = None,
    curvature_stratified: dict[str, dict[str, float | int | str]] | None = None,
) -> ReconstructionMetrics:
    """Evaluate reconstruction quality from triangulation frame results.

    Wraps ``compute_tier1`` internally for reprojection error aggregation,
    and additionally computes low-confidence flag rate and optional Tier 2
    stability from a pre-computed ``Tier2Result``.

    Args:
        frame_results: List of ``(frame_idx, dict[fish_id, Midline3D])`` pairs
            from triangulation results.
        fish_available: Total fish-frame pairs available in the evaluated
            frames. Used to compute reconstruction coverage rate.
        tier2_result: Optional pre-computed Tier 2 leave-one-out displacement
            result. When provided, the overall max displacement is extracted
            as ``tier2_stability``. When omitted or None, ``tier2_stability``
            is set to None.

    Returns:
        ReconstructionMetrics with reprojection error, confidence rates,
        optional tier2_stability, and per-camera/per-fish breakdowns.
    """
    tier1 = compute_tier1(frame_results, fish_available)

    # Count low-confidence flags
    total_fish_frames = 0
    low_confidence_count = 0
    for _frame_idx, midline_dict in frame_results:
        for midline3d in midline_dict.values():
            total_fish_frames += 1
            if midline3d.is_low_confidence:
                low_confidence_count += 1

    if total_fish_frames > 0:
        low_confidence_flag_rate = low_confidence_count / total_fish_frames
    else:
        low_confidence_flag_rate = 0.0
    inlier_ratio = 1.0 - low_confidence_flag_rate

    # Extract tier2_stability from Tier2Result when provided
    tier2_stability: float | None = None
    if tier2_result is not None:
        all_displacements = [
            v
            for cam_map in tier2_result.per_fish_dropout.values()
            for v in cam_map.values()
            if v is not None
        ]
        if all_displacements:
            tier2_stability = float(max(all_displacements))

    # Compute reprojection error percentiles (EVAL-01)
    all_residuals = [
        midline3d.mean_residual
        for _frame_idx, midline_dict in frame_results
        for midline3d in midline_dict.values()
    ]
    if len(all_residuals) > 0:
        pcts = np.percentile(all_residuals, [50, 90, 95])
        p50_err: float | None = float(pcts[0])
        p90_err: float | None = float(pcts[1])
        p95_err: float | None = float(pcts[2])
    else:
        p50_err = None
        p90_err = None
        p95_err = None

    return ReconstructionMetrics(
        mean_reprojection_error=tier1.overall_mean_px,
        max_reprojection_error=tier1.overall_max_px,
        fish_reconstructed=tier1.fish_reconstructed,
        fish_available=tier1.fish_available,
        inlier_ratio=inlier_ratio,
        low_confidence_flag_rate=low_confidence_flag_rate,
        tier2_stability=tier2_stability,
        per_camera_error=tier1.per_camera,
        per_fish_error=tier1.per_fish,
        p50_reprojection_error=p50_err,
        p90_reprojection_error=p90_err,
        p95_reprojection_error=p95_err,
        per_point_error=per_point_error,
        curvature_stratified=curvature_stratified,
    )


@dataclass(frozen=True)
class ZDenoisingMetrics:
    """Z-denoising quality metrics for Components A and B.

    Attributes:
        median_z_range_cm: Median z-range across fish (cm).  The z-range
            for each fish-frame is max(z) - min(z) of the spline evaluated
            at sample points.
        mean_z_profile_rms_cm: Mean frame-to-frame z-profile RMS (cm).
        per_fish_snr: Per-fish signal-to-noise ratio.  Signal = std of
            mean z-profile across time; noise = std of frame-to-frame
            z-profile changes.
        fish_above_snr_1: Count of fish with SNR > 1.
        total_fish: Total number of unique fish evaluated.
        residual_delta_px: Component A gate metric -- mean residual change
            from before to after plane projection.  None when comparison
            data is unavailable.
    """

    median_z_range_cm: float
    mean_z_profile_rms_cm: float
    per_fish_snr: dict[int, float]
    fish_above_snr_1: int
    total_fish: int
    residual_delta_px: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict representation.

        Returns:
            Dict with all fields. Integer keys in per_fish_snr are
            converted to strings for JSON compatibility.
        """
        return {
            "median_z_range_cm": float(self.median_z_range_cm),
            "mean_z_profile_rms_cm": float(self.mean_z_profile_rms_cm),
            "per_fish_snr": {str(k): float(v) for k, v in self.per_fish_snr.items()},
            "fish_above_snr_1": int(self.fish_above_snr_1),
            "total_fish": int(self.total_fish),
            "residual_delta_px": float(self.residual_delta_px)
            if self.residual_delta_px is not None
            else None,
        }


def compute_z_denoising_metrics(
    frame_results: list[tuple[int, dict[int, Midline3D]]],
    n_sample_points: int = 15,
    *,
    residual_delta_px: float | None = None,
) -> ZDenoisingMetrics:
    """Compute z-denoising quality metrics from reconstruction frame results.

    Evaluates z-range, z-profile RMS, and per-fish SNR from the z-coordinates
    of spline-evaluated sample points.

    Args:
        frame_results: List of ``(frame_idx, dict[fish_id, Midline3D])`` pairs.
        n_sample_points: Number of sample points for spline evaluation.
        residual_delta_px: Optional pre-computed residual change metric
            (Component A gate).

    Returns:
        ZDenoisingMetrics with all z-denoising quality indicators.
    """
    # Collect per-fish z-profiles across frames
    # fish_id -> list of (frame_idx, z_profile)
    fish_profiles: dict[int, list[tuple[int, np.ndarray]]] = {}

    u_sample = np.linspace(0.0, 1.0, n_sample_points)

    for frame_idx, midline_dict in frame_results:
        for fish_id, m3d in midline_dict.items():
            if m3d.control_points is None or m3d.knots is None or m3d.degree is None:
                continue
            try:
                spl = scipy.interpolate.BSpline(
                    m3d.knots.astype(np.float64),
                    m3d.control_points.astype(np.float64),
                    SPLINE_K,
                )
                pts = spl(u_sample)  # (n_sample_points, 3)
                z_profile = pts[:, 2]  # z-coordinates
            except Exception:
                continue

            if fish_id not in fish_profiles:
                fish_profiles[fish_id] = []
            fish_profiles[fish_id].append((frame_idx, z_profile))

    if not fish_profiles:
        return ZDenoisingMetrics(
            median_z_range_cm=0.0,
            mean_z_profile_rms_cm=0.0,
            per_fish_snr={},
            fish_above_snr_1=0,
            total_fish=0,
            residual_delta_px=residual_delta_px,
        )

    # Per-fish z-range: median of max(z) - min(z) across frames
    all_z_ranges: list[float] = []
    per_fish_snr: dict[int, float] = {}
    all_z_rms: list[float] = []

    for fish_id, profiles in fish_profiles.items():
        # Sort by frame index
        profiles.sort(key=lambda x: x[0])

        z_profiles = np.array([p[1] for p in profiles])  # (T, n_sample_points)

        # Z-range per frame
        z_ranges = np.ptp(z_profiles, axis=1)  # max(z) - min(z) per frame
        median_z_range = float(np.median(z_ranges))
        all_z_ranges.append(median_z_range)

        # Z-profile RMS (frame-to-frame)
        if len(z_profiles) > 1:
            z_diffs = np.diff(z_profiles, axis=0)  # (T-1, n_sample_points)
            rms_vals = np.sqrt(np.mean(z_diffs**2, axis=1))  # (T-1,)
            mean_rms = float(np.mean(rms_vals))
            all_z_rms.append(mean_rms)

            # SNR: signal = std of mean z-profile, noise = std of diffs
            mean_z = np.mean(z_profiles, axis=0)  # mean z at each sample point
            signal = float(np.std(mean_z))
            noise = float(np.std(z_diffs))
            snr = signal / noise if noise > 1e-12 else 0.0
            per_fish_snr[fish_id] = snr
        else:
            per_fish_snr[fish_id] = 0.0

    median_z_range_cm = float(np.median(all_z_ranges)) * 100.0  # m -> cm
    mean_z_rms_cm = float(np.mean(all_z_rms)) * 100.0 if all_z_rms else 0.0
    fish_above_snr_1 = sum(1 for v in per_fish_snr.values() if v > 1.0)

    return ZDenoisingMetrics(
        median_z_range_cm=median_z_range_cm,
        mean_z_profile_rms_cm=mean_z_rms_cm,
        per_fish_snr=per_fish_snr,
        fish_above_snr_1=fish_above_snr_1,
        total_fish=len(fish_profiles),
        residual_delta_px=residual_delta_px,
    )


def compute_per_point_error(
    frame_results: list[tuple[int, dict[int, Midline3D]]],
    midline_sets_by_frame: dict[int, MidlineSet],
    projection_models: dict[str, Any],
    n_body_points: int = 15,
) -> dict[int, dict[str, float]] | None:
    """Compute per-keypoint reprojection error from 3D spline reprojection.

    For each Midline3D, evaluates the B-spline at ``n_body_points`` uniform
    parameter values, projects through each observing camera's model, and
    measures pixel distance to the corresponding 2D midline points.

    Args:
        frame_results: List of ``(frame_idx, dict[fish_id, Midline3D])`` pairs.
        midline_sets_by_frame: Mapping from frame_idx to MidlineSet
            (fish_id -> camera_id -> Midline2D).
        projection_models: Mapping from camera_id to projection model with
            a ``project(points_3d: Tensor) -> (projected_2d, valid_mask)``
            method.
        n_body_points: Number of uniform sample points on the spline.

    Returns:
        Dict mapping point_index (0..n_body_points-1) to
        ``{"mean_px": float, "p90_px": float}``, or None if no data.
    """
    # Accumulate errors per point index: point_idx -> list of pixel errors
    per_point_errors: dict[int, list[float]] = {i: [] for i in range(n_body_points)}
    u_sample = np.linspace(0.0, 1.0, n_body_points)

    for frame_idx, midline_dict in frame_results:
        if frame_idx not in midline_sets_by_frame:
            continue
        frame_ms = midline_sets_by_frame[frame_idx]

        for fish_id, m3d in midline_dict.items():
            if fish_id not in frame_ms:
                continue

            # Resolve 3D points: spline mode or raw-keypoint mode
            if (
                m3d.control_points is not None
                and m3d.knots is not None
                and m3d.degree is not None
            ):
                # Spline mode: evaluate B-spline at uniform parameter values
                try:
                    spl = scipy.interpolate.BSpline(
                        m3d.knots.astype(np.float64),
                        m3d.control_points.astype(np.float64),
                        SPLINE_K,
                    )
                    pts_3d = spl(u_sample).astype(np.float32)  # (N, 3)
                except Exception:
                    continue
            elif m3d.points is not None and len(m3d.points) == n_body_points:
                # Raw-keypoint mode: use the triangulated points directly
                pts_3d = m3d.points.astype(np.float32)  # (N, 3)
            else:
                # Neither representation available or point count mismatch
                continue

            cam_map = frame_ms[fish_id]
            for camera_id, midline_2d in cam_map.items():
                if camera_id not in projection_models:
                    continue
                model = projection_models[camera_id]

                # Project 3D points
                pts_tensor = torch.from_numpy(pts_3d)
                proj_px, valid = model.project(pts_tensor)
                proj_np = proj_px.detach().cpu().numpy()  # CUDA safety

                # Get observed 2D points — must have same count
                obs_2d = midline_2d.points
                if obs_2d.shape[0] != n_body_points:
                    continue

                valid_np = valid.cpu().numpy()
                for j in range(n_body_points):
                    if not valid_np[j]:
                        continue
                    err = float(
                        np.linalg.norm(proj_np[j] - obs_2d[j].astype(np.float64))
                    )
                    per_point_errors[j].append(err)

    # Check if we collected any data
    has_data = any(len(errs) > 0 for errs in per_point_errors.values())
    if not has_data:
        return None

    result: dict[int, dict[str, float]] = {}
    for pt_idx in range(n_body_points):
        errs = per_point_errors[pt_idx]
        if len(errs) > 0:
            arr = np.array(errs)
            result[pt_idx] = {
                "mean_px": float(np.mean(arr)),
                "p90_px": float(np.percentile(arr, 90)),
            }
        else:
            result[pt_idx] = {"mean_px": 0.0, "p90_px": 0.0}

    return result


def compute_curvature_stratified(
    frame_results: list[tuple[int, dict[int, Midline3D]]],
    midline_sets_by_frame: dict[int, MidlineSet],
) -> dict[str, dict[str, float | int | str]] | None:
    """Compute curvature-stratified reconstruction quality.

    Bins fish-frames by 2D curvature quartile using the camera with highest
    mean keypoint confidence, then reports reprojection error per quartile.

    Args:
        frame_results: List of ``(frame_idx, dict[fish_id, Midline3D])`` pairs.
        midline_sets_by_frame: Mapping from frame_idx to MidlineSet.

    Returns:
        Dict with keys ``"Q1"``..``"Q4"`` mapping to
        ``{"mean_error_px", "p90_error_px", "count", "curvature_range"}``,
        or None if fewer than 4 data points.
    """
    from aquapose.training.pseudo_labels import compute_curvature

    pairs: list[tuple[float, float]] = []  # (curvature, mean_residual)

    for frame_idx, midline_dict in frame_results:
        if frame_idx not in midline_sets_by_frame:
            continue
        frame_ms = midline_sets_by_frame[frame_idx]

        for fish_id, m3d in midline_dict.items():
            if fish_id not in frame_ms:
                continue
            cam_map = frame_ms[fish_id]
            if not cam_map:
                continue

            # Find camera with highest mean point confidence
            best_midline: Midline2D | None = None
            best_conf = -1.0
            for midline_2d in cam_map.values():
                if midline_2d.point_confidence is not None:
                    conf = float(np.mean(midline_2d.point_confidence))
                else:
                    conf = 1.0
                if conf > best_conf:
                    best_conf = conf
                    best_midline = midline_2d

            if best_midline is None:
                continue
            if best_midline.points.shape[0] < 3:
                continue

            curv = compute_curvature(best_midline.points)
            pairs.append((curv, m3d.mean_residual))

    if len(pairs) < 4:
        return None

    curvatures = np.array([p[0] for p in pairs])
    errors = np.array([p[1] for p in pairs])

    # Compute quartile bin edges
    bin_edges = np.quantile(curvatures, [0.25, 0.5, 0.75])
    # np.digitize: values <= edge[0] -> bin 0, etc.
    bin_indices = np.digitize(curvatures, bin_edges)  # 0, 1, 2, 3

    result: dict[str, dict[str, float | int | str]] = {}
    for bin_idx, q_label in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        mask = bin_indices == bin_idx
        count = int(np.sum(mask))
        if count > 0:
            bin_errors = errors[mask]
            bin_curvatures = curvatures[mask]
            result[q_label] = {
                "mean_error_px": float(np.mean(bin_errors)),
                "p90_error_px": float(np.percentile(bin_errors, 90)),
                "count": count,
                "curvature_range": f"{float(np.min(bin_curvatures)):.4f}-{float(np.max(bin_curvatures)):.4f}",
            }
        else:
            result[q_label] = {
                "mean_error_px": 0.0,
                "p90_error_px": 0.0,
                "count": 0,
                "curvature_range": "N/A",
            }

    return result


__all__ = [
    "DEFAULT_GRID",
    "ReconstructionMetrics",
    "ZDenoisingMetrics",
    "compute_curvature_stratified",
    "compute_per_point_error",
    "compute_z_denoising_metrics",
    "evaluate_reconstruction",
]
