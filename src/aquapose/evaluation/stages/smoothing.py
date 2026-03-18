"""Z-smoothing quality evaluator comparing pre- and post-smoothing metrics."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SmoothingMetrics:
    """Metrics for temporal z-smoothing quality.

    Attributes:
        source_file: Name of the smoothed H5 file evaluated.
        sigma_frames: Gaussian smoothing sigma in frames (default 3 if not stored).
        fish_processed: Number of unique fish IDs with data.
        frames_processed: Total fish-frame pairs with centroid z data.
        mean_jitter_before: Mean frame-to-frame centroid z jitter in cm (before).
        mean_jitter_after: Mean frame-to-frame centroid z jitter in cm (after).
        jitter_reduction: Ratio of before/after jitter.
        per_fish_jitter: Per fish: ``{before_cm, after_cm}``.
        reproj_before: Reprojection stats ``{mean, p50, p90, p95, p99}`` in px.
        reproj_after: Same keys, post-smoothing.
        reproj_n_fish_frames: Fish-frames used for reproj evaluation.
        reproj_n_residuals: Total residuals used for reproj evaluation.
    """

    source_file: str
    sigma_frames: int
    fish_processed: int
    frames_processed: int
    mean_jitter_before: float
    mean_jitter_after: float
    jitter_reduction: float
    per_fish_jitter: dict[int, dict[str, float]]
    reproj_before: dict[str, float]
    reproj_after: dict[str, float]
    reproj_n_fish_frames: int
    reproj_n_residuals: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict representation.

        Returns:
            Dict with all fields, per_fish_jitter keyed by string fish IDs.
        """
        return {
            "source_file": self.source_file,
            "sigma_frames": self.sigma_frames,
            "fish_processed": self.fish_processed,
            "frames_processed": self.frames_processed,
            "mean_jitter_before": float(self.mean_jitter_before),
            "mean_jitter_after": float(self.mean_jitter_after),
            "jitter_reduction": float(self.jitter_reduction),
            "per_fish_jitter": {
                str(k): v for k, v in sorted(self.per_fish_jitter.items())
            },
            "reproj_before": {k: float(v) for k, v in self.reproj_before.items()},
            "reproj_after": {k: float(v) for k, v in self.reproj_after.items()},
            "reproj_n_fish_frames": self.reproj_n_fish_frames,
            "reproj_n_residuals": self.reproj_n_residuals,
        }


def _detect_smoothed_h5(run_dir: Path) -> tuple[Path, Path] | None:
    """Detect smoothed H5 file and its unsmoothed counterpart.

    Checks for ``midlines_stitched_smoothed.h5`` first (preferred), then
    ``midlines_smoothed.h5``.

    Args:
        run_dir: Pipeline run directory.

    Returns:
        ``(unsmoothed_path, smoothed_path)`` or ``None`` if no smoothed file.
    """
    for stem in ("midlines_stitched", "midlines"):
        smoothed = run_dir / f"{stem}_smoothed.h5"
        unsmoothed = run_dir / f"{stem}.h5"
        if smoothed.exists() and unsmoothed.exists():
            return unsmoothed, smoothed
    return None


def _compute_jitter_metrics(
    smoothed_path: Path,
) -> tuple[int, int, float, float, float, dict[int, dict[str, float]]]:
    """Compute frame-to-frame centroid z jitter before and after smoothing.

    Args:
        smoothed_path: Path to the smoothed H5 file containing both
            ``centroid_z`` and ``smoothed_centroid_z`` datasets.

    Returns:
        Tuple of ``(fish_processed, frames_processed, mean_jitter_before,
        mean_jitter_after, jitter_reduction, per_fish_jitter)``.
    """
    import h5py

    with h5py.File(str(smoothed_path), "r") as f:
        grp = f["midlines"]
        centroid_z = grp["centroid_z"][:]  # (N, max_fish)
        smoothed_cz = grp["smoothed_centroid_z"][:]  # (N, max_fish)
        fish_ids = grp["fish_id"][:]  # (N, max_fish)
        frame_indices = grp["frame_index"][:]  # (N,)

    # Group by fish_id: list of (frame_index, raw_z, smoothed_z)
    fish_data: dict[int, list[tuple[int, float, float]]] = {}
    frames_processed = 0
    for row_idx in range(len(frame_indices)):
        fi = int(frame_indices[row_idx])
        for slot in range(fish_ids.shape[1]):
            fid = int(fish_ids[row_idx, slot])
            if fid < 0:
                continue
            raw_z = float(centroid_z[row_idx, slot])
            sm_z = float(smoothed_cz[row_idx, slot])
            if np.isnan(raw_z) or np.isnan(sm_z):
                continue
            fish_data.setdefault(fid, []).append((fi, raw_z, sm_z))
            frames_processed += 1

    per_fish_jitter: dict[int, dict[str, float]] = {}
    all_before: list[float] = []
    all_after: list[float] = []

    for fid in sorted(fish_data):
        entries = sorted(fish_data[fid], key=lambda x: x[0])
        before_deltas: list[float] = []
        after_deltas: list[float] = []
        for i in range(1, len(entries)):
            before_deltas.append(abs(entries[i][1] - entries[i - 1][1]))
            after_deltas.append(abs(entries[i][2] - entries[i - 1][2]))
        if before_deltas:
            b = float(np.mean(before_deltas)) * 100.0  # m -> cm
            a = float(np.mean(after_deltas)) * 100.0  # m -> cm
            per_fish_jitter[fid] = {"before_cm": b, "after_cm": a}
            all_before.append(b)
            all_after.append(a)

    mean_before = float(np.mean(all_before)) if all_before else 0.0
    mean_after = float(np.mean(all_after)) if all_after else 0.0
    reduction = mean_before / mean_after if mean_after > 1e-12 else float("inf")

    return (
        len(fish_data),
        frames_processed,
        mean_before,
        mean_after,
        reduction,
        per_fish_jitter,
    )


def _build_dz_lookup(smoothed_path: Path) -> dict[tuple[int, int], float]:
    """Build a (frame_index, fish_id) -> dz lookup from the smoothed H5.

    Args:
        smoothed_path: Path to the smoothed H5 file.

    Returns:
        Dict mapping ``(frame_index, fish_id)`` to ``smoothed_z - raw_z``.
    """
    import h5py

    with h5py.File(str(smoothed_path), "r") as f:
        grp = f["midlines"]
        centroid_z = grp["centroid_z"][:]
        smoothed_cz = grp["smoothed_centroid_z"][:]
        fish_ids = grp["fish_id"][:]
        frame_indices = grp["frame_index"][:]

    dz_all = smoothed_cz - centroid_z
    lookup: dict[tuple[int, int], float] = {}
    for row_idx in range(len(frame_indices)):
        fi = int(frame_indices[row_idx])
        for slot in range(fish_ids.shape[1]):
            fid = int(fish_ids[row_idx, slot])
            if fid < 0:
                continue
            dz_val = float(dz_all[row_idx, slot])
            if not np.isnan(dz_val):
                lookup[(fi, fid)] = dz_val
    return lookup


def _shift_frame_results(
    frame_results: list[tuple[int, dict]],
    dz_lookup: dict[tuple[int, int], float],
) -> list[tuple[int, dict]]:
    """Apply z-shifts to 3D midline points and control points.

    Args:
        frame_results: Pre-smoothing ``(frame_idx, {fish_id: Midline3D})`` list.
        dz_lookup: ``(frame_index, fish_id) -> dz`` mapping.

    Returns:
        New frame_results with shifted z-coordinates.
    """
    shifted: list[tuple[int, dict]] = []
    for frame_idx, fish_dict in frame_results:
        shifted_dict = {}
        for fish_id, m3d in fish_dict.items():
            dz = dz_lookup.get((frame_idx, fish_id))
            if dz is None or abs(dz) < 1e-12:
                shifted_dict[fish_id] = m3d
                continue
            m3d_shifted = copy.copy(m3d)
            if m3d.points is not None:
                pts = m3d.points.copy()
                pts[:, 2] += dz
                m3d_shifted.points = pts
            if m3d.control_points is not None:
                cp = m3d.control_points.copy()
                cp[:, 2] += dz
                m3d_shifted.control_points = cp
            shifted_dict[fish_id] = m3d_shifted
        shifted.append((frame_idx, shifted_dict))
    return shifted


def _compute_reproj_stats(
    frame_results: list[tuple[int, dict]],
    midline_sets_by_frame: dict[int, dict],
    projection_models: dict[str, Any],
    n_body_points: int = 6,
) -> dict[str, float]:
    """Compute reprojection error statistics by projecting 3D points.

    Args:
        frame_results: ``(frame_idx, {fish_id: Midline3D})`` tuples.
        midline_sets_by_frame: ``{frame_idx: {fish_id: {cam_id: Midline2D}}}``.
        projection_models: ``{cam_id: RefractiveProjectionModel}``.
        n_body_points: Expected number of body points per midline.

    Returns:
        Dict with ``mean``, ``p50``, ``p90``, ``p95``, ``p99``, ``n_fish_frames``,
        ``n_residuals``.
    """
    import torch

    all_residuals: list[float] = []
    n_fish_frames = 0

    for frame_idx, fish_dict in frame_results:
        if frame_idx not in midline_sets_by_frame:
            continue
        frame_ms = midline_sets_by_frame[frame_idx]

        for fish_id, m3d in fish_dict.items():
            if fish_id not in frame_ms:
                continue

            # Get 3D points
            if m3d.points is not None and len(m3d.points) == n_body_points:
                pts_3d = m3d.points.astype(np.float32)
            elif (
                m3d.control_points is not None
                and m3d.knots is not None
                and m3d.degree is not None
            ):
                import scipy.interpolate

                from aquapose.core.reconstruction.utils import SPLINE_K

                try:
                    spl = scipy.interpolate.BSpline(
                        m3d.knots.astype(np.float64),
                        m3d.control_points.astype(np.float64),
                        SPLINE_K,
                    )
                    u_sample = np.linspace(0.0, 1.0, n_body_points)
                    pts_3d = spl(u_sample).astype(np.float32)
                except Exception:
                    continue
            else:
                continue

            cam_map = frame_ms[fish_id]
            fish_has_residuals = False
            for camera_id, midline_2d in cam_map.items():
                if camera_id not in projection_models:
                    continue
                model = projection_models[camera_id]
                pts_tensor = torch.from_numpy(pts_3d)
                proj_px, valid = model.project(pts_tensor)
                proj_np = proj_px.detach().cpu().numpy()
                valid_np = valid.cpu().numpy()

                obs_2d = midline_2d.points
                if obs_2d.shape[0] != n_body_points:
                    continue

                for j in range(n_body_points):
                    if not valid_np[j]:
                        continue
                    if np.any(np.isnan(proj_np[j])) or np.any(np.isnan(obs_2d[j])):
                        continue
                    if np.any(np.isnan(pts_3d[j])):
                        continue
                    err = float(
                        np.linalg.norm(proj_np[j] - obs_2d[j].astype(np.float64))
                    )
                    all_residuals.append(err)
                    fish_has_residuals = True

            if fish_has_residuals:
                n_fish_frames += 1

    arr = np.array(all_residuals) if all_residuals else np.array([0.0])
    pcts = np.percentile(arr, [50, 90, 95, 99])
    return {
        "mean": float(np.mean(arr)),
        "p50": float(pcts[0]),
        "p90": float(pcts[1]),
        "p95": float(pcts[2]),
        "p99": float(pcts[3]),
        "n_fish_frames": n_fish_frames,
        "n_residuals": len(all_residuals),
    }


def evaluate_smoothing(
    run_dir: Path,
    frame_results: list[tuple[int, dict]] | None,
    midline_sets_by_frame: dict[int, dict] | None,
    projection_models: dict[str, Any] | None,
    n_body_points: int = 6,
) -> SmoothingMetrics | None:
    """Evaluate z-smoothing quality from a pipeline run directory.

    Returns ``None`` if no ``*_smoothed.h5`` is detected. Computes jitter
    metrics from the smoothed H5, and reprojection error impact if the
    required dependencies (frame_results, midline_sets_by_frame,
    projection_models) are all provided.

    Args:
        run_dir: Pipeline run directory.
        frame_results: Pre-smoothing ``(frame_idx, {fish_id: Midline3D})`` from
            the runner's reconstruction eval. May be ``None`` to skip reproj.
        midline_sets_by_frame: 2D observations by frame from the runner.
        projection_models: Refractive projection models by camera ID.
        n_body_points: Expected number of body points per midline.

    Returns:
        SmoothingMetrics or ``None``.
    """
    detected = _detect_smoothed_h5(run_dir)
    if detected is None:
        return None

    _unsmoothed_path, smoothed_path = detected
    source_file = smoothed_path.name

    # Jitter metrics (always computed — only needs the smoothed H5)
    (
        fish_processed,
        frames_processed,
        mean_jitter_before,
        mean_jitter_after,
        jitter_reduction,
        per_fish_jitter,
    ) = _compute_jitter_metrics(smoothed_path)

    # Reprojection error impact (needs frame_results + projection models)
    reproj_before: dict[str, float] = {}
    reproj_after: dict[str, float] = {}
    reproj_n_fish_frames = 0
    reproj_n_residuals = 0

    if (
        frame_results is not None
        and midline_sets_by_frame is not None
        and projection_models
    ):
        dz_lookup = _build_dz_lookup(smoothed_path)

        reproj_before = _compute_reproj_stats(
            frame_results, midline_sets_by_frame, projection_models, n_body_points
        )
        frame_results_post = _shift_frame_results(frame_results, dz_lookup)
        reproj_after = _compute_reproj_stats(
            frame_results_post, midline_sets_by_frame, projection_models, n_body_points
        )
        reproj_n_fish_frames = reproj_before.get("n_fish_frames", 0)
        reproj_n_residuals = reproj_before.get("n_residuals", 0)
        # Remove count keys from stat dicts
        reproj_before = {
            k: v
            for k, v in reproj_before.items()
            if k not in ("n_fish_frames", "n_residuals")
        }
        reproj_after = {
            k: v
            for k, v in reproj_after.items()
            if k not in ("n_fish_frames", "n_residuals")
        }

    return SmoothingMetrics(
        source_file=source_file,
        sigma_frames=3,  # default; not stored in H5
        fish_processed=fish_processed,
        frames_processed=frames_processed,
        mean_jitter_before=mean_jitter_before,
        mean_jitter_after=mean_jitter_after,
        jitter_reduction=jitter_reduction,
        per_fish_jitter=per_fish_jitter,
        reproj_before=reproj_before,
        reproj_after=reproj_after,
        reproj_n_fish_frames=reproj_n_fish_frames,
        reproj_n_residuals=reproj_n_residuals,
    )


__all__ = [
    "SmoothingMetrics",
    "evaluate_smoothing",
]
