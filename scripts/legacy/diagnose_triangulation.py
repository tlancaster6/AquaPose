"""Systematic triangulation diagnostic: isolate root cause of high residuals.

Runs five hypothesis tests against multi-view triangulation to identify whether
the bottleneck is orientation alignment, epipolar refinement, inlier thresholds,
camera calibration, or raw correspondence quality.

Usage:
    python scripts/diagnose_triangulation.py
    python scripts/diagnose_triangulation.py --stop-frame 30
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Default paths (same as diagnose_pipeline.py)
# ---------------------------------------------------------------------------
DEFAULT_VIDEO_DIR = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/videos/core_videos")
DEFAULT_CALIBRATION = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/calibration.json")
DEFAULT_UNET_WEIGHTS = Path(
    "C:/Users/tucke/Desktop/Aqua/AquaPose/unet/run2/best_model.pth"
)
DEFAULT_YOLO_WEIGHTS = Path(
    "C:/Users/tucke/Desktop/Aqua/AquaPose/yolo/train_v2/weights/best.pt"
)
DEFAULT_OUTPUT_DIR = Path("output/triangulation_diagnostic")
DEFAULT_STOP_FRAME = 30

_SKIP_CAMERA_ID = "e3v8250"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Triangulation diagnostic: isolate root cause of high residuals.",
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--unet-weights", type=Path, default=DEFAULT_UNET_WEIGHTS)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stop-frame", type=int, default=DEFAULT_STOP_FRAME)
    parser.add_argument("--tta", action="store_true", default=False)
    return parser.parse_args()


def check_paths(args: argparse.Namespace) -> bool:
    """Verify all required input paths exist."""
    ok = True
    for label, path in [
        ("Video directory", args.video_dir),
        ("Calibration JSON", args.calibration),
        ("U-Net weights", args.unet_weights),
        ("YOLO weights", args.yolo_weights),
    ]:
        if path.exists():
            print(f"  [OK]   {label}: {path}")
        else:
            print(f"  [MISS] {label}: {path}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Stages 1-4 (reused from diagnose_pipeline.py)
# ---------------------------------------------------------------------------


def run_stages_1_to_4(
    args: argparse.Namespace,
    models: dict,
    undist_maps: dict,
    video_paths: dict[str, Path],
) -> list:
    """Run detection, segmentation, tracking, midline extraction. Returns midline_sets."""
    from aquapose.io.video import VideoSet
    from aquapose.pipeline.stages import (
        run_detection,
        run_midline_extraction,
        run_segmentation,
    )
    from aquapose.reconstruction.midline import MidlineExtractor
    from aquapose.segmentation.model import UNetSegmentor
    from aquapose.tracking.tracker import FishTracker

    segmentor = UNetSegmentor(weights_path=args.unet_weights)
    tracker = FishTracker(expected_count=9)
    extractor = MidlineExtractor()

    print("\n  Stage 1: Detection...")
    with VideoSet(video_paths, undistortion=undist_maps) as vs:
        detections_per_frame = run_detection(
            video_set=vs,
            stop_frame=args.stop_frame,
            detector_kind="yolo",
            model_path=args.yolo_weights,
        )

    print("  Stage 2: Segmentation...")
    with VideoSet(video_paths, undistortion=undist_maps) as vs:
        masks_per_frame = run_segmentation(
            detections_per_frame=detections_per_frame,
            video_set=vs,
            segmentor=segmentor,
            stop_frame=args.stop_frame,
        )

    print("  Stage 3: Tracking...")
    tracks_per_frame: list[list] = []
    for frame_idx, frame_dets in enumerate(detections_per_frame):
        confirmed = tracker.update(frame_dets, models, frame_index=frame_idx)
        tracks_per_frame.append(confirmed)

    print("  Stage 4: Midline Extraction...")
    midline_sets = run_midline_extraction(
        tracks_per_frame=tracks_per_frame,
        masks_per_frame=masks_per_frame,
        detections_per_frame=detections_per_frame,
        extractor=extractor,
    )

    # Debug: camera coverage in midline sets
    cam_counts: dict[str, int] = {}
    for ms in midline_sets:
        for _fish_id, cam_ml in ms.items():
            for cid in cam_ml:
                cam_counts[cid] = cam_counts.get(cid, 0) + 1
    print("  Camera coverage in midline sets:")
    for cid in sorted(cam_counts):
        print(f"    {cid}: {cam_counts[cid]} midlines")
    # Also count how many cameras per fish
    cams_per_fish: list[int] = []
    for ms in midline_sets:
        for _fish_id, cam_ml in ms.items():
            cams_per_fish.append(len(cam_ml))
    if cams_per_fish:
        print(
            f"  Cameras per fish: min={min(cams_per_fish)}, median={np.median(cams_per_fish):.0f}, max={max(cams_per_fish)}"
        )

    return midline_sets


# ---------------------------------------------------------------------------
# Baseline statistics
# ---------------------------------------------------------------------------


def _recompute_residual_nan_safe(
    m3d: object,
    cam_midlines: dict,
    models: dict,
) -> tuple[float, dict[str, float]]:
    """Recompute spline residuals filtering NaN observations.

    Works around the bug in triangulate_midlines where NaN obs_pts
    from epipolar refinement poison the residual computation.
    """
    import scipy.interpolate as _sp

    from aquapose.reconstruction.triangulation import (
        N_SAMPLE_POINTS,
        SPLINE_K,
        SPLINE_KNOTS,
    )

    spline_obj = _sp.BSpline(
        SPLINE_KNOTS, m3d.control_points.astype(np.float64), SPLINE_K
    )
    u_sample = np.linspace(0.0, 1.0, N_SAMPLE_POINTS)
    spline_pts = torch.from_numpy(spline_obj(u_sample).astype(np.float32))

    all_res: list[float] = []
    cam_res: dict[str, float] = {}
    for cid in cam_midlines:
        if cid not in models:
            continue
        proj_px, valid = models[cid].project(spline_pts)
        proj_np = proj_px.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        obs = cam_midlines[cid].points
        errs: list[float] = []
        for j in range(N_SAMPLE_POINTS):
            if (
                valid_np[j]
                and not np.any(np.isnan(proj_np[j]))
                and not np.any(np.isnan(obs[j]))
            ):
                e = float(np.linalg.norm(proj_np[j] - obs[j]))
                errs.append(e)
                all_res.append(e)
        if errs:
            cam_res[cid] = float(np.mean(errs))

    mean_r = float(np.mean(all_res)) if all_res else 0.0
    return mean_r, cam_res


def compute_baseline(
    midline_sets: list,
    models: dict,
) -> dict:
    """Run triangulate_midlines and collect baseline statistics."""
    from aquapose.reconstruction.triangulation import (
        _align_midline_orientations,
        _refine_correspondences_epipolar,
        triangulate_midlines,
    )

    all_residuals: list[float] = []
    all_residuals_corrected: list[float] = []
    low_conf_count = 0
    total_count = 0
    total_input = 0
    cam_residuals_agg: dict[str, list[float]] = {}
    cam_residuals_corrected_agg: dict[str, list[float]] = {}
    arc_lengths: list[float] = []
    n_cameras_list: list[int] = []

    t0 = time.perf_counter()
    all_midlines_3d: list[dict] = []
    for frame_idx, midline_set in enumerate(midline_sets):
        total_input += len(midline_set)
        frame_results = triangulate_midlines(midline_set, models, frame_index=frame_idx)
        all_midlines_3d.append(frame_results)

        for fish_id, m3d in frame_results.items():
            total_count += 1
            all_residuals.append(m3d.mean_residual)
            arc_lengths.append(m3d.arc_length)
            n_cameras_list.append(m3d.n_cameras)
            if m3d.is_low_confidence:
                low_conf_count += 1
            if m3d.per_camera_residuals:
                for cid, res in m3d.per_camera_residuals.items():
                    cam_residuals_agg.setdefault(cid, []).append(res)

            # Recompute with NaN filtering
            cam_ml = midline_set.get(fish_id, {})
            # Re-run alignment+refinement to get the post-processed midlines
            aligned = _align_midline_orientations(cam_ml, models)
            refined = _refine_correspondences_epipolar(aligned, models)
            corr_res, corr_cam = _recompute_residual_nan_safe(m3d, refined, models)
            all_residuals_corrected.append(corr_res)
            for cid, res in corr_cam.items():
                cam_residuals_corrected_agg.setdefault(cid, []).append(res)

    elapsed = time.perf_counter() - t0

    # Filter NaN from raw residuals for reporting
    valid_residuals = [r for r in all_residuals if not np.isnan(r)]
    mean_res = float(np.mean(valid_residuals)) if valid_residuals else 0.0
    mean_res_corrected = (
        float(np.mean(all_residuals_corrected)) if all_residuals_corrected else 0.0
    )
    low_conf_rate = low_conf_count / total_count if total_count else 0.0
    per_cam_mean = {
        cid: float(np.mean(vals)) for cid, vals in sorted(cam_residuals_agg.items())
    }
    per_cam_corrected = {
        cid: float(np.mean(vals))
        for cid, vals in sorted(cam_residuals_corrected_agg.items())
    }

    return {
        "mean_residual": mean_res,
        "mean_residual_corrected": mean_res_corrected,
        "median_residual": float(np.median(valid_residuals))
        if valid_residuals
        else 0.0,
        "nan_residual_count": len(all_residuals) - len(valid_residuals),
        "low_confidence_rate": low_conf_rate,
        "total_fish": total_count,
        "total_input_fish": total_input,
        "low_confidence_count": low_conf_count,
        "per_camera_mean_residual": per_cam_mean,
        "per_camera_corrected_residual": per_cam_corrected,
        "mean_arc_length": float(np.mean(arc_lengths)) if arc_lengths else 0.0,
        "median_n_cameras": int(np.median(n_cameras_list)) if n_cameras_list else 0,
        "elapsed_s": elapsed,
        "all_midlines_3d": all_midlines_3d,
    }


# ---------------------------------------------------------------------------
# H1: Orientation alignment (brute-force vs greedy)
# ---------------------------------------------------------------------------


def test_h1_orientation(
    midline_sets: list,
    models: dict,
) -> dict:
    """Compare greedy orientation alignment against brute-force optimal."""
    from aquapose.reconstruction.triangulation import (
        N_SAMPLE_POINTS,
        _align_midline_orientations,
        _flip_midline,
    )

    sample_indices = [0, N_SAMPLE_POINTS // 2, N_SAMPLE_POINTS - 1]
    n_tested = 0
    n_greedy_optimal = 0
    greedy_residuals: list[float] = []
    brute_residuals: list[float] = []

    for midline_set in midline_sets:
        for _fish_id, cam_midlines in midline_set.items():
            valid_cams = sorted(cid for cid in cam_midlines if cid in models)
            if len(valid_cams) < 3:
                continue

            # Greedy alignment
            greedy_aligned = _align_midline_orientations(cam_midlines, models)
            greedy_res = _mean_residual_for_samples(
                greedy_aligned, models, sample_indices
            )

            # Brute-force: try all 2^(N-1) flip combinations
            others = valid_cams[1:]
            n_others = len(others)
            # Cap at 5 others (32 combos) for tractability
            if n_others > 5:
                others = others[:5]
                n_others = 5

            best_bf_res = float("inf")
            for bits in range(1 << n_others):
                trial = dict(cam_midlines)
                for j, cid in enumerate(others):
                    if bits & (1 << j):
                        trial[cid] = _flip_midline(cam_midlines[cid])
                res = _mean_residual_for_samples(trial, models, sample_indices)
                if res < best_bf_res:
                    best_bf_res = res

            n_tested += 1
            greedy_residuals.append(greedy_res)
            brute_residuals.append(best_bf_res)
            # Both inf = both failed to triangulate, skip comparison
            if np.isinf(greedy_res) and np.isinf(best_bf_res):
                n_greedy_optimal += 1  # neither works, not an alignment issue
            elif greedy_res <= best_bf_res * 1.05:  # 5% tolerance
                n_greedy_optimal += 1

    match_rate = n_greedy_optimal / n_tested if n_tested else 1.0
    verdict = "PASS" if match_rate >= 0.90 else "FAIL"

    # Filter inf for mean computation
    finite_greedy = [r for r in greedy_residuals if np.isfinite(r)]
    finite_brute = [r for r in brute_residuals if np.isfinite(r)]
    n_both_inf = sum(
        1
        for g, b in zip(greedy_residuals, brute_residuals, strict=True)
        if np.isinf(g) and np.isinf(b)
    )
    n_greedy_only_inf = sum(
        1
        for g, b in zip(greedy_residuals, brute_residuals, strict=True)
        if np.isinf(g) and not np.isinf(b)
    )

    return {
        "verdict": verdict,
        "n_tested": n_tested,
        "n_both_inf": n_both_inf,
        "n_greedy_only_inf": n_greedy_only_inf,
        "greedy_optimal_rate": match_rate,
        "mean_greedy_residual": float(np.mean(finite_greedy))
        if finite_greedy
        else float("inf"),
        "mean_brute_residual": float(np.mean(finite_brute))
        if finite_brute
        else float("inf"),
        "greedy_residuals": greedy_residuals,
        "brute_residuals": brute_residuals,
    }


def _mean_residual_for_samples(
    cam_midlines: dict,
    models: dict,
    sample_indices: list[int],
) -> float:
    """Triangulate sample body points and return mean residual."""
    from aquapose.reconstruction.triangulation import _triangulate_body_point

    residuals: list[float] = []
    for i in sample_indices:
        pixels: dict[str, torch.Tensor] = {}
        for cam_id, midline in cam_midlines.items():
            if cam_id not in models:
                continue
            pt = midline.points[i]
            if np.any(np.isnan(pt)):
                continue
            pixels[cam_id] = torch.from_numpy(pt).float()
        result = _triangulate_body_point(pixels, models, inlier_threshold=50.0)
        if result is not None:
            residuals.append(result[2])
    return float(np.mean(residuals)) if residuals else float("inf")


# ---------------------------------------------------------------------------
# H2: Epipolar refinement ablation
# ---------------------------------------------------------------------------


def test_h2_epipolar_ablation(
    midline_sets: list,
    models: dict,
) -> dict:
    """Compare triangulation with/without epipolar refinement."""
    from aquapose.reconstruction.triangulation import (
        _align_midline_orientations,
        _refine_correspondences_epipolar,
        triangulate_midlines,
    )

    # Variant A: alignment only (no epipolar)
    residuals_a: list[float] = []
    nan_count_a = 0
    spline_success_a = 0
    total_fish_a = 0

    # Variant B: full pipeline (baseline)
    residuals_b: list[float] = []
    nan_count_b = 0
    spline_success_b = 0
    total_fish_b = 0

    # Variant C: epipolar only (no alignment)
    residuals_c: list[float] = []
    nan_count_c = 0
    spline_success_c = 0
    total_fish_c = 0

    for frame_idx, midline_set in enumerate(midline_sets):
        # Variant B: full pipeline
        frame_b = triangulate_midlines(midline_set, models, frame_index=frame_idx)
        for m3d in frame_b.values():
            residuals_b.append(m3d.mean_residual)
            spline_success_b += 1
        total_fish_b += len(midline_set)

        for fish_id, cam_midlines in midline_set.items():
            # Variant A: alignment only
            aligned_a = _align_midline_orientations(cam_midlines, models)
            result_a = _triangulate_single_fish(aligned_a, models, frame_idx, fish_id)
            total_fish_a += 1
            if result_a is not None:
                residuals_a.append(result_a["mean_residual"])
                nan_count_a += result_a["nan_count"]
                spline_success_a += 1

            # Variant C: epipolar only (no alignment)
            refined_c = _refine_correspondences_epipolar(cam_midlines, models)
            result_c = _triangulate_single_fish(refined_c, models, frame_idx, fish_id)
            total_fish_c += 1
            if result_c is not None:
                residuals_c.append(result_c["mean_residual"])
                nan_count_c += result_c["nan_count"]
                spline_success_c += 1

    # Count NaNs for variant B by running epipolar separately
    for midline_set in midline_sets:
        for _fish_id, cam_midlines in midline_set.items():
            aligned = _align_midline_orientations(cam_midlines, models)
            refined = _refine_correspondences_epipolar(aligned, models)
            for _cid, ml in refined.items():
                nan_count_b += int(np.sum(np.isnan(ml.points)))

    mean_a = float(np.mean(residuals_a)) if residuals_a else float("inf")
    mean_b = float(np.mean(residuals_b)) if residuals_b else float("inf")
    mean_c = float(np.mean(residuals_c)) if residuals_c else float("inf")

    # FAIL if alignment-only (A) is better than full pipeline (B)
    verdict = "FAIL" if mean_a < mean_b else "PASS"

    return {
        "verdict": verdict,
        "variant_a_align_only": {
            "mean_residual": mean_a,
            "nan_count": nan_count_a,
            "spline_success": spline_success_a,
            "total_fish": total_fish_a,
        },
        "variant_b_full_pipeline": {
            "mean_residual": mean_b,
            "nan_count": nan_count_b,
            "spline_success": spline_success_b,
            "total_fish": total_fish_b,
        },
        "variant_c_epipolar_only": {
            "mean_residual": mean_c,
            "nan_count": nan_count_c,
            "spline_success": spline_success_c,
            "total_fish": total_fish_c,
        },
    }


def _triangulate_single_fish(
    cam_midlines: dict,
    models: dict,
    frame_index: int,
    fish_id: int,
) -> dict | None:
    """Triangulate a single fish without alignment/refinement. Returns stats or None."""
    from aquapose.reconstruction.triangulation import (
        MIN_BODY_POINTS,
        N_SAMPLE_POINTS,
        SPLINE_K,
        SPLINE_KNOTS,
        _fit_spline,
        _triangulate_body_point,
    )

    valid_indices: list[int] = []
    pts_3d_list: list[np.ndarray] = []
    per_point_residuals: list[float] = []
    per_point_n_cams: list[int] = []
    nan_count = 0

    for i in range(N_SAMPLE_POINTS):
        pixels: dict[str, torch.Tensor] = {}
        for cam_id, midline in cam_midlines.items():
            if cam_id not in models:
                continue
            pt = midline.points[i]
            if np.any(np.isnan(pt)):
                nan_count += 1
                continue
            pixels[cam_id] = torch.from_numpy(pt).float()

        result = _triangulate_body_point(pixels, models, inlier_threshold=50.0)
        if result is None:
            continue

        pt3d, inlier_ids, mean_res = result
        pt3d_np = pt3d.detach().cpu().numpy().astype(np.float64)

        valid_indices.append(i)
        pts_3d_list.append(pt3d_np)
        per_point_residuals.append(mean_res)
        per_point_n_cams.append(len(inlier_ids))

    if len(valid_indices) < MIN_BODY_POINTS:
        return None

    u_param = np.array(
        [i / (N_SAMPLE_POINTS - 1) for i in valid_indices], dtype=np.float64
    )
    pts_3d_arr = np.stack(pts_3d_list, axis=0)
    spline_result = _fit_spline(u_param, pts_3d_arr)
    if spline_result is None:
        return None

    control_points, _arc_length = spline_result

    # Compute spline-based residuals
    import scipy.interpolate

    spline_obj = scipy.interpolate.BSpline(
        SPLINE_KNOTS, control_points.astype(np.float64), SPLINE_K
    )
    u_sample = np.linspace(0.0, 1.0, N_SAMPLE_POINTS)
    spline_pts_3d = torch.from_numpy(spline_obj(u_sample).astype(np.float32))

    all_res: list[float] = []
    cam_ids_active = [cid for cid in cam_midlines if cid in models]
    for cid in cam_ids_active:
        proj_px, valid = models[cid].project(spline_pts_3d)
        proj_np = proj_px.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        obs_pts = cam_midlines[cid].points
        for j in range(N_SAMPLE_POINTS):
            if (
                valid_np[j]
                and not np.any(np.isnan(proj_np[j]))
                and not np.any(np.isnan(obs_pts[j]))
            ):
                err = float(np.linalg.norm(proj_np[j] - obs_pts[j]))
                all_res.append(err)

    mean_residual = float(np.mean(all_res)) if all_res else 0.0

    return {
        "mean_residual": mean_residual,
        "nan_count": nan_count,
        "n_valid_points": len(valid_indices),
        "per_point_n_cams": per_point_n_cams,
    }


# ---------------------------------------------------------------------------
# H3: Inlier threshold sweep
# ---------------------------------------------------------------------------


def test_h3_threshold_sweep(
    midline_sets: list,
    models: dict,
) -> dict:
    """Sweep inlier_threshold and report NaN-filtered residuals vs camera count."""
    from aquapose.reconstruction.triangulation import (
        _align_midline_orientations,
        _refine_correspondences_epipolar,
        triangulate_midlines,
    )

    thresholds = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    results: list[dict] = []

    for thresh in thresholds:
        print(f"    threshold={thresh}px...", end=" ", flush=True)
        corrected_residuals: list[float] = []
        n_cams_list: list[int] = []
        total = 0
        success = 0

        for frame_idx, midline_set in enumerate(midline_sets):
            frame_results = triangulate_midlines(
                midline_set, models, frame_index=frame_idx, inlier_threshold=thresh
            )
            total += len(midline_set)
            for fish_id, m3d in frame_results.items():
                n_cams_list.append(m3d.n_cameras)
                success += 1
                # Recompute NaN-filtered residual
                cam_ml = midline_set.get(fish_id, {})
                aligned = _align_midline_orientations(cam_ml, models, thresh)
                refined = _refine_correspondences_epipolar(
                    aligned, models, snap_threshold=thresh
                )
                corr_res, _ = _recompute_residual_nan_safe(m3d, refined, models)
                corrected_residuals.append(corr_res)

        mean_res = (
            float(np.mean(corrected_residuals)) if corrected_residuals else float("inf")
        )
        median_cams = float(np.median(n_cams_list)) if n_cams_list else 0.0
        print(
            f"mean_res={mean_res:.1f}px, median_cams={median_cams:.1f}, success={success}/{total}"
        )
        results.append(
            {
                "threshold": thresh,
                "mean_residual": mean_res,
                "median_residual": float(np.median(corrected_residuals))
                if corrected_residuals
                else float("inf"),
                "median_n_cameras": median_cams,
                "success_rate": success / total if total else 0.0,
            }
        )

    # Find optimal
    best = min(results, key=lambda r: r["mean_residual"])

    return {
        "verdict": "INFORMATIONAL",
        "sweep_results": results,
        "optimal_threshold": best["threshold"],
        "optimal_mean_residual": best["mean_residual"],
        "note": "snap_threshold is coupled to inlier_threshold in _refine_correspondences_epipolar",
    }


# ---------------------------------------------------------------------------
# H4: Camera-pair residual matrix
# ---------------------------------------------------------------------------


def test_h4_camera_pair_matrix(
    midline_sets: list,
    models: dict,
) -> dict:
    """Build NxN camera-pair residual matrix using direct 2-camera triangulation."""
    from aquapose.calibration.projection import triangulate_rays
    from aquapose.reconstruction.triangulation import N_SAMPLE_POINTS

    cam_ids = sorted(models.keys())
    pair_residuals: dict[tuple[str, str], list[float]] = {}

    for midline_set in midline_sets:
        for _fish_id, cam_midlines in midline_set.items():
            available = [c for c in cam_ids if c in cam_midlines and c in models]
            if len(available) < 2:
                continue

            for ca, cb in itertools.combinations(available, 2):
                ml_a = cam_midlines[ca]
                ml_b = cam_midlines[cb]
                pair_key = (ca, cb)

                for i in range(N_SAMPLE_POINTS):
                    pt_a = ml_a.points[i]
                    pt_b = ml_b.points[i]
                    if np.any(np.isnan(pt_a)) or np.any(np.isnan(pt_b)):
                        continue

                    px_a = torch.from_numpy(pt_a).float().unsqueeze(0)
                    px_b = torch.from_numpy(pt_b).float().unsqueeze(0)

                    o_a, d_a = models[ca].cast_ray(px_a)
                    o_b, d_b = models[cb].cast_ray(px_b)

                    origs = torch.stack([o_a[0], o_b[0]])
                    dirs = torch.stack([d_a[0], d_b[0]])
                    pt3d = triangulate_rays(origs, dirs)

                    # Reproject into both cameras
                    pt3d_batch = pt3d.unsqueeze(0)
                    errs: list[float] = []
                    for cid, px_orig in [(ca, pt_a), (cb, pt_b)]:
                        proj, valid = models[cid].project(pt3d_batch)
                        if valid[0]:
                            err = float(
                                np.linalg.norm(proj[0].detach().cpu().numpy() - px_orig)
                            )
                            errs.append(err)
                    if errs:
                        pair_residuals.setdefault(pair_key, []).append(
                            float(np.mean(errs))
                        )

    # Build matrix
    matrix: dict[str, dict[str, float]] = {}
    for ca in cam_ids:
        matrix[ca] = {}
        for cb in cam_ids:
            if ca == cb:
                matrix[ca][cb] = 0.0
            else:
                key = (min(ca, cb), max(ca, cb))
                vals = pair_residuals.get(key, [])
                matrix[ca][cb] = float(np.mean(vals)) if vals else float("nan")

    # Per-camera marginal
    per_cam_marginal: dict[str, float] = {}
    for ca in cam_ids:
        all_vals: list[float] = []
        for key, vals in pair_residuals.items():
            if ca in key:
                all_vals.extend(vals)
        per_cam_marginal[ca] = float(np.mean(all_vals)) if all_vals else float("nan")

    # Verdict: FAIL if any camera marginal > 2x median
    marginals = [v for v in per_cam_marginal.values() if not np.isnan(v)]
    if marginals:
        median_marginal = float(np.median(marginals))
        outlier_cams = {
            cid: val
            for cid, val in per_cam_marginal.items()
            if not np.isnan(val) and val > 2.0 * median_marginal
        }
        verdict = "FAIL" if outlier_cams else "PASS"
    else:
        median_marginal = 0.0
        outlier_cams = {}
        verdict = "PASS"

    return {
        "verdict": verdict,
        "per_camera_marginal": per_cam_marginal,
        "median_marginal": median_marginal,
        "outlier_cameras": outlier_cams,
        "matrix": matrix,
    }


# ---------------------------------------------------------------------------
# H5: Raw correspondence quality
# ---------------------------------------------------------------------------


def test_h5_raw_correspondence(
    midline_sets: list,
    models: dict,
) -> dict:
    """Measure epipolar distance of raw (unaligned, unrefined) correspondences."""
    from aquapose.reconstruction.triangulation import (
        N_SAMPLE_POINTS,
        _select_reference_camera,
        _trace_epipolar_curve,
    )

    depth_samples = torch.linspace(0.5, 3.0, 50)
    all_epi_dists: list[float] = []
    per_camera_dists: dict[str, list[float]] = {}

    for midline_set in midline_sets:
        for _fish_id, cam_midlines in midline_set.items():
            valid_cams = {cid: ml for cid, ml in cam_midlines.items() if cid in models}
            if len(valid_cams) < 2:
                continue

            ref_id = _select_reference_camera(valid_cams)
            ref_midline = valid_cams[ref_id]

            for tgt_id, tgt_midline in valid_cams.items():
                if tgt_id == ref_id:
                    continue

                tgt_pts = torch.from_numpy(tgt_midline.points).float()

                for i in range(N_SAMPLE_POINTS):
                    ref_px = torch.from_numpy(ref_midline.points[i]).float()
                    epi_curve = _trace_epipolar_curve(
                        ref_px, models[ref_id], models[tgt_id], depth_samples
                    )
                    if epi_curve is None:
                        continue

                    # Distance from target body point i to epipolar curve
                    tgt_pt = tgt_pts[i].unsqueeze(0)  # (1, 2)
                    dists = torch.cdist(tgt_pt.unsqueeze(0), epi_curve.unsqueeze(0))[
                        0
                    ]  # (1, S')
                    min_dist = float(dists.min().item())
                    all_epi_dists.append(min_dist)
                    per_camera_dists.setdefault(tgt_id, []).append(min_dist)

    median_dist = float(np.median(all_epi_dists)) if all_epi_dists else 0.0
    mean_dist = float(np.mean(all_epi_dists)) if all_epi_dists else 0.0
    verdict = "FAIL" if median_dist > 100.0 else "PASS"

    per_cam_median = {
        cid: float(np.median(vals)) for cid, vals in sorted(per_camera_dists.items())
    }

    return {
        "verdict": verdict,
        "median_epipolar_distance": median_dist,
        "mean_epipolar_distance": mean_dist,
        "p90_epipolar_distance": float(np.percentile(all_epi_dists, 90))
        if all_epi_dists
        else 0.0,
        "per_camera_median": per_cam_median,
        "n_measurements": len(all_epi_dists),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_report(
    output_path: Path,
    baseline: dict,
    h1: dict,
    h2: dict,
    h3: dict,
    h4: dict,
    h5: dict,
) -> None:
    """Write Markdown diagnostic report."""
    lines: list[str] = []
    lines.append("# Triangulation Diagnostic Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Baseline
    lines.append("## Baseline\n")
    lines.append(
        f"- Mean residual (NaN-filtered): **{baseline['mean_residual_corrected']:.1f}px**"
    )
    lines.append(f"- Mean residual (raw): {baseline['mean_residual']:.1f}px")
    lines.append(
        f"- NaN residual count: {baseline['nan_residual_count']}/{baseline['total_fish']}"
    )
    lines.append(f"- Median residual: {baseline['median_residual']:.1f}px")
    lines.append(
        f"- Spline success rate: {baseline['total_fish']}/{baseline['total_input_fish']}"
    )
    lines.append(
        f"- Low-confidence rate: **{baseline['low_confidence_rate']:.1%}** ({baseline['low_confidence_count']}/{baseline['total_fish']})"
    )
    lines.append(f"- Mean arc length: {baseline['mean_arc_length']:.4f}m")
    lines.append(f"- Median min cameras: {baseline['median_n_cameras']}")
    lines.append(f"- Triangulation time: {baseline['elapsed_s']:.2f}s\n")
    lines.append("### Per-camera mean residual (NaN-filtered)\n")
    lines.append("| Camera | Corrected (px) | Raw (px) |")
    lines.append("|--------|---------------|---------|")
    all_cams = sorted(
        set(baseline["per_camera_mean_residual"])
        | set(baseline["per_camera_corrected_residual"])
    )
    for cid in all_cams:
        raw = baseline["per_camera_mean_residual"].get(cid, float("nan"))
        corr = baseline["per_camera_corrected_residual"].get(cid, float("nan"))
        lines.append(f"| {cid} | {corr:.1f} | {raw:.1f} |")
    lines.append("")

    # H1
    lines.append(f"## H1: Orientation Alignment — **{h1['verdict']}**\n")
    lines.append(f"- Tested: {h1['n_tested']} fish instances (>=3 cameras)")
    lines.append(f"- Greedy matches optimal: **{h1['greedy_optimal_rate']:.1%}**")
    lines.append(f"- Both-inf (untestable): {h1['n_both_inf']}/{h1['n_tested']}")
    lines.append(
        f"- Greedy-only-inf (greedy fails, brute works): {h1['n_greedy_only_inf']}"
    )
    lines.append(f"- Mean greedy residual (finite): {h1['mean_greedy_residual']:.1f}px")
    lines.append(
        f"- Mean brute-force residual (finite): {h1['mean_brute_residual']:.1f}px"
    )
    if h1["verdict"] == "FAIL":
        lines.append(
            f"- **Root cause candidate**: Greedy alignment suboptimal in {1 - h1['greedy_optimal_rate']:.1%} of cases"
        )
    lines.append("")

    # H2
    lines.append(f"## H2: Epipolar Refinement Ablation — **{h2['verdict']}**\n")
    for label, key in [
        ("A (align only)", "variant_a_align_only"),
        ("B (full pipeline)", "variant_b_full_pipeline"),
        ("C (epipolar only)", "variant_c_epipolar_only"),
    ]:
        v = h2[key]
        lines.append(f"### Variant {label}")
        lines.append(f"- Mean residual: **{v['mean_residual']:.1f}px**")
        lines.append(f"- NaN count: {v['nan_count']}")
        lines.append(f"- Spline success: {v['spline_success']}/{v['total_fish']}")
        lines.append("")
    if h2["verdict"] == "FAIL":
        lines.append(
            "**Root cause candidate**: Epipolar refinement increases residuals (alignment-only is better)\n"
        )

    # H3
    lines.append(f"## H3: Inlier Threshold Sweep — **{h3['verdict']}**\n")
    lines.append(
        f"- Optimal threshold: **{h3['optimal_threshold']}px** (mean residual: {h3['optimal_mean_residual']:.1f}px)"
    )
    lines.append(f"- Note: {h3['note']}\n")
    lines.append(
        "| Threshold (px) | Mean residual (px) | Median residual (px) | Median cameras | Success rate |"
    )
    lines.append(
        "|---------------|-------------------|---------------------|---------------|-------------|"
    )
    for r in h3["sweep_results"]:
        lines.append(
            f"| {r['threshold']} | {r['mean_residual']:.1f} | {r['median_residual']:.1f} "
            f"| {r['median_n_cameras']:.1f} | {r['success_rate']:.1%} |"
        )
    lines.append("")

    # H4
    lines.append(f"## H4: Camera-Pair Residual Matrix — **{h4['verdict']}**\n")
    lines.append(f"- Median camera marginal: {h4['median_marginal']:.1f}px\n")
    lines.append("### Per-camera marginal residual\n")
    lines.append("| Camera | Marginal residual (px) |")
    lines.append("|--------|----------------------|")
    for cid, val in sorted(h4["per_camera_marginal"].items()):
        flag = " **OUTLIER**" if cid in h4.get("outlier_cameras", {}) else ""
        lines.append(f"| {cid} | {val:.1f}{flag} |")
    lines.append("")

    if h4["outlier_cameras"]:
        lines.append(
            f"**Root cause candidate**: Cameras {list(h4['outlier_cameras'].keys())} have >2x median marginal residual\n"
        )

    # Pair matrix
    cam_ids = sorted(h4["matrix"].keys())
    if cam_ids:
        lines.append("### Pairwise residual matrix (px)\n")
        header = "| |" + "|".join(f" {c} " for c in cam_ids) + "|"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(cam_ids) + 1)) + "|")
        for ca in cam_ids:
            row = f"| {ca} |"
            for cb in cam_ids:
                val = h4["matrix"][ca][cb]
                if np.isnan(val) or ca == cb:
                    row += " - |"
                else:
                    row += f" {val:.0f} |"
            lines.append(row)
        lines.append("")

    # H5
    lines.append(f"## H5: Raw Correspondence Quality — **{h5['verdict']}**\n")
    lines.append(
        f"- Median epipolar distance: **{h5['median_epipolar_distance']:.1f}px**"
    )
    lines.append(f"- Mean epipolar distance: {h5['mean_epipolar_distance']:.1f}px")
    lines.append(f"- P90 epipolar distance: {h5['p90_epipolar_distance']:.1f}px")
    lines.append(f"- Measurements: {h5['n_measurements']}\n")
    lines.append("### Per-camera median epipolar distance\n")
    lines.append("| Camera | Median dist (px) |")
    lines.append("|--------|-----------------|")
    for cid, val in sorted(h5["per_camera_median"].items()):
        lines.append(f"| {cid} | {val:.1f} |")
    lines.append("")

    if h5["verdict"] == "FAIL":
        lines.append(
            "**Root cause candidate**: Raw index-based correspondences have poor epipolar consistency\n"
        )

    # Summary
    lines.append("## Root Cause Summary\n")
    verdicts = {
        "H1": h1["verdict"],
        "H2": h2["verdict"],
        "H3": h3["verdict"],
        "H4": h4["verdict"],
        "H5": h5["verdict"],
    }
    failures = [k for k, v in verdicts.items() if v == "FAIL"]
    if failures:
        lines.append(f"**Failed tests**: {', '.join(failures)}\n")
        for f in failures:
            if f == "H1":
                lines.append(
                    f"- H1: Greedy orientation alignment is suboptimal ({h1['greedy_optimal_rate']:.0%} match rate)"
                )
            elif f == "H2":
                lines.append(
                    "- H2: Epipolar refinement hurts — consider disabling or fixing snap logic"
                )
            elif f == "H4":
                lines.append(
                    f"- H4: Camera calibration outliers: {list(h4.get('outlier_cameras', {}).keys())}"
                )
            elif f == "H5":
                lines.append(
                    f"- H5: Raw correspondences inconsistent (median epipolar dist: {h5['median_epipolar_distance']:.0f}px)"
                )
    else:
        lines.append("All tests passed. High residuals may stem from:")
        lines.append("- Insufficient cameras per body point")
        lines.append("- Mask/midline extraction quality")
        lines.append("- Refractive model accuracy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run triangulation diagnostics."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("=== Triangulation Diagnostic ===\n")
    print("Checking input paths...")
    if not check_paths(args):
        print("\nAborting: one or more required paths are missing.")
        return 1

    # Load calibration
    from aquapose.calibration.loader import (
        compute_undistortion_maps,
        load_calibration_data,
    )
    from aquapose.calibration.projection import RefractiveProjectionModel

    video_paths: dict[str, Path] = {}
    for suffix in ("*.avi", "*.mp4"):
        for p in args.video_dir.glob(suffix):
            camera_id = p.stem.split("-")[0]
            if camera_id == _SKIP_CAMERA_ID:
                continue
            video_paths[camera_id] = p

    if not video_paths:
        print(f"No .avi/.mp4 files found in {args.video_dir}")
        return 1

    print(f"Found {len(video_paths)} cameras: {sorted(video_paths)}")

    calib = load_calibration_data(args.calibration)
    undist_maps = {}
    models: dict[str, RefractiveProjectionModel] = {}
    for cam_id in video_paths:
        if cam_id not in calib.cameras:
            logger.warning("Camera %r not in calibration; skipping", cam_id)
            continue
        cam_data = calib.cameras[cam_id]
        maps = compute_undistortion_maps(cam_data)
        undist_maps[cam_id] = maps
        models[cam_id] = RefractiveProjectionModel(
            K=maps.K_new,
            R=cam_data.R,
            t=cam_data.t,
            water_z=calib.water_z,
            normal=calib.interface_normal,
            n_air=calib.n_air,
            n_water=calib.n_water,
        )

    if not models:
        print("No cameras matched between video_dir and calibration.")
        return 1

    # Run stages 1-4
    print("\nRunning upstream stages...")
    midline_sets = run_stages_1_to_4(args, models, undist_maps, video_paths)

    n_fish_total = sum(len(ms) for ms in midline_sets)
    print(
        f"\nMidline sets: {len(midline_sets)} frames, {n_fish_total} fish-frame instances"
    )

    if n_fish_total == 0:
        print("No midlines extracted. Cannot run diagnostics.")
        return 1

    # --- Run tests ---
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Baseline ---")
    baseline = compute_baseline(midline_sets, models)
    print(f"  Mean residual (raw, may have NaN): {baseline['mean_residual']:.1f}px")
    print(
        f"  Mean residual (NaN-filtered): {baseline['mean_residual_corrected']:.1f}px"
    )
    print(f"  NaN residual count: {baseline['nan_residual_count']}")
    print(f"  Success rate: {baseline['total_fish']}/{baseline['total_input_fish']}")
    print(f"  Low-confidence rate: {baseline['low_confidence_rate']:.1%}")

    print("\n--- H1: Orientation Alignment ---")
    h1 = test_h1_orientation(midline_sets, models)
    print(
        f"  Verdict: {h1['verdict']} (greedy optimal: {h1['greedy_optimal_rate']:.1%})"
    )
    print(f"  Both-inf (untestable): {h1['n_both_inf']}/{h1['n_tested']}")
    print(f"  Greedy-only-inf: {h1['n_greedy_only_inf']}")
    print(f"  Mean greedy residual (finite): {h1['mean_greedy_residual']:.1f}px")
    print(f"  Mean brute-force residual (finite): {h1['mean_brute_residual']:.1f}px")

    print("\n--- H2: Epipolar Refinement Ablation ---")
    h2 = test_h2_epipolar_ablation(midline_sets, models)
    print(f"  Verdict: {h2['verdict']}")
    print(f"  A (align only): {h2['variant_a_align_only']['mean_residual']:.1f}px")
    print(f"  B (full):       {h2['variant_b_full_pipeline']['mean_residual']:.1f}px")
    print(f"  C (epi only):   {h2['variant_c_epipolar_only']['mean_residual']:.1f}px")

    print("\n--- H3: Inlier Threshold Sweep ---")
    h3 = test_h3_threshold_sweep(midline_sets, models)
    print(
        f"  Optimal: {h3['optimal_threshold']}px ({h3['optimal_mean_residual']:.1f}px)"
    )

    print("\n--- H4: Camera-Pair Residual Matrix ---")
    h4 = test_h4_camera_pair_matrix(midline_sets, models)
    print(
        f"  Verdict: {h4['verdict']} (median marginal: {h4['median_marginal']:.1f}px)"
    )

    print("\n--- H5: Raw Correspondence Quality ---")
    h5 = test_h5_raw_correspondence(midline_sets, models)
    print(
        f"  Verdict: {h5['verdict']} (median epi dist: {h5['median_epipolar_distance']:.1f}px)"
    )

    # Write report
    write_report(output_dir / "report.md", baseline, h1, h2, h3, h4, h5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
