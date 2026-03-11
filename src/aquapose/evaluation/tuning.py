"""TuningOrchestrator -- grid sweep engine for association and reconstruction parameter tuning."""

from __future__ import annotations

import copy
import dataclasses
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from aquapose.core.context import PipelineContext
from aquapose.evaluation.metrics import select_frames
from aquapose.evaluation.runner import load_run_context
from aquapose.evaluation.stages.association import (
    DEFAULT_GRID as ASSOCIATION_DEFAULT_GRID,
)
from aquapose.evaluation.stages.association import (
    AssociationMetrics,
    evaluate_association,
)
from aquapose.evaluation.stages.reconstruction import (
    DEFAULT_GRID as RECONSTRUCTION_DEFAULT_GRID,
)
from aquapose.evaluation.stages.reconstruction import (
    ReconstructionMetrics,
    evaluate_reconstruction,
)

__all__ = [
    "TuningOrchestrator",
    "TuningResult",
    "format_comparison_table",
    "format_config_diff",
    "format_yield_matrix",
]


@dataclass(frozen=True)
class TuningResult:
    """Result of a parameter tuning sweep.

    Attributes:
        stage: Stage name -- ``"association"`` or ``"reconstruction"``.
        winner_params: The winning parameter combination.
        baseline_metrics: Key metrics from the baseline run.
        winner_metrics: Key metrics from the winner.
        all_combos: All sweep results for reporting.
        joint_grid_results: For 2D yield matrix (association only). None for
            reconstruction sweeps.
    """

    stage: str
    winner_params: dict[str, float]
    baseline_metrics: dict[str, float]
    winner_metrics: dict[str, float]
    all_combos: list[dict[str, Any]]
    joint_grid_results: list[dict[str, Any]] | None


def _compute_association_score(
    assoc_metrics: AssociationMetrics,
    recon_metrics: ReconstructionMetrics,
) -> tuple[float, float]:
    """Score an association sweep combo.

    Primary: maximize fish yield (negate for min-sort).
    Secondary: minimize mean reprojection error.

    Args:
        assoc_metrics: Association metrics for this combo.
        recon_metrics: Reconstruction metrics for this combo.

    Returns:
        Tuple of (neg yield ratio, mean reprojection error) for sorting.
    """
    return (-assoc_metrics.fish_yield_ratio, recon_metrics.mean_reprojection_error)


def _compute_reconstruction_score(
    recon_metrics: ReconstructionMetrics,
) -> tuple[float, float]:
    """Score a reconstruction sweep combo.

    Primary: minimize mean reprojection error.
    Secondary: maximize reconstruction coverage (negate for min-sort).

    Args:
        recon_metrics: Reconstruction metrics for this combo.

    Returns:
        Tuple of (mean reproj error, neg coverage ratio) for sorting.
    """
    coverage = recon_metrics.fish_reconstructed / max(recon_metrics.fish_available, 1)
    return (recon_metrics.mean_reprojection_error, -coverage)


def _build_midline_sets(
    ctx: PipelineContext,
    tracklet_groups: list[Any],
    sampled_indices: list[int],
) -> list[dict[int, dict[str, Any]]]:
    """Build per-frame MidlineSets from context data and tracklet groups.

    Adapted from EvalRunner._build_midline_sets but accepts explicit
    tracklet_groups so the caller can pass new groups from a sweep combo
    while reusing the context's detection data.

    Reads keypoints from context.detections (v3.7+). Also supports legacy
    contexts with annotated_detections for backward compatibility with
    old diagnostic runs.

    Args:
        ctx: PipelineContext with detections (v3.7) or annotated_detections
            (legacy) from the midline cache.
        tracklet_groups: TrackletGroup list (may be from original cache or
            from an association sweep combo).
        sampled_indices: Frame indices to include.

    Returns:
        List of MidlineSet dicts, one per sampled frame.
    """
    import math

    import numpy as np

    from aquapose.core.reconstruction.stage import (
        _KEYPOINT_T_VALUES,
        _keypoints_to_midline,
    )
    from aquapose.core.types.midline import Midline2D

    _CENTROID_MATCH_TOLERANCE_PX = 5.0

    # v3.7: read from detections; fall back to annotated_detections for legacy runs
    detections = ctx.detections or []
    annotated_detections: list = getattr(ctx, "annotated_detections", None) or []
    use_detections = bool(detections)

    data_source = detections if use_detections else annotated_detections
    frame_count = ctx.frame_count or len(data_source)
    effective_indices = sampled_indices if sampled_indices else list(range(frame_count))
    effective_set = set(effective_indices)

    collected: dict[int, dict[int, dict[str, Any]]] = {}

    for group in tracklet_groups:
        if not group.tracklets:
            continue
        fish_id = group.fish_id
        for tracklet in group.tracklets:
            cam_id: str = tracklet.camera_id
            for tidx, (frame_idx, status) in enumerate(
                zip(tracklet.frames, tracklet.frame_status, strict=False)
            ):
                if status != "detected":
                    continue
                if frame_idx not in effective_set:
                    continue
                if frame_idx >= len(data_source):
                    continue

                centroid: tuple[float, float] = tracklet.centroids[tidx]
                cx, cy = centroid

                if use_detections:
                    # v3.7: read keypoints from Detection objects
                    frame_dets = data_source[frame_idx]
                    if not isinstance(frame_dets, dict):
                        continue
                    cam_list = frame_dets.get(cam_id)
                    if not cam_list:
                        continue

                    best = None
                    best_dist = _CENTROID_MATCH_TOLERANCE_PX
                    for det in cam_list:
                        x, y, w, h = det.bbox
                        det_cx = x + w / 2.0
                        det_cy = y + h / 2.0
                        dist = math.sqrt((det_cx - cx) ** 2 + (det_cy - cy) ** 2)
                        if dist <= best_dist:
                            best_dist = dist
                            best = det

                    if best is None or best.keypoints is None:
                        continue

                    kpts_conf = (
                        best.keypoint_conf
                        if best.keypoint_conf is not None
                        else np.ones(len(best.keypoints), dtype=np.float32)
                    )
                    points, point_conf = _keypoints_to_midline(
                        best.keypoints,
                        _KEYPOINT_T_VALUES,
                        kpts_conf,
                        n_points=15,
                    )
                    midline = Midline2D(
                        points=points,
                        half_widths=np.zeros(len(points), dtype=np.float32),
                        fish_id=fish_id,
                        camera_id=cam_id,
                        frame_index=frame_idx,
                        point_confidence=point_conf,
                    )
                else:
                    # Legacy: read midline from AnnotatedDetection objects
                    frame_annot = data_source[frame_idx]
                    if not isinstance(frame_annot, dict):
                        continue
                    cam_list = frame_annot.get(cam_id)
                    if not cam_list:
                        continue

                    best = None
                    best_dist = _CENTROID_MATCH_TOLERANCE_PX
                    for ann_det in cam_list:
                        x, y, w, h = ann_det.detection.bbox
                        det_cx = x + w / 2.0
                        det_cy = y + h / 2.0
                        dist = math.sqrt((det_cx - cx) ** 2 + (det_cy - cy) ** 2)
                        if dist <= best_dist:
                            best_dist = dist
                            best = ann_det

                    if best is None:
                        continue
                    midline = best.midline
                    if midline is None or not isinstance(midline, Midline2D):
                        continue

                collected.setdefault(frame_idx, {}).setdefault(fish_id, {})[cam_id] = (
                    midline
                )

    return [collected.get(fi, {}) for fi in effective_indices]


def _compute_centroid_reprojection(
    tracklet_groups: list[Any],
    models: dict[str, Any],
    sampled_indices: list[int],
    n_animals: int,
) -> ReconstructionMetrics:
    """Compute lightweight reprojection error from consensus centroids.

    Projects each TrackletGroup's pre-computed 3D consensus centroids back
    into the cameras that observed them, and compares against the observed 2D
    tracklet centroids. Much faster than full reconstruction since it skips
    body-point triangulation and spline fitting.

    Args:
        tracklet_groups: TrackletGroup list from AssociationStage output.
        models: Dict mapping camera_id to RefractiveProjectionModel.
        sampled_indices: Frame indices to evaluate.
        n_animals: Expected number of fish (for fish_available).

    Returns:
        ReconstructionMetrics populated with centroid-based reprojection
        error. Fields not applicable to centroid reprojection (tier2_stability,
        low_confidence_flag_rate) are set to None/0.
    """
    sampled_set = set(sampled_indices)
    all_residuals: list[float] = []
    per_camera_residuals: dict[str, list[float]] = {}
    per_fish_residuals: dict[int, list[float]] = {}
    fish_frames_reconstructed = 0

    for group in tracklet_groups:
        if group.consensus_centroids is None:
            continue

        # Build per-frame observed 2D centroids keyed by camera
        cam_frame_centroids: dict[str, dict[int, tuple[float, float]]] = {}
        for tracklet in group.tracklets:
            cam_id: str = tracklet.camera_id
            cam_frame_centroids.setdefault(cam_id, {})
            for frame_idx, centroid, status in zip(
                tracklet.frames, tracklet.centroids, tracklet.frame_status, strict=False
            ):
                if status == "detected" and frame_idx in sampled_set:
                    cam_frame_centroids[cam_id][frame_idx] = centroid

        for frame_idx, point_3d in group.consensus_centroids:
            if point_3d is None or frame_idx not in sampled_set:
                continue

            pt_tensor = torch.from_numpy(
                np.asarray(point_3d, dtype=np.float32)
            ).unsqueeze(0)
            has_residual = False

            for cam_id, frame_map in cam_frame_centroids.items():
                if frame_idx not in frame_map or cam_id not in models:
                    continue
                observed = frame_map[frame_idx]
                proj_px, valid = models[cam_id].project(pt_tensor)
                if not valid[0]:
                    continue
                obs_tensor = torch.tensor(observed, dtype=torch.float32)
                err = float(torch.linalg.norm(proj_px[0] - obs_tensor).item())
                all_residuals.append(err)
                per_camera_residuals.setdefault(cam_id, []).append(err)
                per_fish_residuals.setdefault(group.fish_id, []).append(err)
                has_residual = True

            if has_residual:
                fish_frames_reconstructed += 1

    if not all_residuals:
        logging.getLogger(__name__).warning(
            "Centroid reprojection: no residuals computed — all groups lack "
            "consensus_centroids (likely over-merged then fully evicted)"
        )
        mean_err = float("inf")
        max_err = float("inf")
    else:
        mean_err = float(np.mean(all_residuals))
        max_err = float(np.max(all_residuals))

    per_camera_error = {
        cam_id: {"mean_px": float(np.mean(res)), "max_px": float(np.max(res))}
        for cam_id, res in per_camera_residuals.items()
    }
    per_fish_error = {
        fish_id: {"mean_px": float(np.mean(res)), "max_px": float(np.max(res))}
        for fish_id, res in per_fish_residuals.items()
    }

    return ReconstructionMetrics(
        mean_reprojection_error=mean_err,
        max_reprojection_error=max_err,
        fish_reconstructed=fish_frames_reconstructed,
        fish_available=n_animals * len(sampled_indices),
        inlier_ratio=1.0,
        low_confidence_flag_rate=0.0,
        tier2_stability=None,
        per_camera_error=per_camera_error,
        per_fish_error=per_fish_error,
    )


class TuningOrchestrator:
    """Orchestrates parameter grid sweeps for association and reconstruction stages.

    Uses per-stage pickle caches to skip upstream re-execution. Supports
    joint 2D grid sweeps, sequential carry-forward, two-tier validation,
    and formatted output.

    Example::

        orchestrator = TuningOrchestrator(config_path)
        result = orchestrator.sweep_association()
        print(format_comparison_table(result.baseline_metrics, result.winner_metrics))
    """

    def __init__(self, config_path: Path) -> None:
        """Initialize with a run config path.

        Args:
            config_path: Path to the run-generated exhaustive config YAML.
                The config's parent directory is treated as the run directory.

        Raises:
            FileNotFoundError: If config or required caches are missing.
            StaleCacheError: If caches are incompatible with current codebase.
        """
        from aquapose.engine.config import load_config

        self._config_path = Path(config_path)
        self._run_dir = self._config_path.parent
        self._config = load_config(config_path)

        # Lazy-loaded projection models for centroid reprojection
        self._projection_models: dict[str, Any] | None = None

        # Try chunk-based loading first (Phase 54+), fall back to legacy
        # per-stage cache files for older diagnostic runs.
        ctx, _manifest = load_run_context(self._run_dir)

        self._caches: dict[str, PipelineContext] = {}
        if ctx is not None:
            # Build stage-keyed cache dict from merged context fields
            if ctx.detections is not None:
                self._caches["detection"] = ctx
            if ctx.tracks_2d is not None:
                self._caches["tracking"] = ctx
            if ctx.tracklet_groups is not None:
                self._caches["association"] = ctx
            # v3.7: pose data lives in detections (keypoints on Detection objects)
            # Legacy runs may have annotated_detections; check both.
            if (
                ctx.detections is not None
                or getattr(ctx, "annotated_detections", None) is not None
            ):
                self._caches["midline"] = ctx
            if ctx.midlines_3d is not None:
                self._caches["reconstruction"] = ctx
        else:
            # Legacy: load individual stage caches
            from aquapose.core.context import load_stage_cache

            diag_dir = self._run_dir / "diagnostics"
            for stage_key in (
                "detection",
                "tracking",
                "association",
                "midline",
                "reconstruction",
            ):
                cache_path = diag_dir / f"{stage_key}_cache.pkl"
                if cache_path.exists():
                    self._caches[stage_key] = load_stage_cache(cache_path)

    def sweep_association(self) -> TuningResult:
        """Sweep association parameters over a joint 2D grid then carry-forward.

        Evaluates metrics on all frames (no subsampling) since centroid
        reprojection is cheap compared to re-running association.

        Returns:
            TuningResult with winner, baseline, and all combo results.

        Raises:
            FileNotFoundError: If required caches are missing.
        """
        from aquapose.core.association.stage import AssociationStage

        # Load required caches
        tracking_ctx = self._require_cache("tracking")
        midline_ctx = self._require_cache("midline")

        frame_count = tracking_ctx.frame_count or 0
        all_indices = list(range(frame_count))

        grid = ASSOCIATION_DEFAULT_GRID
        n_animals = self._config.n_animals

        # Compute baseline metrics
        baseline_assoc, baseline_recon = self._evaluate_association_combo(
            tracking_ctx, midline_ctx, self._config, all_indices, n_animals
        )
        _compute_association_score(baseline_assoc, baseline_recon)

        # Phase 1: Joint 3D grid over ray_distance_threshold x score_min x keypoint_confidence_floor
        joint_params = [
            "ray_distance_threshold",
            "score_min",
            "keypoint_confidence_floor",
        ]
        joint_values = [grid[p] for p in joint_params]
        joint_combos = list(itertools.product(*joint_values))

        all_results: list[dict[str, Any]] = []
        joint_grid_results: list[dict[str, Any]] = []

        print(f"Joint grid: {len(joint_combos)} combos ({frame_count} frames)")
        for combo_vals in joint_combos:
            params = dict(zip(joint_params, combo_vals, strict=True))
            config = self._patch_association_config(params)
            ctx_copy = copy.copy(tracking_ctx)

            try:
                assoc_stage = AssociationStage(config=config)
                ctx_copy = assoc_stage.run(ctx_copy)
            except Exception as exc:
                print(f"  {params} -> ERROR: {exc}")
                continue

            assoc_m, recon_m = self._evaluate_association_with_ctx(
                ctx_copy, midline_ctx, config, all_indices, n_animals
            )
            score = _compute_association_score(assoc_m, recon_m)
            entry = {
                "params": params,
                "assoc_metrics": assoc_m,
                "recon_metrics": recon_m,
                "score": score,
            }
            all_results.append(entry)
            joint_grid_results.append(entry)
            print(
                f"  ray_dist={params['ray_distance_threshold']:.3f} "
                f"score_min={params['score_min']:.3f} "
                f"kpt_floor={params['keypoint_confidence_floor']:.2f} -> "
                f"yield={assoc_m.fish_yield_ratio:.1%} "
                f"error={recon_m.mean_reprojection_error:.2f}px"
            )

        # Find best from joint grid
        if all_results:
            all_results.sort(key=lambda r: r["score"])
            best_params = dict(all_results[0]["params"])
        else:
            best_params = {
                p: getattr(self._config.association, p) for p in joint_params
            }

        # Phase 2: Sequential carry-forward for remaining params
        carry_forward_params = [
            "eviction_reproj_threshold",
            "leiden_resolution",
            "early_k",
        ]
        for param in carry_forward_params:
            values = grid[param]
            best_score = None
            best_val = getattr(self._config.association, param)

            for val in values:
                candidate_params = {**best_params, param: val}
                config = self._patch_association_config(candidate_params)
                ctx_copy = copy.copy(tracking_ctx)

                try:
                    assoc_stage = AssociationStage(config=config)
                    ctx_copy = assoc_stage.run(ctx_copy)
                except Exception as exc:
                    print(f"  {param}={val} -> ERROR: {exc}")
                    continue

                assoc_m, recon_m = self._evaluate_association_with_ctx(
                    ctx_copy, midline_ctx, config, all_indices, n_animals
                )
                score = _compute_association_score(assoc_m, recon_m)
                entry = {
                    "params": candidate_params,
                    "assoc_metrics": assoc_m,
                    "recon_metrics": recon_m,
                    "score": score,
                }
                all_results.append(entry)
                print(
                    f"  {param}={val} -> "
                    f"yield={assoc_m.fish_yield_ratio:.1%} "
                    f"error={recon_m.mean_reprojection_error:.2f}px"
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_val = val

            # Cast early_k to int
            if param == "early_k":
                best_val = int(best_val)
            best_params[param] = best_val

        # Find overall winner
        all_results.sort(key=lambda r: r["score"])
        winner = all_results[0] if all_results else None

        if winner is not None:
            winner_params = winner["params"]
            w_assoc = winner["assoc_metrics"]
            w_recon = winner["recon_metrics"]
            winner_metrics = {
                "fish_yield_ratio": w_assoc.fish_yield_ratio,
                "mean_reprojection_error": w_recon.mean_reprojection_error,
                "max_reprojection_error": w_recon.max_reprojection_error,
                "singleton_rate": w_assoc.singleton_rate,
                "tier2_stability": w_recon.tier2_stability,
            }
        else:
            winner_params = best_params
            winner_metrics = {
                "fish_yield_ratio": baseline_assoc.fish_yield_ratio,
                "mean_reprojection_error": baseline_recon.mean_reprojection_error,
                "max_reprojection_error": baseline_recon.max_reprojection_error,
                "singleton_rate": baseline_assoc.singleton_rate,
                "tier2_stability": baseline_recon.tier2_stability,
            }

        baseline_metrics_dict = {
            "fish_yield_ratio": baseline_assoc.fish_yield_ratio,
            "mean_reprojection_error": baseline_recon.mean_reprojection_error,
            "max_reprojection_error": baseline_recon.max_reprojection_error,
            "singleton_rate": baseline_assoc.singleton_rate,
            "tier2_stability": baseline_recon.tier2_stability,
        }

        # Normalize winner_params values
        normalized_params: dict[str, float] = {}
        for k, v in winner_params.items():
            if k == "early_k":
                normalized_params[k] = float(int(v))
            else:
                normalized_params[k] = float(v)

        return TuningResult(
            stage="association",
            winner_params=normalized_params,
            baseline_metrics=baseline_metrics_dict,
            winner_metrics=winner_metrics,
            all_combos=[
                {"params": r["params"], "score": r["score"]} for r in all_results
            ],
            joint_grid_results=[
                {
                    "params": r["params"],
                    "fish_yield_ratio": r["assoc_metrics"].fish_yield_ratio,
                    "score": r["score"],
                }
                for r in joint_grid_results
            ],
        )

    def sweep_reconstruction(
        self,
        n_frames: int = 30,
        n_frames_validate: int = 100,
        top_n: int = 3,
    ) -> TuningResult:
        """Sweep reconstruction parameters over a 1D sequential carry-forward grid.

        Args:
            n_frames: Frame count for fast sweep tier.
            n_frames_validate: Frame count for top-N validation tier.
            top_n: Number of top candidates for full validation.

        Returns:
            TuningResult with winner, baseline, and all combo results.

        Raises:
            FileNotFoundError: If required caches are missing.
        """
        midline_ctx = self._require_cache("midline")

        frame_count = midline_ctx.frame_count or 0
        sampled_indices = select_frames(tuple(range(frame_count)), n_frames)
        n_animals = self._config.n_animals

        # Compute baseline
        baseline_recon = self._evaluate_reconstruction_combo(
            midline_ctx, self._config, sampled_indices, n_animals
        )

        grid = RECONSTRUCTION_DEFAULT_GRID
        best_params: dict[str, Any] = {}
        all_results: list[dict[str, Any]] = []

        # 1D sequential carry-forward: outlier_threshold first, then n_points
        sweep_order = ["outlier_threshold", "n_points"]

        for param in sweep_order:
            values = grid[param]
            best_score = None
            best_val: float = (
                getattr(self._config.reconstruction, param)
                if param != "n_points"
                else float(self._config.reconstruction.n_sample_points)
            )

            print(f"Sweeping {param}: {len(values)} values")
            for val in values:
                candidate_params = {**best_params, param: val}
                config = self._patch_reconstruction_config(candidate_params)

                recon_m = self._evaluate_reconstruction_combo(
                    midline_ctx, config, sampled_indices, n_animals
                )
                score = _compute_reconstruction_score(recon_m)
                entry = {
                    "params": dict(candidate_params),
                    "recon_metrics": recon_m,
                    "score": score,
                }
                all_results.append(entry)
                print(
                    f"  {param}={val} -> "
                    f"error={recon_m.mean_reprojection_error:.2f}px "
                    f"fish={recon_m.fish_reconstructed}/{recon_m.fish_available}"
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_val = val

            best_params[param] = best_val

        # Sort and take top-N
        all_results.sort(key=lambda r: r["score"])
        top_candidates = all_results[:top_n]

        # Top-N validation
        validate_indices = select_frames(tuple(range(frame_count)), n_frames_validate)
        best_validated = None

        print(
            f"\nValidating top {len(top_candidates)} candidates at {n_frames_validate} frames"
        )
        for candidate in top_candidates:
            params = candidate["params"]
            config = self._patch_reconstruction_config(params)
            recon_m = self._evaluate_reconstruction_combo(
                midline_ctx, config, validate_indices, n_animals
            )
            score = _compute_reconstruction_score(recon_m)
            candidate["validated_recon"] = recon_m
            candidate["validated_score"] = score
            print(
                f"  {params} -> "
                f"error={recon_m.mean_reprojection_error:.2f}px "
                f"fish={recon_m.fish_reconstructed}/{recon_m.fish_available}"
            )

            if best_validated is None or score < best_validated["validated_score"]:
                best_validated = candidate

        # Build result
        if best_validated is not None:
            winner_params_raw = best_validated["params"]
            v_recon = best_validated["validated_recon"]
            winner_metrics = {
                "mean_reprojection_error": v_recon.mean_reprojection_error,
                "max_reprojection_error": v_recon.max_reprojection_error,
                "fish_reconstructed": float(v_recon.fish_reconstructed),
                "fish_available": float(v_recon.fish_available),
                "inlier_ratio": v_recon.inlier_ratio,
                "tier2_stability": v_recon.tier2_stability,
            }
        else:
            winner_params_raw = best_params
            winner_metrics = {
                "mean_reprojection_error": baseline_recon.mean_reprojection_error,
                "max_reprojection_error": baseline_recon.max_reprojection_error,
                "fish_reconstructed": float(baseline_recon.fish_reconstructed),
                "fish_available": float(baseline_recon.fish_available),
                "inlier_ratio": baseline_recon.inlier_ratio,
                "tier2_stability": baseline_recon.tier2_stability,
            }

        # Normalize params: map n_points -> n_sample_points for output
        normalized_params: dict[str, float] = {}
        for k, v in winner_params_raw.items():
            if k == "n_points":
                normalized_params["n_sample_points"] = float(v)
            else:
                normalized_params[k] = float(v)

        baseline_metrics_dict = {
            "mean_reprojection_error": baseline_recon.mean_reprojection_error,
            "max_reprojection_error": baseline_recon.max_reprojection_error,
            "fish_reconstructed": float(baseline_recon.fish_reconstructed),
            "fish_available": float(baseline_recon.fish_available),
            "inlier_ratio": baseline_recon.inlier_ratio,
            "tier2_stability": baseline_recon.tier2_stability,
        }

        return TuningResult(
            stage="reconstruction",
            winner_params=normalized_params,
            baseline_metrics=baseline_metrics_dict,
            winner_metrics=winner_metrics,
            all_combos=[
                {"params": r["params"], "score": r["score"]} for r in all_results
            ],
            joint_grid_results=None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_cache(self, stage_key: str) -> PipelineContext:
        """Return a loaded cache or raise FileNotFoundError.

        Args:
            stage_key: Cache stage name (e.g. ``"tracking"``).

        Returns:
            PipelineContext from the cache.

        Raises:
            FileNotFoundError: If the cache is not present.
        """
        if stage_key not in self._caches:
            diag_dir = self._run_dir / "diagnostics"
            raise FileNotFoundError(
                f"Required cache '{stage_key}' not found in '{diag_dir}'. "
                f"Re-run the pipeline in diagnostic mode to generate it."
            )
        return self._caches[stage_key]

    def _patch_association_config(self, params: dict[str, Any]) -> Any:
        """Create a PipelineConfig with patched association params.

        Args:
            params: Association parameter overrides.

        Returns:
            PipelineConfig with patched AssociationConfig.
        """
        # Cast early_k to int if present
        clean_params = dict(params)
        if "early_k" in clean_params:
            clean_params["early_k"] = int(clean_params["early_k"])

        patched_assoc = dataclasses.replace(self._config.association, **clean_params)
        return dataclasses.replace(self._config, association=patched_assoc)

    def _patch_reconstruction_config(self, params: dict[str, Any]) -> Any:
        """Create a PipelineConfig with patched reconstruction params.

        Args:
            params: Reconstruction parameter overrides. The key ``"n_points"``
                is mapped to ``"n_sample_points"`` in ReconstructionConfig.

        Returns:
            PipelineConfig with patched ReconstructionConfig.
        """
        clean_params: dict[str, Any] = {}
        for k, v in params.items():
            if k == "n_points":
                clean_params["n_sample_points"] = int(v)
            else:
                clean_params[k] = v

        patched_recon = dataclasses.replace(self._config.reconstruction, **clean_params)
        return dataclasses.replace(self._config, reconstruction=patched_recon)

    def _evaluate_association_combo(
        self,
        tracking_ctx: PipelineContext,
        midline_ctx: PipelineContext,
        config: Any,
        sampled_indices: list[int],
        n_animals: int,
    ) -> tuple[AssociationMetrics, ReconstructionMetrics]:
        """Evaluate a single association parameter combo from scratch.

        Args:
            tracking_ctx: Context with tracking data.
            midline_ctx: Context with annotated_detections.
            config: PipelineConfig with the combo's association params.
            sampled_indices: Frame indices to evaluate.
            n_animals: Expected number of fish.

        Returns:
            Tuple of (AssociationMetrics, ReconstructionMetrics).
        """
        from aquapose.core.association.stage import AssociationStage

        ctx_copy = copy.copy(tracking_ctx)
        assoc_stage = AssociationStage(config=config)
        ctx_copy = assoc_stage.run(ctx_copy)

        return self._evaluate_association_with_ctx(
            ctx_copy, midline_ctx, config, sampled_indices, n_animals
        )

    def _get_projection_models(self) -> dict[str, Any]:
        """Lazily load projection models for centroid reprojection.

        Returns:
            Dict mapping camera_id to RefractiveProjectionModel.
        """
        if self._projection_models is None:
            from aquapose.calibration.loader import (
                compute_undistortion_maps,
                load_calibration_data,
            )
            from aquapose.calibration.projection import RefractiveProjectionModel

            calib = load_calibration_data(str(self._config.calibration_path))
            self._projection_models = {}
            for cam_id, cam_data in calib.cameras.items():
                maps = compute_undistortion_maps(cam_data)
                self._projection_models[cam_id] = RefractiveProjectionModel(
                    K=maps.K_new,
                    R=cam_data.R,
                    t=cam_data.t,
                    water_z=calib.water_z,
                    normal=calib.interface_normal,
                    n_air=calib.n_air,
                    n_water=calib.n_water,
                )
        return self._projection_models

    def _evaluate_association_with_ctx(
        self,
        assoc_ctx: PipelineContext,
        midline_ctx: PipelineContext,
        config: Any,
        sampled_indices: list[int],
        n_animals: int,
    ) -> tuple[AssociationMetrics, ReconstructionMetrics]:
        """Evaluate association metrics and centroid reprojection error.

        Uses lightweight centroid reprojection from consensus_centroids
        instead of full reconstruction. Much faster per combo.

        Args:
            assoc_ctx: Context after AssociationStage.run (has tracklet_groups).
            midline_ctx: Context with annotated_detections.
            config: PipelineConfig (unused, kept for interface compatibility).
            sampled_indices: Frame indices to evaluate.
            n_animals: Expected number of fish.

        Returns:
            Tuple of (AssociationMetrics, ReconstructionMetrics).
        """
        tracklet_groups = assoc_ctx.tracklet_groups or []
        midline_sets = _build_midline_sets(
            midline_ctx, tracklet_groups, sampled_indices
        )
        assoc_m = evaluate_association(midline_sets, n_animals)

        models = self._get_projection_models()
        recon_m = _compute_centroid_reprojection(
            tracklet_groups, models, sampled_indices, n_animals
        )

        return assoc_m, recon_m

    def _evaluate_reconstruction_combo(
        self,
        midline_ctx: PipelineContext,
        config: Any,
        sampled_indices: list[int],
        n_animals: int,
    ) -> ReconstructionMetrics:
        """Evaluate a single reconstruction parameter combo.

        Args:
            midline_ctx: Context with midline data (tracklet_groups + annotated_detections).
            config: PipelineConfig with the combo's reconstruction params.
            sampled_indices: Frame indices to evaluate.
            n_animals: Expected number of fish.

        Returns:
            ReconstructionMetrics for this combo.
        """
        from aquapose.core.reconstruction.stage import ReconstructionStage

        ctx_copy = copy.copy(midline_ctx)
        recon_stage = ReconstructionStage(
            calibration_path=config.calibration_path,
            backend=config.reconstruction.backend,
            min_cameras=config.reconstruction.min_cameras,
            max_interp_gap=config.reconstruction.max_interp_gap,
            n_control_points=config.reconstruction.n_control_points,
            outlier_threshold=config.reconstruction.outlier_threshold,
        )
        ctx_copy = recon_stage.run(ctx_copy)

        midlines_3d = ctx_copy.midlines_3d or []
        frame_results = [
            (i, midlines_3d[i])
            for i in sampled_indices
            if i < len(midlines_3d) and midlines_3d[i]
        ]
        fish_available = n_animals * len(sampled_indices)
        return evaluate_reconstruction(frame_results, fish_available)


# ------------------------------------------------------------------
# Output formatting functions
# ------------------------------------------------------------------


def format_comparison_table(
    baseline_metrics: dict[str, Any],
    winner_metrics: dict[str, Any],
) -> str:
    """Format a before/after comparison table with deltas.

    Args:
        baseline_metrics: Key metrics from the baseline run.
        winner_metrics: Key metrics from the winner.

    Returns:
        Multi-line string with a tabular comparison.
    """
    lines: list[str] = []
    lines.append("Comparison: Baseline vs Winner")
    lines.append("-" * 60)
    lines.append(f"{'Metric':<30} {'Baseline':>10} {'Winner':>10} {'Delta':>8}")
    lines.append("-" * 60)

    for key in baseline_metrics:
        b_val = baseline_metrics.get(key)
        w_val = winner_metrics.get(key)

        if b_val is None and w_val is None:
            lines.append(f"{key:<30} {'N/A':>10} {'N/A':>10} {'':>8}")
            continue

        if b_val is None:
            b_str = "N/A"
            w_str = f"{w_val:.4f}" if isinstance(w_val, float) else str(w_val)
            lines.append(f"{key:<30} {b_str:>10} {w_str:>10} {'':>8}")
            continue

        if w_val is None:
            b_str = f"{b_val:.4f}" if isinstance(b_val, float) else str(b_val)
            w_str = "N/A"
            lines.append(f"{key:<30} {b_str:>10} {w_str:>10} {'':>8}")
            continue

        if isinstance(b_val, float) and isinstance(w_val, float):
            delta = w_val - b_val
            # Format as percentage for ratio fields
            if "ratio" in key or "rate" in key:
                b_str = f"{b_val:.1%}"
                w_str = f"{w_val:.1%}"
                d_str = f"{delta:+.1%}"
            else:
                b_str = f"{b_val:.4f}"
                w_str = f"{w_val:.4f}"
                d_str = f"{delta:+.4f}"
            lines.append(f"{key:<30} {b_str:>10} {w_str:>10} {d_str:>8}")
        else:
            b_str = str(b_val)
            w_str = str(w_val)
            lines.append(f"{key:<30} {b_str:>10} {w_str:>10} {'':>8}")

    lines.append("-" * 60)
    return "\n".join(lines)


def format_yield_matrix(
    joint_results: list[dict[str, Any]],
    param_a: str,
    values_a: list[float],
    param_b: str,
    values_b: list[float],
) -> str:
    """Format a 2D yield percentage matrix for association sweeps.

    Args:
        joint_results: List of dicts with ``"params"`` and ``"fish_yield_ratio"`` keys.
        param_a: Name of the row parameter.
        values_a: Row parameter values.
        param_b: Name of the column parameter.
        values_b: Column parameter values.

    Returns:
        Multi-line string with a 2D grid of yield percentages.
    """
    # Build lookup: (val_a, val_b) -> yield ratio
    lookup: dict[tuple[float, float], float] = {}
    for entry in joint_results:
        params = entry["params"]
        yield_ratio = entry.get("fish_yield_ratio", 0.0)
        a_val = params.get(param_a, 0.0)
        b_val = params.get(param_b, 0.0)
        lookup[(a_val, b_val)] = yield_ratio

    lines: list[str] = []
    lines.append(f"Yield Matrix: {param_a} (rows) x {param_b} (cols)")
    lines.append("")

    # Header row
    col_width = 8
    header = f"{'':>{col_width}}"
    for b_val in values_b:
        header += f"{b_val:>{col_width}.3f}"
    lines.append(header)

    # Data rows
    for a_val in values_a:
        row = f"{a_val:>{col_width}.3f}"
        for b_val in values_b:
            yield_pct = lookup.get((a_val, b_val))
            if yield_pct is not None:
                row += f"{yield_pct:>{col_width}.1%}"
            else:
                row += f"{'---':>{col_width}}"
        lines.append(row)

    return "\n".join(lines)


def format_config_diff(
    stage_name: str,
    winner_params: dict[str, float],
    baseline_config: Any,
) -> str:
    """Format a YAML snippet showing only changed parameters.

    Args:
        stage_name: Stage name (``"association"`` or ``"reconstruction"``).
        winner_params: Winning parameter values.
        baseline_config: Baseline stage config dataclass (e.g. AssociationConfig).

    Returns:
        YAML snippet string with only changed parameters, or a message
        indicating no changes.
    """
    changes: dict[str, Any] = {}
    for param, value in winner_params.items():
        baseline_val = getattr(baseline_config, param, None)
        if baseline_val is None:
            changes[param] = _clean_yaml_value(value)
            continue

        # Compare with tolerance for floats
        if isinstance(value, float) and isinstance(baseline_val, (int, float)):
            if abs(value - float(baseline_val)) > 1e-9:
                changes[param] = _clean_yaml_value(value)
        elif value != baseline_val:
            changes[param] = _clean_yaml_value(value)

    if not changes:
        return f"# No parameter changes for {stage_name}"

    diff = {stage_name: changes}
    yaml_str = yaml.dump(diff, default_flow_style=False, sort_keys=False)
    return f"# Recommended config changes:\n{yaml_str}"


def _clean_yaml_value(value: float) -> int | float:
    """Convert whole-number floats to int for cleaner YAML output.

    Args:
        value: Numeric value.

    Returns:
        int if value is a whole number, otherwise float.
    """
    if isinstance(value, float) and value == int(value):
        return int(value)
    return value
