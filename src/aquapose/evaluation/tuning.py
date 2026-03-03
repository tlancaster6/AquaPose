"""TuningOrchestrator -- grid sweep engine for association and reconstruction parameter tuning."""

from __future__ import annotations

import copy
import dataclasses
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aquapose.core.context import PipelineContext, load_stage_cache
from aquapose.evaluation.metrics import select_frames
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
    while reusing the midline cache's annotated_detections.

    Args:
        ctx: PipelineContext with annotated_detections (from midline cache).
        tracklet_groups: TrackletGroup list (may be from original cache or
            from an association sweep combo).
        sampled_indices: Frame indices to include.

    Returns:
        List of MidlineSet dicts, one per sampled frame.
    """
    import math

    from aquapose.core.types.midline import Midline2D

    _CENTROID_MATCH_TOLERANCE_PX = 5.0
    annotated_detections = ctx.annotated_detections or []
    frame_count = ctx.frame_count or len(annotated_detections)
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
                if frame_idx >= len(annotated_detections):
                    continue

                frame_annot = annotated_detections[frame_idx]
                if not isinstance(frame_annot, dict):
                    continue
                cam_list = frame_annot.get(cam_id)
                if not cam_list:
                    continue

                centroid: tuple[float, float] = tracklet.centroids[tidx]

                # Match by centroid proximity
                cx, cy = centroid
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
                if midline is None:
                    continue
                if not isinstance(midline, Midline2D):
                    continue

                collected.setdefault(frame_idx, {}).setdefault(fish_id, {})[cam_id] = (
                    midline
                )

    return [collected.get(fi, {}) for fi in effective_indices]


class TuningOrchestrator:
    """Orchestrates parameter grid sweeps for association and reconstruction stages.

    Uses per-stage pickle caches to skip upstream re-execution. Supports
    joint 2D grid sweeps, sequential carry-forward, two-tier validation,
    and formatted output.

    Example::

        orchestrator = TuningOrchestrator(config_path)
        result = orchestrator.sweep_association(n_frames=30, top_n=3)
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

        # Discover and load caches
        self._caches: dict[str, PipelineContext] = {}
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

    def sweep_association(
        self,
        n_frames: int = 30,
        n_frames_validate: int = 100,
        top_n: int = 3,
    ) -> TuningResult:
        """Sweep association parameters over a joint 2D grid then carry-forward.

        Args:
            n_frames: Frame count for fast sweep tier.
            n_frames_validate: Frame count for top-N validation tier.
            top_n: Number of top candidates for full validation.

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
        sampled_indices = select_frames(tuple(range(frame_count)), n_frames)

        grid = ASSOCIATION_DEFAULT_GRID
        n_animals = self._config.n_animals

        # Compute baseline metrics
        baseline_assoc, baseline_recon = self._evaluate_association_combo(
            tracking_ctx, midline_ctx, self._config, sampled_indices, n_animals
        )
        _compute_association_score(baseline_assoc, baseline_recon)

        # Phase 1: Joint 2D grid over ray_distance_threshold x score_min
        joint_params = ["ray_distance_threshold", "score_min"]
        joint_values = [grid[p] for p in joint_params]
        joint_combos = list(itertools.product(*joint_values))

        all_results: list[dict[str, Any]] = []
        joint_grid_results: list[dict[str, Any]] = []

        print(f"Joint grid: {len(joint_combos)} combos")
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
                ctx_copy, midline_ctx, config, sampled_indices, n_animals
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
                f"score_min={params['score_min']:.3f} -> "
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
                    ctx_copy, midline_ctx, config, sampled_indices, n_animals
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

        # Sort all results and take top-N for validation
        all_results.sort(key=lambda r: r["score"])
        top_candidates = all_results[:top_n]

        # Top-N validation at higher frame count
        validate_indices = select_frames(tuple(range(frame_count)), n_frames_validate)
        best_validated = None

        print(
            f"\nValidating top {len(top_candidates)} candidates at {n_frames_validate} frames"
        )
        for candidate in top_candidates:
            params = candidate["params"]
            config = self._patch_association_config(params)
            ctx_copy = copy.copy(tracking_ctx)

            try:
                assoc_stage = AssociationStage(config=config)
                ctx_copy = assoc_stage.run(ctx_copy)
            except Exception:
                continue

            assoc_m, recon_m = self._evaluate_association_with_ctx(
                ctx_copy, midline_ctx, config, validate_indices, n_animals
            )
            score = _compute_association_score(assoc_m, recon_m)
            candidate["validated_assoc"] = assoc_m
            candidate["validated_recon"] = recon_m
            candidate["validated_score"] = score
            print(
                f"  {params} -> "
                f"yield={assoc_m.fish_yield_ratio:.1%} "
                f"error={recon_m.mean_reprojection_error:.2f}px"
            )

            if best_validated is None or score < best_validated["validated_score"]:
                best_validated = candidate

        # Build result
        if best_validated is not None:
            winner_params = best_validated["params"]
            v_assoc = best_validated["validated_assoc"]
            v_recon = best_validated["validated_recon"]
            winner_metrics = {
                "fish_yield_ratio": v_assoc.fish_yield_ratio,
                "mean_reprojection_error": v_recon.mean_reprojection_error,
                "max_reprojection_error": v_recon.max_reprojection_error,
                "singleton_rate": v_assoc.singleton_rate,
                "tier2_stability": v_recon.tier2_stability,
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
            cache_path = self._run_dir / "diagnostics" / f"{stage_key}_cache.pkl"
            raise FileNotFoundError(
                f"Required cache '{stage_key}' not found at {cache_path}. "
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

    def _evaluate_association_with_ctx(
        self,
        assoc_ctx: PipelineContext,
        midline_ctx: PipelineContext,
        config: Any,
        sampled_indices: list[int],
        n_animals: int,
    ) -> tuple[AssociationMetrics, ReconstructionMetrics]:
        """Evaluate association + reconstruction metrics from an already-run context.

        Args:
            assoc_ctx: Context after AssociationStage.run (has tracklet_groups).
            midline_ctx: Context with annotated_detections.
            config: PipelineConfig for reconstruction stage construction.
            sampled_indices: Frame indices to evaluate.
            n_animals: Expected number of fish.

        Returns:
            Tuple of (AssociationMetrics, ReconstructionMetrics).
        """
        from aquapose.core.reconstruction.stage import ReconstructionStage

        tracklet_groups = assoc_ctx.tracklet_groups or []
        midline_sets = _build_midline_sets(
            midline_ctx, tracklet_groups, sampled_indices
        )
        assoc_m = evaluate_association(midline_sets, n_animals)

        # Run reconstruction to get reprojection errors
        recon_ctx = copy.copy(assoc_ctx)
        # Copy annotated_detections from midline cache
        recon_ctx.annotated_detections = midline_ctx.annotated_detections

        recon_stage = ReconstructionStage(
            calibration_path=config.calibration_path,
            backend=config.reconstruction.backend,
            min_cameras=config.reconstruction.min_cameras,
            max_interp_gap=config.reconstruction.max_interp_gap,
            n_control_points=config.reconstruction.n_control_points,
            outlier_threshold=config.reconstruction.outlier_threshold,
        )
        recon_ctx = recon_stage.run(recon_ctx)

        midlines_3d = recon_ctx.midlines_3d or []
        frame_results = [
            (i, midlines_3d[i])
            for i in sampled_indices
            if i < len(midlines_3d) and midlines_3d[i]
        ]
        fish_available = n_animals * len(sampled_indices)
        recon_m = evaluate_reconstruction(frame_results, fish_available)

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
