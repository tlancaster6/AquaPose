"""Pure-function evaluator for the reconstruction stage."""

from __future__ import annotations

from dataclasses import dataclass

from aquapose.core.types.reconstruction import Midline3D
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
        }


def evaluate_reconstruction(
    frame_results: list[tuple[int, dict[int, Midline3D]]],
    fish_available: int = 0,
    *,
    tier2_result: Tier2Result | None = None,
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
    )


__all__ = ["DEFAULT_GRID", "ReconstructionMetrics", "evaluate_reconstruction"]
