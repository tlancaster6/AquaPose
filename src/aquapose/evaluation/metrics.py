"""Frame selection and metric computation for offline evaluation.

Provides select_frames for deterministic frame sampling, and compute_tier1
for reprojection error metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from aquapose.core.types.reconstruction import Midline3D


@dataclass(frozen=True)
class Tier1Result:
    """Aggregated Tier 1 (reprojection error) metrics.

    Attributes:
        per_camera: Per-camera reprojection error aggregates.
            Maps camera_id to dict with keys ``"mean_px"`` and ``"max_px"``.
        per_fish: Per-fish reprojection error aggregates.
            Maps fish_id to dict with keys ``"mean_px"`` and ``"max_px"``.
        overall_mean_px: Mean reprojection error across all fish and cameras.
        overall_max_px: Maximum reprojection error across all fish and cameras.
    """

    per_camera: dict[str, dict[str, float]]
    per_fish: dict[int, dict[str, float]]
    overall_mean_px: float
    overall_max_px: float
    fish_reconstructed: int
    fish_available: int


@dataclass(frozen=True)
class Tier2Result:
    """Aggregated Tier 2 (leave-one-out displacement) metrics.

    Attributes:
        per_fish_dropout: Per-fish per-dropout-camera max control-point
            displacement. Maps fish_id to a dict of dropout_camera_id to max
            displacement in metres (or None when reconstruction failed).
    """

    per_fish_dropout: dict[int, dict[str, float | None]]


def select_frames(frame_indices: tuple[int, ...], n_frames: int = 100) -> list[int]:
    """Select a deterministic subset of frame indices via np.linspace.

    Args:
        frame_indices: All available frame indices from a fixture.
        n_frames: Number of frames to select.

    Returns:
        List of selected frame indices.  If fewer frames are available than
        requested, all frames are returned and a warning is emitted.  If the
        input is empty, an empty list is returned.
    """
    if len(frame_indices) == 0:
        return []

    available = list(frame_indices)

    if len(available) < n_frames:
        warnings.warn(
            f"Fixture has fewer frames ({len(available)}) than requested "
            f"({n_frames}); returning all {len(available)} frames.",
            stacklevel=2,
        )
        return available

    positions = np.linspace(0, len(available) - 1, n_frames, dtype=int)
    return [available[int(p)] for p in positions]


def compute_tier1(
    frame_results: list[tuple[int, dict[int, Midline3D]]],
    fish_available: int = 0,
) -> Tier1Result:
    """Compute Tier 1 reprojection error metrics from triangulation results.

    Aggregates per-camera and per-fish mean/max reprojection errors across all
    evaluated frames.

    Args:
        frame_results: List of ``(frame_idx, dict[fish_id, Midline3D])`` pairs
            from triangulation results.
        fish_available: Total fish-frame pairs available in the evaluated frames.
            Used to compute reconstruction coverage rate.

    Returns:
        Tier1Result with per-camera, per-fish, and overall aggregates.
    """
    # Accumulate per-camera errors: cam_id -> list[float]
    cam_errors: dict[str, list[float]] = {}
    # Accumulate per-fish residuals: fish_id -> list[float] (mean_residuals)
    fish_mean_residuals: dict[int, list[float]] = {}
    fish_max_residuals: dict[int, list[float]] = {}

    for _frame_idx, midline_dict in frame_results:
        for fish_id, midline3d in midline_dict.items():
            # Per-camera accumulation
            for cam_id, err in (midline3d.per_camera_residuals or {}).items():
                if cam_id not in cam_errors:
                    cam_errors[cam_id] = []
                cam_errors[cam_id].append(err)

            # Per-fish accumulation
            if fish_id not in fish_mean_residuals:
                fish_mean_residuals[fish_id] = []
                fish_max_residuals[fish_id] = []
            fish_mean_residuals[fish_id].append(midline3d.mean_residual)
            fish_max_residuals[fish_id].append(midline3d.max_residual)

    # Build per-camera aggregates
    per_camera: dict[str, dict[str, float]] = {}
    for cam_id, errors in cam_errors.items():
        per_camera[cam_id] = {
            "mean_px": float(np.mean(errors)),
            "max_px": float(np.max(errors)),
        }

    # Build per-fish aggregates
    per_fish: dict[int, dict[str, float]] = {}
    for fish_id in fish_mean_residuals:
        per_fish[fish_id] = {
            "mean_px": float(np.mean(fish_mean_residuals[fish_id])),
            "max_px": float(np.max(fish_max_residuals[fish_id])),
        }

    # Overall aggregates
    all_means = [v["mean_px"] for v in per_fish.values()]
    all_maxes = [v["max_px"] for v in per_fish.values()]
    overall_mean = float(np.mean(all_means)) if all_means else 0.0
    overall_max = float(np.max(all_maxes)) if all_maxes else 0.0

    # Reconstruction coverage
    fish_reconstructed = sum(len(midline_dict) for _, midline_dict in frame_results)

    return Tier1Result(
        per_camera=per_camera,
        per_fish=per_fish,
        overall_mean_px=overall_mean,
        overall_max_px=overall_max,
        fish_reconstructed=fish_reconstructed,
        fish_available=fish_available,
    )
