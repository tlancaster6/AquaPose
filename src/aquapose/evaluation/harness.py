"""Evaluation harness orchestrator for offline reconstruction quality metrics.

Provides run_evaluation which loads a self-contained MidlineFixture, builds
projection models from the bundled CalibBundle, selects frames, runs Tier 1
and Tier 2 metric computation, and writes output files.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.core.reconstruction.backends.dlt import DltBackend
from aquapose.core.reconstruction.backends.triangulation import TriangulationBackend
from aquapose.core.types.reconstruction import MidlineSet
from aquapose.evaluation.metrics import (
    Tier1Result,
    Tier2Result,
    compute_tier1,
    compute_tier2,
    select_frames,
)
from aquapose.evaluation.output import format_summary_table, write_regression_json
from aquapose.io.midline_fixture import CalibBundle, load_midline_fixture


@dataclass(frozen=True)
class EvalResults:
    """Results from a completed evaluation run.

    Attributes:
        tier1: Aggregated Tier 1 reprojection error metrics.
        tier2: Aggregated Tier 2 leave-one-out displacement metrics.
        summary_table: Human-readable ASCII summary string.
        json_path: Path to the written regression JSON file.
        frames_evaluated: Number of frames evaluated.
        frames_available: Total frames available in the fixture.
    """

    tier1: Tier1Result
    tier2: Tier2Result
    summary_table: str
    json_path: Path
    frames_evaluated: int
    frames_available: int


def _build_models_from_calib(
    calib_bundle: CalibBundle,
) -> dict[str, RefractiveProjectionModel]:
    """Build per-camera projection models from a CalibBundle.

    Args:
        calib_bundle: Calibration data bundled in a v2.0 fixture.

    Returns:
        Dict mapping camera_id to RefractiveProjectionModel (CPU tensors).
    """
    normal = torch.from_numpy(calib_bundle.interface_normal).float()
    models: dict[str, RefractiveProjectionModel] = {}
    for cam_id in calib_bundle.camera_ids:
        models[cam_id] = RefractiveProjectionModel(
            K=torch.from_numpy(calib_bundle.K_new[cam_id]).float(),
            R=torch.from_numpy(calib_bundle.R[cam_id]).float(),
            t=torch.from_numpy(calib_bundle.t[cam_id]).float(),
            water_z=calib_bundle.water_z,
            normal=normal,
            n_air=calib_bundle.n_air,
            n_water=calib_bundle.n_water,
        )
    return models


def run_evaluation(
    fixture_path: Path | str,
    n_frames: int = 15,
    output_dir: Path | None = None,
    backend: str = "triangulation",
    outlier_threshold: float | None = None,
) -> EvalResults:
    """Run offline evaluation on a self-contained MidlineFixture.

    Loads the fixture, builds projection models from the bundled CalibBundle,
    selects frames, computes Tier 1 reprojection error and Tier 2 leave-one-out
    displacement metrics, formats output, and writes JSON regression data.

    Args:
        fixture_path: Path to the NPZ fixture file (must be v2.0 with calib/).
        n_frames: Number of frames to evaluate (default 15).
        output_dir: Directory for writing eval_results.json.  If None, uses
            the fixture's parent directory.
        backend: Reconstruction backend to use. Supported values:
            ``"triangulation"`` (default) and ``"dlt"``.
        outlier_threshold: Maximum reprojection error (pixels) for DLT backend
            outlier rejection. When None, uses the DltBackend default. Only
            applies when ``backend="dlt"``.

    Returns:
        EvalResults containing metric data, summary table, and JSON path.

    Raises:
        ValueError: If the fixture has no CalibBundle (v1.0 format) or if
            *backend* is not a recognized backend identifier.
    """
    fixture_path = Path(fixture_path)

    # 1. Load fixture
    fixture = load_midline_fixture(fixture_path)

    # 2. Validate calibration bundle
    if fixture.calib_bundle is None:
        raise ValueError(
            f"Fixture at {fixture_path} has no calib bundle (v1.0 format). "
            "Re-export the fixture with DiagnosticObserver to generate a "
            "v2.0 fixture with bundled calibration data."
        )

    # 3. Build projection models and reconstruction backend
    models = _build_models_from_calib(fixture.calib_bundle)

    if backend == "triangulation":
        recon_backend = TriangulationBackend.from_models(models)
    elif backend == "dlt":
        if outlier_threshold is not None:
            recon_backend = DltBackend.from_models(
                models, outlier_threshold=outlier_threshold
            )
        else:
            recon_backend = DltBackend.from_models(models)
    else:
        raise ValueError(
            f"Unknown evaluation backend: {backend!r}. "
            f"Supported backends: ['triangulation', 'dlt']"
        )

    # 4. Select frames
    selected_frame_indices = select_frames(fixture.frame_indices, n_frames)
    frames_available = len(fixture.frame_indices)
    frames_evaluated = len(selected_frame_indices)

    # 5. Build position lookup: frame_index -> position in fixture.frames
    frame_to_pos = {fi: pos for pos, fi in enumerate(fixture.frame_indices)}

    # 6. Tier 1: triangulate each selected frame
    frame_results: list[tuple[int, dict]] = []
    # Keep baseline results for Tier 2
    baseline_by_frame: dict[int, dict] = {}
    fish_available = 0

    for fi in selected_frame_indices:
        midline_set: MidlineSet = fixture.frames[frame_to_pos[fi]]
        fish_available += len(midline_set)
        result = recon_backend.reconstruct_frame(fi, midline_set)
        frame_results.append((fi, result))
        baseline_by_frame[fi] = result

    # 7. Tier 2: leave-one-out per selected frame, per fish, per observing camera
    # Accumulate: fish_id -> dropout_cam_id -> list[float | None]
    tier2_data: dict[int, dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for fi in selected_frame_indices:
        midline_set = fixture.frames[frame_to_pos[fi]]
        baseline = baseline_by_frame[fi]

        for fish_id, cam_map in midline_set.items():
            if fish_id not in baseline:
                continue  # Fish not reconstructable with all cameras

            baseline_ctrl: np.ndarray = baseline[fish_id].control_points

            for dropout_cam in cam_map:
                # Build reduced midline set excluding dropout_cam for ALL fish
                reduced: MidlineSet = {
                    fid: remaining
                    for fid, cams in midline_set.items()
                    if (
                        remaining := {c: m for c, m in cams.items() if c != dropout_cam}
                    )
                }
                if not reduced:
                    tier2_data[fish_id][dropout_cam].append(None)
                    continue
                dropout_result = recon_backend.reconstruct_frame(fi, reduced)

                if fish_id not in dropout_result:
                    tier2_data[fish_id][dropout_cam].append(None)
                else:
                    dropout_ctrl = dropout_result[fish_id].control_points
                    diffs = np.linalg.norm(
                        dropout_ctrl - baseline_ctrl, axis=1
                    )  # shape (7,)
                    max_displacement = float(np.max(diffs))
                    tier2_data[fish_id][dropout_cam].append(max_displacement)

    # 8. Compute metrics
    tier1 = compute_tier1(frame_results, fish_available=fish_available)
    # Convert defaultdict to plain dict for compute_tier2
    tier2_plain: dict[int, dict[str, list[float | None]]] = {
        fid: dict(cam_dict) for fid, cam_dict in tier2_data.items()
    }
    tier2 = compute_tier2(tier2_plain)

    # 9. Format summary
    fixture_name = fixture_path.name
    summary_table = format_summary_table(
        tier1, tier2, fixture_name, frames_evaluated, frames_available
    )

    # 10. Determine output directory and write JSON
    if output_dir is None:
        output_dir = fixture_path.parent
    json_path = write_regression_json(
        tier1,
        tier2,
        fixture_name=fixture_name,
        frames_evaluated=frames_evaluated,
        frames_available=frames_available,
        output_path=output_dir / "eval_results.json",
    )

    return EvalResults(
        tier1=tier1,
        tier2=tier2,
        summary_table=summary_table,
        json_path=json_path,
        frames_evaluated=frames_evaluated,
        frames_available=frames_available,
    )
