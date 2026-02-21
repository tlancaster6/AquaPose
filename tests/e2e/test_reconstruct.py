"""End-to-end integration tests for the full 5-stage reconstruction pipeline.

Slow test requires GPU and real 13-camera video data at the path defined by
``RAW_VIDEOS_DIR``. The import-only test is fast and CI-safe.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths to real data (CI will skip if absent)
# ---------------------------------------------------------------------------
RAW_VIDEOS_DIR = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/raw_videos")
CALIBRATION_PATH = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/calibration.json")
UNET_WEIGHTS_PATH = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/unet/best_model.pth")

# All five stage names expected in stage_timing dict
_EXPECTED_STAGES = {
    "detection",
    "segmentation",
    "tracking",
    "midline_extraction",
    "triangulation",
}


# ---------------------------------------------------------------------------
# Fast CI-safe test: verify importability and signature
# ---------------------------------------------------------------------------


def test_reconstruct_import() -> None:
    """Verify that reconstruct is importable and has the expected parameters.

    This test is fast and GPU-free — suitable for CI without real data.
    """
    from aquapose.pipeline import reconstruct

    sig = inspect.signature(reconstruct)
    params = set(sig.parameters.keys())

    # Required positional params
    assert "video_dir" in params, "reconstruct() must accept video_dir"
    assert "calibration_path" in params, "reconstruct() must accept calibration_path"
    assert "output_dir" in params, "reconstruct() must accept output_dir"

    # Optional keyword params
    assert "stop_frame" in params, "reconstruct() must accept stop_frame"
    assert "mode" in params, "reconstruct() must accept mode"
    assert "unet_weights" in params, "reconstruct() must accept unet_weights"

    # Verify return type annotation references ReconstructResult
    from aquapose.pipeline import ReconstructResult

    assert "output_dir" in ReconstructResult.__dataclass_fields__
    assert "midlines_3d" in ReconstructResult.__dataclass_fields__
    assert "stage_timing" in ReconstructResult.__dataclass_fields__


# ---------------------------------------------------------------------------
# Slow E2E test: full pipeline on real data
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reconstruct_e2e(tmp_path: Path) -> None:
    """Run the full 5-stage pipeline on 10 frames of real 13-camera video data.

    Asserts:
    - HDF5 output file ``midlines_3d.h5`` is written and readable.
    - ``stage_timing`` contains all five stage names.
    - ``midlines_3d`` list is non-empty.
    - ``report.md`` exists in the output directory.
    - At least one overlay video exists under ``output_dir/overlays/``.

    Skips if real data paths are not available.
    """
    if not RAW_VIDEOS_DIR.exists():
        pytest.skip(f"Raw video directory not found: {RAW_VIDEOS_DIR}")
    if not CALIBRATION_PATH.exists():
        pytest.skip(f"Calibration JSON not found: {CALIBRATION_PATH}")
    if not UNET_WEIGHTS_PATH.exists():
        pytest.skip(f"U-Net weights not found: {UNET_WEIGHTS_PATH}")

    from aquapose.io import read_midline3d_results
    from aquapose.pipeline import reconstruct

    result = reconstruct(
        video_dir=RAW_VIDEOS_DIR,
        calibration_path=CALIBRATION_PATH,
        output_dir=tmp_path,
        stop_frame=10,
        mode="diagnostic",
        unet_weights=UNET_WEIGHTS_PATH,
    )

    # --- HDF5 output ---
    h5_path = result.output_dir / "midlines_3d.h5"
    assert result.output_dir.exists(), "output_dir does not exist"
    assert h5_path.exists(), "midlines_3d.h5 was not written"

    # --- HDF5 readability ---
    data = read_midline3d_results(h5_path)
    assert "frame_index" in data, "HDF5 missing frame_index dataset"
    assert "control_points" in data, "HDF5 missing control_points dataset"
    assert len(data["frame_index"]) > 0, "HDF5 has no frames"

    # --- Stage timing ---
    assert _EXPECTED_STAGES.issubset(result.stage_timing.keys()), (
        f"stage_timing missing stages: {_EXPECTED_STAGES - result.stage_timing.keys()!r}"
    )

    # --- Non-empty midlines list ---
    assert len(result.midlines_3d) > 0, "midlines_3d result list is empty"

    # --- Diagnostic report ---
    report_path = result.output_dir / "report.md"
    assert report_path.exists(), "report.md was not written in diagnostic mode"

    # --- Overlay videos ---
    overlays_dir = result.output_dir / "overlays"
    overlay_videos = list(overlays_dir.glob("*.mp4"))
    assert len(overlay_videos) >= 1, f"No overlay .mp4 files found in {overlays_dir}"

    # --- Print timing table for manual review ---
    print("\n=== Stage Timing Summary ===")
    total = sum(result.stage_timing.values())
    print(f"{'Stage':<25} {'Seconds':>10} {'% Total':>10}")
    print("-" * 47)
    for stage_name, elapsed in result.stage_timing.items():
        pct = 100.0 * elapsed / total if total > 0 else 0.0
        print(f"{stage_name:<25} {elapsed:>10.2f} {pct:>9.1f}%")
    print("-" * 47)
    print(f"{'TOTAL':<25} {total:>10.2f} {'100.0':>9}%")

    print(f"\nOutput directory: {result.output_dir}")
    print(f"Frames processed: {len(result.midlines_3d)}")
    non_empty = sum(1 for f in result.midlines_3d if f)
    print(f"Frames with midlines: {non_empty}/{len(result.midlines_3d)}")
    print(f"Overlay videos: {[v.name for v in overlay_videos]}")
