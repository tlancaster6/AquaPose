"""End-to-end tests for the AquaPose v2.1 5-stage pipeline.

Two test classes:

- TestSyntheticSmoke: Fast, NOT marked @slow. Exercises the full pipeline on
  known-geometry synthetic inputs. Runs in the normal ``hatch run test`` suite.
  Uses the real AquaCal calibration to generate realistic projections.

- TestRealData: Marked @slow and @e2e. Exercises the pipeline on 4 real
  cameras with ~100 frames. Only runs with ``hatch run test-all``.
  Saves reprojection overlay artifacts to tests/e2e/output/ for human review.

Run synthetic tests only::

    hatch run test tests/e2e/test_smoke.py -k synthetic -v

Run all e2e tests (includes real-data)::

    hatch run test-all tests/e2e/test_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aquapose.engine.config import (
    PipelineConfig,
    load_config,
)
from aquapose.engine.pipeline import PosePipeline, build_stages

# ---------------------------------------------------------------------------
# Synthetic smoke tests (NOT @slow -- run in normal test suite)
# ---------------------------------------------------------------------------


class TestSyntheticSmoke:
    """Synthetic pipeline smoke tests.

    Exercises the 4-stage synthetic pipeline:
    SyntheticDataStage -> TrackingStage -> AssociationStage -> ReconstructionStage

    These tests do NOT require YOLO weights, U-Net weights, or video files --
    only the AquaCal calibration JSON. They are NOT marked @slow and run in
    the standard ``hatch run test`` suite.
    """

    def _build_synthetic_config(
        self,
        output_dir: Path,
        calibration_path: Path,
    ) -> PipelineConfig:
        """Construct a PipelineConfig for synthetic mode.

        Args:
            output_dir: Temporary directory for run artifacts.
            calibration_path: Path to the AquaCal calibration JSON.

        Returns:
            Frozen PipelineConfig with synthetic mode and default SyntheticConfig.
        """
        return load_config(
            cli_overrides={
                "mode": "synthetic",
                "calibration_path": str(calibration_path),
                "output_dir": str(output_dir),
            }
        )

    def test_synthetic_pipeline_completes(
        self,
        tmp_path: Path,
        calibration_path: Path,
    ) -> None:
        """Synthetic pipeline completes without exception.

        Builds a synthetic config, runs the pipeline, and asserts no
        exception is raised. This is a pure smoke test -- no output assertions.

        Args:
            tmp_path: Pytest temporary directory for run artifacts.
            calibration_path: Session-scoped calibration file path.
        """
        config = self._build_synthetic_config(tmp_path, calibration_path)
        stages = build_stages(config)
        pipeline = PosePipeline(stages=stages, config=config)
        # If this raises, the test fails with the exception traceback.
        pipeline.run()

    def test_synthetic_output_validation(
        self,
        tmp_path: Path,
        calibration_path: Path,
    ) -> None:
        """Synthetic pipeline produces expected output fields.

        Runs the synthetic pipeline and validates that:
        - context.tracks_2d is populated with at least 1 camera that has tracklets
        - context.tracklet_groups is not None (may be empty if LUTs not cached)
        - context.midlines_3d is not None (may be empty if no groups formed)

        Note: AssociationStage requires pre-built LUTs for grouping. Without
        cached LUTs, tracklet_groups will be [] (empty list, not None). This is
        expected behaviour -- the stage degrades gracefully. When LUTs are built
        (via CLI ``aquapose build-luts``), association and reconstruction will
        produce non-empty results.

        Args:
            tmp_path: Pytest temporary directory for run artifacts.
            calibration_path: Session-scoped calibration file path.
        """
        config = self._build_synthetic_config(tmp_path, calibration_path)
        stages = build_stages(config)
        pipeline = PosePipeline(stages=stages, config=config)
        context = pipeline.run()

        # --- Stage 2 output: per-camera 2D tracklets ---
        assert context.tracks_2d is not None, "tracks_2d should be set by TrackingStage"
        cameras_with_tracklets = [
            cam for cam, trks in context.tracks_2d.items() if trks
        ]
        assert len(cameras_with_tracklets) >= 1, (
            f"Expected at least 1 camera with tracklets, got none. "
            f"tracks_2d cameras: {list(context.tracks_2d.keys())}"
        )

        # --- Stage 3 output: cross-camera identity groups ---
        assert context.tracklet_groups is not None, (
            "tracklet_groups should be a list (possibly empty) after AssociationStage"
        )

        # --- Stage 5 output: 3D midlines ---
        assert context.midlines_3d is not None, (
            "midlines_3d should be a list (possibly empty) after ReconstructionStage"
        )

        # --- Frame count sanity ---
        assert context.frame_count == config.synthetic.frame_count, (
            f"Expected frame_count={config.synthetic.frame_count}, "
            f"got {context.frame_count}"
        )


# ---------------------------------------------------------------------------
# Real-data tests (@slow @e2e -- only run with hatch run test-all)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.e2e
class TestRealData:
    """Real-data pipeline validation on 4-camera video subset.

    Exercises the full 5-stage pipeline:
    DetectionStage -> TrackingStage -> AssociationStage -> MidlineStage ->
    ReconstructionStage

    Requires: calibration_path, test_video_dir, yolo_weights, unet_weights
    fixtures (provided by conftest.py). Tests skip if any required file is
    missing.

    Artifacts (reprojection overlay videos) are saved to tests/e2e/output/
    for human review after the tests complete.

    Run with::

        hatch run test-all tests/e2e/test_smoke.py -v
    """

    def _build_real_config(
        self,
        output_dir: Path,
        calibration_path: Path,
        video_dir: Path,
        yolo_weights: Path,
        unet_weights: Path,
        stop_frame: int = 100,
    ) -> PipelineConfig:
        """Construct a PipelineConfig for real-data diagnostic mode.

        Args:
            output_dir: Directory for run artifacts (reprojection videos, etc.).
            calibration_path: Path to AquaCal calibration JSON.
            video_dir: Directory containing the 4-camera test MP4 files.
            yolo_weights: Path to YOLO fish detection model weights.
            unet_weights: Path to U-Net segmentation model weights.
            stop_frame: Maximum frame index to process (default 100 ~= 3s).

        Returns:
            Frozen PipelineConfig for diagnostic mode with real data.
        """
        return load_config(
            cli_overrides={
                "mode": "diagnostic",
                "calibration_path": str(calibration_path),
                "video_dir": str(video_dir),
                "output_dir": str(output_dir),
                "detection": {
                    "detector_kind": "yolo",
                    "model_path": str(yolo_weights),
                    "stop_frame": stop_frame,
                    "device": "cpu",
                },
                "midline": {
                    "weights_path": str(unet_weights),
                    "backend": "segment_then_extract",
                    "device": "cpu",
                },
            }
        )

    def test_real_pipeline_completes(
        self,
        e2e_output_dir: Path,
        calibration_path: Path,
        test_video_dir: Path,
        yolo_weights: Path,
        unet_weights: Path,
    ) -> None:
        """Real-data pipeline completes without exception on 4-camera subset.

        Processes ~100 frames from 4 cameras in diagnostic mode, which
        triggers the TrackletTrailObserver and other diagnostic observers.
        Artifacts are saved to tests/e2e/output/<run_id>/ for human review.

        Args:
            e2e_output_dir: Session-scoped output directory for artifacts.
            calibration_path: Path to AquaCal calibration JSON.
            test_video_dir: Path to directory with 4-camera test videos.
            yolo_weights: Path to YOLO detection model weights.
            unet_weights: Path to U-Net segmentation model weights.
        """
        run_output_dir = e2e_output_dir / "real_data_run"
        config = self._build_real_config(
            output_dir=run_output_dir,
            calibration_path=calibration_path,
            video_dir=test_video_dir,
            yolo_weights=yolo_weights,
            unet_weights=unet_weights,
            stop_frame=100,
        )
        stages = build_stages(config)
        pipeline = PosePipeline(stages=stages, config=config)
        # If this raises, the test fails with the full traceback.
        context = pipeline.run()

        # Smoke assertions: stage outputs are set (possibly empty)
        assert context.detections is not None, (
            "detections should be set by DetectionStage"
        )
        assert context.tracks_2d is not None, "tracks_2d should be set by TrackingStage"
        assert context.tracklet_groups is not None, (
            "tracklet_groups should be a list after AssociationStage"
        )
        assert context.midlines_3d is not None, (
            "midlines_3d should be a list after ReconstructionStage"
        )

    def test_real_output_has_3d_splines(
        self,
        e2e_output_dir: Path,
        calibration_path: Path,
        test_video_dir: Path,
        yolo_weights: Path,
        unet_weights: Path,
    ) -> None:
        """At least 1 fish produces valid 3D splines spanning 3+ contiguous frames.

        This test validates that the pipeline is producing useful 3D output,
        not just completing without crashing. It checks:
        - At least 1 non-empty frame in midlines_3d
        - At least 1 fish has splines in 3+ contiguous frames

        Note: This test depends on AssociationStage having valid LUTs. If LUTs
        are not built, tracklet_groups will be empty and no 3D output is expected.
        If this test fails with 0 splines, run ``aquapose build-luts`` first.

        Args:
            e2e_output_dir: Session-scoped output directory for artifacts.
            calibration_path: Path to AquaCal calibration JSON.
            test_video_dir: Path to directory with 4-camera test videos.
            yolo_weights: Path to YOLO detection model weights.
            unet_weights: Path to U-Net segmentation model weights.
        """
        run_output_dir = e2e_output_dir / "real_data_spline_check"
        config = self._build_real_config(
            output_dir=run_output_dir,
            calibration_path=calibration_path,
            video_dir=test_video_dir,
            yolo_weights=yolo_weights,
            unet_weights=unet_weights,
            stop_frame=100,
        )
        stages = build_stages(config)
        pipeline = PosePipeline(stages=stages, config=config)
        context = pipeline.run()

        midlines_3d = context.midlines_3d
        assert midlines_3d is not None

        # Count frames with at least one fish spline
        non_empty_frames = [frame_dict for frame_dict in midlines_3d if frame_dict]
        assert len(non_empty_frames) >= 1, (
            "Expected at least 1 frame with 3D splines, got 0. "
            "This typically means LUTs are not built (run 'aquapose build-luts') "
            "or no fish were detected in the test video subset."
        )

        # Find fish that appear in 3+ contiguous frames
        fish_frame_sets: dict[int, list[int]] = {}
        for frame_idx, frame_dict in enumerate(midlines_3d):
            for fish_id in frame_dict:
                fish_frame_sets.setdefault(fish_id, []).append(frame_idx)

        # Check for 3+ contiguous frames for any fish
        def _max_contiguous_run(frame_indices: list[int]) -> int:
            if not frame_indices:
                return 0
            sorted_frames = sorted(frame_indices)
            max_run = current_run = 1
            for i in range(1, len(sorted_frames)):
                if sorted_frames[i] == sorted_frames[i - 1] + 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            return max_run

        best_fish_id = max(
            fish_frame_sets, key=lambda fid: _max_contiguous_run(fish_frame_sets[fid])
        )
        best_run = _max_contiguous_run(fish_frame_sets[best_fish_id])

        assert best_run >= 3, (
            f"Expected at least 1 fish with 3D splines in 3+ contiguous frames. "
            f"Best fish (id={best_fish_id}) only had {best_run} contiguous frames. "
            f"Fish frame counts: {dict((k, len(v)) for k, v in fish_frame_sets.items())}"
        )


# ---------------------------------------------------------------------------
# Known issues / non-blocking bugs
# ---------------------------------------------------------------------------
# The following issues were observed during Phase 28 e2e testing and are
# non-blocking (pipeline completes but with degraded output):
#
# 1. AssociationStage requires pre-built LUTs (forward + inverse .npz files).
#    Without LUTs, association produces empty tracklet_groups and no 3D output.
#    Fix: run `aquapose build-luts --calibration <path>` before real-data tests.
#    Tracked in: .planning/phases/28-e2e-testing/28-BUGS.md
#
# 2. SyntheticDataStage originally placed fish at z=0.02-0.12m (above water
#    interface at z=1.03m). Fixed in Phase 28: fish now placed at water_z+0.05
#    to water_z+0.35m (5-35cm below water surface).
#    Fix applied: src/aquapose/core/synthetic.py _generate_fish_splines()
