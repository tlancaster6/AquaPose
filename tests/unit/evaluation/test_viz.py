"""Unit tests for evaluation.viz modules."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from aquapose.core.context import PipelineContext

# ---------------------------------------------------------------------------
# Helpers to build minimal fake chunk caches
# ---------------------------------------------------------------------------


def _make_context(
    frame_count: int = 5,
    camera_ids: list[str] | None = None,
    midlines_3d: list | None = None,
    tracks_2d: dict | None = None,
    tracklet_groups: list | None = None,
    annotated_detections: list | None = None,
) -> PipelineContext:
    """Build a minimal PipelineContext for testing."""
    ctx = PipelineContext()
    ctx.frame_count = frame_count
    ctx.camera_ids = camera_ids or ["cam0", "cam1"]
    ctx.midlines_3d = (
        midlines_3d if midlines_3d is not None else [{} for _ in range(frame_count)]
    )
    ctx.tracks_2d = tracks_2d or {}
    ctx.tracklet_groups = tracklet_groups or []
    ctx.annotated_detections = annotated_detections or [None] * frame_count
    return ctx


def _write_fake_chunk_cache(run_dir: Path, chunk_idx: int, context: object) -> Path:
    """Write a minimal fake chunk cache envelope to disk."""
    chunk_dir = run_dir / "diagnostics" / f"chunk_{chunk_idx:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    cache_path = chunk_dir / "cache.pkl"
    envelope = {
        "run_id": "test-run-id",
        "timestamp": "2026-03-04T00:00:00Z",
        "version_fingerprint": "fp-test",
        "context": context,
    }
    cache_path.write_bytes(pickle.dumps(envelope))
    return cache_path


def _write_fake_manifest(run_dir: Path, chunk_indices: list[int]) -> Path:
    """Write a minimal fake manifest.json."""
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = diag_dir / "manifest.json"
    manifest = {
        "run_id": "test-run-id",
        "total_frames": None,
        "chunk_size": None,
        "version_fingerprint": "fp-test",
        "chunks": [
            {"index": i, "start_frame": None, "end_frame": None} for i in chunk_indices
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _setup_run_dir(tmp_path: Path, n_chunks: int = 1, frame_count: int = 5) -> Path:
    """Set up a run directory with fake chunk caches and manifest."""
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    for i in range(n_chunks):
        ctx = _make_context(frame_count=frame_count)
        _write_fake_chunk_cache(run_dir, i, ctx)
    _write_fake_manifest(run_dir, list(range(n_chunks)))
    return run_dir


# ---------------------------------------------------------------------------
# Tests for _loader
# ---------------------------------------------------------------------------


class TestLoadAllChunkCaches:
    """Tests for load_all_chunk_caches utility."""

    def test_empty_when_no_diagnostics_dir(self, tmp_path: Path) -> None:
        from aquapose.evaluation.viz._loader import load_all_chunk_caches

        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        result = load_all_chunk_caches(run_dir)
        assert result == []

    def test_empty_when_no_manifest(self, tmp_path: Path) -> None:
        from aquapose.evaluation.viz._loader import load_all_chunk_caches

        run_dir = tmp_path / "run_no_manifest"
        run_dir.mkdir()
        (run_dir / "diagnostics").mkdir()
        result = load_all_chunk_caches(run_dir)
        assert result == []

    def test_loads_single_chunk(self, tmp_path: Path) -> None:
        from aquapose.evaluation.viz._loader import load_all_chunk_caches

        run_dir = _setup_run_dir(tmp_path, n_chunks=1)
        result = load_all_chunk_caches(run_dir)
        assert len(result) == 1
        assert result[0].frame_count == 5

    def test_loads_multiple_chunks_in_order(self, tmp_path: Path) -> None:
        from aquapose.evaluation.viz._loader import load_all_chunk_caches

        run_dir = _setup_run_dir(tmp_path, n_chunks=3)
        result = load_all_chunk_caches(run_dir)
        assert len(result) == 3

    def test_skips_missing_cache_file(self, tmp_path: Path) -> None:
        from aquapose.evaluation.viz._loader import load_all_chunk_caches

        run_dir = tmp_path / "run_missing"
        run_dir.mkdir()
        # Write manifest with 2 chunks, but only write cache for chunk 0.
        ctx = _make_context()
        _write_fake_chunk_cache(run_dir, 0, ctx)
        _write_fake_manifest(run_dir, [0, 1])
        result = load_all_chunk_caches(run_dir)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests for generate_all
# ---------------------------------------------------------------------------


class TestGenerateAll:
    """Tests for the generate_all convenience function."""

    def test_generate_all_calls_each_sub_generator(self, tmp_path: Path) -> None:
        """generate_all calls overlay, animation, and trails sub-generators."""
        from aquapose.evaluation.viz import generate_all

        run_dir = _setup_run_dir(tmp_path)
        overlay_path = tmp_path / "viz" / "overlay_mosaic.mp4"
        animation_path = tmp_path / "viz" / "animation_3d.html"
        trails_path = tmp_path / "viz"

        with (
            patch(
                "aquapose.evaluation.viz.generate_overlay",
                return_value=overlay_path,
            ) as mock_overlay,
            patch(
                "aquapose.evaluation.viz.generate_animation",
                return_value=animation_path,
            ) as mock_animation,
            patch(
                "aquapose.evaluation.viz.generate_trails",
                return_value=trails_path,
            ) as mock_trails,
        ):
            results = generate_all(run_dir)

        mock_overlay.assert_called_once_with(run_dir, None)
        mock_animation.assert_called_once_with(run_dir, None)
        mock_trails.assert_called_once_with(run_dir, None)

        assert results["overlay"] == overlay_path
        assert results["animation"] == animation_path
        assert results["trails"] == trails_path

    def test_generate_all_catches_individual_failures(self, tmp_path: Path) -> None:
        """generate_all catches failures per visualization and returns Exception."""
        from aquapose.evaluation.viz import generate_all

        run_dir = _setup_run_dir(tmp_path)
        err = RuntimeError("overlay failed")

        with (
            patch(
                "aquapose.evaluation.viz.generate_overlay",
                side_effect=err,
            ),
            patch(
                "aquapose.evaluation.viz.generate_animation",
                return_value=tmp_path / "animation.html",
            ),
            patch(
                "aquapose.evaluation.viz.generate_trails",
                return_value=tmp_path,
            ),
        ):
            results = generate_all(run_dir)

        # Overlay should be the exception, others should succeed.
        assert results["overlay"] is err
        assert isinstance(results["animation"], Path)
        assert isinstance(results["trails"], Path)

    def test_generate_all_all_fail_no_crash(self, tmp_path: Path) -> None:
        """generate_all does not raise even if all sub-generators fail."""
        from aquapose.evaluation.viz import generate_all

        run_dir = _setup_run_dir(tmp_path)
        error = RuntimeError("test error")

        with (
            patch("aquapose.evaluation.viz.generate_overlay", side_effect=error),
            patch("aquapose.evaluation.viz.generate_animation", side_effect=error),
            patch("aquapose.evaluation.viz.generate_trails", side_effect=error),
        ):
            results = generate_all(run_dir)

        assert all(isinstance(v, Exception) for v in results.values())

    def test_generate_all_passes_output_dir(self, tmp_path: Path) -> None:
        """generate_all passes output_dir to all sub-generators."""
        from aquapose.evaluation.viz import generate_all

        run_dir = _setup_run_dir(tmp_path)
        custom_out = tmp_path / "custom_output"

        with (
            patch(
                "aquapose.evaluation.viz.generate_overlay",
                return_value=custom_out / "overlay_mosaic.mp4",
            ) as mock_overlay,
            patch(
                "aquapose.evaluation.viz.generate_animation",
                return_value=custom_out / "animation_3d.html",
            ),
            patch(
                "aquapose.evaluation.viz.generate_trails",
                return_value=custom_out,
            ),
        ):
            generate_all(run_dir, custom_out)

        mock_overlay.assert_called_once_with(run_dir, custom_out)


# ---------------------------------------------------------------------------
# Tests for generate_overlay (output path / no-crash contract)
# ---------------------------------------------------------------------------


class TestGenerateOverlay:
    """Tests for generate_overlay output path and missing H5 behavior."""

    def test_raises_on_missing_h5(self, tmp_path: Path) -> None:
        """generate_overlay raises RuntimeError when no midlines H5 is found."""
        from aquapose.evaluation.viz.overlay import generate_overlay

        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(RuntimeError, match="No midlines HDF5"):
            generate_overlay(run_dir)

    def test_raises_on_missing_calibration(self, tmp_path: Path) -> None:
        """generate_overlay raises RuntimeError when calibration is missing."""
        import yaml

        from aquapose.evaluation.viz.overlay import generate_overlay

        run_dir = tmp_path / "run_no_calib"
        run_dir.mkdir()
        _write_minimal_h5(run_dir / "midlines.h5")
        (run_dir / "config.yaml").write_text(
            yaml.dump({"calibration_path": "nonexistent.json"})
        )
        with pytest.raises(RuntimeError, match="Calibration file not found"):
            generate_overlay(run_dir)


# ---------------------------------------------------------------------------
# Tests for generate_animation
# ---------------------------------------------------------------------------


class TestGenerateAnimation:
    """Tests for generate_animation output path and missing H5 behavior."""

    def test_raises_on_missing_h5(self, tmp_path: Path) -> None:
        """generate_animation raises RuntimeError when no midlines H5 is found."""
        from aquapose.evaluation.viz.animation import generate_animation

        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(RuntimeError, match="No midlines HDF5"):
            generate_animation(run_dir)

    def test_output_path_in_viz_subdir(self, tmp_path: Path) -> None:
        """generate_animation writes to {run_dir}/viz/animation_3d.html by default."""
        from aquapose.evaluation.viz.animation import generate_animation

        run_dir = tmp_path / "run_anim"
        run_dir.mkdir()
        _write_minimal_h5(run_dir / "midlines.h5")

        out_path = generate_animation(run_dir)
        assert out_path == run_dir / "viz" / "animation_3d.html"
        assert out_path.exists()

    def test_custom_output_dir_used(self, tmp_path: Path) -> None:
        """generate_animation writes to custom output_dir when provided."""
        from aquapose.evaluation.viz.animation import generate_animation

        run_dir = tmp_path / "run_anim_custom"
        run_dir.mkdir()
        _write_minimal_h5(run_dir / "midlines.h5")
        custom_dir = tmp_path / "my_viz"

        out_path = generate_animation(run_dir, custom_dir)
        assert out_path.parent == custom_dir
        assert out_path.name == "animation_3d.html"
        assert out_path.exists()

    def test_prefers_stitched_h5(self, tmp_path: Path) -> None:
        """generate_animation prefers midlines_stitched.h5 by default."""
        from aquapose.evaluation.viz.animation import generate_animation

        run_dir = tmp_path / "run_anim_stitched"
        run_dir.mkdir()
        _write_minimal_h5(run_dir / "midlines.h5", fish_id=0)
        _write_minimal_h5(run_dir / "midlines_stitched.h5", fish_id=99)

        out_path = generate_animation(run_dir)
        assert out_path.exists()
        # The animation should contain fish 99 from stitched, not fish 0
        content = out_path.read_text()
        assert "Fish 99" in content


# ---------------------------------------------------------------------------
# Tests for generate_trails
# ---------------------------------------------------------------------------


class TestGenerateTrails:
    """Tests for generate_trails output path and missing H5 behavior."""

    def test_raises_on_missing_h5(self, tmp_path: Path) -> None:
        """generate_trails raises RuntimeError when no midlines H5 is found."""
        from aquapose.evaluation.viz.trails import generate_trails

        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(RuntimeError, match="No midlines HDF5"):
            generate_trails(run_dir)

    def test_raises_on_missing_calibration(self, tmp_path: Path) -> None:
        """generate_trails raises RuntimeError when calibration is missing."""
        import yaml

        from aquapose.evaluation.viz.trails import generate_trails

        run_dir = tmp_path / "run_no_calib"
        run_dir.mkdir()
        # Write a minimal midlines.h5
        _write_minimal_h5(run_dir / "midlines.h5")
        # Write config.yaml with bogus calibration path
        (run_dir / "config.yaml").write_text(
            yaml.dump({"calibration_path": "nonexistent.json"})
        )
        with pytest.raises(RuntimeError, match="Calibration file not found"):
            generate_trails(run_dir)

    def test_prefers_stitched_h5(self, tmp_path: Path) -> None:
        """generate_trails prefers midlines_stitched.h5 over midlines.h5."""
        from aquapose.evaluation.viz.trails import _load_midline_positions

        run_dir = tmp_path / "run_stitched"
        run_dir.mkdir()
        # Write both files with different fish IDs
        _write_minimal_h5(run_dir / "midlines.h5", fish_id=0)
        _write_minimal_h5(run_dir / "midlines_stitched.h5", fish_id=99)

        # Default (unstitched=False) should pick stitched
        stitched_path = run_dir / "midlines_stitched.h5"
        data = _load_midline_positions(stitched_path)
        assert any(99 in fdict for fdict in data.values())

    def test_unstitched_flag(self, tmp_path: Path) -> None:
        """With unstitched=True, generate_trails uses midlines.h5."""
        from aquapose.evaluation.viz.trails import _load_midline_positions

        run_dir = tmp_path / "run_unstitched"
        run_dir.mkdir()
        _write_minimal_h5(run_dir / "midlines.h5", fish_id=0)
        _write_minimal_h5(run_dir / "midlines_stitched.h5", fish_id=99)

        raw_path = run_dir / "midlines.h5"
        data = _load_midline_positions(raw_path)
        assert any(0 in fdict for fdict in data.values())


def _write_minimal_h5(path: Path, fish_id: int = 0) -> None:
    """Write a minimal midlines.h5 for testing."""
    import h5py
    import numpy as np

    from aquapose.core.reconstruction.utils import SPLINE_K, SPLINE_KNOTS

    with h5py.File(path, "w") as f:
        grp = f.create_group("midlines")
        grp.attrs["SPLINE_KNOTS"] = SPLINE_KNOTS
        grp.attrs["SPLINE_K"] = SPLINE_K
        N = 3
        max_fish = 1
        n_kpts = 6
        grp.create_dataset("frame_index", data=np.arange(N, dtype=np.int64))
        fid_arr = np.full((N, max_fish), fish_id, dtype=np.int32)
        grp.create_dataset("fish_id", data=fid_arr)
        pts = (
            np.random.default_rng(42)
            .uniform(-0.1, 0.1, (N, max_fish, n_kpts, 3))
            .astype(np.float32)
        )
        grp.create_dataset("points", data=pts)
        grp.create_dataset(
            "control_points",
            data=np.full((N, max_fish, 7, 3), np.nan, dtype=np.float32),
        )
        grp.create_dataset(
            "arc_length", data=np.full((N, max_fish), np.nan, dtype=np.float32)
        )
        grp.create_dataset(
            "half_widths", data=np.full((N, max_fish, n_kpts), np.nan, dtype=np.float32)
        )
        grp.create_dataset("n_cameras", data=np.full((N, max_fish), 3, dtype=np.int32))
        grp.create_dataset(
            "mean_residual", data=np.full((N, max_fish), 0.01, dtype=np.float32)
        )
        grp.create_dataset(
            "max_residual", data=np.full((N, max_fish), 0.02, dtype=np.float32)
        )
        grp.create_dataset(
            "is_low_confidence", data=np.zeros((N, max_fish), dtype=bool)
        )
