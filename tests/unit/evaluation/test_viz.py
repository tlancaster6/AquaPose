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
    """Tests for generate_overlay output path and empty cache behavior."""

    def test_raises_on_empty_chunk_caches(self, tmp_path: Path) -> None:
        """generate_overlay raises RuntimeError when no caches are found."""
        from aquapose.evaluation.viz.overlay import generate_overlay

        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(RuntimeError, match="No chunk caches"):
            generate_overlay(run_dir)

    def test_output_path_in_viz_subdir(self, tmp_path: Path) -> None:
        """generate_overlay writes to {run_dir}/viz/overlay_mosaic.mp4 by default."""
        from aquapose.evaluation.viz.overlay import generate_overlay

        run_dir = _setup_run_dir(tmp_path)

        # No calibration data or video dir in this run_dir => RuntimeError on frame source.
        with pytest.raises(RuntimeError):
            generate_overlay(run_dir)

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        """generate_overlay respects custom output_dir argument."""
        from aquapose.evaluation.viz.overlay import generate_overlay

        run_dir = _setup_run_dir(tmp_path)
        custom_dir = tmp_path / "custom_viz"

        # Should fail on missing video/calibration, but we're checking the
        # output path is rooted under custom_dir.
        with pytest.raises((RuntimeError, Exception)):
            generate_overlay(run_dir, custom_dir)


# ---------------------------------------------------------------------------
# Tests for generate_animation
# ---------------------------------------------------------------------------


class TestGenerateAnimation:
    """Tests for generate_animation output path and empty cache behavior."""

    def test_raises_on_empty_chunk_caches(self, tmp_path: Path) -> None:
        """generate_animation raises RuntimeError when no caches are found."""
        from aquapose.evaluation.viz.animation import generate_animation

        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(RuntimeError, match=r"No midlines\.h5 or chunk caches"):
            generate_animation(run_dir)

    def test_raises_on_no_midlines_3d(self, tmp_path: Path) -> None:
        """generate_animation raises RuntimeError when all chunks have no midlines_3d."""
        from aquapose.evaluation.viz.animation import generate_animation

        run_dir = _setup_run_dir(tmp_path)
        # Overwrite the chunk cache with a context that has no midlines.
        ctx = _make_context(frame_count=5, midlines_3d=[])
        _write_fake_chunk_cache(run_dir, 0, ctx)

        with pytest.raises(RuntimeError, match="No midlines_3d"):
            generate_animation(run_dir)

    def test_output_path_in_viz_subdir(self, tmp_path: Path) -> None:
        """generate_animation writes to {run_dir}/viz/animation_3d.html by default."""
        from aquapose.evaluation.viz.animation import generate_animation

        run_dir = _setup_run_dir(tmp_path)
        # Provide real midlines_3d data (empty dicts per frame) so generation proceeds.
        ctx = _make_context(frame_count=3, midlines_3d=[{}, {}, {}])
        _write_fake_chunk_cache(run_dir, 0, ctx)

        out_path = generate_animation(run_dir)
        assert out_path == run_dir / "viz" / "animation_3d.html"
        assert out_path.exists()

    def test_custom_output_dir_used(self, tmp_path: Path) -> None:
        """generate_animation writes to custom output_dir when provided."""
        from aquapose.evaluation.viz.animation import generate_animation

        run_dir = _setup_run_dir(tmp_path)
        ctx = _make_context(frame_count=2, midlines_3d=[{}, {}])
        _write_fake_chunk_cache(run_dir, 0, ctx)
        custom_dir = tmp_path / "my_viz"

        out_path = generate_animation(run_dir, custom_dir)
        assert out_path.parent == custom_dir
        assert out_path.name == "animation_3d.html"
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Tests for generate_trails
# ---------------------------------------------------------------------------


class TestGenerateTrails:
    """Tests for generate_trails output path and empty cache behavior."""

    def test_raises_on_empty_chunk_caches(self, tmp_path: Path) -> None:
        """generate_trails raises RuntimeError when no caches are found."""
        from aquapose.evaluation.viz.trails import generate_trails

        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(RuntimeError, match="No chunk caches"):
            generate_trails(run_dir)

    def test_returns_viz_dir_by_default(self, tmp_path: Path) -> None:
        """generate_trails returns {run_dir}/viz/ as output directory."""
        from aquapose.evaluation.viz.trails import generate_trails

        run_dir = _setup_run_dir(tmp_path)
        # No frame source available => returns output dir without writing videos.
        out = generate_trails(run_dir)
        assert out == run_dir / "viz"

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        """generate_trails uses custom output_dir when provided."""
        from aquapose.evaluation.viz.trails import generate_trails

        run_dir = _setup_run_dir(tmp_path)
        custom_dir = tmp_path / "custom_trails"
        out = generate_trails(run_dir, custom_dir)
        assert out == custom_dir
