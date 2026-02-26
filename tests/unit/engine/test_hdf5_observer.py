"""Unit tests for HDF5ExportObserver."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from aquapose.engine.events import PipelineComplete, PipelineStart
from aquapose.engine.hdf5_observer import HDF5ExportObserver
from aquapose.engine.observers import Observer


def _make_spline(seed: int = 0) -> SimpleNamespace:
    """Create a mock Spline3D with control_points and arc_length."""
    rng = np.random.RandomState(seed)
    return SimpleNamespace(
        control_points=rng.randn(7, 3).astype(np.float32),
        arc_length=float(rng.uniform(10.0, 50.0)),
    )


def _make_context(n_frames: int = 2, n_fish: int = 2) -> SimpleNamespace:
    """Create a mock PipelineContext with midlines_3d."""
    midlines_3d: list[dict[int, SimpleNamespace]] = []
    for frame_idx in range(n_frames):
        frame_dict: dict[int, SimpleNamespace] = {}
        for fish_id in range(n_fish):
            frame_dict[fish_id] = _make_spline(seed=frame_idx * 10 + fish_id)
        midlines_3d.append(frame_dict)
    return SimpleNamespace(midlines_3d=midlines_3d)


def test_hdf5_observer_satisfies_protocol(tmp_path: Path) -> None:
    """HDF5ExportObserver satisfies the Observer protocol."""
    observer = HDF5ExportObserver(output_dir=tmp_path)
    assert isinstance(observer, Observer)


def test_hdf5_writes_on_pipeline_complete(tmp_path: Path) -> None:
    """HDF5 file is written on PipelineComplete with correct structure."""
    observer = HDF5ExportObserver(output_dir=tmp_path)
    context = _make_context(n_frames=2, n_fish=2)

    observer.on_event(
        PipelineComplete(
            run_id="test_run",
            elapsed_seconds=5.0,
            context=context,
        )
    )

    output_path = tmp_path / "outputs.h5"
    assert output_path.exists()

    with h5py.File(str(output_path), "r") as f:
        assert "frames" in f
        assert "0000" in f["frames"]
        assert "fish_0" in f["frames"]["0000"]
        cp = f["frames"]["0000"]["fish_0"]["control_points"][:]
        assert cp.shape == (7, 3)
        assert cp.dtype == np.float32


def test_hdf5_frame_major_layout(tmp_path: Path) -> None:
    """All expected frame/fish groups exist and values match input."""
    observer = HDF5ExportObserver(output_dir=tmp_path)
    context = _make_context(n_frames=3, n_fish=2)

    observer.on_event(
        PipelineComplete(run_id="layout_test", elapsed_seconds=1.0, context=context)
    )

    with h5py.File(str(tmp_path / "outputs.h5"), "r") as f:
        for frame_idx in range(3):
            frame_name = f"{frame_idx:04d}"
            assert frame_name in f["frames"], f"Missing frame {frame_name}"
            for fish_id in range(2):
                fish_name = f"fish_{fish_id}"
                assert fish_name in f["frames"][frame_name]
                cp = f["frames"][frame_name][fish_name]["control_points"][:]
                expected = context.midlines_3d[frame_idx][fish_id].control_points
                np.testing.assert_array_almost_equal(cp, expected)


def test_hdf5_metadata_attributes(tmp_path: Path) -> None:
    """Root attrs include run_id, frame_count, and fish_ids."""
    observer = HDF5ExportObserver(output_dir=tmp_path)
    context = _make_context(n_frames=2, n_fish=3)

    observer.on_event(
        PipelineComplete(run_id="meta_test", elapsed_seconds=2.0, context=context)
    )

    with h5py.File(str(tmp_path / "outputs.h5"), "r") as f:
        assert f.attrs["run_id"] == "meta_test"
        assert f.attrs["frame_count"] == 2
        np.testing.assert_array_equal(f.attrs["fish_ids"], [0, 1, 2])


def test_hdf5_config_hash(tmp_path: Path) -> None:
    """Config hash root attr is a non-empty hex string when PipelineStart fires."""
    from aquapose.engine.config import PipelineConfig

    observer = HDF5ExportObserver(output_dir=tmp_path)
    config = PipelineConfig(run_id="hash_test")

    observer.on_event(PipelineStart(run_id="hash_test", config=config))
    observer.on_event(
        PipelineComplete(
            run_id="hash_test",
            elapsed_seconds=1.0,
            context=_make_context(n_frames=1, n_fish=1),
        )
    )

    with h5py.File(str(tmp_path / "outputs.h5"), "r") as f:
        config_hash = f.attrs["config_hash"]
        assert isinstance(config_hash, str)
        assert len(config_hash) == 32  # MD5 hex length


def test_hdf5_skips_if_no_context(tmp_path: Path) -> None:
    """No file written and no error when PipelineComplete has no context."""
    observer = HDF5ExportObserver(output_dir=tmp_path)
    observer.on_event(PipelineComplete(run_id="no_ctx", elapsed_seconds=1.0))

    assert not (tmp_path / "outputs.h5").exists()


def test_hdf5_empty_frames(tmp_path: Path) -> None:
    """File written with frame_count=0 when midlines_3d is empty list."""
    observer = HDF5ExportObserver(output_dir=tmp_path)
    context = SimpleNamespace(midlines_3d=[])

    observer.on_event(
        PipelineComplete(run_id="empty", elapsed_seconds=0.5, context=context)
    )

    output_path = tmp_path / "outputs.h5"
    assert output_path.exists()

    with h5py.File(str(output_path), "r") as f:
        assert f.attrs["frame_count"] == 0
