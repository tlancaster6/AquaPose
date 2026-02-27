"""Unit tests for HDF5ExportObserver."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from aquapose.engine.events import PipelineComplete, PipelineStart
from aquapose.engine.hdf5_observer import HDF5ExportObserver
from aquapose.engine.observers import Observer


def _make_spline(seed: int = 0, is_low_confidence: bool = False) -> SimpleNamespace:
    """Create a mock Spline3D with control_points and arc_length."""
    rng = np.random.RandomState(seed)
    return SimpleNamespace(
        control_points=rng.randn(7, 3).astype(np.float32),
        arc_length=float(rng.uniform(10.0, 50.0)),
        is_low_confidence=is_low_confidence,
    )


def _make_context(
    n_frames: int = 2,
    n_fish: int = 2,
    tracklet_groups: list | None = None,
) -> SimpleNamespace:
    """Create a mock PipelineContext with midlines_3d."""
    midlines_3d: list[dict[int, SimpleNamespace]] = []
    for frame_idx in range(n_frames):
        frame_dict: dict[int, SimpleNamespace] = {}
        for fish_id in range(n_fish):
            frame_dict[fish_id] = _make_spline(seed=frame_idx * 10 + fish_id)
        midlines_3d.append(frame_dict)
    return SimpleNamespace(midlines_3d=midlines_3d, tracklet_groups=tracklet_groups)


def _make_tracklet_group(fish_id: int, frames: tuple[int, ...]) -> SimpleNamespace:
    """Create a mock TrackletGroup with tracklets."""
    tracklet = SimpleNamespace(
        camera_id="cam1",
        track_id=fish_id,
        frames=frames,
        centroids=tuple((50.0, 50.0) for _ in frames),
        bboxes=tuple((40.0, 40.0, 20.0, 20.0) for _ in frames),
        frame_status=tuple("detected" for _ in frames),
    )
    return SimpleNamespace(
        fish_id=fish_id,
        tracklets=(tracklet,),
        confidence=0.9,
        per_frame_confidence=None,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_hdf5_observer_satisfies_protocol(tmp_path: Path) -> None:
    """HDF5ExportObserver satisfies the Observer protocol."""
    observer = HDF5ExportObserver(output_dir=tmp_path)
    assert isinstance(observer, Observer)


# ---------------------------------------------------------------------------
# Frame-major layout (legacy, when tracklet_groups absent)
# ---------------------------------------------------------------------------


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
    context = SimpleNamespace(midlines_3d=[], tracklet_groups=None)

    observer.on_event(
        PipelineComplete(run_id="empty", elapsed_seconds=0.5, context=context)
    )

    output_path = tmp_path / "outputs.h5"
    assert output_path.exists()

    with h5py.File(str(output_path), "r") as f:
        assert f.attrs["frame_count"] == 0


# ---------------------------------------------------------------------------
# Fish-first layout (v2.1, when tracklet_groups present)
# ---------------------------------------------------------------------------


def test_hdf5_fish_first_layout(tmp_path: Path) -> None:
    """Fish-first layout has /fish_{id}/spline_controls and /fish_{id}/confidence."""
    groups = [
        _make_tracklet_group(0, (0, 1)),
        _make_tracklet_group(1, (0, 1)),
    ]
    context = _make_context(n_frames=2, n_fish=2, tracklet_groups=groups)

    observer = HDF5ExportObserver(output_dir=tmp_path)
    observer.on_event(
        PipelineComplete(run_id="fish_first", elapsed_seconds=1.0, context=context)
    )

    with h5py.File(str(tmp_path / "outputs.h5"), "r") as f:
        assert f.attrs["layout"] == "fish_first"
        assert "fish_0" in f
        assert "fish_1" in f
        assert "spline_controls" in f["fish_0"]
        assert "confidence" in f["fish_0"]

        sc = f["fish_0"]["spline_controls"][:]
        assert sc.shape == (2, 7, 3)
        assert sc.dtype == np.float32

        conf = f["fish_0"]["confidence"][:]
        assert conf.shape == (2,)
        assert conf.dtype == np.float32


def test_hdf5_fish_first_confidence_values(tmp_path: Path) -> None:
    """Interpolated frames (is_low_confidence=True) have confidence=0."""
    groups = [_make_tracklet_group(0, (0, 1, 2))]

    # Frame 1 is interpolated
    midlines_3d = [
        {0: _make_spline(seed=0, is_low_confidence=False)},
        {0: _make_spline(seed=1, is_low_confidence=True)},
        {0: _make_spline(seed=2, is_low_confidence=False)},
    ]
    context = SimpleNamespace(midlines_3d=midlines_3d, tracklet_groups=groups)

    observer = HDF5ExportObserver(output_dir=tmp_path)
    observer.on_event(
        PipelineComplete(run_id="conf_test", elapsed_seconds=1.0, context=context)
    )

    with h5py.File(str(tmp_path / "outputs.h5"), "r") as f:
        conf = f["fish_0"]["confidence"][:]
        assert conf[0] == 1.0  # Normal frame
        assert conf[1] == 0.0  # Interpolated
        assert conf[2] == 1.0  # Normal frame


def test_hdf5_fish_first_root_attributes(tmp_path: Path) -> None:
    """Fish-first layout includes root attributes: config_hash, run_timestamp."""
    from aquapose.engine.config import PipelineConfig

    groups = [_make_tracklet_group(0, (0,))]
    context = _make_context(n_frames=1, n_fish=1, tracklet_groups=groups)
    config = PipelineConfig(
        run_id="root_attr_test", calibration_path="/fake/calib.json"
    )

    observer = HDF5ExportObserver(output_dir=tmp_path)
    observer.on_event(PipelineStart(run_id="root_attr_test", config=config))
    observer.on_event(
        PipelineComplete(run_id="root_attr_test", elapsed_seconds=1.0, context=context)
    )

    with h5py.File(str(tmp_path / "outputs.h5"), "r") as f:
        assert "config_hash" in f.attrs
        assert len(f.attrs["config_hash"]) == 32
        assert "run_timestamp" in f.attrs
        assert "calibration_path" in f.attrs
        assert f.attrs["calibration_path"] == "/fake/calib.json"


def test_hdf5_fish_first_empty_midlines(tmp_path: Path) -> None:
    """Fish-first layout with empty midlines produces valid HDF5."""
    groups = [_make_tracklet_group(0, (0,))]
    midlines_3d: list[dict] = [{}]
    context = SimpleNamespace(midlines_3d=midlines_3d, tracklet_groups=groups)

    observer = HDF5ExportObserver(output_dir=tmp_path)
    observer.on_event(
        PipelineComplete(run_id="empty_fish", elapsed_seconds=1.0, context=context)
    )

    output_path = tmp_path / "outputs.h5"
    assert output_path.exists()

    with h5py.File(str(output_path), "r") as f:
        assert f.attrs["frame_count"] == 1
        assert f.attrs["layout"] == "fish_first"


def test_hdf5_backward_compat_no_tracklet_groups(tmp_path: Path) -> None:
    """When tracklet_groups is None, falls back to frame-major layout."""
    context = _make_context(n_frames=2, n_fish=1, tracklet_groups=None)

    observer = HDF5ExportObserver(output_dir=tmp_path)
    observer.on_event(
        PipelineComplete(run_id="compat", elapsed_seconds=1.0, context=context)
    )

    with h5py.File(str(tmp_path / "outputs.h5"), "r") as f:
        assert "frames" in f
        assert "fish_0" not in f  # No fish-first groups
        assert "0000" in f["frames"]


def test_hdf5_fish_first_spline_values_match(tmp_path: Path) -> None:
    """Fish-first spline_controls values match the input Midline3D."""
    groups = [_make_tracklet_group(0, (0,))]
    spline = _make_spline(seed=42)
    midlines_3d = [{0: spline}]
    context = SimpleNamespace(midlines_3d=midlines_3d, tracklet_groups=groups)

    observer = HDF5ExportObserver(output_dir=tmp_path)
    observer.on_event(
        PipelineComplete(run_id="values", elapsed_seconds=1.0, context=context)
    )

    with h5py.File(str(tmp_path / "outputs.h5"), "r") as f:
        sc = f["fish_0"]["spline_controls"][0]
        np.testing.assert_array_almost_equal(sc, spline.control_points)


def test_hdf5_fish_first_missing_frames_are_nan(tmp_path: Path) -> None:
    """Frames where a fish is absent have NaN in spline_controls and confidence."""
    groups = [_make_tracklet_group(0, (0, 1, 2))]
    # Fish 0 only in frames 0 and 2, missing in frame 1
    midlines_3d = [
        {0: _make_spline(seed=0)},
        {},
        {0: _make_spline(seed=2)},
    ]
    context = SimpleNamespace(midlines_3d=midlines_3d, tracklet_groups=groups)

    observer = HDF5ExportObserver(output_dir=tmp_path)
    observer.on_event(
        PipelineComplete(run_id="nan_test", elapsed_seconds=1.0, context=context)
    )

    with h5py.File(str(tmp_path / "outputs.h5"), "r") as f:
        sc = f["fish_0"]["spline_controls"][:]
        conf = f["fish_0"]["confidence"][:]

        # Frame 0 and 2: valid
        assert not np.any(np.isnan(sc[0]))
        assert not np.any(np.isnan(sc[2]))
        assert not np.isnan(conf[0])
        assert not np.isnan(conf[2])

        # Frame 1: NaN
        assert np.all(np.isnan(sc[1]))
        assert np.isnan(conf[1])
