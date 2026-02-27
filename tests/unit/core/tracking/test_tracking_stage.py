"""Interface tests for TrackingStage — Stage 4 of the AquaPose pipeline.

Validates:
- TrackingStage satisfies the Stage Protocol via structural typing
- run() correctly populates PipelineContext.tracks from context.associated_bundles
- Tracker state persists across frames (fish_id continuity)
- FishTrack objects have required attributes (fish_id, positions, state)
- Backend registry raises ValueError for unknown kinds
- Import boundary (ENG-07): no engine/ runtime imports in core/tracking/
- ValueError raised when context.associated_bundles is None (Stage 3 precondition)
"""

from __future__ import annotations

import importlib
import inspect
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aquapose.core.association.types import AssociationBundle
from aquapose.core.context import PipelineContext, Stage
from aquapose.core.tracking import FishTrack, TrackingStage, TrackState
from aquapose.core.tracking.backends import get_backend

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_tracking_stage_satisfies_protocol(tmp_path: Path) -> None:
    """TrackingStage is a Stage (structural typing) even before run() is called."""
    stage = _build_stage(tmp_path)
    assert isinstance(stage, Stage), (
        "TrackingStage must satisfy the Stage Protocol via structural typing"
    )


# ---------------------------------------------------------------------------
# Context population
# ---------------------------------------------------------------------------


def test_tracking_stage_populates_tracks(tmp_path: Path) -> None:
    """run() populates context.tracks from context.associated_bundles."""
    # Three frames of bundles
    synthetic_bundles: list[list[AssociationBundle]] = [
        [_make_bundle(0, [0.1, 0.0, 0.5]), _make_bundle(1, [0.5, 0.0, 0.5])],
        [_make_bundle(0, [0.1, 0.0, 0.5])],
        [],
    ]

    # Create one FishTrack per frame to return from the mock
    def _make_track(fish_id: int) -> FishTrack:
        track = FishTrack(fish_id=fish_id)
        track.positions.append(np.array([float(fish_id), 0.0, 0.5]))
        track.state = TrackState.CONFIRMED
        return track

    per_frame_tracks = [
        [_make_track(0), _make_track(1)],  # frame 0: 2 confirmed fish
        [_make_track(0)],  # frame 1: 1 confirmed fish
        [],  # frame 2: no confirmed fish
    ]

    stage = _build_stage(tmp_path, per_frame_tracks=per_frame_tracks)

    ctx = PipelineContext()
    ctx.associated_bundles = synthetic_bundles
    result = stage.run(ctx)

    assert result is ctx, "run() must return the same context object"
    assert ctx.tracks is not None
    assert isinstance(ctx.tracks, list)
    assert len(ctx.tracks) == len(synthetic_bundles), (
        "tracks must have one entry per frame"
    )

    # Validate per-frame structure
    assert len(ctx.tracks[0]) == 2, "frame 0 should have 2 tracks"
    assert len(ctx.tracks[1]) == 1, "frame 1 should have 1 track"
    assert len(ctx.tracks[2]) == 0, "frame 2 should have 0 tracks"


def test_tracking_requires_bundles(tmp_path: Path) -> None:
    """run() raises ValueError if context.associated_bundles is None."""
    stage = _build_stage(tmp_path)
    ctx = PipelineContext()
    # associated_bundles is None by default

    with pytest.raises(ValueError, match=r"associated_bundles"):
        stage.run(ctx)


def test_tracking_stage_no_detections_raises(tmp_path: Path) -> None:
    """run() raises ValueError if context.associated_bundles is not set.

    Confirms the error message references Stage 3 (AssociationStage).
    """
    stage = _build_stage(tmp_path)
    ctx = PipelineContext()

    with pytest.raises(ValueError, match=r"AssociationStage"):
        stage.run(ctx)


def test_tracking_stage_returns_same_context(tmp_path: Path) -> None:
    """run() returns the same context object it received (not a copy)."""
    stage = _build_stage(tmp_path, per_frame_tracks=[[]])

    ctx = PipelineContext()
    ctx.associated_bundles = [[_make_bundle(0, [0.0, 0.0, 0.5])]]
    result = stage.run(ctx)

    assert result is ctx


# ---------------------------------------------------------------------------
# State persistence across frames
# ---------------------------------------------------------------------------


def test_tracking_state_persists_across_frames(tmp_path: Path) -> None:
    """Tracker state persists: same stage instance produces continuity across frames.

    Runs multiple sequential run() calls (simulating streaming frames) and
    verifies that the backend.track_frame is called for each frame — i.e., the
    tracker is invoked per-frame without resetting between calls.
    """
    bundle = _make_bundle(0, [0.0, 0.0, 0.5])

    # Two runs, each with 2 frames
    run1_tracks = [
        [FishTrack(fish_id=0)],
        [FishTrack(fish_id=0), FishTrack(fish_id=1)],
    ]
    run2_tracks = [
        [FishTrack(fish_id=0), FishTrack(fish_id=1)],
        [FishTrack(fish_id=0), FishTrack(fish_id=1), FishTrack(fish_id=2)],
    ]

    all_tracks = run1_tracks + run2_tracks
    call_idx = [0]

    stage = _build_stage(tmp_path)
    mock_backend = stage._backend  # type: ignore[attr-defined]

    def _track_frame(**kwargs: object) -> list[FishTrack]:
        result = all_tracks[call_idx[0]]
        call_idx[0] += 1
        return result

    mock_backend.track_frame = MagicMock(side_effect=lambda **kw: _track_frame(**kw))

    # First run: 2 frames
    ctx1 = PipelineContext()
    ctx1.associated_bundles = [[bundle], [bundle]]
    stage.run(ctx1)

    # Second run: 2 more frames through the SAME stage instance
    ctx2 = PipelineContext()
    ctx2.associated_bundles = [[bundle], [bundle]]
    stage.run(ctx2)

    # Total calls: 4 (2 per run)
    assert mock_backend.track_frame.call_count == 4, (
        "track_frame should be called once per frame across both runs"
    )

    # Verify the growing fish count indicates state persistence
    assert len(ctx1.tracks[0]) == 1  # type: ignore[index]
    assert len(ctx1.tracks[1]) == 2  # type: ignore[index]
    assert len(ctx2.tracks[0]) == 2  # type: ignore[index]
    assert len(ctx2.tracks[1]) == 3  # type: ignore[index]


# ---------------------------------------------------------------------------
# FishTrack attribute validation
# ---------------------------------------------------------------------------


def test_tracks_have_required_attributes(tmp_path: Path) -> None:
    """FishTrack objects in output have fish_id, positions, and state attributes."""
    track = FishTrack(fish_id=42)
    track.positions.append(np.array([1.0, 2.0, 0.5]))
    track.state = TrackState.CONFIRMED

    stage = _build_stage(tmp_path, per_frame_tracks=[[track]])

    ctx = PipelineContext()
    ctx.associated_bundles = [[_make_bundle(0, [1.0, 2.0, 0.5])]]
    stage.run(ctx)

    assert ctx.tracks is not None
    assert len(ctx.tracks) == 1
    assert len(ctx.tracks[0]) == 1

    result_track = ctx.tracks[0][0]
    assert hasattr(result_track, "fish_id"), "FishTrack must have fish_id"
    assert hasattr(result_track, "positions"), "FishTrack must have positions"
    assert hasattr(result_track, "state"), "FishTrack must have state"
    assert result_track.fish_id == 42
    assert isinstance(result_track.positions, deque)
    assert result_track.state == TrackState.CONFIRMED


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


def test_backend_registry_unknown_raises() -> None:
    """get_backend raises ValueError for an unrecognized backend kind."""
    with pytest.raises(ValueError, match="Unknown tracking backend kind"):
        get_backend("nonexistent_backend")


def test_backend_registry_hungarian_constructs(tmp_path: Path) -> None:
    """get_backend('hungarian') constructs without error (mocked calibration)."""
    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    backend = get_backend("hungarian", calibration_path=str(calib_path))
    assert backend is not None


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------


_CORE_TRACKING_MODULES = [
    "aquapose.core.tracking",
    "aquapose.core.tracking.stage",
    "aquapose.core.tracking.types",
    "aquapose.core.tracking.backends",
    "aquapose.core.tracking.backends.hungarian",
]


def test_import_boundary_no_engine_imports() -> None:
    """No core/tracking/ module may import from aquapose.engine at module level.

    TYPE_CHECKING-guarded imports are permitted, but no runtime import of
    aquapose.engine is allowed (ENG-07).
    """
    for mod_name in _CORE_TRACKING_MODULES:
        module = importlib.import_module(mod_name)
        source = inspect.getsource(module)
        lines = source.splitlines()

        in_type_checking_block = False
        for line in lines:
            stripped = line.strip()

            # Detect entry into TYPE_CHECKING block
            if "TYPE_CHECKING" in stripped and "if" in stripped:
                in_type_checking_block = True
                continue

            # Exit TYPE_CHECKING block when we return to top-level indentation
            if in_type_checking_block and stripped and not line.startswith(" "):
                in_type_checking_block = False

            if not in_type_checking_block:
                assert "from aquapose.engine" not in stripped, (
                    f"{mod_name}: runtime import from aquapose.engine found: "
                    f"{line!r}. Use TYPE_CHECKING guard for annotation-only "
                    f"imports (ENG-07)."
                )
                assert stripped != "import aquapose.engine", (
                    f"{mod_name}: runtime import found: {line!r}. "
                    "Use TYPE_CHECKING guard (ENG-07)."
                )
                assert not stripped.startswith("import aquapose.engine."), (
                    f"{mod_name}: runtime import found: {line!r}. "
                    "Use TYPE_CHECKING guard (ENG-07)."
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundle(fish_idx: int, centroid: list[float]) -> AssociationBundle:
    """Create a minimal AssociationBundle for testing.

    Args:
        fish_idx: Per-frame fish index.
        centroid: 3D centroid as a list of 3 floats.

    Returns:
        An AssociationBundle with realistic attributes.
    """
    return AssociationBundle(
        fish_idx=fish_idx,
        centroid_3d=np.array(centroid, dtype=np.float32),
        camera_detections={"cam1": 0, "cam2": 0, "cam3": 0},
        n_cameras=3,
        reprojection_residual=2.5,
        confidence=1.0,
    )


def _build_stage(
    tmp_path: Path,
    per_frame_tracks: list[list[FishTrack]] | None = None,
) -> TrackingStage:
    """Build a TrackingStage with all heavy I/O mocked.

    The calibration loading is not patched — HungarianBackend now only checks
    that the calibration file exists (no longer loads models). The backend's
    ``track_frame`` is replaced with a mock that returns ``per_frame_tracks``
    entries in order.

    Args:
        tmp_path: Temporary directory for fake calibration file.
        per_frame_tracks: Per-frame track lists to return from the mock
            backend. If None, returns empty lists for each call.

    Returns:
        A configured TrackingStage ready for testing.
    """
    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    stage = TrackingStage(calibration_path=str(calib_path))

    # Replace backend.track_frame with a side_effect that returns per-frame data
    if per_frame_tracks is not None:
        call_results = iter(per_frame_tracks)
        stage._backend.track_frame = MagicMock(  # type: ignore[union-attr]
            side_effect=lambda **kw: next(call_results)
        )
    else:
        stage._backend.track_frame = MagicMock(return_value=[])  # type: ignore[union-attr]

    return stage
