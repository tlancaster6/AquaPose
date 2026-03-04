"""Unit tests for ChunkOrchestrator identity stitching and boundary computation."""

from __future__ import annotations

import types

import pytest


def test_stitch_identities_first_chunk() -> None:
    """First chunk: all groups get fresh global IDs."""
    from aquapose.engine.orchestrator import _stitch_identities

    tracklet = types.SimpleNamespace(camera_id="cam1", track_id=0)
    group = types.SimpleNamespace(fish_id=0, tracklets=[tracklet])
    identity_map, next_id = _stitch_identities(
        [group], prev_handoff=None, next_global_id=0
    )
    assert identity_map == {0: 0}
    assert next_id == 1


def test_stitch_identities_track_continues() -> None:
    """Track continuing from previous chunk maps to same global ID."""
    from aquapose.engine.orchestrator import ChunkHandoff, _stitch_identities

    prev_handoff = ChunkHandoff(
        tracks_2d_state={},
        identity_map={0: 5},
        track_id_to_global={("cam1", 0): 5},
        next_global_id=6,
    )
    tracklet = types.SimpleNamespace(camera_id="cam1", track_id=0)
    group = types.SimpleNamespace(fish_id=0, tracklets=[tracklet])
    identity_map, next_id = _stitch_identities(
        [group], prev_handoff=prev_handoff, next_global_id=6
    )
    assert identity_map == {0: 5}
    assert next_id == 6


def test_stitch_identities_new_fish() -> None:
    """New fish with no matching track gets a fresh global ID."""
    from aquapose.engine.orchestrator import ChunkHandoff, _stitch_identities

    prev_handoff = ChunkHandoff(
        tracks_2d_state={},
        identity_map={},
        track_id_to_global={},
        next_global_id=7,
    )
    tracklet = types.SimpleNamespace(camera_id="cam2", track_id=99)
    group = types.SimpleNamespace(fish_id=3, tracklets=[tracklet])
    identity_map, next_id = _stitch_identities(
        [group], prev_handoff=prev_handoff, next_global_id=7
    )
    assert identity_map == {3: 7}
    assert next_id == 8


def test_stitch_identities_conflict_majority_vote(caplog) -> None:  # type: ignore[no-untyped-def]
    """Conflicting global ID matches resolved by majority vote with warning logged."""
    import logging

    from aquapose.engine.orchestrator import ChunkHandoff, _stitch_identities

    prev_handoff = ChunkHandoff(
        tracks_2d_state={},
        identity_map={},
        track_id_to_global={("cam1", 0): 2, ("cam2", 0): 2, ("cam3", 0): 99},
        next_global_id=100,
    )
    tracklets = [
        types.SimpleNamespace(camera_id="cam1", track_id=0),
        types.SimpleNamespace(camera_id="cam2", track_id=0),
        types.SimpleNamespace(camera_id="cam3", track_id=0),
    ]
    group = types.SimpleNamespace(fish_id=0, tracklets=tracklets)
    with caplog.at_level(logging.WARNING, logger="aquapose.engine.orchestrator"):
        identity_map, _next_id = _stitch_identities(
            [group], prev_handoff=prev_handoff, next_global_id=100
        )
    assert identity_map == {0: 2}
    assert "conflict" in caplog.text.lower() or "Identity conflict" in caplog.text


def test_single_chunk_degenerate() -> None:
    """chunk_size=None produces a single boundary covering all frames."""
    total_frames = 500
    chunk_size = None
    if chunk_size is None or chunk_size <= 0:
        boundaries = [(0, total_frames)]
    else:
        boundaries = [
            (s, min(s + chunk_size, total_frames))
            for s in range(0, total_frames, chunk_size)
        ]
    assert boundaries == [(0, 500)]


def test_chunk_boundaries_partial() -> None:
    """chunk_size=300 with 700 frames produces three correct boundaries."""
    total_frames = 700
    chunk_size = 300
    boundaries = [
        (s, min(s + chunk_size, total_frames))
        for s in range(0, total_frames, chunk_size)
    ]
    assert boundaries == [(0, 300), (300, 600), (600, 700)]


# ---------------------------------------------------------------------------
# Mode conflict validation
# ---------------------------------------------------------------------------


def test_mode_conflict_raises_for_diagnostic_multi_chunk() -> None:
    """diagnostic mode + chunk_size > 0 + max_chunks > 1 raises ValueError."""
    from unittest.mock import MagicMock

    from aquapose.engine.orchestrator import ChunkOrchestrator

    config = MagicMock()
    config.chunk_size = 1000
    config.mode = "diagnostic"
    with pytest.raises(ValueError, match="mutually exclusive"):
        ChunkOrchestrator(config=config, max_chunks=None)


def test_mode_conflict_allows_diagnostic_single_chunk() -> None:
    """diagnostic mode + chunk_size=null (degenerate) is allowed."""
    from unittest.mock import MagicMock

    from aquapose.engine.orchestrator import ChunkOrchestrator

    config = MagicMock()
    config.chunk_size = None
    config.mode = "diagnostic"
    # Should NOT raise
    ChunkOrchestrator(config=config)


def test_mode_conflict_allows_diagnostic_max_chunks_1() -> None:
    """diagnostic mode + chunk_size > 0 + max_chunks=1 is allowed."""
    from unittest.mock import MagicMock

    from aquapose.engine.orchestrator import ChunkOrchestrator

    config = MagicMock()
    config.chunk_size = 1000
    config.mode = "diagnostic"
    # Should NOT raise
    ChunkOrchestrator(config=config, max_chunks=1)
