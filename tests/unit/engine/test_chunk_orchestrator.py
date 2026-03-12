"""Unit tests for ChunkOrchestrator identity stitching and boundary computation."""

from __future__ import annotations

import json
import pathlib
import types
import unittest.mock


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
        fish_tracklet_sets={5: frozenset({("cam1", 0)})},
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
        fish_tracklet_sets={},
        next_global_id=7,
    )
    tracklet = types.SimpleNamespace(camera_id="cam2", track_id=99)
    group = types.SimpleNamespace(fish_id=3, tracklets=[tracklet])
    identity_map, next_id = _stitch_identities(
        [group], prev_handoff=prev_handoff, next_global_id=7
    )
    assert identity_map == {3: 7}
    assert next_id == 8


def test_stitch_identities_set_overlap_picks_higher() -> None:
    """Group overlapping two previous fish picks the one with higher overlap."""
    from aquapose.engine.orchestrator import ChunkHandoff, _stitch_identities

    prev_handoff = ChunkHandoff(
        tracks_2d_state={},
        identity_map={},
        fish_tracklet_sets={
            2: frozenset({("cam1", 0), ("cam2", 0)}),
            99: frozenset({("cam3", 0)}),
        },
        next_global_id=100,
    )
    tracklets = [
        types.SimpleNamespace(camera_id="cam1", track_id=0),
        types.SimpleNamespace(camera_id="cam2", track_id=0),
        types.SimpleNamespace(camera_id="cam3", track_id=0),
    ]
    group = types.SimpleNamespace(fish_id=0, tracklets=tracklets)
    identity_map, _next_id = _stitch_identities(
        [group], prev_handoff=prev_handoff, next_global_id=100
    )
    # Overlap with fish 2 is 2 (cam1, cam2), overlap with fish 99 is 1 (cam3)
    assert identity_map == {0: 2}


def test_stitch_identities_collision_resolved() -> None:
    """Two groups overlapping the same previous fish — higher overlap wins, other gets fresh."""
    from aquapose.engine.orchestrator import ChunkHandoff, _stitch_identities

    prev_handoff = ChunkHandoff(
        tracks_2d_state={},
        identity_map={},
        fish_tracklet_sets={
            5: frozenset({("cam1", 0), ("cam2", 0), ("cam3", 0)}),
        },
        next_global_id=6,
    )
    # Group A: overlaps fish 5 by 2 tracklets
    group_a = types.SimpleNamespace(
        fish_id=0,
        tracklets=[
            types.SimpleNamespace(camera_id="cam1", track_id=0),
            types.SimpleNamespace(camera_id="cam2", track_id=0),
        ],
    )
    # Group B: overlaps fish 5 by 1 tracklet
    group_b = types.SimpleNamespace(
        fish_id=1,
        tracklets=[
            types.SimpleNamespace(camera_id="cam3", track_id=0),
        ],
    )
    identity_map, _next_id = _stitch_identities(
        [group_a, group_b], prev_handoff=prev_handoff, next_global_id=6
    )
    assert identity_map[0] == 5  # group A wins fish 5
    assert identity_map[1] == 6  # group B gets fresh ID
    assert len(set(identity_map.values())) == len(identity_map)  # all unique


def test_stitch_identities_bijection_invariant() -> None:
    """Multiple groups contending for same global_id — all output IDs are unique."""
    from aquapose.engine.orchestrator import ChunkHandoff, _stitch_identities

    prev_handoff = ChunkHandoff(
        tracks_2d_state={},
        identity_map={},
        fish_tracklet_sets={
            10: frozenset({("cam1", 0), ("cam2", 0)}),
        },
        next_global_id=11,
    )
    groups = [
        types.SimpleNamespace(
            fish_id=i,
            tracklets=[types.SimpleNamespace(camera_id=f"cam{i + 1}", track_id=0)],
        )
        for i in range(3)  # groups 0, 1, 2 — only 0 and 1 overlap fish 10
    ]
    identity_map, _next_id = _stitch_identities(
        groups, prev_handoff=prev_handoff, next_global_id=11
    )
    assert len(set(identity_map.values())) == len(identity_map)


def test_stitch_identities_no_overlap_gets_fresh() -> None:
    """Group with no tracklet overlap with any previous fish gets a fresh ID."""
    from aquapose.engine.orchestrator import ChunkHandoff, _stitch_identities

    prev_handoff = ChunkHandoff(
        tracks_2d_state={},
        identity_map={},
        fish_tracklet_sets={
            5: frozenset({("cam1", 0)}),
        },
        next_global_id=6,
    )
    group = types.SimpleNamespace(
        fish_id=0,
        tracklets=[types.SimpleNamespace(camera_id="cam9", track_id=99)],
    )
    identity_map, next_id = _stitch_identities(
        [group], prev_handoff=prev_handoff, next_global_id=6
    )
    assert identity_map == {0: 6}
    assert next_id == 7


def test_stitch_identities_loser_gets_fresh_not_second_choice() -> None:
    """Group overlapping two prev fish, both claimed by stronger contenders, gets fresh ID."""
    from aquapose.engine.orchestrator import ChunkHandoff, _stitch_identities

    prev_handoff = ChunkHandoff(
        tracks_2d_state={},
        identity_map={},
        fish_tracklet_sets={
            10: frozenset({("cam1", 0), ("cam2", 0), ("cam3", 0)}),
            20: frozenset({("cam4", 0), ("cam5", 0), ("cam6", 0)}),
        },
        next_global_id=21,
    )
    # Group A: strong overlap with fish 10 (3 tracklets)
    group_a = types.SimpleNamespace(
        fish_id=0,
        tracklets=[
            types.SimpleNamespace(camera_id="cam1", track_id=0),
            types.SimpleNamespace(camera_id="cam2", track_id=0),
            types.SimpleNamespace(camera_id="cam3", track_id=0),
        ],
    )
    # Group B: strong overlap with fish 20 (3 tracklets)
    group_b = types.SimpleNamespace(
        fish_id=1,
        tracklets=[
            types.SimpleNamespace(camera_id="cam4", track_id=0),
            types.SimpleNamespace(camera_id="cam5", track_id=0),
            types.SimpleNamespace(camera_id="cam6", track_id=0),
        ],
    )
    # Group C: weak overlap with both fish 10 and 20 (1 tracklet each)
    # but both are already claimed by stronger groups
    group_c = types.SimpleNamespace(
        fish_id=2,
        tracklets=[
            types.SimpleNamespace(camera_id="cam1", track_id=0),
            types.SimpleNamespace(camera_id="cam4", track_id=0),
        ],
    )
    identity_map, next_id = _stitch_identities(
        [group_a, group_b, group_c], prev_handoff=prev_handoff, next_global_id=21
    )
    assert identity_map[0] == 10  # group A → fish 10
    assert identity_map[1] == 20  # group B → fish 20
    assert identity_map[2] == 21  # group C → fresh (both prev fish claimed)
    assert next_id == 22


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
# Diagnostic + chunk mode co-existence (mutual exclusion removed in 54-01)
# ---------------------------------------------------------------------------


def test_diagnostic_mode_allows_multi_chunk() -> None:
    """diagnostic mode + chunk_size > 0 + max_chunks > 1 no longer raises ValueError."""
    from unittest.mock import MagicMock

    from aquapose.engine.orchestrator import ChunkOrchestrator

    config = MagicMock()
    config.chunk_size = 1000
    config.mode = "diagnostic"
    # Should NOT raise after mutual exclusion removed
    ChunkOrchestrator(config=config, max_chunks=None)


def test_diagnostic_mode_allows_single_chunk() -> None:
    """diagnostic mode + chunk_size=null (degenerate) is allowed."""
    from unittest.mock import MagicMock

    from aquapose.engine.orchestrator import ChunkOrchestrator

    config = MagicMock()
    config.chunk_size = None
    config.mode = "diagnostic"
    # Should NOT raise
    ChunkOrchestrator(config=config)


def test_diagnostic_mode_allows_max_chunks_1() -> None:
    """diagnostic mode + chunk_size > 0 + max_chunks=1 is allowed."""
    from unittest.mock import MagicMock

    from aquapose.engine.orchestrator import ChunkOrchestrator

    config = MagicMock()
    config.chunk_size = 1000
    config.mode = "diagnostic"
    # Should NOT raise
    ChunkOrchestrator(config=config, max_chunks=1)


# ---------------------------------------------------------------------------
# INTEG-03: Degenerate single-chunk, multi-chunk, and manifest start_frame
# ---------------------------------------------------------------------------


def test_degenerate_single_chunk_output(tmp_path: pathlib.Path) -> None:
    """INTEG-03: chunk_size=None — single boundary, offset 0, HDF5 written once.

    Mocks PosePipeline.run() to return a canned PipelineContext with one fish
    and verifies the orchestrator calls Midline3DWriter.write_frame with
    global_frame_idx=0 (no offset), and the identity_map is {0: 0} (first chunk,
    no remapping).
    """

    from aquapose.core.context import PipelineContext
    from aquapose.engine.orchestrator import ChunkOrchestrator

    # Minimal canned context: 1 frame, 1 fish with global_id 0
    fish_id = 0
    fake_midline = types.SimpleNamespace(
        fish_id=fish_id
    )  # must have fish_id (mutated by orchestrator)
    context = PipelineContext()
    context.frame_count = 1
    context.camera_ids = ["cam1"]
    context.midlines_3d = [{fish_id: fake_midline}]
    context.tracklet_groups = []  # no tracklets → fresh global IDs from _stitch_identities

    # Mock PosePipeline so no real stages run
    mock_writer = unittest.mock.MagicMock()
    mock_writer.__enter__ = unittest.mock.MagicMock(return_value=mock_writer)
    mock_writer.__exit__ = unittest.mock.MagicMock(return_value=False)

    mock_video_source = unittest.mock.MagicMock()
    mock_video_source.__len__ = unittest.mock.MagicMock(return_value=1)
    mock_video_source.__enter__ = unittest.mock.MagicMock(
        return_value=mock_video_source
    )
    mock_video_source.__exit__ = unittest.mock.MagicMock(return_value=False)

    mock_pipeline = unittest.mock.MagicMock()
    mock_pipeline.run.return_value = context

    config = unittest.mock.MagicMock()
    config.chunk_size = None
    config.mode = "production"
    config.n_animals = 2
    config.n_sample_points = 15
    config.output_dir = str(tmp_path)
    config.video_dir = str(tmp_path)
    config.calibration_path = str(tmp_path / "calib.json")

    with (
        unittest.mock.patch(
            "aquapose.engine.orchestrator._build_stages_for_chunk",
            return_value=[],
        ),
        unittest.mock.patch(
            "aquapose.engine.pipeline.PosePipeline",
            return_value=mock_pipeline,
        ),
        unittest.mock.patch(
            "aquapose.core.types.frame_source.VideoFrameSource",
            return_value=mock_video_source,
        ),
        unittest.mock.patch(
            "aquapose.io.midline_writer.Midline3DWriter",
            return_value=mock_writer,
        ),
        unittest.mock.patch(
            "aquapose.logging.setup_file_logging",
        ),
        unittest.mock.patch(
            "aquapose.engine.orchestrator.write_handoff",
        ),
        unittest.mock.patch(
            "aquapose.engine.observer_factory.build_observers",
            return_value=[],
        ),
    ):
        orchestrator = ChunkOrchestrator(config=config)
        orchestrator.run()

    # Single chunk → write_frame called exactly once with global_frame_idx=0
    mock_writer.write_frame.assert_called_once()
    call_args = mock_writer.write_frame.call_args
    global_frame_idx = call_args[0][0]
    assert global_frame_idx == 0, f"Expected global_frame_idx=0, got {global_frame_idx}"


def test_multi_chunk_mechanical_correctness(tmp_path: pathlib.Path) -> None:
    """INTEG-03: Multi-chunk — correct frame offsets, identity stitching, HDF5 writes.

    chunk_size=100, total_frames=200 produces two chunks: [0,100) and [100,200).
    Verifies:
    - Chunk 1: write_frame called with indices 0-0 (1 frame stub), fish global_id=0
    - Chunk 2: write_frame called with indices 100-100 (1 frame stub), same global_id
      when track IDs match across chunks
    - Handoff from chunk 1 carries identity state into chunk 2 stitching
    """

    from aquapose.core.association.types import TrackletGroup
    from aquapose.core.context import PipelineContext
    from aquapose.engine.orchestrator import ChunkOrchestrator

    # Chunk 1: one fish at local_id=0, one tracklet (cam1, track_id=0)
    tracklet_ch1 = types.SimpleNamespace(camera_id="cam1", track_id=0)
    group_ch1 = TrackletGroup(fish_id=0, tracklets=(tracklet_ch1,))
    ctx_ch1 = PipelineContext()
    ctx_ch1.frame_count = 1
    ctx_ch1.camera_ids = ["cam1"]
    ctx_ch1.midlines_3d = [{0: types.SimpleNamespace(fish_id=0)}]  # must have fish_id
    ctx_ch1.tracklet_groups = [group_ch1]

    # Chunk 2: same fish continues, local_id=0 should stitch to global_id=0
    tracklet_ch2 = types.SimpleNamespace(camera_id="cam1", track_id=0)
    group_ch2 = TrackletGroup(fish_id=0, tracklets=(tracklet_ch2,))
    ctx_ch2 = PipelineContext()
    ctx_ch2.frame_count = 1
    ctx_ch2.camera_ids = ["cam1"]
    ctx_ch2.midlines_3d = [{0: types.SimpleNamespace(fish_id=0)}]  # must have fish_id
    ctx_ch2.tracklet_groups = [group_ch2]

    # Side-effect: first call returns ch1, second returns ch2
    mock_pipeline = unittest.mock.MagicMock()
    mock_pipeline.run.side_effect = [ctx_ch1, ctx_ch2]

    mock_writer = unittest.mock.MagicMock()
    mock_writer.__enter__ = unittest.mock.MagicMock(return_value=mock_writer)
    mock_writer.__exit__ = unittest.mock.MagicMock(return_value=False)

    mock_video_source = unittest.mock.MagicMock()
    mock_video_source.__len__ = unittest.mock.MagicMock(return_value=200)
    mock_video_source.__enter__ = unittest.mock.MagicMock(
        return_value=mock_video_source
    )
    mock_video_source.__exit__ = unittest.mock.MagicMock(return_value=False)

    config = unittest.mock.MagicMock()
    config.chunk_size = 100
    config.mode = "production"
    config.n_animals = 2
    config.n_sample_points = 15
    config.output_dir = str(tmp_path)
    config.video_dir = str(tmp_path)
    config.calibration_path = str(tmp_path / "calib.json")

    with (
        unittest.mock.patch(
            "aquapose.engine.orchestrator._build_stages_for_chunk",
            return_value=[],
        ),
        unittest.mock.patch(
            "aquapose.engine.pipeline.PosePipeline",
            return_value=mock_pipeline,
        ),
        unittest.mock.patch(
            "aquapose.core.types.frame_source.VideoFrameSource",
            return_value=mock_video_source,
        ),
        unittest.mock.patch(
            "aquapose.io.midline_writer.Midline3DWriter",
            return_value=mock_writer,
        ),
        unittest.mock.patch(
            "aquapose.logging.setup_file_logging",
        ),
        unittest.mock.patch(
            "aquapose.engine.orchestrator.write_handoff",
        ),
        unittest.mock.patch(
            "aquapose.engine.observer_factory.build_observers",
            return_value=[],
        ),
    ):
        orchestrator = ChunkOrchestrator(config=config)
        orchestrator.run()

    # Two write_frame calls total (1 frame per chunk)
    assert mock_writer.write_frame.call_count == 2, (
        f"Expected 2 write_frame calls, got {mock_writer.write_frame.call_count}"
    )

    calls = mock_writer.write_frame.call_args_list
    # Chunk 1: global_frame_idx = 0 + 0 = 0
    assert calls[0][0][0] == 0, (
        f"Chunk 1 global_frame_idx should be 0, got {calls[0][0][0]}"
    )
    # Chunk 2: global_frame_idx = 100 + 0 = 100
    assert calls[1][0][0] == 100, (
        f"Chunk 2 global_frame_idx should be 100, got {calls[1][0][0]}"
    )

    # Identity stitching: chunk 2 fish (local_id=0, cam1/track_id=0) should
    # continue as global_id=0 (same track as chunk 1)
    ch2_remapped = calls[1][0][1]  # remapped dict passed to write_frame
    assert 0 in ch2_remapped, "Global fish_id=0 should be in chunk 2 remapped dict"


def test_diagnostic_observer_flushed_after_remap(tmp_path: pathlib.Path) -> None:
    """Integration: orchestrator remaps context IDs to global, then flushes diagnostic cache.

    Verifies that cache.pkl written by DiagnosticObserver contains globally-consistent
    fish IDs (not chunk-local IDs) in both midlines_3d and tracklet_groups.
    """
    import pickle

    from aquapose.core.association.types import TrackletGroup
    from aquapose.core.context import PipelineContext
    from aquapose.engine.diagnostic_observer import DiagnosticObserver
    from aquapose.engine.orchestrator import ChunkOrchestrator

    # Local fish_id=5 will be stitched to global_id=0 (first chunk, fresh ID)
    # Use SimpleNamespace (picklable, mutable) to simulate Midline3D
    local_fish_id = 5
    fake_midline = types.SimpleNamespace(fish_id=local_fish_id)

    tracklet = types.SimpleNamespace(camera_id="cam1", track_id=0)
    group = TrackletGroup(
        fish_id=local_fish_id,
        tracklets=(tracklet,),
        confidence=None,
        per_frame_confidence=None,
        consensus_centroids=None,
    )

    context = PipelineContext()
    context.frame_count = 1
    context.camera_ids = ["cam1"]
    context.midlines_3d = [{local_fish_id: fake_midline}]
    context.tracklet_groups = [group]

    diag_observer = DiagnosticObserver(output_dir=tmp_path, chunk_idx=0, chunk_start=0)

    def _mock_pipeline_run(initial_context: object = None) -> PipelineContext:
        """Simulate pipeline: fire events then return context."""
        from aquapose.engine.events import PipelineComplete, PipelineStart

        diag_observer.on_event(PipelineStart(run_id="test-remap-run"))
        diag_observer.on_event(PipelineComplete(context=context))
        return context

    mock_pipeline = unittest.mock.MagicMock()
    mock_pipeline.run.side_effect = _mock_pipeline_run

    mock_writer = unittest.mock.MagicMock()
    mock_writer.__enter__ = unittest.mock.MagicMock(return_value=mock_writer)
    mock_writer.__exit__ = unittest.mock.MagicMock(return_value=False)

    mock_video_source = unittest.mock.MagicMock()
    mock_video_source.__len__ = unittest.mock.MagicMock(return_value=1)
    mock_video_source.__enter__ = unittest.mock.MagicMock(
        return_value=mock_video_source
    )
    mock_video_source.__exit__ = unittest.mock.MagicMock(return_value=False)

    config = unittest.mock.MagicMock()
    config.chunk_size = None
    config.mode = "diagnostic"
    config.n_animals = 2
    config.n_sample_points = 15
    config.output_dir = str(tmp_path)
    config.video_dir = str(tmp_path)
    config.calibration_path = str(tmp_path / "calib.json")

    with (
        unittest.mock.patch(
            "aquapose.engine.orchestrator._build_stages_for_chunk",
            return_value=[],
        ),
        unittest.mock.patch(
            "aquapose.engine.pipeline.PosePipeline",
            return_value=mock_pipeline,
        ),
        unittest.mock.patch(
            "aquapose.core.types.frame_source.VideoFrameSource",
            return_value=mock_video_source,
        ),
        unittest.mock.patch(
            "aquapose.io.midline_writer.Midline3DWriter",
            return_value=mock_writer,
        ),
        unittest.mock.patch(
            "aquapose.logging.setup_file_logging",
        ),
        unittest.mock.patch(
            "aquapose.engine.orchestrator.write_handoff",
        ),
        unittest.mock.patch(
            "aquapose.engine.observer_factory.build_observers",
            return_value=[diag_observer],
        ),
    ):
        orchestrator = ChunkOrchestrator(config=config)
        orchestrator.run()

    # Verify cache was written
    cache_path = tmp_path / "diagnostics" / "chunk_000" / "cache.pkl"
    assert cache_path.exists(), "cache.pkl should have been written by orchestrator"

    envelope = pickle.loads(cache_path.read_bytes())
    cached_ctx = envelope["context"]

    # midlines_3d should have global fish_id (0) as key, not local (5)
    assert len(cached_ctx.midlines_3d) == 1
    frame_midlines = cached_ctx.midlines_3d[0]
    assert 0 in frame_midlines, (
        f"Global fish_id=0 should be key in midlines_3d, got keys: {list(frame_midlines.keys())}"
    )
    assert local_fish_id not in frame_midlines, (
        f"Local fish_id={local_fish_id} should NOT be key in midlines_3d"
    )
    # Midline3D.fish_id should also be updated
    assert frame_midlines[0].fish_id == 0, (
        f"Midline3D.fish_id should be global 0, got {frame_midlines[0].fish_id}"
    )

    # tracklet_groups should have global fish_id (0)
    assert len(cached_ctx.tracklet_groups) == 1
    assert cached_ctx.tracklet_groups[0].fish_id == 0, (
        f"TrackletGroup.fish_id should be global 0, got {cached_ctx.tracklet_groups[0].fish_id}"
    )


def test_manifest_start_frame(tmp_path: pathlib.Path) -> None:
    """INTEG-03: DiagnosticObserver with chunk_start=500 writes start_frame=500 in manifest."""

    from aquapose.core.context import PipelineContext
    from aquapose.engine.diagnostic_observer import DiagnosticObserver
    from aquapose.engine.events import PipelineComplete, PipelineStart

    observer = DiagnosticObserver(
        output_dir=tmp_path,
        chunk_idx=2,
        chunk_start=500,
    )

    # Simulate PipelineStart so run_id is set
    observer.on_event(PipelineStart(run_id="test-run"))

    # Build a minimal context
    ctx = PipelineContext()
    ctx.frame_count = 100

    # Simulate PipelineComplete (no longer writes to disk — deferred to flush_cache)
    observer.on_event(
        PipelineComplete(run_id="test-run", elapsed_seconds=1.0, context=ctx)
    )

    # Flush the cache (orchestrator does this after identity stitching)
    observer.flush_cache()

    manifest_path = tmp_path / "diagnostics" / "manifest.json"
    assert manifest_path.exists(), (
        "manifest.json should have been written after flush_cache()"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks", [])
    assert len(chunks) == 1, f"Expected 1 chunk entry, got {len(chunks)}"

    chunk_entry = chunks[0]
    assert chunk_entry["index"] == 2, (
        f"Expected chunk index=2, got {chunk_entry['index']}"
    )
    assert chunk_entry["start_frame"] == 500, (
        f"Expected start_frame=500, got {chunk_entry['start_frame']}"
    )
