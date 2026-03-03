"""Unit tests for ChunkHandoff, write_handoff, ChunkFrameSource, and PipelineConfig.chunk_size."""

import pickle

import numpy as np
import pytest

from aquapose.core.types.frame_source import ChunkFrameSource
from aquapose.engine.orchestrator import ChunkHandoff, write_handoff

# ---------------------------------------------------------------------------
# ChunkHandoff tests
# ---------------------------------------------------------------------------


def test_chunk_handoff_pickle_round_trip():
    handoff = ChunkHandoff(
        tracks_2d_state={"cam1": {"next_local_id": 5}},
        identity_map={0: 0, 1: 1},
        next_global_id=2,
    )
    data = pickle.dumps(handoff)
    restored = pickle.loads(data)
    assert restored.tracks_2d_state == handoff.tracks_2d_state
    assert restored.identity_map == handoff.identity_map
    assert restored.next_global_id == handoff.next_global_id


def test_chunk_handoff_frozen():
    handoff = ChunkHandoff(tracks_2d_state={}, identity_map={}, next_global_id=0)
    with pytest.raises((Exception,)):
        handoff.next_global_id = 99  # type: ignore


def test_write_handoff(tmp_path):
    dest = tmp_path / "handoff.pkl"
    handoff = ChunkHandoff(tracks_2d_state={}, identity_map={0: 5}, next_global_id=6)
    write_handoff(dest, handoff)
    assert dest.exists()
    restored = pickle.loads(dest.read_bytes())
    assert isinstance(restored, ChunkHandoff)
    assert restored.identity_map == {0: 5}
    assert restored.next_global_id == 6


# ---------------------------------------------------------------------------
# PipelineConfig.chunk_size tests
# ---------------------------------------------------------------------------


def test_pipeline_config_chunk_size_field():
    from aquapose.engine.config import PipelineConfig

    cfg = PipelineConfig(run_id="test", output_dir="/tmp", n_animals=1, chunk_size=1000)
    assert cfg.chunk_size == 1000


def test_pipeline_config_chunk_size_default_none():
    from aquapose.engine.config import PipelineConfig

    cfg = PipelineConfig(run_id="test", output_dir="/tmp", n_animals=1)
    assert cfg.chunk_size is None


# ---------------------------------------------------------------------------
# Mock VideoFrameSource for ChunkFrameSource tests
# ---------------------------------------------------------------------------


class _MockVideoFrameSource:
    """Minimal VideoFrameSource stand-in for testing ChunkFrameSource."""

    def __init__(self, n_frames: int, camera_ids: list):
        self._n = n_frames
        self._camera_ids = camera_ids

    @property
    def camera_ids(self) -> list:
        return list(self._camera_ids)

    def __len__(self) -> int:
        return self._n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def __iter__(self):
        for i in range(self._n):
            yield (
                i,
                {cam: np.zeros((4, 4, 3), dtype=np.uint8) for cam in self._camera_ids},
            )

    def read_frame(self, idx: int) -> dict:
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Frame {idx} out of range")
        return {
            cam: np.full((4, 4, 3), idx, dtype=np.uint8) for cam in self._camera_ids
        }


# ---------------------------------------------------------------------------
# ChunkFrameSource tests
# ---------------------------------------------------------------------------


def test_chunk_frame_source_iteration():
    source = _MockVideoFrameSource(n_frames=10, camera_ids=["cam1"])
    chunk = ChunkFrameSource(source, start_frame=3, end_frame=7)
    assert len(chunk) == 4
    assert chunk.global_frame_offset == 3
    frames_seen = list(chunk)
    assert len(frames_seen) == 4
    assert [fi for fi, _ in frames_seen] == [0, 1, 2, 3]
    assert frames_seen[0][1]["cam1"][0, 0, 0] == 3  # global idx 3
    assert frames_seen[3][1]["cam1"][0, 0, 0] == 6  # global idx 6


def test_chunk_frame_source_read_frame():
    source = _MockVideoFrameSource(n_frames=10, camera_ids=["cam1"])
    chunk = ChunkFrameSource(source, start_frame=5, end_frame=9)
    frames = chunk.read_frame(0)  # local 0 -> global 5
    assert frames["cam1"][0, 0, 0] == 5
    frames = chunk.read_frame(3)  # local 3 -> global 8
    assert frames["cam1"][0, 0, 0] == 8


def test_chunk_frame_source_context_manager_noop():
    source = _MockVideoFrameSource(n_frames=5, camera_ids=["cam1"])
    chunk = ChunkFrameSource(source, start_frame=0, end_frame=5)
    with chunk as c:
        assert c is chunk  # no-op: returns self
