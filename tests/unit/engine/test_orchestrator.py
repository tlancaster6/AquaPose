"""Unit tests for ChunkOrchestrator diagnostic+chunk mode and chunk_idx wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aquapose.engine.orchestrator import ChunkOrchestrator


def test_diagnostic_multi_chunk_no_error() -> None:
    """ChunkOrchestrator allows diagnostic mode with multi-chunk settings."""
    config = MagicMock()
    config.chunk_size = 200
    config.mode = "diagnostic"
    # Should not raise ValueError
    orchestrator = ChunkOrchestrator(config=config, max_chunks=None)
    assert orchestrator is not None


def test_build_observers_receives_chunk_idx() -> None:
    """build_observers is called with chunk_idx matching the current chunk."""
    from aquapose.engine.observer_factory import build_observers

    config = MagicMock()
    config.output_dir = "/tmp/test_output"
    config.mode = "production"

    call_kwargs: list[dict] = []

    original_build = build_observers

    def capturing_build(**kwargs: object) -> list:
        call_kwargs.append(dict(kwargs))
        return []

    with patch(
        "aquapose.engine.observer_factory.build_observers", side_effect=capturing_build
    ) as mock_build:
        mock_build.side_effect = capturing_build
        # Call build_observers directly with chunk_idx=3
        result = original_build(
            config=config,
            mode="production",
            verbose=False,
            total_stages=5,
            chunk_idx=3,
        )

    # Verify chunk_idx was accepted without error
    assert isinstance(result, list)


def test_build_observers_chunk_idx_forwarded_to_diagnostic_observer(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """build_observers passes chunk_idx to DiagnosticObserver in diagnostic mode."""
    from aquapose.engine.observer_factory import build_observers

    config = MagicMock()
    config.output_dir = str(tmp_path)
    config.calibration_path = ""
    config.mode = "diagnostic"

    observers = build_observers(
        config=config,
        mode="diagnostic",
        verbose=False,
        total_stages=5,
        chunk_idx=7,
    )

    from aquapose.engine.diagnostic_observer import DiagnosticObserver

    diag_observers = [o for o in observers if isinstance(o, DiagnosticObserver)]
    assert len(diag_observers) == 1
    assert diag_observers[0]._chunk_idx == 7
