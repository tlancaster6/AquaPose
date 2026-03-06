"""Unit tests for the OOM retry utility in core.inference."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aquapose.core.inference import BatchState, predict_with_oom_retry


class _FakeCudaOOM(RuntimeError):
    """Stand-in for torch.cuda.OutOfMemoryError when CUDA is unavailable."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_predict(items: list) -> list:
    """Pass-through predict_fn that returns its input unchanged."""
    return list(items)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictWithOomRetry:
    """Tests for predict_with_oom_retry."""

    def test_no_oom_correct_chunk_sizes(self) -> None:
        """predict_fn is called on all inputs in correct chunk sizes."""
        calls: list[list[int]] = []

        def tracking_predict(items: list[int]) -> list[int]:
            calls.append(list(items))
            return [x * 2 for x in items]

        state = BatchState()
        result = predict_with_oom_retry(
            tracking_predict, [1, 2, 3, 4, 5], max_batch_size=2, state=state
        )

        assert result == [2, 4, 6, 8, 10]
        assert calls == [[1, 2], [3, 4], [5]]

    def test_batch_size_zero_sends_all(self) -> None:
        """batch_size=0 sends all inputs in one call."""
        calls: list[list[int]] = []

        def tracking_predict(items: list[int]) -> list[int]:
            calls.append(list(items))
            return items

        state = BatchState()
        inputs = [1, 2, 3, 4, 5]
        predict_with_oom_retry(tracking_predict, inputs, max_batch_size=0, state=state)

        assert len(calls) == 1
        assert calls[0] == inputs

    @patch("aquapose.core.inference.torch")
    def test_cuda_oom_error_halves_batch(self, mock_torch: MagicMock) -> None:
        """When predict_fn raises torch.cuda.OutOfMemoryError, batch halves and retry succeeds."""
        mock_torch.cuda.OutOfMemoryError = _FakeCudaOOM
        mock_torch.cuda.empty_cache = MagicMock()

        call_count = 0

        def oom_then_succeed(items: list[int]) -> list[int]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeCudaOOM("CUDA out of memory")
            return [x * 10 for x in items]

        state = BatchState()
        result = predict_with_oom_retry(
            oom_then_succeed, [1, 2, 3, 4], max_batch_size=4, state=state
        )

        # After halving from 4 -> 2, should process in chunks of 2
        assert result == [10, 20, 30, 40]
        assert state.effective_batch_size == 2
        mock_torch.cuda.empty_cache.assert_called()

    @patch("aquapose.core.inference.torch")
    def test_runtime_error_cuda_oom_message_halves(self, mock_torch: MagicMock) -> None:
        """RuntimeError with 'CUDA out of memory' triggers halving."""
        mock_torch.cuda.OutOfMemoryError = _FakeCudaOOM
        mock_torch.cuda.empty_cache = MagicMock()

        call_count = 0

        def oom_runtime(items: list[int]) -> list[int]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("CUDA out of memory: tried to allocate 2GiB")
            return items

        state = BatchState()
        result = predict_with_oom_retry(
            oom_runtime, [1, 2, 3, 4], max_batch_size=4, state=state
        )

        assert result == [1, 2, 3, 4]
        assert state.effective_batch_size == 2

    @patch("aquapose.core.inference.torch")
    def test_non_cuda_runtime_error_propagates(self, mock_torch: MagicMock) -> None:
        """RuntimeError NOT containing 'CUDA out of memory' propagates immediately."""
        mock_torch.cuda.OutOfMemoryError = _FakeCudaOOM

        def bad_predict(items: list[int]) -> list[int]:
            raise RuntimeError("some other error")

        state = BatchState()
        with pytest.raises(RuntimeError, match="some other error"):
            predict_with_oom_retry(bad_predict, [1, 2], max_batch_size=2, state=state)

    @patch("aquapose.core.inference.torch")
    def test_batch_size_one_oom_propagates(self, mock_torch: MagicMock) -> None:
        """When batch_size=1 fails with OOM, exception propagates (no infinite loop)."""
        mock_torch.cuda.OutOfMemoryError = _FakeCudaOOM
        mock_torch.cuda.empty_cache = MagicMock()

        def always_oom(items: list[int]) -> list[int]:
            raise _FakeCudaOOM("CUDA out of memory")

        state = BatchState()
        with pytest.raises(_FakeCudaOOM):
            predict_with_oom_retry(always_oom, [1], max_batch_size=1, state=state)

    @patch("aquapose.core.inference.torch")
    def test_state_persists_across_calls(self, mock_torch: MagicMock) -> None:
        """BatchState.effective_batch_size persists -- second call uses reduced size."""
        mock_torch.cuda.OutOfMemoryError = _FakeCudaOOM
        mock_torch.cuda.empty_cache = MagicMock()

        call_count = 0

        def oom_first_call(items: list[int]) -> list[int]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeCudaOOM("CUDA out of memory")
            return items

        state = BatchState()
        # First call: OOM reduces from 4 -> 2
        predict_with_oom_retry(
            oom_first_call, [1, 2, 3, 4], max_batch_size=4, state=state
        )
        assert state.effective_batch_size == 2

        # Second call: should use 2 from the start (no OOM)
        second_calls: list[list[int]] = []

        def tracking_predict(items: list[int]) -> list[int]:
            second_calls.append(list(items))
            return items

        predict_with_oom_retry(
            tracking_predict, [10, 20, 30, 40], max_batch_size=4, state=state
        )
        # Should chunk by 2 (persisted effective_batch_size)
        assert second_calls == [[10, 20], [30, 40]]

    @patch("aquapose.core.inference.torch")
    def test_oom_occurred_flag(self, mock_torch: MagicMock) -> None:
        """BatchState.oom_occurred is True after any OOM event."""
        mock_torch.cuda.OutOfMemoryError = _FakeCudaOOM
        mock_torch.cuda.empty_cache = MagicMock()

        call_count = 0

        def oom_once(items: list[int]) -> list[int]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeCudaOOM("CUDA out of memory")
            return items

        state = BatchState()
        assert state.oom_occurred is False

        predict_with_oom_retry(oom_once, [1, 2], max_batch_size=2, state=state)
        assert state.oom_occurred is True

    def test_empty_inputs_returns_empty(self) -> None:
        """Empty inputs list returns empty list without calling predict_fn."""
        predict_fn = MagicMock()
        state = BatchState()

        result = predict_with_oom_retry(predict_fn, [], max_batch_size=4, state=state)

        assert result == []
        predict_fn.assert_not_called()
