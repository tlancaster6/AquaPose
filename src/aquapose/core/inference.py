"""Shared inference utilities for batched GPU prediction with OOM recovery.

Provides :class:`BatchState` for tracking adaptive batch sizes across calls and
:func:`predict_with_oom_retry` for chunked inference with automatic halving on
CUDA OOM errors.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class BatchState:
    """Mutable state tracking adaptive batch size across inference calls.

    Attributes:
        effective_batch_size: Current batch size after OOM adaptation.
            ``None`` means not yet determined (will use ``max_batch_size``
            from the first call).
        original_batch_size: The ``max_batch_size`` from the first call.
            Stored for diagnostics.
        oom_occurred: ``True`` if any OOM event has been encountered.
    """

    effective_batch_size: int | None = None
    original_batch_size: int | None = None
    oom_occurred: bool = False


def predict_with_oom_retry(
    predict_fn: Callable[[list[Any]], list[Any]],
    inputs: list[Any],
    max_batch_size: int,
    state: BatchState,
) -> list[Any]:
    """Run *predict_fn* on *inputs* in chunks with OOM retry and batch halving.

    On CUDA out-of-memory errors the effective batch size is halved and the
    entire input list is retried from scratch.  The reduced batch size is
    persisted in *state* so subsequent calls start at the smaller size.

    Args:
        predict_fn: Callable that accepts a list of inputs and returns a list
            of results (same length).
        inputs: Full input list to process.
        max_batch_size: Maximum chunk size.  ``0`` means no limit (send all
            inputs in a single call).
        state: Mutable :class:`BatchState` shared across calls.

    Returns:
        Concatenated results from all chunks.

    Raises:
        torch.cuda.OutOfMemoryError: When OOM occurs with effective batch
            size already at 1 (cannot halve further).
        RuntimeError: When a non-OOM ``RuntimeError`` is raised by
            *predict_fn*.
    """
    if not inputs:
        return []

    # Determine effective batch size
    if state.effective_batch_size is not None:
        effective = state.effective_batch_size
    elif max_batch_size > 0:
        effective = max_batch_size
    else:
        effective = len(inputs)

    # Record original on first call
    if state.original_batch_size is None:
        state.original_batch_size = effective

    while True:
        try:
            results: list[Any] = []
            for i in range(0, len(inputs), effective):
                chunk = inputs[i : i + effective]
                results.extend(predict_fn(chunk))
            return results
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # Only handle genuine CUDA OOM; re-raise other RuntimeErrors
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)
            )
            if not is_oom:
                raise

            if effective <= 1:
                raise

            torch.cuda.empty_cache()
            effective = effective // 2
            state.effective_batch_size = effective
            state.oom_occurred = True
            logger.warning(
                "CUDA OOM during inference: halving batch size to %d",
                effective,
            )
            # Retry from scratch with smaller batch size
