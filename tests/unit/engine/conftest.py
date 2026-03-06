"""Shared fixtures for engine unit tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_lut_loading() -> object:
    """Auto-mock LUT loading so build_stages doesn't require real LUT files."""
    with (
        patch(
            "aquapose.engine.pipeline.load_forward_luts",
            return_value={"cam1": object()},
        ),
        patch(
            "aquapose.engine.pipeline.load_inverse_luts",
            return_value=object(),
        ),
    ):
        yield
