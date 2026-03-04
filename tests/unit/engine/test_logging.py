"""Tests for aquapose.engine.logging."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from aquapose.logging import setup_file_logging


def test_setup_file_logging_creates_file_and_captures_messages(tmp_path: Path) -> None:
    """setup_file_logging creates the log file and captures debug messages."""
    log_path = setup_file_logging(tmp_path, "test")

    assert log_path == tmp_path / "logs" / "test.log"
    assert log_path.exists()

    # Log a debug message via a child logger (simulating a module logger)
    child = logging.getLogger("aquapose.engine.test_child")
    child.debug("hello from test")

    # Flush handlers
    root = logging.getLogger("aquapose")
    for handler in root.handlers:
        handler.flush()

    content = log_path.read_text()
    assert "hello from test" in content
    # Version header should be present
    assert "AquaPose" in content

    # Cleanup: remove the handler we added to avoid leaking into other tests
    for handler in list(root.handlers):
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.close()
            root.removeHandler(handler)
