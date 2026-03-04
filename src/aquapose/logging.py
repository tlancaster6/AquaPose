"""File-based debug logging for AquaPose CLI commands."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = (
    "%(asctime)s %(levelname)-8s [%(relativeCreated)6dms] %(name)s %(message)s"
)
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 2


def setup_file_logging(log_dir: Path, log_name: str) -> Path:
    """Configure rotating file handler on the ``aquapose`` root logger.

    Args:
        log_dir: Base directory (e.g. run directory). A ``logs/`` subdirectory
            is created underneath.
        log_name: Stem for the log file (e.g. ``"run"`` -> ``logs/run.log``).

    Returns:
        Absolute path to the log file.
    """
    logs_dir = log_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{log_name}.log"

    logger = logging.getLogger("aquapose")
    logger.setLevel(logging.DEBUG)

    handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)

    # Version header
    _log_version_header(logger)

    # Install uncaught-exception hook
    _install_excepthook(logger)

    return log_path


def _log_version_header(logger: logging.Logger) -> None:
    """Log runtime version information."""
    import aquapose

    lines = [f"AquaPose {aquapose.__version__}", f"Python {sys.version}"]

    try:
        import torch

        lines.append(f"PyTorch {torch.__version__}")
    except ImportError:
        lines.append("PyTorch not installed")

    try:
        from importlib.metadata import version

        lines.append(f"AquaCal {version('aquacal')}")
    except Exception:
        lines.append("AquaCal version unknown")

    logger.info("--- %s ---", " | ".join(lines))


def _install_excepthook(logger: logging.Logger) -> None:
    """Install ``sys.excepthook`` that logs uncaught exceptions."""
    _original_hook = sys.excepthook

    def _hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: object,
    ) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            logger.critical(
                "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
            )  # type: ignore[arg-type]
        _original_hook(exc_type, exc_value, exc_tb)  # type: ignore[arg-type]

    sys.excepthook = _hook  # type: ignore[assignment]
