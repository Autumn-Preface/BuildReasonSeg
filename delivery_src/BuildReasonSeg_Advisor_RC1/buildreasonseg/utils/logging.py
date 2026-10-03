"""Logging: short console messages for users, full tracebacks into `logs/`."""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path

from buildreasonseg import paths

LOGGER_NAME = "buildreasonseg"


def setup_logger(name: str = LOGGER_NAME, *, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    return logger


def log_exception(error: BaseException, *, context: str = "") -> Path:
    """Append the full traceback to a timestamped file in `logs/` and return its path."""

    logs = paths.logs_dir()
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs / f"error_{stamp}.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"=== {datetime.now().isoformat()} | context: {context or '-'} ===\n")
        handle.write("".join(traceback.format_exception(type(error), error, error.__traceback__)))
        handle.write("\n")
    return path


__all__ = ["LOGGER_NAME", "log_exception", "setup_logger"]
