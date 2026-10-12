"""Structured logging configuration.

Produces single-line, key=value logs that are easy to grep and to ingest into a
log aggregator. Patient feature values are never logged here; callers are
responsible for not passing sensitive payloads to log records.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def configure_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Configure root logging once per process."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(_LEVEL_MAP.get(level.upper(), logging.INFO))

    # Uvicorn access logs are noisy for health checks; keep them at WARNING.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
