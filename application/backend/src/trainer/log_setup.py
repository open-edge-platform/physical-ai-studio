# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration for the trainer service.

Routes stdlib logging (including uvicorn's own loggers) through loguru so the
remote trainer's console output uses the same timestamped, leveled format as
the local backend, instead of uvicorn's bare `INFO:     ...` lines.
"""

from __future__ import annotations

import inspect
import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """Forward stdlib logging records to loguru, preserving caller info.

    Mirrors ``core.logging.handlers.InterceptHandler`` in the backend so both
    services emit console logs in the same format.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Log the record via loguru at the matching level and caller depth."""
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru's console sink with the same format used by the backend."""
    logger.remove()
    logger.add(sys.stderr, level=level)


def setup_uvicorn_logging() -> None:
    """Redirect uvicorn's stdlib loggers through loguru's InterceptHandler.

    Without this, uvicorn's startup/access logs use its own bare formatter
    (no timestamp), which looks inconsistent next to the trainer's own
    loguru-formatted log lines.
    """
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = [InterceptHandler()]
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.propagate = False
    for logger_name in ("uvicorn.access", "uvicorn.error"):
        child_logger = logging.getLogger(logger_name)
        child_logger.handlers.clear()
        child_logger.propagate = True
