# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging

from loguru import logger


class InterceptHandler(logging.Handler):
    """Intercepts standard logging calls and forwards them to loguru.

    Preserves the original caller information so that log messages show the
    correct source file and line number rather than the handler itself.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class LoggerStdoutWriter:
    """Wrapper for redirecting stdout to logger."""

    @staticmethod
    def write(msg: str) -> None:
        msg = msg.rstrip("\n")
        if msg:
            logger.info(msg)

    @staticmethod
    def flush() -> None:
        pass
