# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from uuid import UUID

from loguru import logger

from core.logging.handlers import InterceptHandler, LoggerStdoutWriter
from core.logging.setup import global_log_config


def _validate_uuid(value: str | UUID) -> str | UUID:
    """Validate that a value is a valid UUID (prevents path traversal).

    Args:
        value: The identifier to validate

    Returns:
        Validated value

    Raises:
        ValueError: If value is not a valid UUID
    """
    try:
        UUID(str(value))
    except ValueError as e:
        raise ValueError(
            f"Invalid id '{value}'. Only valid UUIDs are allowed.",
        ) from e
    return value


def get_job_logs_path(job_id: str | UUID) -> str:
    """Get the path to the log file for a specific job.

    Args:
        job_id: Unique identifier for the job
        job_type: The job type (e.g. "training", "import", "export")

    Returns:
        str: Path to the job's log file (e.g. logs/jobs/{type}_{job_id}.log)

    Raises:
        ValueError: If job_id contains invalid characters or job_type is unknown
    """
    job_id = _validate_uuid(job_id)
    jobs_folder = os.path.join(global_log_config.log_folder, "jobs")
    try:
        os.makedirs(jobs_folder, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create jobs log directory: {e}") from e
    return os.path.join(jobs_folder, f"{job_id}.log")


@contextmanager
def job_logging_ctx(job_id: str | UUID) -> Generator[str]:
    """Add a temporary log sink for a specific job.

    Captures all logs emitted during the context to
    logs/jobs/{type}_{job_id}.log.  The sink is automatically removed on
    exit, but the log file persists.  Logs also continue to go to other
    configured sinks.

    Args:
        job_id: Unique identifier for the job, used as the log filename
        job_type: The job type (e.g. "training", "import", "export")

    Yields:
        str: Path to the created log file (e.g. logs/jobs/{type}_{job_id}.log)

    Raises:
        ValueError: If job_id contains invalid characters or job_type is unknown
        RuntimeError: If log directory creation or sink addition fails
    """
    job_id = _validate_uuid(job_id)

    log_file = get_job_logs_path(job_id)

    root_logger = logging.getLogger()
    original_root_handlers = list(root_logger.handlers)
    original_root_level = root_logger.level
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        sink_id = logger.add(
            log_file,
            rotation=global_log_config.rotation,
            retention=global_log_config.retention,
            level=global_log_config.level,
            serialize=global_log_config.serialize,
            enqueue=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add log sink for job {job_id}: {e}") from e

    try:
        root_logger.handlers = [InterceptHandler()]
        root_logger.setLevel(logging.NOTSET)
        sys.stdout = LoggerStdoutWriter(level="INFO")  # type: ignore[assignment]
        sys.stderr = LoggerStdoutWriter(level="WARNING")  # type: ignore[assignment]

        logger.info(f"Started logging to {log_file}")
        yield log_file
    finally:
        logger.info(f"Stopped logging to {log_file}")
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        root_logger.handlers = original_root_handlers
        root_logger.setLevel(original_root_level)
        logger.remove(sink_id)
