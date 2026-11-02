# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-job log sinks.

Concurrent training jobs -- one per remote trainer, plus one local -- run as
asyncio tasks inside a *single* worker process, so a per-job log sink has to be
scoped to its job rather than to the process. :func:`job_logging_ctx` tags every
record emitted inside its block with the job id (via a loguru context variable,
which each asyncio task carries independently) and filters the job's sink on that
tag, so overlapping jobs cannot write into each other's log files.
"""

import logging
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import IO, TYPE_CHECKING
from uuid import UUID

from loguru import logger

from core.logging.handlers import InterceptHandler, LoggerStdoutWriter
from core.logging.setup import global_log_config

if TYPE_CHECKING:
    from loguru import Record

# Guards the shared bookkeeping below. Job contexts are normally opened from a
# worker's event loop thread, but nothing guarantees that, so every mutation of
# the module state is serialized.
_state_lock = threading.Lock()

# job id -> number of open contexts for that job. A job is normally entered once;
# the count only keeps accidental re-entry from unbalancing the bookkeeping.
_active_jobs: dict[str, int] = {}

# The single active job id, or None when zero or several jobs are active.
_sole_active_job: str | None = None

_redirect_depth = 0
_saved_stdout: IO[str] | None = None
_saved_stderr: IO[str] | None = None
_saved_root_handlers: list[logging.Handler] = []
_saved_root_level = logging.NOTSET


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

    Returns:
        str: Path to the job's log file (e.g. logs/jobs/{job_id}.log)

    Raises:
        ValueError: If job_id is not a valid UUID
        RuntimeError: If the jobs log directory cannot be created
    """
    job_id = _validate_uuid(job_id)
    jobs_folder = Path(global_log_config.log_folder) / "jobs"
    try:
        jobs_folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create jobs log directory: {e}") from e
    return str(jobs_folder / f"{job_id}.log")


def _job_sink_filter(record: "Record", job_id: str) -> bool:
    """Decide whether a record belongs in ``job_id``'s log file.

    Records tagged by :func:`job_logging_ctx` are routed strictly by their tag,
    so two jobs running side by side never mix.

    Untagged records come from the few places that cannot inherit the context
    variable -- most notably plain ``threading.Thread`` workers started by
    third-party code, whose stdout is bridged by :class:`LoggerStdoutWriter`.
    They are attributed to the running job only while it is the *only* active
    one. With several jobs active their origin is unknowable, so they are left
    out of the per-job files rather than duplicated into all of them; they are
    still captured by the worker and application log sinks.
    """
    tagged = record["extra"].get("job_id")
    if tagged is not None:
        return tagged == job_id
    return _sole_active_job == job_id


def _recompute_sole_active_job() -> None:
    """Refresh the single-active-job shortcut. Caller must hold ``_state_lock``."""
    global _sole_active_job  # noqa: PLW0603
    _sole_active_job = next(iter(_active_jobs)) if len(_active_jobs) == 1 else None


def _register_job(job_id: str) -> None:
    """Mark ``job_id`` as having one more open logging context."""
    with _state_lock:
        _active_jobs[job_id] = _active_jobs.get(job_id, 0) + 1
        _recompute_sole_active_job()


def _unregister_job(job_id: str) -> None:
    """Mark ``job_id`` as having one fewer open logging context."""
    with _state_lock:
        remaining = _active_jobs.get(job_id, 0) - 1
        if remaining > 0:
            _active_jobs[job_id] = remaining
        else:
            _active_jobs.pop(job_id, None)
        _recompute_sole_active_job()


def _install_redirects() -> None:
    """Route stdout/stderr and stdlib logging into loguru (first context only)."""
    global _redirect_depth, _saved_stdout, _saved_stderr, _saved_root_handlers, _saved_root_level  # noqa: PLW0603
    root_logger = logging.getLogger()
    with _state_lock:
        _redirect_depth += 1
        if _redirect_depth > 1:
            return
        _saved_stdout = sys.stdout
        _saved_stderr = sys.stderr
        _saved_root_handlers = list(root_logger.handlers)
        _saved_root_level = root_logger.level
        root_logger.handlers = [InterceptHandler()]
        root_logger.setLevel(logging.NOTSET)
        sys.stdout = LoggerStdoutWriter(level="INFO")  # type: ignore[assignment]
        sys.stderr = LoggerStdoutWriter(level="WARNING")  # type: ignore[assignment]


def _remove_redirects() -> None:
    """Restore the real streams and root logger (last context only)."""
    global _redirect_depth, _saved_stdout, _saved_stderr, _saved_root_handlers  # noqa: PLW0603
    root_logger = logging.getLogger()
    with _state_lock:
        _redirect_depth = max(0, _redirect_depth - 1)
        if _redirect_depth > 0:
            return
        if _saved_stdout is not None:
            sys.stdout = _saved_stdout
        if _saved_stderr is not None:
            sys.stderr = _saved_stderr
        root_logger.handlers = _saved_root_handlers
        root_logger.setLevel(_saved_root_level)
        _saved_stdout = None
        _saved_stderr = None
        _saved_root_handlers = []


@contextmanager
def job_logging_ctx(job_id: str | UUID) -> Generator[str]:
    """Add a log sink scoped to a single job.

    Captures logs emitted inside the context -- including from asyncio tasks and
    ``asyncio.to_thread`` calls started within it -- to
    ``logs/jobs/{job_id}.log``. The sink is removed on exit, but the log file
    persists. Logs also continue to go to the other configured sinks.

    Safe to run concurrently with contexts for other jobs: each job's records are
    tagged and its sink filters on that tag, so overlapping jobs (e.g. one per
    remote trainer) keep separate logs. See :func:`_job_sink_filter` for how the
    few records that cannot carry a tag are handled.

    Args:
        job_id: Unique identifier for the job, used as the log filename

    Yields:
        str: Path to the created log file (e.g. logs/jobs/{job_id}.log)

    Raises:
        ValueError: If job_id is not a valid UUID
        RuntimeError: If log directory creation or sink addition fails
    """
    job_key = str(_validate_uuid(job_id))
    log_file = get_job_logs_path(job_key)

    _register_job(job_key)
    try:
        sink_id = logger.add(
            log_file,
            rotation=global_log_config.rotation,
            retention=global_log_config.retention,
            level=global_log_config.level,
            filter=partial(_job_sink_filter, job_id=job_key),
            serialize=global_log_config.serialize,
            enqueue=True,
        )
    except Exception as e:
        _unregister_job(job_key)
        raise RuntimeError(f"Failed to add log sink for job {job_key}: {e}") from e

    try:
        _install_redirects()
        # Tag every record emitted in this block -- and in the asyncio tasks and
        # threads it spawns -- so the sink above claims exactly its own records.
        with logger.contextualize(job_id=job_key):
            logger.info(f"Started logging to {log_file}")
            try:
                yield log_file
            finally:
                logger.info(f"Stopped logging to {log_file}")
    finally:
        _remove_redirects()
        _unregister_job(job_key)
        logger.remove(sink_id)
