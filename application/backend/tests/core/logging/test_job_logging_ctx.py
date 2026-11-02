# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-job log isolation.

The regression these guard: with several remote trainers, the training worker
runs one asyncio task per trainer inside a single process, so an unfiltered
per-job sink captured every concurrent job's records into every job's file.
"""

import asyncio
import json
import logging
import sys
from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import pytest
from loguru import logger

from core.logging import utils as logging_utils
from core.logging.utils import job_logging_ctx


@pytest.fixture(autouse=True)
def _job_logs_in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Point job logs at tmp_path and keep sink/stream state from leaking."""
    monkeypatch.setattr(logging_utils.global_log_config, "log_folder", tmp_path, raising=False)
    monkeypatch.setattr(logging_utils.global_log_config, "serialize", True, raising=False)
    monkeypatch.setattr(logging_utils.global_log_config, "level", "INFO", raising=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    root_logger = logging.getLogger()
    original_handlers, original_level = list(root_logger.handlers), root_logger.level
    yield
    sys.stdout, sys.stderr = original_stdout, original_stderr
    root_logger.handlers, root_logger.level = original_handlers, original_level


def _read_messages(log_file: str | Path) -> list[str]:
    """Return the ``message`` field of every serialized record in a job log."""
    # enqueue=True writes from a background thread; loguru flushes it on remove().
    lines = Path(log_file).read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["record"]["message"] for line in lines if line.strip()]


def test_rejects_non_uuid_job_id() -> None:
    with pytest.raises(ValueError, match="Only valid UUIDs are allowed"), job_logging_ctx(job_id="../../etc/passwd"):
        pass


def test_single_job_captures_its_logs() -> None:
    job_id = str(uuid4())

    with job_logging_ctx(job_id=job_id) as log_file:
        logger.info("hello from the job")

    assert "hello from the job" in _read_messages(log_file)


def test_untagged_thread_logs_land_in_the_sole_active_job() -> None:
    """A lone job still collects records from threads that miss the context var."""
    job_id = str(uuid4())

    with job_logging_ctx(job_id=job_id) as log_file:
        # logger.contextualize() uses a contextvar, which a bare thread started
        # from a snapshot-free context (as third-party libs do) does not inherit.
        ctx_free = logger.bind()
        ctx_free.patch(lambda record: record["extra"].pop("job_id", None)).info("untagged line")

    assert "untagged line" in _read_messages(log_file)


@pytest.mark.anyio
async def test_concurrent_jobs_do_not_mix_logs() -> None:
    """Two overlapping jobs must each log only their own lines."""
    job_a, job_b = str(uuid4()), str(uuid4())
    both_open = asyncio.Event()
    files: dict[str, str] = {}

    async def run_job(job_id: str, other_started: asyncio.Event) -> None:
        with job_logging_ctx(job_id=job_id) as log_file:
            files[job_id] = log_file
            logger.info("start {}", job_id)
            other_started.set()
            await both_open.wait()
            # Both sinks are open at this point: the interleaving that used to
            # duplicate every line into both files.
            logger.info("work {}", job_id)
            await asyncio.sleep(0)
            logger.info("done {}", job_id)

    a_started, b_started = asyncio.Event(), asyncio.Event()
    task_a = asyncio.create_task(run_job(job_a, a_started))
    task_b = asyncio.create_task(run_job(job_b, b_started))
    await a_started.wait()
    await b_started.wait()
    both_open.set()
    await asyncio.gather(task_a, task_b)

    messages_a = _read_messages(files[job_a])
    messages_b = _read_messages(files[job_b])

    assert f"work {job_a}" in messages_a
    assert f"done {job_a}" in messages_a
    assert f"work {job_b}" in messages_b
    assert f"done {job_b}" in messages_b
    # The actual bug: neither file may contain the other job's lines.
    assert not [message for message in messages_a if job_b in message]
    assert not [message for message in messages_b if job_a in message]


@pytest.mark.anyio
async def test_tasks_spawned_inside_a_job_inherit_its_tag() -> None:
    """Background work started inside a job context stays attributed to it."""
    job_a, job_b = str(uuid4()), str(uuid4())
    files: dict[str, str] = {}
    ready = asyncio.Event()

    async def nested_work(job_id: str) -> None:
        await ready.wait()
        logger.info("nested {}", job_id)

    async def run_job(job_id: str, started: asyncio.Event) -> None:
        with job_logging_ctx(job_id=job_id) as log_file:
            files[job_id] = log_file
            # create_task copies the current context, tag included.
            nested = asyncio.create_task(nested_work(job_id))
            started.set()
            await asyncio.sleep(0)
            ready.set()
            await nested

    a_started, b_started = asyncio.Event(), asyncio.Event()
    await asyncio.gather(run_job(job_a, a_started), run_job(job_b, b_started))

    assert f"nested {job_a}" in _read_messages(files[job_a])
    assert f"nested {job_b}" in _read_messages(files[job_b])
    assert f"nested {job_b}" not in _read_messages(files[job_a])
    assert f"nested {job_a}" not in _read_messages(files[job_b])


@pytest.mark.anyio
async def test_to_thread_calls_inherit_the_job_tag() -> None:
    """asyncio.to_thread copies the context, so offloaded work keeps its tag."""
    job_id = str(uuid4())

    with job_logging_ctx(job_id=job_id) as log_file:
        await asyncio.to_thread(logger.info, "from a worker thread")

    assert "from a worker thread" in _read_messages(log_file)


@pytest.mark.anyio
async def test_streams_restored_only_after_the_last_job_exits() -> None:
    """An inner job's exit must not hand the real stdout back early."""
    real_stdout = sys.stdout
    job_a, job_b = str(uuid4()), str(uuid4())

    with job_logging_ctx(job_id=job_a):
        assert sys.stdout is not real_stdout
        with job_logging_ctx(job_id=job_b):
            assert sys.stdout is not real_stdout
        # job_b closed, job_a is still running: stdout must stay redirected.
        assert sys.stdout is not real_stdout

    assert sys.stdout is real_stdout


def test_sink_is_removed_and_state_cleaned_after_exit() -> None:
    job_id = str(uuid4())

    with job_logging_ctx(job_id=job_id) as log_file:
        pass

    logger.info("after the job closed")

    assert "after the job closed" not in _read_messages(log_file)
    assert logging_utils._active_jobs == {}
    assert logging_utils._sole_active_job is None
    assert logging_utils._redirect_depth == 0
