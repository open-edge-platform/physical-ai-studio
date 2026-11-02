# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from schemas.base_job import JobStatus
from schemas.dataset_import_job import DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob
from workers.dataset_import_worker import DatasetImportWorker


@pytest.mark.anyio
@pytest.mark.parametrize("status", [JobStatus.RUNNING, JobStatus.PENDING, JobStatus.FAILED])
async def test_worker_executes_only_assigned_claimed_import(status):
    job = DatasetImportJob(
        project_id=uuid4(),
        status=status,
        payload=DatasetImportJobPayload(archive_staging_id=str(uuid4()), step=ImportStep.QUEUED_FOR_DETECTION),
    )
    worker = object.__new__(DatasetImportWorker)
    worker.job_id = job.id
    worker.should_stop = MagicMock(return_value=False)
    worker._process_job = AsyncMock()

    @asynccontextmanager
    async def session_scope():
        yield MagicMock()

    with (
        patch("workers.dataset_import_worker.get_async_db_session_ctx", session_scope),
        patch("workers.dataset_import_worker.JobService.get_job_by_id", AsyncMock(return_value=job)) as get_job,
    ):
        await worker.run_loop()

    get_job.assert_awaited_once_with(job.id)
    if status == JobStatus.RUNNING:
        worker._process_job.assert_awaited_once_with(job)
    else:
        worker._process_job.assert_not_awaited()


@pytest.mark.anyio
async def test_worker_does_not_start_import_during_shutdown():
    worker = object.__new__(DatasetImportWorker)
    worker.should_stop = MagicMock(return_value=True)
    worker._process_job = AsyncMock()
    await worker.run_loop()
    worker._process_job.assert_not_awaited()
