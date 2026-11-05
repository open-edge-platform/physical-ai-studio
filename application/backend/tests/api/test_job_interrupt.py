# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from api.job import interrupt_job
from core.scheduler import Scheduler
from schemas.base_job import JobStatus, JobType
from schemas.job import RemoteTrainJobPayload, TrainJob
from services.job_service import JobService


@pytest.mark.anyio
@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.RUNNING])
@pytest.mark.parametrize("remote_job_id", [None, uuid4()])
async def test_stop_signals_worker_when_queued_cancellation_requires_cooperation(status, remote_job_id):
    job = TrainJob(
        project_id=uuid4(),
        status=status,
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            policy="act",
            model_name="remote",
            remote_trainer_id=uuid4(),
            remote_job_id=remote_job_id,
        ),
    )
    service = JobService(MagicMock())
    service.get_job_by_id = AsyncMock(return_value=job)
    service.cancel_pending_job = AsyncMock(return_value=None)
    scheduler = MagicMock(spec=Scheduler)

    await interrupt_job(job.id, service, scheduler)

    service.cancel_pending_job.assert_awaited_once_with(job.id, JobType.TRAINING)
    scheduler.cancel_job.assert_called_once_with(job.id)
