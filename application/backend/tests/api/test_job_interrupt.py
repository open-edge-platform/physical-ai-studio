# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from api.job import interrupt_job
from schemas.base_job import JobStatus
from schemas.job import RemoteTrainJobPayload, TrainJob


@pytest.mark.anyio
async def test_pending_reattachment_stop_cancels_remote_job() -> None:
    remote_job_id = uuid4()
    job = TrainJob(
        project_id=uuid4(),
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            policy="act",
            model_name="remote",
            remote_trainer_id=uuid4(),
            remote_trainer_url="http://127.0.0.1:8001",
            remote_job_id=remote_job_id,
        ),
    )
    service = MagicMock()
    service.get_job_by_id = AsyncMock(return_value=job)
    service.transition_pending_train_job = AsyncMock(return_value=True)
    scheduler = MagicMock(job_interrupt_flags={})

    with patch("services.training_backends.remote.RemoteTrainingBackend.cancel", new_callable=AsyncMock) as cancel:
        await interrupt_job(job.id, service, scheduler)

    service.transition_pending_train_job.assert_awaited_once_with(job.id, JobStatus.CANCELED)
    cancel.assert_awaited_once_with(remote_job_id)
    assert scheduler.job_interrupt_flags == {}


@pytest.mark.anyio
async def test_pending_stop_claim_lost_to_worker_sets_interrupt_flag() -> None:
    job = TrainJob(
        project_id=uuid4(),
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            policy="act",
            model_name="remote",
            remote_trainer_id=uuid4(),
        ),
    )
    service = MagicMock()
    service.get_job_by_id = AsyncMock(side_effect=[job, job.model_copy(update={"status": JobStatus.RUNNING})])
    service.transition_pending_train_job = AsyncMock(return_value=False)
    scheduler = MagicMock(job_interrupt_flags={})

    await interrupt_job(job.id, service, scheduler)

    assert scheduler.job_interrupt_flags[str(job.id)] is True
