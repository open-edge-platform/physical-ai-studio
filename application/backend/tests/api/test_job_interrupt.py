# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from api.job import interrupt_job
from core.scheduler import Scheduler
from exceptions import AwsFeatureDisabledError
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


@pytest.mark.anyio
@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.RUNNING])
async def test_submitted_batch_stop_persists_cancellation_for_worker(status: JobStatus) -> None:
    job = TrainJob(
        project_id=uuid4(),
        status=status,
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            policy="act",
            model_name="batch",
            remote_trainer_id=uuid4(),
            remote_trainer_url="aws-batch://trainer",
            remote_job_id=uuid4(),
        ),
    )
    service = JobService(MagicMock())
    service.get_job_by_id = AsyncMock(return_value=job)
    service.update_remote_job_state = AsyncMock()
    service.cancel_pending_job = AsyncMock()
    scheduler = MagicMock(spec=Scheduler)

    with patch("services.training_backends.remote.RemoteTrainingBackend.cancel", new_callable=AsyncMock) as cancel:
        await interrupt_job(job.id, service, scheduler)

    service.update_remote_job_state.assert_awaited_once_with(job.id, cancel_requested=True)
    service.cancel_pending_job.assert_not_awaited()
    cancel.assert_not_awaited()
    scheduler.cancel_job.assert_called_once_with(job.id)


@pytest.mark.anyio
@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.RUNNING])
async def test_batch_stop_rejects_exposure_before_signaling_worker(status: JobStatus) -> None:
    job = TrainJob(
        project_id=uuid4(),
        status=status,
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            policy="act",
            model_name="batch",
            remote_trainer_id=uuid4(),
            remote_trainer_url="aws-batch://trainer",
            remote_job_id=uuid4(),
        ),
    )
    service = JobService(MagicMock())
    service.get_job_by_id = AsyncMock(return_value=job)
    service.update_remote_job_state = AsyncMock()
    service.cancel_pending_job = AsyncMock()
    scheduler = MagicMock(spec=Scheduler)
    with (
        patch("services.job_handlers.require_aws_feature_active", side_effect=AwsFeatureDisabledError),
        pytest.raises(AwsFeatureDisabledError),
    ):
        await interrupt_job(job.id, service, scheduler)
    service.update_remote_job_state.assert_not_awaited()
    service.cancel_pending_job.assert_not_awaited()
    scheduler.cancel_job.assert_not_called()
