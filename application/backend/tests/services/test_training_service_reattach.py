# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reattach-aware orphan reconciliation for remote training jobs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from schemas.base_job import JobStatus
from schemas.job import TrainingTarget, TrainJobPayload
from services.training_service import TrainingService

MODULE = "services.training_service"


def _payload(*, remote_job_id: UUID | None = None, target: TrainingTarget = TrainingTarget.REMOTE) -> TrainJobPayload:
    return TrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="m",
        training_target=target,
        remote_trainer_id=uuid4() if target is TrainingTarget.REMOTE else None,
        remote_trainer_url="https://trainer.test" if target is TrainingTarget.REMOTE else None,
        remote_job_id=remote_job_id,
    )


def _job(payload: TrainJobPayload) -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.payload = payload
    return job


class TestReattachOrphans:
    @pytest.mark.anyio
    async def test_remote_running_job_with_remote_id_is_requeued(self):
        """A RUNNING remote job that recorded its remote id is requeued, not failed."""
        job = _job(_payload(remote_job_id=uuid4()))

        service = MagicMock()
        service.get_job_list = AsyncMock(return_value=[job])
        service.update_job_status = AsyncMock(return_value=MagicMock())

        await TrainingService.abort_orphan_jobs(service)

        service.update_job_status.assert_awaited_once()
        assert service.update_job_status.call_args.kwargs["status"] == JobStatus.PENDING

    @pytest.mark.anyio
    async def test_remote_running_job_without_remote_id_is_failed(self):
        """A RUNNING remote job that never got a remote id cannot resume and is failed."""
        job = _job(_payload(remote_job_id=None))

        service = MagicMock()
        service.get_job_list = AsyncMock(return_value=[job])
        service.update_job_status = AsyncMock(return_value=MagicMock())

        await TrainingService.abort_orphan_jobs(service)

        assert service.update_job_status.call_args.kwargs["status"] == JobStatus.FAILED

    @pytest.mark.anyio
    async def test_local_job_always_fails_orphans(self):
        """A local job cannot reattach, even if a stale remote id is present."""
        job = _job(_payload(remote_job_id=uuid4(), target=TrainingTarget.LOCAL))

        service = MagicMock()
        service.get_job_list = AsyncMock(return_value=[job])
        service.update_job_status = AsyncMock(return_value=MagicMock())

        await TrainingService.abort_orphan_jobs(service)

        assert service.update_job_status.call_args.kwargs["status"] == JobStatus.FAILED

    def test_reattachable_remote_job_id_reads_dict_payload(self):
        """Payloads persisted as plain dicts are also understood."""
        job = MagicMock()
        remote_job_id = uuid4()
        job.payload = {"training_target": "remote", "remote_job_id": str(remote_job_id)}
        assert TrainingService._reattachable_remote_job_id(job) == remote_job_id

        job.payload = {"training_target": "remote", "remote_job_id": "not-a-uuid"}
        assert TrainingService._reattachable_remote_job_id(job) is None
        job.payload = {"training_target": "remote", "remote_job_id": None}
        assert TrainingService._reattachable_remote_job_id(job) is None
