# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reattach-aware orphan reconciliation for remote training jobs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from schemas.base_job import JobStatus
from schemas.job import TrainJobPayload
from services.training_service import TrainingService

MODULE = "services.training_service"


def _payload(*, remote_job_id: UUID | None = None) -> TrainJobPayload:
    return TrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="m",
        remote_job_id=remote_job_id,
    )


def _job(payload: TrainJobPayload) -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.payload = payload
    return job


def _settings(*, training_mode: str) -> MagicMock:
    settings = MagicMock()
    settings.training_mode = training_mode
    return settings


class TestReattachOrphans:
    @pytest.mark.anyio
    async def test_remote_running_job_with_remote_id_is_requeued(self):
        """A RUNNING remote job that recorded its remote id is requeued, not failed."""
        job = _job(_payload(remote_job_id=uuid4()))

        with (
            patch(f"{MODULE}.get_settings", return_value=_settings(training_mode="remote")),
            patch(f"{MODULE}.JobService") as MockJobService,
        ):
            MockJobService.get_job_list = AsyncMock(return_value=[job])
            MockJobService.update_job_status = AsyncMock(return_value=MagicMock())

            await TrainingService.abort_orphan_jobs()

            MockJobService.update_job_status.assert_awaited_once()
            assert MockJobService.update_job_status.call_args.kwargs["status"] == JobStatus.PENDING

    @pytest.mark.anyio
    async def test_remote_running_job_without_remote_id_is_failed(self):
        """A RUNNING remote job that never got a remote id cannot resume and is failed."""
        job = _job(_payload(remote_job_id=None))

        with (
            patch(f"{MODULE}.get_settings", return_value=_settings(training_mode="remote")),
            patch(f"{MODULE}.JobService") as MockJobService,
        ):
            MockJobService.get_job_list = AsyncMock(return_value=[job])
            MockJobService.update_job_status = AsyncMock(return_value=MagicMock())

            await TrainingService.abort_orphan_jobs()

            assert MockJobService.update_job_status.call_args.kwargs["status"] == JobStatus.FAILED

    @pytest.mark.anyio
    async def test_local_mode_always_fails_orphans(self):
        """In local mode a remote id is irrelevant; orphan jobs are failed."""
        job = _job(_payload(remote_job_id=uuid4()))

        with (
            patch(f"{MODULE}.get_settings", return_value=_settings(training_mode="local")),
            patch(f"{MODULE}.JobService") as MockJobService,
        ):
            MockJobService.get_job_list = AsyncMock(return_value=[job])
            MockJobService.update_job_status = AsyncMock(return_value=MagicMock())

            await TrainingService.abort_orphan_jobs()

            assert MockJobService.update_job_status.call_args.kwargs["status"] == JobStatus.FAILED

    def test_reattachable_remote_job_id_reads_dict_payload(self):
        """Payloads persisted as plain dicts are also understood."""
        job = MagicMock()
        remote_job_id = uuid4()
        job.payload = {"remote_job_id": str(remote_job_id)}
        assert TrainingService._reattachable_remote_job_id(job) == remote_job_id

        job.payload = {"remote_job_id": "not-a-uuid"}
        assert TrainingService._reattachable_remote_job_id(job) is None
        job.payload = {"remote_job_id": None}
        assert TrainingService._reattachable_remote_job_id(job) is None
