from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceInUseError, ResourceNotFoundError
from schemas.base_job import JobStatus
from schemas.job import TrainJob
from services.job_service import JobService


def _job(*, status: JobStatus = JobStatus.PENDING) -> TrainJob:
    project_id = uuid4()
    return TrainJob(
        id=uuid4(),
        project_id=project_id,
        status=status,
        payload={
            "type": "training",
            "project_id": str(project_id),
            "dataset_id": str(uuid4()),
            "policy": "act",
            "model_name": "test-model",
            "max_steps": 100,
            "batch_size": 8,
            "base_model_id": None,
        },
    )


def test_job_service_uses_injected_session() -> None:
    session = MagicMock(spec=AsyncSession)

    with patch("services.job_service.JobRepository") as repository_type:
        service = JobService(session)

    repository_type.assert_called_once_with(session)
    assert service.session is session
    assert service.repo is repository_type.return_value


@pytest.mark.anyio
async def test_create_job_uses_instance_repository() -> None:
    session = MagicMock(spec=AsyncSession)
    job = _job()

    with patch("services.job_service.JobRepository") as repository_type:
        repository_type.return_value.save = AsyncMock(return_value=job)
        service = JobService(session)
        result = await service.create_job(job)

    assert result is job
    repository_type.return_value.save.assert_awaited_once_with(job)


@pytest.mark.anyio
async def test_get_job_by_id_raises_when_missing() -> None:
    session = MagicMock(spec=AsyncSession)
    job_id = uuid4()

    with patch("services.job_service.JobRepository") as repository_type:
        repository_type.return_value.get_by_id = AsyncMock(return_value=None)
        service = JobService(session)
        with pytest.raises(ResourceNotFoundError):
            await service.get_job_by_id(job_id)

    repository_type.return_value.get_by_id.assert_awaited_once_with(job_id)


@pytest.mark.anyio
async def test_delete_job_rejects_active_job() -> None:
    session = MagicMock(spec=AsyncSession)
    job = _job(status=JobStatus.RUNNING)

    with patch("services.job_service.JobRepository") as repository_type:
        repository_type.return_value.get_by_id = AsyncMock(return_value=job)
        repository_type.return_value.delete_by_id = AsyncMock()
        service = JobService(session)
        with pytest.raises(ResourceInUseError):
            await service.delete_job(job.id)

    repository_type.return_value.delete_by_id.assert_not_awaited()
