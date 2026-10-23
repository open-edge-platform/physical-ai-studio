import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from exceptions import InvalidResourceError, ResourceInUseError, ResourceNotFoundError
from schemas.base_job import JobStatus
from schemas.job import LocalTrainJobPayload, RemoteTrainJobPayload, TrainJob
from schemas.model import Model
from services import remote_trainer_service as remote_trainer_service_module
from services.job_service import JobService
from services.remote_trainer_service import RemoteTrainerService
from services.ssh import persistent_trainer


def _job(*, status: JobStatus = JobStatus.PENDING) -> TrainJob:
    project_id = uuid4()
    return TrainJob(
        id=uuid4(),
        project_id=project_id,
        status=status,
        payload={
            "training_target": "local",
            "project_id": str(project_id),
            "dataset_id": str(uuid4()),
            "policy": "act",
            "model_name": "test-model",
            "max_steps": 100,
            "batch_size": 8,
            "base_model_id": None,
        },
    )


@pytest.mark.anyio
async def test_local_submission_rejects_unsupported_policy_before_saving() -> None:
    payload = LocalTrainJobPayload(project_id=uuid4(), dataset_id=uuid4(), policy="cosmos3", model_name="model")
    with (
        patch("services.job_service.JobRepository") as repository_type,
        patch("services.training_targets.local.available_policy_classes", return_value={"act": (object, [])}),
    ):
        service = JobService(MagicMock(spec=AsyncSession))
        with pytest.raises(InvalidResourceError, match="not available on this machine"):
            await service.submit_train_job(payload)
        repository_type.return_value.save.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("optimized", [False, True])
async def test_local_submission_validates_optimized_base_model_before_saving(optimized, tmp_path):
    project_id = uuid4()
    base = Model(
        project_id=project_id,
        name="base",
        policy="act",
        path=str(tmp_path),
        properties={"optimization": {"source_model_id": str(uuid4()), "mode": "int8_sym"}} if optimized else {},
    )
    payload = LocalTrainJobPayload(
        project_id=project_id, dataset_id=uuid4(), policy="act", model_name="retrain", base_model_id=base.id
    )
    with (
        patch("services.job_service.JobRepository") as repository_type,
        patch("services.job_service.ModelService") as model_service,
        patch("services.training_targets.local.available_policy_classes", return_value={"act": (object, [])}),
    ):
        model_service.return_value.get_model_by_id = AsyncMock(return_value=base)
        repository = repository_type.return_value
        repository.is_job_duplicate = AsyncMock(return_value=False)
        repository.save = AsyncMock(side_effect=lambda job: job)
        service = JobService(MagicMock(spec=AsyncSession))
        if optimized:
            with pytest.raises(InvalidResourceError, match="inference-only"):
                await service.submit_train_job(payload)
            repository.save.assert_not_awaited()
            repository.is_job_duplicate.assert_not_awaited()
        else:
            job = await service.submit_train_job(payload)
            assert job.payload.base_model_id == base.id
            repository.save.assert_awaited_once()
        model_service.return_value.get_model_by_id.assert_awaited_once_with(base.id)


@pytest.mark.anyio
async def test_remote_submission_blocks_setup_until_job_is_persisted() -> None:
    trainer_id = uuid4()
    payload = RemoteTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        remote_trainer_id=trainer_id,
    )
    saving = asyncio.Event()
    release = asyncio.Event()

    async def save(_job):
        saving.set()
        await release.wait()

    handler = MagicMock()
    handler.prepare = AsyncMock(return_value=payload)
    with (
        patch("services.job_service.JobRepository") as repository_type,
        patch("services.job_service.get_training_target_handler", return_value=handler),
        patch.object(remote_trainer_service_module, "_setup_lock", asyncio.Lock()),
    ):
        repository_type.return_value.is_job_duplicate = AsyncMock(return_value=False)
        repository_type.return_value.save = AsyncMock(side_effect=save)
        service = JobService(MagicMock(spec=AsyncSession))
        submission = asyncio.create_task(service.submit_train_job(payload))
        await saving.wait()

        async def competing_submission():
            async with RemoteTrainerService.allow_job_submission(trainer_id):
                return True

        setup = asyncio.create_task(competing_submission())
        await asyncio.sleep(0)
        assert not setup.done()
        release.set()
        await submission
        # The reservation has been released after the job commit.
        await setup
        assert setup.result() is True


@pytest.mark.anyio
async def test_remote_submission_rejects_trainer_being_installed() -> None:
    trainer_id = uuid4()
    payload = RemoteTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        remote_trainer_id=trainer_id,
    )
    running = asyncio.create_task(asyncio.sleep(60))
    remote_trainer_service_module._background_installs[trainer_id] = running
    try:
        with patch("services.job_service.JobRepository") as repository_type:
            service = JobService(MagicMock(spec=AsyncSession))
            with pytest.raises(ResourceInUseError):
                await service.submit_train_job(payload)
            repository_type.return_value.save.assert_not_called()
    finally:
        remote_trainer_service_module._background_installs.pop(trainer_id)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.anyio
@pytest.mark.parametrize("reason", ["reboot_required", "relogin_required", "reboot_blocked_active_containers"])
async def test_remote_submission_rejects_unfinished_setup(reason: str) -> None:
    trainer_id = uuid4()
    payload = RemoteTrainJobPayload(
        project_id=uuid4(), dataset_id=uuid4(), policy="act", model_name="model", remote_trainer_id=trainer_id
    )
    persistent_trainer.set_install_failure(trainer_id, reason)
    try:
        with patch("services.job_service.JobRepository") as repository_type:
            with pytest.raises(ResourceInUseError):
                await JobService(MagicMock(spec=AsyncSession)).submit_train_job(payload)
            repository_type.return_value.save.assert_not_called()
    finally:
        persistent_trainer.set_install_failure(trainer_id, None)


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
@pytest.mark.parametrize("first", [JobStatus.RUNNING, JobStatus.CANCELED])
async def test_only_one_pending_train_job_transition_wins(first: JobStatus) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    job_id = uuid4()
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE jobs (id TEXT, type TEXT, status TEXT, message TEXT, "
                "start_time DATETIME, end_time DATETIME, payload JSON)"
            )
        )
        await connection.execute(
            text("INSERT INTO jobs (id, type, status) VALUES (:id, 'training', 'pending')"), {"id": str(job_id)}
        )
    try:
        async with AsyncSession(engine) as session:
            service = JobService(session)
            other = JobStatus.CANCELED if first is JobStatus.RUNNING else JobStatus.RUNNING
            transitions = {
                JobStatus.RUNNING: service.claim_pending_train_job,
                JobStatus.CANCELED: service.cancel_pending_train_job,
            }
            with patch.object(service.repo, "get_by_id", AsyncMock(return_value=_job(status=first))):
                assert await transitions[first](job_id) is not None
                assert await transitions[other](job_id) is None
            result = await session.execute(
                text("SELECT status, start_time, end_time FROM jobs WHERE id = :id"), {"id": str(job_id)}
            )
            status, start_time, end_time = result.one()
            assert status == first.value
            assert bool(start_time) == (first is JobStatus.RUNNING)
            assert bool(end_time) == (first is JobStatus.CANCELED)
    finally:
        await engine.dispose()


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
