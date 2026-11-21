# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from dataclasses import replace
from threading import Event
from unittest.mock import MagicMock, patch
from uuid import uuid4

import modal
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.job import interrupt_job
from core.scheduler import Scheduler
from db.schema import Base, ProjectDB
from exceptions import InvalidJobStateError
from schemas.base_job import JobStatus, JobType
from schemas.dataset_import_job import DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob, RemoteTrainJobPayload, TrainJob
from services.job_handlers import get_job_handler
from services.job_service import JobService
from services.training_backends.base import TrainingCanceledError, TrainingContext, TrainingSuspendedError
from services.training_backends.modal import ModalTrainingBackend


@pytest.fixture
async def jobs():
    engine = create_async_engine("sqlite+aiosqlite://")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with factory() as session:
        project_id = uuid4()
        session.add(ProjectDB(id=str(project_id), name="Job policies"))
        await session.commit()
        yield JobService(session), project_id
    await engine.dispose()


def import_job(project_id, step, status=JobStatus.PENDING):
    return DatasetImportJob(
        project_id=project_id,
        status=status,
        payload=DatasetImportJobPayload(archive_staging_id=str(uuid4()), step=step),
    )


@pytest.mark.anyio
async def test_modal_stop_survives_shutdown_during_cancellation_outage(jobs, tmp_path):
    service, project_id = jobs
    job = TrainJob(
        project_id=project_id,
        status=JobStatus.RUNNING,
        payload=RemoteTrainJobPayload(
            project_id=project_id,
            dataset_id=uuid4(),
            policy="act",
            model_name="Modal",
            remote_trainer_id=uuid4(),
            remote_trainer_url="modal://trainer",
            remote_job_id="fc-test",
        ),
    )
    await service.create_job(job)
    cancel_event = Event()
    await service.interrupt_job(job.id, lambda _: cancel_event.set())
    assert cancel_event.is_set()
    # A worker saving the call identity must not erase the concurrent Stop request.
    await service.update_remote_job_state(job.id, remote_job_id="fc-test")
    context = TrainingContext(
        job=job,
        model=MagicMock(),
        snapshot=None,
        payload=job.payload,
        base_model=None,
        output_dir=tmp_path / "model",
        cache_dir=tmp_path / "cache",
        progress=MagicMock(),
        should_stop=cancel_event.is_set,
        should_cancel_job=cancel_event.is_set,
        should_suspend=lambda: True,
        remote_job_id=job.payload.remote_job_id,
    )
    call = MagicMock()
    call.cancel.side_effect = modal.exception.AuthError("credentials unavailable")
    with pytest.raises(TrainingSuspendedError):
        await ModalTrainingBackend._check_stop(context, call)
    await service.update_job_status(job.id, JobStatus.PENDING)

    # Restart drops the process-local event, but not the saved cancellation request.
    cancel_event.clear()
    restored = await JobService(service.session).get_job_by_id(job.id)
    assert restored.status == JobStatus.PENDING
    assert restored.payload.cancel_requested is True
    assert restored.payload.remote_job_id == "fc-test"
    reattached = replace(
        context,
        job=restored,
        payload=restored.payload,
        should_suspend=lambda: False,
        should_cancel_job=lambda: bool(restored.payload.cancel_requested),
    )
    call.cancel.side_effect = None
    with pytest.raises(TrainingCanceledError):
        await ModalTrainingBackend._check_stop(reattached, call)
    assert call.cancel.call_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize("step", list(ImportStep))
async def test_import_claim_checks_runnable_step_atomically(jobs, step):
    service, project_id = jobs
    job = import_job(project_id, step)
    await service.create_job(job)
    handler = get_job_handler(job.type)
    eligible = step in {ImportStep.QUEUED_FOR_DETECTION, ImportStep.QUEUED_FOR_IMPORT}
    assert handler.is_runnable(job) == eligible

    claimed = await service.claim_pending_job(job.id, job.type)

    assert (claimed is not None) == eligible
    if eligible:
        assert claimed.status == JobStatus.RUNNING
        assert claimed.message == "Preparing dataset import"
        assert claimed.start_time is not None
        assert await service.claim_pending_job(job.id, job.type) is None
    else:
        assert (await service.get_job_by_id(job.id)).status == JobStatus.PENDING


@pytest.mark.anyio
async def test_import_claim_does_not_trust_stale_runnable_selection(jobs):
    service, project_id = jobs
    job = import_job(project_id, ImportStep.QUEUED_FOR_IMPORT)
    await service.create_job(job)
    assert get_job_handler(job.type).is_runnable(job)
    payload = job.payload.model_copy(update={"step": ImportStep.AWAITING_USER_REVIEW})
    await service.update_job_payload(job.id, payload)

    assert await service.claim_pending_job(job.id, job.type) is None
    assert (await service.get_job_by_id(job.id)).status == JobStatus.PENDING


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("step", "status"),
    [
        (ImportStep.AWAITING_ARCHIVE_UPLOAD, JobStatus.PENDING),
        (ImportStep.QUEUED_FOR_DETECTION, JobStatus.PENDING),
        (ImportStep.DETECTING_FORMAT, JobStatus.RUNNING),
        (ImportStep.AWAITING_USER_REVIEW, JobStatus.PENDING),
        (ImportStep.QUEUED_FOR_IMPORT, JobStatus.PENDING),
        (ImportStep.IMPORTING_DATASET, JobStatus.RUNNING),
    ],
)
async def test_generic_interruption_cannot_fake_import_cancellation(jobs, step, status):
    service, project_id = jobs
    job = import_job(project_id, step, status)
    await service.create_job(job)
    scheduler = MagicMock(spec=Scheduler)

    with pytest.raises(InvalidJobStateError, match="does not support generic interruption"):
        await interrupt_job(job.id, service, scheduler)

    current = await service.get_job_by_id(job.id)
    assert current.status == status
    assert current.payload == job.payload
    assert current.end_time is None
    scheduler.cancel_job.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("assigned_step", "current_step", "failed"),
    [
        (ImportStep.QUEUED_FOR_DETECTION, ImportStep.QUEUED_FOR_IMPORT, False),
        (ImportStep.QUEUED_FOR_DETECTION, ImportStep.QUEUED_FOR_DETECTION, True),
        (ImportStep.QUEUED_FOR_IMPORT, ImportStep.QUEUED_FOR_IMPORT, True),
    ],
)
async def test_reaping_import_distinguishes_next_step_from_unclaimed_assignment(
    jobs, assigned_step, current_step, failed
):
    service, project_id = jobs
    assigned = import_job(project_id, assigned_step)
    await service.create_job(assigned)
    payload = assigned.payload.model_copy(update={"step": current_step})
    await service.update_job_payload(assigned.id, payload)

    @asynccontextmanager
    async def session_scope():
        yield service.session

    exited = MagicMock()
    exited.is_alive.return_value = False
    exited.pid = None if failed else 123
    with (
        patch("core.scheduler.get_async_db_session_ctx", session_scope),
        patch("core.scheduler.mp.Queue"),
        patch("core.scheduler.mp.Process") as process_factory,
    ):
        scheduler = Scheduler()
        scheduler.processes["dataset_import"] = (exited, assigned)
        try:
            await scheduler._reap_processes()
            current = await service.get_job_by_id(assigned.id)
            assert current.status == (JobStatus.FAILED if failed else JobStatus.PENDING)
            await scheduler._schedule_jobs()
            if failed:
                process_factory.assert_not_called()
            else:
                process_factory.assert_called_once()
                assert scheduler.processes["dataset_import"][1].payload.step == ImportStep.QUEUED_FOR_IMPORT
        finally:
            scheduler.processes.clear()
            scheduler.shutdown()


@pytest.mark.parametrize(
    ("job_type", "worker_path", "cancelable"),
    [
        (JobType.TRAINING, "workers.training_worker.TrainingWorker", True),
        (JobType.DATASET_IMPORT, "workers.dataset_import_worker.DatasetImportWorker", False),
    ],
)
def test_registered_handler_constructs_its_worker(job_type, worker_path, cancelable):
    handler = get_job_handler(job_type)
    assert handler.job_type == job_type
    assert handler.supports_running_cancel == cancelable
    job_id, stop, cancel, queue = uuid4(), MagicMock(), MagicMock(), MagicMock()
    with patch(worker_path) as worker:
        assert handler.create_worker(job_id, stop, cancel, queue) is worker.return_value
        expected_args = (job_id, stop, cancel, queue) if cancelable else (job_id, stop, queue)
        worker.assert_called_once_with(*expected_args)


@pytest.mark.parametrize("job_type", [JobType.TRAINING])
def test_cooperative_worker_requires_cancellation_event(job_type):
    with pytest.raises(ValueError, match="require a cancellation event"):
        get_job_handler(job_type).create_worker(uuid4(), MagicMock(), None, MagicMock())


def test_every_job_type_has_a_lifecycle_handler():
    assert {get_job_handler(job_type).job_type for job_type in JobType} == set(JobType)
