# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.job import interrupt_job
from core.scheduler import Scheduler
from db.schema import Base, ProjectDB
from exceptions import InvalidJobStateError
from schemas.base_job import JobStatus, JobType
from schemas.dataset_import_job import DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob, LocalTrainJobPayload, OptimizeModelJob, OptimizeModelJobPayload, TrainJob
from services.job_handlers import get_job_handler
from services.job_service import JobService


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
@pytest.mark.parametrize("optimization_first", [True, False])
async def test_shared_local_slot_uses_submission_order_across_job_types(jobs, optimization_first):
    service, project_id = jobs
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 1, 2, tzinfo=UTC)
    optimization = OptimizeModelJob(
        project_id=project_id,
        payload=OptimizeModelJobPayload(model_id=uuid4()),
        created_at=older if optimization_first else newer,
    )
    training = TrainJob(
        project_id=project_id,
        payload=LocalTrainJobPayload(project_id=project_id, dataset_id=uuid4(), policy="act", model_name="test"),
        created_at=newer if optimization_first else older,
    )
    review = import_job(project_id, ImportStep.AWAITING_USER_REVIEW)
    upload = import_job(project_id, ImportStep.AWAITING_ARCHIVE_UPLOAD)
    runnable_import = import_job(project_id, ImportStep.QUEUED_FOR_DETECTION)
    # Deliberately insert in the opposite order to the requested FIFO order.
    for job in (training, optimization, review, upload, runnable_import):
        await service.create_job(job)

    @asynccontextmanager
    async def session_scope():
        yield service.session

    with (
        patch("core.scheduler.get_async_db_session_ctx", session_scope),
        patch("core.scheduler.mp.Queue"),
        patch("core.scheduler.mp.Process"),
    ):
        scheduler = Scheduler()
        try:
            await scheduler._schedule_jobs()
            expected = optimization if optimization_first else training
            assert scheduler.processes["local"][1].id == expected.id
            assert scheduler.processes["dataset_import"][1].id == runnable_import.id
            assert len(scheduler.processes) == 2
            assert (await service.get_job_by_id(runnable_import.id)).status == JobStatus.PENDING
        finally:
            scheduler.processes.clear()
            scheduler.shutdown()


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
        (JobType.OPTIMIZE_MODEL, "workers.model_optimization_worker.ModelOptimizationWorker", True),
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


@pytest.mark.parametrize("job_type", [JobType.TRAINING, JobType.OPTIMIZE_MODEL])
def test_cooperative_worker_requires_cancellation_event(job_type):
    with pytest.raises(ValueError, match="require a cancellation event"):
        get_job_handler(job_type).create_worker(uuid4(), MagicMock(), None, MagicMock())


def test_every_job_type_has_a_lifecycle_handler():
    assert {get_job_handler(job_type).job_type for job_type in JobType} == set(JobType)
