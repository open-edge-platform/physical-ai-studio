# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from core.scheduler import Scheduler, _run_job
from schemas.base_job import JobStatus, JobType
from schemas.dataset_import_job import DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob, LocalTrainJobPayload, RemoteTrainJobPayload, TrainingDevice, TrainJob
from services.event_processor import EventType
from services.training_targets import target_key

MODULE = "core.scheduler"


def _training_job(*, trainer_id=None, gpu=None):
    common = {"project_id": uuid4(), "dataset_id": uuid4(), "policy": "act", "model_name": "test"}
    if trainer_id is None:
        payload = LocalTrainJobPayload(**common)
    else:
        payload = RemoteTrainJobPayload(**common, remote_trainer_id=trainer_id)
        if gpu is not None:
            payload.device = TrainingDevice(type="cuda", index=gpu)
    return TrainJob(project_id=payload.project_id, payload=payload)


def _import_job():
    return DatasetImportJob(
        project_id=uuid4(),
        status=JobStatus.RUNNING,
        payload=DatasetImportJobPayload(archive_staging_id=str(uuid4()), step=ImportStep.QUEUED_FOR_DETECTION),
    )


@pytest.fixture
def scheduler():
    @asynccontextmanager
    async def session_scope():
        yield MagicMock()

    with (
        patch(f"{MODULE}.mp.Queue"),
        patch(f"{MODULE}.get_async_db_session_ctx", session_scope),
    ):
        instance = Scheduler()
        yield instance
        instance.shutdown()


@pytest.fixture
def services():
    with (
        patch(f"{MODULE}.JobService") as job_service,
        patch(f"{MODULE}.DatasetImportService") as import_service,
        patch(f"{MODULE}.TrainingService.abort_orphan_jobs", new_callable=AsyncMock) as recover,
    ):
        recover.return_value = []
        jobs = job_service.return_value
        jobs.get_pending_train_jobs = AsyncMock(return_value=[])
        jobs.get_job_list = AsyncMock(return_value=[])
        jobs.get_jobs_by_ids = AsyncMock(return_value=[])
        jobs.update_job_status = AsyncMock()
        imports = import_service.return_value
        imports.claim_pending_dataset_import_job = AsyncMock(return_value=None)
        yield jobs, imports, recover


@pytest.mark.anyio
async def test_idle_startup_does_not_launch_workers(scheduler, services):
    scheduler.mp_stop_event.wait = MagicMock(side_effect=lambda _: scheduler.mp_stop_event.set())
    with patch(f"{MODULE}.mp.Process") as process:
        scheduler.start_workers()
        scheduler._thread.join(timeout=5)
    assert not scheduler._thread.is_alive()
    process.assert_not_called()
    services[2].assert_awaited()


@pytest.mark.anyio
async def test_startup_recovery_failure_is_retried_before_scheduling(scheduler):
    async def stop_after_scheduling():
        scheduler.mp_stop_event.set()

    scheduler.mp_stop_event.wait = MagicMock()
    with (
        patch.object(
            scheduler, "_recover_jobs", AsyncMock(side_effect=[RuntimeError("database busy"), None, None])
        ) as recover,
        patch.object(scheduler, "_reap_processes", AsyncMock()),
        patch.object(scheduler, "_schedule_jobs", AsyncMock(side_effect=stop_after_scheduling)) as schedule,
        patch.object(scheduler, "_shutdown_processes") as shutdown,
    ):
        await scheduler._run_loop()
    assert recover.await_count == 3
    schedule.assert_awaited_once()
    shutdown.assert_called_once()


@pytest.mark.anyio
async def test_shutdown_recovery_failure_does_not_escape_the_scheduler_thread(scheduler):
    scheduler.mp_stop_event.set()
    with (
        patch.object(
            scheduler, "_recover_jobs", AsyncMock(side_effect=RuntimeError("database unavailable"))
        ) as recover,
        patch.object(scheduler, "_shutdown_processes") as shutdown,
    ):
        await scheduler._run_loop()
    recover.assert_awaited_once()
    shutdown.assert_called_once()


@pytest.mark.anyio
async def test_schedules_training_per_target_alongside_one_import(scheduler, services):
    jobs, imports, _ = services
    trainer = uuid4()
    pending = [
        _training_job(),
        _training_job(),
        _training_job(trainer_id=trainer, gpu=0),
        _training_job(trainer_id=trainer, gpu=0),
        _training_job(trainer_id=trainer, gpu=1),
        _training_job(trainer_id=uuid4()),
    ]
    jobs.get_pending_train_jobs.return_value = pending
    import_job = _import_job()
    imports.claim_pending_dataset_import_job.return_value = import_job
    with patch(f"{MODULE}.mp.Process") as process:
        await scheduler._schedule_jobs()
        await scheduler._schedule_jobs()
        assert process.call_count == 5
        assert process.return_value.start.call_count == 5
        assert {job.id for _, job in scheduler.processes.values()} == {
            pending[0].id,
            pending[2].id,
            pending[4].id,
            pending[5].id,
            import_job.id,
        }
        imports.claim_pending_dataset_import_job.assert_awaited_once()
        call = process.call_args_list[0]
        assert call.kwargs["target"] is _run_job
        assert call.kwargs["args"][:2] == (JobType.TRAINING, pending[0].id)
        scheduler._shutdown_processes()


@pytest.mark.anyio
async def test_same_policy_batch_jobs_start_independently_without_duplicate_starts(scheduler, services):
    trainer_id = uuid4()
    pending = [_training_job(trainer_id=trainer_id), _training_job(trainer_id=trainer_id)]
    for job in pending:
        job.payload.remote_trainer_url = "aws-batch://us-east-1/jobs-bucket"
    services[0].get_pending_train_jobs.return_value = [*pending, pending[0]]

    with patch(f"{MODULE}.mp.Process") as process:
        await scheduler._schedule_jobs()
        await scheduler._schedule_jobs()

        assert process.call_count == 2
        assert {job.id for _, job in scheduler.processes.values()} == {job.id for job in pending}
        scheduler.cancel_job(pending[0].id)
        assert scheduler._cancel_events[pending[0].id].is_set()
        assert not scheduler._cancel_events[pending[1].id].is_set()
        scheduler._shutdown_processes()


def test_cancellation_signals_only_the_assigned_training_process(scheduler):
    first, second = _training_job(), _training_job(trainer_id=uuid4())
    with patch(f"{MODULE}.mp.Process") as process:
        scheduler._start_job("local", first)
        scheduler._start_job(target_key(second.payload), second)
        first_event = process.call_args_list[0].kwargs["args"][3]
        second_event = process.call_args_list[1].kwargs["args"][3]
        scheduler.cancel_job(first.id)
        scheduler.cancel_job(uuid4())
        assert first_event.is_set()
        assert not second_event.is_set()
        assert not scheduler.mp_stop_event.is_set()
        scheduler._shutdown_processes()
    assert not scheduler._cancel_events


@pytest.mark.parametrize(
    ("target", "active", "busy"),
    [
        ("local", {"local": None}, True),
        ("remote:trainer:cuda:1", {"remote:trainer:cuda:0": None}, False),
        ("remote:trainer:cuda:0", {"remote:trainer:cuda:0": None}, True),
        ("remote:trainer", {"remote:trainer:cuda:0": None}, True),
        ("remote:trainer:cuda:1", {"remote:trainer": None}, True),
    ],
)
def test_target_reservations(target, active, busy):
    assert Scheduler._target_is_busy(target, active) is busy


@pytest.mark.anyio
async def test_reap_releases_target_clears_interrupt_and_starts_next_job(scheduler, services):
    jobs, _, recover = services
    first, second = _training_job(), _training_job()
    first.status = JobStatus.COMPLETED
    process = MagicMock()
    process.is_alive.return_value = False
    scheduler.processes["local"] = (process, first)
    scheduler._cancel_events[first.id] = MagicMock()
    jobs.get_jobs_by_ids.return_value = [first]
    jobs.get_pending_train_jobs.return_value = [second]
    await scheduler._reap_processes()
    process.join.assert_called_once_with()
    process.close.assert_called_once()
    assert not scheduler.processes
    assert not scheduler._cancel_events
    recover.assert_awaited_once()
    with patch(f"{MODULE}.mp.Process"):
        await scheduler._schedule_jobs()
        assert scheduler.processes["local"][1].id == second.id
        scheduler._shutdown_processes()


@pytest.mark.anyio
async def test_cancellation_after_completion_and_reaping_is_pruned(scheduler, services):
    job = _training_job()
    job.status = JobStatus.COMPLETED
    services[0].get_jobs_by_ids.return_value = [job]
    process = MagicMock()
    process.is_alive.return_value = False
    scheduler.processes["local"] = (process, job)
    await scheduler._reap_processes()
    scheduler.cancel_job(job.id)
    assert scheduler._cancel_events[job.id].is_set()
    await scheduler._reap_processes()
    assert not scheduler._cancel_events


@pytest.mark.anyio
@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.CANCELED, None])
async def test_unassigned_cancellation_is_retained_only_for_nonterminal_jobs(scheduler, services, status):
    job = _training_job(trainer_id=uuid4())
    if status is not None:
        job.status = status
    services[0].get_jobs_by_ids.return_value = [job] if status is not None else []
    scheduler.cancel_job(job.id)
    await scheduler._reap_processes()
    assert (job.id in scheduler._cancel_events) == (status in {JobStatus.PENDING, JobStatus.RUNNING})


@pytest.mark.anyio
async def test_deleted_terminal_job_still_releases_target(scheduler, services):
    job = _training_job()
    process = MagicMock()
    process.is_alive.return_value = False
    scheduler.processes["local"] = (process, job)
    await scheduler._reap_processes()
    assert not scheduler.processes
    services[0].update_job_status.assert_not_awaited()


@pytest.mark.anyio
async def test_failed_preparation_does_not_leave_training_pending(scheduler, services):
    jobs, _, _ = services
    job = _training_job()
    process = MagicMock()
    process.is_alive.return_value = False
    scheduler.processes["local"] = (process, job)
    jobs.get_jobs_by_ids.return_value = [job]
    await scheduler._reap_processes()
    jobs.update_job_status.assert_awaited_once_with(
        job.id, JobStatus.FAILED, message="Training worker exited before completing the job"
    )
    scheduler.event_queue.put.assert_called_once()
    assert not scheduler.processes


@pytest.mark.anyio
async def test_process_start_failure_is_reaped(scheduler, services):
    jobs, _, _ = services
    job = _import_job()
    jobs.get_job_list.return_value = [job]
    with patch(f"{MODULE}.mp.Process") as process:
        process.return_value.start.side_effect = OSError("cannot start")
        process.return_value.pid = None
        process.return_value.is_alive.return_value = False
        with pytest.raises(OSError):
            scheduler._start_job("dataset_import", job)
        await scheduler._reap_processes()
        process.return_value.join.assert_not_called()
    assert not scheduler.processes
    jobs.update_job_status.assert_awaited_once_with(
        job.id, JobStatus.FAILED, message="Dataset import worker stopped before completing the job"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("status", [JobStatus.FAILED, JobStatus.PENDING])
async def test_recovery_emits_updated_training_jobs(scheduler, services, status):
    job = _training_job(trainer_id=uuid4() if status == JobStatus.PENDING else None)
    job.status = status
    services[2].return_value = [job]

    await scheduler._recover_jobs()

    scheduler.event_queue.put.assert_called_once_with((EventType.JOB_UPDATE, job))


@pytest.mark.anyio
async def test_recovery_excludes_live_jobs_and_fails_only_abandoned_imports(scheduler, services):
    jobs, _, recover = services
    training = _training_job()
    active_import, orphan_import = _import_job(), _import_job()
    process = MagicMock()
    scheduler.processes = {"local": (process, training), "dataset_import": (process, active_import)}
    jobs.get_job_list.return_value = [active_import, orphan_import]
    await scheduler._recover_jobs()
    assert recover.await_args.kwargs["exclude_job_ids"] == frozenset({training.id, active_import.id})
    jobs.update_job_status.assert_awaited_once_with(
        orphan_import.id, JobStatus.FAILED, message="Dataset import worker stopped before completing the job"
    )
    scheduler._shutdown_processes()


@pytest.mark.anyio
async def test_review_wait_does_not_fail_or_restart_import(scheduler, services):
    jobs, imports, _ = services
    job = _import_job()
    job.status = JobStatus.PENDING
    job.payload.step = ImportStep.AWAITING_USER_REVIEW
    process = MagicMock()
    process.is_alive.return_value = False
    scheduler.processes["dataset_import"] = (process, job)
    await scheduler._reap_processes()
    with patch(f"{MODULE}.mp.Process") as factory:
        await scheduler._schedule_jobs()
        factory.assert_not_called()
    jobs.update_job_status.assert_not_awaited()
    imports.claim_pending_dataset_import_job.assert_awaited_once()


def test_shutdown_joins_then_terminates_and_kills_unresponsive_workers(scheduler):
    process = MagicMock()
    process.is_alive.return_value = True
    scheduler.processes["local"] = (process, _training_job())
    scheduler.mp_stop_event.set()
    scheduler._shutdown_processes()
    assert process.join.call_args_list[0].kwargs == {"timeout": 10}
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.join.call_args_list[-1].args == ()
    process.close.assert_called_once()
    assert not scheduler.processes


@pytest.mark.anyio
async def test_shutdown_prevents_new_processes(scheduler, services):
    services[0].get_pending_train_jobs.return_value = [_training_job()]
    scheduler.mp_stop_event.set()
    with patch(f"{MODULE}.mp.Process") as process:
        await scheduler._schedule_jobs()
    process.assert_not_called()
    services[1].claim_pending_dataset_import_job.assert_not_awaited()


@pytest.mark.parametrize("job", [_training_job(), _training_job(trainer_id=uuid4(), gpu=1)])
def test_training_target_keys_still_use_shared_registry(job):
    expected = (
        "local" if isinstance(job.payload, LocalTrainJobPayload) else f"remote:{job.payload.remote_trainer_id}:cuda:1"
    )
    assert target_key(job.payload) == expected
