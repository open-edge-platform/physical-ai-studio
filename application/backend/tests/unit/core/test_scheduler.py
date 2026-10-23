# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from core.scheduler import Scheduler, _run_job
from schemas.base_job import JobStatus, JobType
from schemas.dataset_import_job import DatasetImportJobPayload, ImportStep
from schemas.job import (
    DatasetImportJob,
    LocalTrainJobPayload,
    OptimizeModelJob,
    OptimizeModelJobPayload,
    RemoteTrainJobPayload,
    TrainingDevice,
    TrainJob,
)
from services.event_processor import EventType
from services.job_handlers import get_job_handler
from services.training_service import TrainingService
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
        patch(
            "services.training_service.TrainingService.recover_orphan_job",
            new_callable=AsyncMock,
            wraps=TrainingService.recover_orphan_job,
        ) as recover,
    ):
        jobs = job_service.return_value
        jobs.get_pending_jobs = AsyncMock(return_value=[])
        jobs.get_job_list = AsyncMock(return_value=[])
        jobs.get_jobs_by_ids = AsyncMock(return_value=[])
        jobs.update_job_status = AsyncMock()
        yield jobs, None, recover


@pytest.mark.anyio
async def test_idle_startup_does_not_launch_workers(scheduler, services):
    scheduler.mp_stop_event.wait = MagicMock(side_effect=lambda _: scheduler.mp_stop_event.set())
    with patch(f"{MODULE}.mp.Process") as process:
        scheduler.start_workers()
        scheduler._thread.join(timeout=5)
    assert not scheduler._thread.is_alive()
    process.assert_not_called()
    services[0].get_job_list.assert_awaited()


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
    jobs, _, _ = services
    trainer = uuid4()
    pending = [
        _training_job(),
        _training_job(),
        _training_job(trainer_id=trainer, gpu=0),
        _training_job(trainer_id=trainer, gpu=0),
        _training_job(trainer_id=trainer, gpu=1),
        _training_job(trainer_id=uuid4()),
    ]
    import_job = _import_job()
    import_job.status = JobStatus.PENDING
    jobs.get_pending_jobs.return_value = [*pending, import_job]
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
        jobs.get_pending_jobs.assert_awaited_with()
        call = process.call_args_list[0]
        assert call.kwargs["target"] is _run_job
        assert call.kwargs["args"][:2] == (JobType.TRAINING, pending[0].id)
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
    assert get_job_handler(JobType.TRAINING).resource_is_busy(target, active) is busy


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
    jobs.get_pending_jobs.return_value = [second]
    await scheduler._reap_processes()
    process.join.assert_called_once_with()
    process.close.assert_called_once()
    assert not scheduler.processes
    assert not scheduler._cancel_events
    recover.assert_not_awaited()
    jobs.get_job_list.assert_awaited_once()
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
        job_id=job.id, status=JobStatus.FAILED, message="Training worker exited before completing the job"
    )
    scheduler.event_queue.put.assert_called_once()
    assert not scheduler.processes


@pytest.mark.anyio
async def test_process_start_failure_is_reaped(scheduler, services):
    jobs, _, _ = services
    job = _import_job()
    jobs.get_jobs_by_ids.return_value = [job]
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
    job.status = JobStatus.RUNNING
    services[0].get_job_list.return_value = [job]
    services[2].return_value = job.model_copy(update={"status": status})

    await scheduler._recover_jobs()

    scheduler.event_queue.put.assert_called_once_with((EventType.JOB_UPDATE, services[2].return_value))


@pytest.mark.anyio
async def test_recovery_excludes_live_jobs_and_fails_only_abandoned_imports(scheduler, services):
    jobs, _, recover = services
    training = _training_job()
    active_import, orphan_import = _import_job(), _import_job()
    process = MagicMock()
    scheduler.processes = {"local": (process, training), "dataset_import": (process, active_import)}
    jobs.get_job_list.return_value = [training, active_import, orphan_import]
    await scheduler._recover_jobs()
    recover.assert_not_awaited()
    jobs.update_job_status.assert_awaited_once_with(
        orphan_import.id, JobStatus.FAILED, message="Dataset import worker stopped before completing the job"
    )
    scheduler._shutdown_processes()


@pytest.mark.anyio
async def test_review_wait_does_not_fail_or_restart_import(scheduler, services):
    jobs, _, _ = services
    job = _import_job()
    job.status = JobStatus.PENDING
    job.payload.step = ImportStep.AWAITING_USER_REVIEW
    process = MagicMock()
    process.is_alive.return_value = False
    scheduler.processes["dataset_import"] = (process, job)
    jobs.get_jobs_by_ids.return_value = [job]
    jobs.get_pending_jobs.return_value = [job]
    await scheduler._reap_processes()
    with patch(f"{MODULE}.mp.Process") as factory:
        await scheduler._schedule_jobs()
        factory.assert_not_called()
    jobs.update_job_status.assert_not_awaited()
    jobs.get_pending_jobs.assert_awaited_once_with()


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
    services[0].get_pending_jobs.return_value = [_training_job()]
    scheduler.mp_stop_event.set()
    with patch(f"{MODULE}.mp.Process") as process:
        await scheduler._schedule_jobs()
    process.assert_not_called()
    services[0].update_job_status.assert_not_awaited()


@pytest.mark.parametrize("job", [_training_job(), _training_job(trainer_id=uuid4(), gpu=1)])
def test_training_target_keys_still_use_shared_registry(job):
    expected = (
        "local" if isinstance(job.payload, LocalTrainJobPayload) else f"remote:{job.payload.remote_trainer_id}:cuda:1"
    )
    assert target_key(job.payload) == expected


def _optimization_job():
    return OptimizeModelJob(project_id=uuid4(), payload=OptimizeModelJobPayload(model_id=uuid4()))


@pytest.mark.anyio
async def test_optimization_shares_local_slot_and_gets_its_own_cancellation_event(scheduler, services):
    jobs, _, _ = services
    optimization = _optimization_job()
    jobs.get_pending_jobs.return_value = [optimization, _optimization_job()]
    with patch(f"{MODULE}.mp.Process") as process:
        await scheduler._schedule_jobs()
        await scheduler._schedule_jobs()
        process.assert_called_once()
        args = process.call_args.kwargs["args"]
        assert args[:2] == (JobType.OPTIMIZE_MODEL, optimization.id)
        assert args[3] is scheduler._cancel_events[optimization.id]
        scheduler.cancel_job(optimization.id)
        assert args[3].is_set()
        jobs.get_pending_jobs.return_value = [_training_job()]
        await scheduler._schedule_jobs()
        process.assert_called_once()
        scheduler._shutdown_processes()


@pytest.mark.anyio
async def test_local_training_blocks_optimization_but_remote_training_does_not(scheduler, services):
    jobs, _, _ = services
    optimization = _optimization_job()
    jobs.get_pending_jobs.return_value = [_training_job(), optimization]
    with patch(f"{MODULE}.mp.Process") as process:
        await scheduler._schedule_jobs()
        process.assert_called_once()
        assert scheduler.processes["local"][1].type == JobType.TRAINING
        scheduler._shutdown_processes()
        jobs.get_pending_jobs.return_value = [_training_job(trainer_id=uuid4()), optimization]
        await scheduler._schedule_jobs()
        assert process.call_count == 3
        assert scheduler.processes["local"][1].id == optimization.id
        scheduler._shutdown_processes()


@pytest.mark.anyio
async def test_recovery_cleans_only_abandoned_optimization_output_and_notifies(scheduler, services):
    jobs, _, _ = services
    active, orphan = _optimization_job(), _optimization_job()
    active.status = orphan.status = JobStatus.RUNNING
    scheduler.processes["local"] = (MagicMock(), active)
    jobs.get_job_list.return_value = [active, orphan]
    with patch("services.model_optimization_service.ModelOptimizationService.cleanup") as cleanup:
        await scheduler._recover_jobs()
        cleanup.assert_called_once_with(orphan.id)
    jobs.update_job_status.assert_awaited_once_with(
        orphan.id, JobStatus.FAILED, message="Model optimization worker stopped before completing the job"
    )
    scheduler.event_queue.put.assert_called_once()
    scheduler._shutdown_processes()


@pytest.mark.parametrize("job_type", list(JobType))
def test_worker_registry_imports_only_the_assigned_job_type(job_type):
    handler = get_job_handler(job_type)
    with patch.object(handler, "create_worker") as create_worker:
        cancel = MagicMock()
        job_id, stop, queue = uuid4(), MagicMock(), MagicMock()
        _run_job(job_type, job_id, stop, cancel, queue)
    create_worker.assert_called_once_with(job_id, stop, cancel, queue)
    create_worker.return_value.run.assert_called_once()


@pytest.mark.anyio
async def test_reap_notifies_clients_if_optimization_exits_after_publication(scheduler, services):
    job = _optimization_job()
    job.status = JobStatus.COMPLETED
    services[0].get_jobs_by_ids.return_value = [job]
    process = MagicMock()
    process.is_alive.return_value = False
    scheduler.processes["local"] = (process, job)
    with patch("services.model_optimization_service.ModelOptimizationService.cleanup") as cleanup:
        await scheduler._reap_processes()
        cleanup.assert_not_called()
    scheduler.event_queue.put.assert_called_once_with((EventType.JOB_UPDATE, job))
    services[0].update_job_status.assert_not_awaited()
