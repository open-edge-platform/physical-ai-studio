# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Exercise on-demand workers with real spawn processes and an isolated SQLite DB."""

import asyncio
import json
import multiprocessing as mp
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from api.job import interrupt_job
from core.scheduler import Scheduler
from db import get_async_db_session_ctx
from db.schema import ProjectDB
from exceptions import ResourceInUseError
from schemas import Snapshot
from schemas.base_job import JobStatus
from schemas.dataset_import_job import DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob, LocalTrainJobPayload, RemoteTrainJobPayload, TrainJob
from services.event_processor import EventProcessor, EventType
from services.job_service import JobService


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.parametrize("stale_remote_id", [None, uuid4()])
async def test_cancel_during_snapshot_preparation_never_starts_training(
    migrated_db, monkeypatch, tmp_path, stale_remote_id
):
    from workers import training_worker as module

    project_id = uuid4()
    job = TrainJob(
        project_id=project_id,
        payload=LocalTrainJobPayload(
            project_id=project_id,
            dataset_id=uuid4(),
            policy="act",
            model_name="Cancel copy",
            remote_job_id=stale_remote_id,
        ),
        created_at=datetime.now(tz=UTC),
    )
    async with get_async_db_session_ctx() as session:
        session.add(ProjectDB(id=str(project_id), name="Preparation cancellation"))
        await session.commit()
        await JobService(session).create_job(job)

    copying, resume = asyncio.Event(), asyncio.Event()

    async def paused_snapshot(*args, **kwargs):
        copying.set()
        await resume.wait()
        return Snapshot(id=uuid4(), dataset_id=job.payload.dataset_id, path=str(tmp_path))

    monkeypatch.setattr(module.DatasetService, "get_dataset_by_id", AsyncMock())
    monkeypatch.setattr(module.SnapshotService, "create_snapshot_for_dataset", AsyncMock(side_effect=paused_snapshot))
    backend = AsyncMock()
    monkeypatch.setattr(module, "get_training_backend", backend)
    scheduler = Scheduler()
    cancel_event = mp.Event()
    scheduler._cancel_events[job.id] = cancel_event
    worker = module.TrainingWorker(
        job_id=job.id, stop_event=scheduler.mp_stop_event, cancel_event=cancel_event, event_queue=Queue()
    )
    task = asyncio.create_task(worker.run_loop())
    try:
        await asyncio.wait_for(copying.wait(), timeout=5)
        async with get_async_db_session_ctx() as session:
            service = JobService(session)
            await interrupt_job(job.id, service, scheduler)
            assert (await service.get_job_by_id(job.id)).status == JobStatus.RUNNING
            with pytest.raises(ResourceInUseError):
                await service.delete_job(job.id)
        resume.set()
        await asyncio.wait_for(task, timeout=5)
        backend.assert_not_awaited()
        async with get_async_db_session_ctx() as session:
            assert (await JobService(session).get_job_by_id(job.id)).status == JobStatus.CANCELED
    finally:
        resume.set()
        await asyncio.gather(task, return_exceptions=True)
        scheduler.shutdown()
        scheduler.event_queue.close()


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.parametrize("claim_first", [False, True])
@pytest.mark.parametrize("stale_remote_id", [None, uuid4()])
async def test_queued_cancellation_and_claim_cannot_overwrite_each_other(migrated_db, claim_first, stale_remote_id):
    project_id = uuid4()
    job = TrainJob(
        project_id=project_id,
        payload=LocalTrainJobPayload(
            project_id=project_id, dataset_id=uuid4(), policy="act", model_name="Race", remote_job_id=stale_remote_id
        ),
        created_at=datetime.now(tz=UTC),
    )
    async with get_async_db_session_ctx() as session:
        session.add(ProjectDB(id=str(project_id), name="Claim cancellation race"))
        await session.commit()
        await JobService(session).create_job(job)
    async with get_async_db_session_ctx() as first, get_async_db_session_ctx() as second:
        if claim_first:
            claimed = await JobService(first).claim_pending_train_job(job.id)
            assert claimed.status == JobStatus.RUNNING
            assert claimed.start_time is not None
            assert await JobService(second).cancel_pending_train_job(job.id) is None
        else:
            canceled = await JobService(first).cancel_pending_train_job(job.id)
            assert canceled.status == JobStatus.CANCELED
            assert canceled.end_time is not None
            assert await JobService(second).claim_pending_train_job(job.id) is None


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancellation_with_stale_pending_read_signals_claimed_worker(migrated_db, monkeypatch):
    project_id = uuid4()
    job = TrainJob(
        project_id=project_id,
        payload=LocalTrainJobPayload(project_id=project_id, dataset_id=uuid4(), policy="act", model_name="Stale read"),
        created_at=datetime.now(tz=UTC),
    )
    async with get_async_db_session_ctx() as session:
        session.add(ProjectDB(id=str(project_id), name="Stale cancellation read"))
        await session.commit()
        service = JobService(session)
        await service.create_job(job)
        original_cancel = service.cancel_pending_job

        async def claim_before_cancellation(job_id, job_type):
            async with get_async_db_session_ctx() as worker_session:
                assert await JobService(worker_session).claim_pending_train_job(job_id) is not None
            return await original_cancel(job_id, job_type)

        monkeypatch.setattr(service, "cancel_pending_job", claim_before_cancellation)
        scheduler = Scheduler()
        cancel_event = mp.Event()
        scheduler._cancel_events[job.id] = cancel_event
        try:
            await interrupt_job(job.id, service, scheduler)
            assert cancel_event.is_set()
            assert (await service.get_job_by_id(job.id)).status == JobStatus.RUNNING
        finally:
            scheduler.shutdown()
            scheduler.event_queue.close()


def _block_claimed_training(job_type, job_id, stop_event, cancel_event, result_queue):
    from workers.training_worker import TrainingWorker

    worker = TrainingWorker(job_id=job_id, stop_event=stop_event, cancel_event=cancel_event, event_queue=result_queue)

    async def block(job, payload):
        result_queue.put("claimed")
        await asyncio.sleep(120)

    worker._run_training_job = block
    worker.run()


@pytest.mark.integration
@pytest.mark.anyio
async def test_killed_worker_releases_target_and_next_job_runs(migrated_db, monkeypatch):
    project_id = uuid4()
    jobs = [
        TrainJob(
            project_id=project_id,
            created_at=datetime.now(tz=UTC),
            payload=LocalTrainJobPayload(project_id=project_id, dataset_id=uuid4(), policy="act", model_name=name),
        )
        for name in ("Kill this worker", "Next job")
    ]
    async with get_async_db_session_ctx() as session:
        session.add(ProjectDB(id=str(project_id), name="Worker crash"))
        await session.commit()
        for job in jobs:
            await JobService(session).create_job(job)
    monkeypatch.setattr("core.scheduler.mp", mp.get_context("spawn"))
    scheduler = Scheduler()
    try:
        with monkeypatch.context() as first_worker:
            first_worker.setattr("core.scheduler._run_job", _block_claimed_training)
            await scheduler._schedule_jobs()
        process = scheduler.processes["local"][0]
        assert await asyncio.to_thread(scheduler.event_queue.get, True, 30) == "claimed"
        async with get_async_db_session_ctx() as session:
            assert (await JobService(session).get_job_by_id(jobs[0].id)).status == JobStatus.RUNNING
        process.kill()
        await asyncio.to_thread(process.join, 10)
        assert process.exitcode is not None and process.exitcode < 0
        await scheduler._reap_processes()
        assert not scheduler.processes
        assert not scheduler._cancel_events
        while True:
            event, recovered = await asyncio.to_thread(scheduler.event_queue.get, True, 5)
            if recovered.id == jobs[0].id:
                break
        assert event == EventType.JOB_UPDATE
        assert recovered.status == JobStatus.FAILED
        async with get_async_db_session_ctx() as session:
            assert (await JobService(session).get_job_by_id(jobs[0].id)).status == JobStatus.FAILED
        await scheduler._schedule_jobs()
        process, assigned = scheduler.processes["local"]
        assert assigned.id == jobs[1].id
        await asyncio.to_thread(process.join, 30)
        assert not process.is_alive()
        await scheduler._reap_processes()
        assert not scheduler.processes
        async with get_async_db_session_ctx() as session:
            # The next real worker ran and reported its missing dataset, rather than staying queued.
            assert (await JobService(session).get_job_by_id(jobs[1].id)).status == JobStatus.FAILED
    finally:
        scheduler.mp_stop_event.set()
        scheduler._shutdown_processes()
        scheduler.shutdown()
        scheduler.event_queue.close()


@pytest.mark.integration
@pytest.mark.anyio
async def test_killed_remote_worker_reattaches_without_resubmitting(migrated_db, monkeypatch):
    project_id, remote_job_id = uuid4(), uuid4()
    attached = Queue()
    finish = threading.Event()
    requests = []

    class TrainerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(("GET", self.path))
            if self.path == "/health":
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == f"/jobs/{remote_job_id}/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                attached.put(self.path)
                if finish.wait(timeout=30):
                    state = json.dumps({"status": "canceled", "progress": 50})
                    try:
                        self.wfile.write(f"event: state\ndata: {state}\n\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
            else:
                self.send_error(404)

        def do_POST(self):
            requests.append(("POST", self.path))
            if self.path == f"/jobs/{remote_job_id}/cancel":
                body = b'{"status":"canceled"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                finish.set()
            else:
                self.send_error(400, "Reattachment must not submit or upload another job")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), TrainerHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    started = datetime(2026, 9, 23, 14, 0, tzinfo=UTC)
    job = TrainJob(
        project_id=project_id,
        status=JobStatus.RUNNING,
        start_time=started,
        created_at=datetime.now(tz=UTC),
        payload=RemoteTrainJobPayload(
            project_id=project_id,
            dataset_id=uuid4(),
            policy="act",
            model_name="Reattach",
            remote_trainer_id=uuid4(),
            remote_trainer_url=f"http://127.0.0.1:{server.server_port}",
            remote_job_id=remote_job_id,
        ),
    )
    async with get_async_db_session_ctx() as session:
        session.add(ProjectDB(id=str(project_id), name="Remote crash recovery"))
        await session.commit()
        await JobService(session).create_job(job)
    monkeypatch.setattr("core.scheduler.mp", mp.get_context("spawn"))
    scheduler = Scheduler()
    processor = EventProcessor(scheduler.event_queue)
    updates = asyncio.Queue()

    def capture_update(event, updated):
        if updated.id == job.id:
            updates.put_nowait(updated)

    processor.subscribe([EventType.JOB_UPDATE], capture_update)
    try:
        await scheduler._recover_jobs()
        recovered_update = await asyncio.wait_for(updates.get(), timeout=5)
        assert recovered_update.id == job.id
        assert recovered_update.status == JobStatus.PENDING
        await scheduler._schedule_jobs()
        assert await asyncio.to_thread(attached.get, True, 30) == f"/jobs/{remote_job_id}/events"
        process = next(iter(scheduler.processes.values()))[0]
        original_pid = process.pid
        process.kill()
        await asyncio.to_thread(process.join, 10)
        async with get_async_db_session_ctx() as session:
            service = JobService(session)
            original_cancel = service.cancel_pending_job

            async def recover_before_cancellation(job_id, job_type):
                # Reproduce a RUNNING API read followed by crash recovery to PENDING.
                await scheduler._reap_processes()
                return await original_cancel(job_id, job_type)

            monkeypatch.setattr(service, "cancel_pending_job", recover_before_cancellation)
            await interrupt_job(job.id, service, scheduler)
            recovered = await service.get_job_by_id(job.id)
            assert recovered.status == JobStatus.PENDING
            assert recovered.payload.remote_job_id == remote_job_id
        recovered_update = await asyncio.wait_for(updates.get(), timeout=5)
        assert recovered_update.id == job.id
        assert recovered_update.status == JobStatus.PENDING
        await scheduler._schedule_jobs()
        assert await asyncio.to_thread(finish.wait, 30)
        process = next(iter(scheduler.processes.values()))[0]
        assert process.pid != original_pid
        await asyncio.to_thread(process.join, 30)
        assert not process.is_alive()
        # A canceled reattachment now sends cancel before opening another event stream.
        assert attached.empty()
        await scheduler._reap_processes()
        async with get_async_db_session_ctx() as session:
            final = await JobService(session).get_job_by_id(job.id)
            assert final.status == JobStatus.CANCELED
            assert final.start_time == started
            assert final.payload.remote_job_id == remote_job_id
        assert [(method, path) for method, path in requests if method != "GET"] == [
            ("POST", f"/jobs/{remote_job_id}/cancel")
        ]
        assert not scheduler._cancel_events
    finally:
        finish.set()
        scheduler.mp_stop_event.set()
        await asyncio.to_thread(scheduler._shutdown_processes)
        scheduler.shutdown()
        processor.shutdown()
        await asyncio.gather(processor.task, return_exceptions=True)
        scheduler.event_queue.close()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        server_thread.join(timeout=5)


def _probe_cancellation(job_id, stop_event, cancel_event, result_queue):
    from workers.training_worker import TrainingWorker

    worker = TrainingWorker(job_id=job_id, stop_event=stop_event, cancel_event=cancel_event, event_queue=result_queue)
    result_queue.put("ready")
    assert cancel_event.wait(timeout=10)
    result_queue.put((worker._should_interrupt(), worker.should_stop()))


@pytest.mark.integration
def test_cancellation_event_reaches_spawned_worker_without_signaling_shutdown(monkeypatch):
    context = mp.get_context("spawn")
    monkeypatch.setattr("core.scheduler.mp", context)
    scheduler = Scheduler()
    job_id = uuid4()
    cancel_event = context.Event()
    scheduler._cancel_events[job_id] = cancel_event
    process = context.Process(
        target=_probe_cancellation, args=(job_id, scheduler.mp_stop_event, cancel_event, scheduler.event_queue)
    )
    try:
        process.start()
        assert scheduler.event_queue.get(timeout=30) == "ready"
        scheduler.cancel_job(job_id)
        assert scheduler.event_queue.get(timeout=10) == (True, False)
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        process.close()
        scheduler.shutdown()
        scheduler.event_queue.close()
        scheduler.event_queue.join_thread()


@pytest.mark.integration
@pytest.mark.anyio
async def test_spawned_workers_exit_and_release_reservations_on_failure(migrated_db, monkeypatch):
    project_id = uuid4()
    training = TrainJob(
        project_id=project_id,
        created_at=datetime.now(tz=UTC),
        payload=LocalTrainJobPayload(
            project_id=project_id, dataset_id=uuid4(), policy="act", model_name="missing-dataset"
        ),
    )
    importing = DatasetImportJob(
        project_id=project_id,
        created_at=datetime.now(tz=UTC),
        payload=DatasetImportJobPayload(archive_staging_id=str(uuid4()), step=ImportStep.QUEUED_FOR_DETECTION),
    )
    async with get_async_db_session_ctx() as session:
        session.add(ProjectDB(id=str(project_id), name="Scheduler process test"))
        await session.commit()
        service = JobService(session)
        await service.create_job(training)
        await service.create_job(importing)

    # Match production's spawn policy without changing the test runner's global context.
    monkeypatch.setattr("core.scheduler.mp", mp.get_context("spawn"))
    children_before = {process.pid for process in mp.active_children()}
    scheduler = Scheduler()
    try:
        assert {process.pid for process in mp.active_children()} == children_before
        assert not scheduler.processes
        await scheduler._schedule_jobs()
        assert set(scheduler.processes) == {"local", "dataset_import"}
        for process, _ in scheduler.processes.values():
            await asyncio.to_thread(process.join, 45)
            assert not process.is_alive(), f"Worker {process.name} did not exit after its single job"
        await scheduler._reap_processes()
        assert not scheduler.processes
        assert not scheduler._cancel_events
        async with get_async_db_session_ctx() as session:
            service = JobService(session)
            assert (await service.get_job_by_id(training.id)).status == JobStatus.FAILED
            assert (await service.get_job_by_id(importing.id)).status == JobStatus.FAILED
    finally:
        scheduler.mp_stop_event.set()
        scheduler._shutdown_processes()
        scheduler.shutdown()
