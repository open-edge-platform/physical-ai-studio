# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import multiprocessing as mp
import threading
from typing import TYPE_CHECKING

from loguru import logger

from db import get_async_db_session_ctx
from schemas.base_job import JobStatus
from services.event_processor import EventType
from services.job_handlers import get_job_handler
from services.job_service import JobService

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass
    from uuid import UUID

    from schemas import Job
    from schemas.base_job import JobType


def _run_job(
    job_type: JobType, job_id: UUID, stop_event: EventClass, cancel_event: EventClass | None, queue: mp.Queue
) -> None:
    """Construct and execute the assigned worker only inside its child process."""
    get_job_handler(job_type).create_worker(job_id, stop_event, cancel_event, queue).run()


class Scheduler:
    """Own short-lived processes; delegate job-specific lifecycle rules to handlers.

    Runnable candidates are considered in submission order. Handlers define their
    resource reservations, cancellation support, and abandoned-job recovery.
    """

    def __init__(self) -> None:
        self.mp_stop_event = mp.Event()
        self._cancel_events: dict[UUID, EventClass] = {}
        self.event_queue: mp.Queue = mp.Queue()
        self.processes: dict[str, tuple[mp.Process, Job]] = {}
        self._thread = threading.Thread(target=self._run, name="JobScheduler")

    def start_workers(self) -> None:
        """Start polling without launching any job processes."""
        self._thread.start()

    def _run(self) -> None:
        asyncio.run(self._run_loop())

    async def _run_loop(self) -> None:
        recovered = False
        try:
            while not self.mp_stop_event.is_set():
                try:
                    if not recovered:
                        await self._recover_jobs()
                        recovered = True
                    await self._reap_processes()
                    await self._schedule_jobs()
                except Exception:
                    recovered = False
                    logger.exception("Job scheduling failed; retrying on the next poll")
                self.mp_stop_event.wait(0.5)
        finally:
            self._shutdown_processes()
            try:
                await self._recover_jobs()
            except Exception:
                logger.exception("Shutdown recovery failed; abandoned jobs will be reconciled on the next startup")

    async def _schedule_jobs(self) -> None:
        async with get_async_db_session_ctx() as session:
            jobs = await JobService(session).get_pending_jobs()
        for job in jobs:
            if self.mp_stop_event.is_set():
                break
            handler = get_job_handler(job.type)
            if not handler.is_runnable(job):
                continue
            resource = handler.resource_key(job)
            if not handler.resource_is_busy(resource, self.processes):
                self._start_job(resource, job)

    def cancel_job(self, job_id: UUID) -> None:
        """Signal this job, retaining requests until a remote worker can reattach."""
        self._cancel_events.setdefault(job_id, mp.Event()).set()

    def _start_job(self, resource: str, job: Job) -> None:
        if self.mp_stop_event.is_set():
            return
        cancel_event = None
        if get_job_handler(job.type).supports_running_cancel:
            cancel_event = self._cancel_events.setdefault(job.id, mp.Event())
        process = mp.Process(
            target=_run_job,
            args=(job.type, job.id, self.mp_stop_event, cancel_event, self.event_queue),
            name=f"{job.type.value}-{job.id}",
            daemon=False,
        )
        # Keep the reservation even if start fails so the next reap reconciles the job.
        self.processes[resource] = (process, job)
        process.start()

    async def _reap_processes(self) -> None:
        for resource, (process, job) in list(self.processes.items()):
            if process.is_alive():
                continue
            if process.pid is not None:
                process.join()
            async with get_async_db_session_ctx() as session:
                service = JobService(session)
                current_jobs = await service.get_jobs_by_ids([job.id])
                current = current_jobs[0] if current_jobs else None
                if current is not None:
                    handler = get_job_handler(current.type)
                    if handler.needs_exit_recovery(job, current) and (
                        current.status == JobStatus.RUNNING or not self.mp_stop_event.is_set()
                    ):
                        current = await handler.recover(current, service)
                    # Reconcile committed state even if the worker exited before sending its event.
                    self.event_queue.put((EventType.JOB_UPDATE, current))
            if current is None or current.status not in {JobStatus.PENDING, JobStatus.RUNNING}:
                self._cancel_events.pop(job.id, None)
            if process.pid is not None:
                process.close()
            del self.processes[resource]
            await self._recover_jobs()

        unassigned = set(self._cancel_events.copy()) - {job.id for _, job in self.processes.values()}
        if unassigned:
            async with get_async_db_session_ctx() as session:
                jobs = await JobService(session).get_jobs_by_ids(list(unassigned))
            waiting = {job.id for job in jobs if job.status in {JobStatus.PENDING, JobStatus.RUNNING}}
            for job_id in unassigned - waiting:
                self._cancel_events.pop(job_id, None)

    async def _recover_jobs(self) -> None:
        """Reconcile abandoned executions without touching any live worker's job."""
        active_ids = frozenset(job.id for _, job in self.processes.values())
        async with get_async_db_session_ctx() as session:
            service = JobService(session)
            jobs = await service.get_job_list(extra_filters={"status": JobStatus.RUNNING})
            for job in jobs:
                if job.id not in active_ids:
                    updated = await get_job_handler(job.type).recover(job, service)
                    self.event_queue.put((EventType.JOB_UPDATE, updated))

    def _shutdown_processes(self) -> None:
        for process, _ in self.processes.values():
            if process.pid is None:
                continue
            process.join(timeout=10)
            if process.is_alive():
                logger.warning("Force terminating process: {}", process.name)
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()
        self.processes.clear()
        self._cancel_events.clear()

    def shutdown(self) -> None:
        """Stop polling, join or terminate active jobs, then release shared state."""
        self.mp_stop_event.set()
        if self._thread.ident is not None:
            self._thread.join()
