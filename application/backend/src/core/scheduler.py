# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import multiprocessing as mp
import threading
from typing import TYPE_CHECKING

from loguru import logger

from db import get_async_db_session_ctx
from schemas.base_job import JobStatus, JobType
from schemas.job import DatasetImportJob, RemoteTrainJobPayload, TrainJobPayloadAdapter
from services.dataset_import.service import DatasetImportService
from services.event_processor import EventType
from services.job_service import JobService
from services.remote_trainer_service import RemoteTrainerService
from services.training_service import TrainingService
from services.training_targets import target_key

if TYPE_CHECKING:
    from collections.abc import Mapping
    from multiprocessing.synchronize import Event as EventClass
    from uuid import UUID

    from schemas import Job
    from workers.base import BaseProcessWorker


def _run_job(
    job_type: JobType, job_id: UUID, stop_event: EventClass, cancel_event: EventClass | None, queue: mp.Queue
) -> None:
    """Import and run the assigned worker only inside its child process."""
    worker: BaseProcessWorker
    if job_type == JobType.TRAINING:
        from workers.training_worker import TrainingWorker

        if cancel_event is None:
            raise ValueError("Training jobs require a cancellation event")
        worker = TrainingWorker(job_id=job_id, stop_event=stop_event, cancel_event=cancel_event, event_queue=queue)
    elif job_type == JobType.DATASET_IMPORT:
        from workers.dataset_import_worker import DatasetImportWorker

        worker = DatasetImportWorker(job_id=job_id, stop_event=stop_event, event_queue=queue)
    else:
        raise ValueError(f"Unsupported job type: {job_type}")
    worker.run()


class Scheduler:
    """Poll jobs in a thread and own the short-lived processes that execute them.

    Local training is exclusive; remote training retains per-trainer/GPU
    reservations. One dataset import can run alongside training.
    """

    def __init__(self) -> None:
        self.mp_stop_event = mp.Event()
        self._cancel_events: dict[UUID, EventClass] = {}
        self.event_queue: mp.Queue = mp.Queue()
        self.processes: dict[str, tuple[mp.Process, Job]] = {}
        self._thread = threading.Thread(target=self._run, name="JobScheduler")

    def start_workers(self) -> None:
        """Start polling without launching training or import processes."""
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

    @staticmethod
    def _target_is_busy(target: str, active: Mapping[str, object]) -> bool:
        return target in active or (
            target.startswith("remote:")
            and any(key.startswith(f"{target}:") or target.startswith(f"{key}:") for key in active)
        )

    async def _schedule_jobs(self) -> None:
        async with get_async_db_session_ctx() as session:
            jobs = await JobService(session).get_pending_train_jobs()
        for job in jobs:
            target = target_key(TrainJobPayloadAdapter.validate_python(job.payload))
            if target.startswith("batch:"):
                target = f"{target}:{job.id}"
            if not self._target_is_busy(target, self.processes):
                self._start_job(target, job)

        if "dataset_import" not in self.processes and not self.mp_stop_event.is_set():
            async with get_async_db_session_ctx() as session:
                job = await DatasetImportService(session).claim_pending_dataset_import_job()
            if isinstance(job, DatasetImportJob):
                self._start_job("dataset_import", job)

    def cancel_job(self, job_id: UUID) -> None:
        """Signal this job, retaining requests until a remote worker can reattach."""
        self._cancel_events.setdefault(job_id, mp.Event()).set()

    def _start_job(self, target: str, job: Job) -> None:
        if self.mp_stop_event.is_set():
            return
        cancel_event = None
        if job.type == JobType.TRAINING:
            cancel_event = self._cancel_events.setdefault(job.id, mp.Event())
        process = mp.Process(
            target=_run_job,
            args=(job.type, job.id, self.mp_stop_event, cancel_event, self.event_queue),
            name=f"{job.type.value}-{job.id}",
            daemon=False,
        )
        # Keep the reservation even if start fails so the next reap reconciles the job.
        self.processes[target] = (process, job)
        process.start()

    async def _reap_processes(self) -> None:
        for target, (process, job) in list(self.processes.items()):
            if process.is_alive():
                continue
            if process.pid is not None:
                process.join()
            reattaching = False
            # Startup failures can leave training PENDING before the worker claims it.
            if job.type == JobType.TRAINING:
                async with get_async_db_session_ctx() as session:
                    service = JobService(session)
                    current_jobs = await service.get_jobs_by_ids([job.id])
                    if current_jobs:
                        current = current_jobs[0]
                        reattaching = (
                            current.status in {JobStatus.PENDING, JobStatus.RUNNING}
                            and isinstance(current.payload, RemoteTrainJobPayload)
                            and current.payload.remote_job_id is not None
                        )
                    if (
                        current_jobs
                        and current_jobs[0].status == JobStatus.PENDING
                        and not reattaching
                        and not self.mp_stop_event.is_set()
                    ):
                        failed = await service.update_job_status(
                            job.id, JobStatus.FAILED, message="Training worker exited before completing the job"
                        )
                        self.event_queue.put((EventType.JOB_UPDATE, failed))
            if not reattaching:
                self._cancel_events.pop(job.id, None)
            if process.pid is not None:
                process.close()
            del self.processes[target]
            await self._recover_jobs()

        unassigned = set(self._cancel_events.copy()) - {job.id for _, job in self.processes.values()}
        if unassigned:
            async with get_async_db_session_ctx() as session:
                jobs = await JobService(session).get_jobs_by_ids(list(unassigned))
            waiting = {job.id for job in jobs if job.status in {JobStatus.PENDING, JobStatus.RUNNING}}
            for job_id in unassigned - waiting:
                self._cancel_events.pop(job_id, None)

    async def _recover_jobs(self) -> None:
        """Reconcile abandoned jobs without touching any live worker's job."""
        active_ids = frozenset(job.id for _, job in self.processes.values())
        async with get_async_db_session_ctx() as session:
            service = JobService(session, RemoteTrainerService(session))
            recovered = await TrainingService.abort_orphan_jobs(service, exclude_job_ids=active_ids)
            for job in recovered:
                self.event_queue.put((EventType.JOB_UPDATE, job))
            imports = await service.get_job_list(
                extra_filters={"status": JobStatus.RUNNING, "type": JobType.DATASET_IMPORT}
            )
            for job in imports:
                if job.id not in active_ids:
                    failed = await service.update_job_status(
                        job.id, JobStatus.FAILED, message="Dataset import worker stopped before completing the job"
                    )
                    self.event_queue.put((EventType.JOB_UPDATE, failed))

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
