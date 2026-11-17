# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import func, update

from core.logging.utils import job_logging_ctx
from db import get_async_db_session_ctx
from db.schema import JobDB
from schemas.base_job import JobStatus, JobType
from schemas.job import OptimizeModelJob
from schemas.model import Model
from schemas.model_variant import ExportPackage
from services.event_processor import EventType
from services.job_service import JobService
from services.model_optimization_service import ModelOptimizationService
from services.model_service import ModelService
from workers.base import ProcessWorker

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass
    from uuid import UUID

    from schemas import Job


class ModelOptimizationWorker(ProcessWorker[None]):
    """Build one export variant in a short-lived, memory-isolated process."""

    ROLE = "ModelOptimizationWorker"

    def __init__(self, job_id: UUID, stop_event: EventClass, cancel_event: EventClass, event_queue: mp.Queue):
        super().__init__(name=self.ROLE)
        self.shutdown_event = stop_event
        self.job_id = job_id
        self.cancel_event = cancel_event
        self.queue = event_queue

    def should_stop(self) -> bool:
        """Stop on the worker's own stop or the scheduler's shutdown."""
        return super().should_stop() or self.shutdown_event.is_set()

    async def run_loop(self, context: None = None) -> None:  # noqa: ARG002
        """Claim the job and attach a variant only after its complete package succeeds."""
        if self.should_stop():
            return
        async with get_async_db_session_ctx() as session:
            job = await JobService(session).claim_pending_job(self.job_id, JobType.OPTIMIZE_MODEL)
        if not isinstance(job, OptimizeModelJob):
            return
        self.queue.put((EventType.JOB_UPDATE, job))
        with job_logging_ctx(job_id=str(job.id)):
            try:
                result = await self._optimize(job)
            except Exception as exc:
                logger.exception("Model optimization worker failed")
                async with get_async_db_session_ctx() as session:
                    current = await JobService(session).get_job_by_id(job.id)
                    if current.status == JobStatus.COMPLETED:
                        return
                    await ModelOptimizationService(session).cleanup(job)
                await self._finish(job, JobStatus.FAILED, f"Model optimization failed: {exc}")
                return
            if result is not None:
                model, completed = result
                self.queue.put((EventType.MODEL_UPDATE, model))
                self.queue.put((EventType.JOB_UPDATE, completed))

    async def _optimize(self, job: OptimizeModelJob) -> tuple[Model, Job] | None:
        async with get_async_db_session_ctx() as session:
            source = await ModelService(session).get_model_by_id(job.payload.model_id)
        staging = ModelOptimizationService.staging(job.id)
        destination = ExportPackage.for_model(Path(source.path), "openvino", "int8").directory
        if self.should_stop() or self.cancel_event.is_set():
            async with get_async_db_session_ctx() as session:
                await ModelOptimizationService(session).cleanup(job)
            await self._finish(job, JobStatus.CANCELED, "Model optimization canceled")
            return None
        ModelOptimizationService.compress(source, staging, job.id)
        # NNCF cannot be interrupted mid-call; never publish its output after cancellation.
        if self.should_stop() or self.cancel_event.is_set():
            async with get_async_db_session_ctx() as session:
                await ModelOptimizationService(session).cleanup(job)
            await self._finish(job, JobStatus.CANCELED, "Model optimization canceled")
            return None
        completed = job.model_copy(
            update={
                "status": JobStatus.COMPLETED,
                "message": "INT8 export created",
                "progress": 100,
                "end_time": datetime.now(UTC),
            }
        )
        async with get_async_db_session_ctx() as session:
            # Cancellation and publication compete for the same RUNNING row.
            publication = await session.execute(
                update(JobDB)
                .where(
                    JobDB.id == str(job.id),
                    JobDB.type == JobType.OPTIMIZE_MODEL,
                    JobDB.status == JobStatus.RUNNING,
                    func.coalesce(func.json_extract(JobDB.payload, "$.cancellation_requested"), False).is_(False),
                )
                .values(
                    status=JobStatus.COMPLETED,
                    message=completed.message,
                    progress=completed.progress,
                    end_time=completed.end_time,
                )
                .returning(JobDB.id)
            )
            canceled = publication.scalar_one_or_none() is None or self.should_stop() or self.cancel_event.is_set()
            if canceled:
                await session.rollback()
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(f"Export variant already exists: {destination.name}")
                await session.commit()
        if canceled:
            async with get_async_db_session_ctx() as session:
                await ModelOptimizationService(session).cleanup(job)
            await self._finish(job, JobStatus.CANCELED, "Model optimization canceled")
            return None
        ModelOptimizationService.publish(source, job.id)
        return source, completed

    async def _finish(self, job: OptimizeModelJob, status: JobStatus, message: str) -> None:
        async with get_async_db_session_ctx() as session:
            updated = await JobService(session).update_job_status(job.id, status, message=message)
        self.queue.put((EventType.JOB_UPDATE, updated))
