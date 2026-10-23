# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from core.logging.utils import job_logging_ctx
from db import get_async_db_session_ctx
from repositories.mappers.job_mapper import JobMapper
from repositories.mappers.model_mapper import ModelMapper
from schemas.base_job import JobStatus, JobType
from schemas.job import OptimizeModelJob
from schemas.model import Model
from services.event_processor import EventType
from services.job_service import JobService
from services.model_optimization_service import ModelOptimizationService
from services.model_service import ModelService
from settings import get_settings
from workers.base import BaseProcessWorker

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass
    from uuid import UUID

    from schemas import Job


class ModelOptimizationWorker(BaseProcessWorker):
    """Compress one assigned model in a short-lived, memory-isolated process."""

    ROLE = "ModelOptimizationWorker"

    def __init__(self, job_id: UUID, stop_event: EventClass, cancel_event: EventClass, event_queue: mp.Queue):
        super().__init__(stop_event=stop_event)
        self.job_id = job_id
        self.cancel_event = cancel_event
        self.queue = event_queue
        self._published = False

    async def run_loop(self) -> None:
        """Claim the job and publish a separate model only after compression succeeds."""
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
                if self._published:
                    return
                ModelOptimizationService.cleanup(job.id)
                await self._finish(job, JobStatus.FAILED, f"Model optimization failed: {exc}")
                return
            if result is not None:
                model, completed = result
                self.queue.put((EventType.MODEL_UPDATE, model))
                self.queue.put((EventType.JOB_UPDATE, completed))

    async def _optimize(self, job: OptimizeModelJob) -> tuple[Model, Job] | None:
        async with get_async_db_session_ctx() as session:
            source = await ModelService(session).get_model_by_id(job.payload.model_id)
        models_dir = get_settings().models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        staging = models_dir / f".optimize-{job.id}"
        destination = models_dir / str(job.id)
        if self.should_stop() or self.cancel_event.is_set():
            await self._finish(job, JobStatus.CANCELED, "Model optimization canceled")
            return None
        metadata = ModelOptimizationService.optimize_export(source, staging)
        # NNCF cannot be interrupted mid-call; never publish its output after cancellation.
        if self.should_stop() or self.cancel_event.is_set():
            ModelOptimizationService.cleanup(job.id)
            await self._finish(job, JobStatus.CANCELED, "Model optimization canceled")
            return None
        staging.rename(destination)
        model = Model(
            id=job.id,
            name=f"{source.name} (INT8)",
            path=str(destination),
            policy=source.policy,
            project_id=source.project_id,
            dataset_id=source.dataset_id,
            # The optimized model owns no checkpoint or dataset snapshot.
            properties={**source.properties, "optimization": metadata.model_dump(mode="json")},
        )
        completed = job.model_copy(
            update={
                "status": JobStatus.COMPLETED,
                "message": "INT8 model optimization completed",
                "progress": 100,
                "end_time": datetime.now(UTC),
            }
        )
        async with get_async_db_session_ctx() as session:
            session.add(ModelMapper.to_schema(model))
            await session.merge(JobMapper.to_schema(completed))
            # Commit last: a failing refresh must not trigger cleanup of published artifacts.
            await session.commit()
            self._published = True
        return model, completed

    async def _finish(self, job: OptimizeModelJob, status: JobStatus, message: str) -> None:
        async with get_async_db_session_ctx() as session:
            updated = await JobService(session).update_job_status(job.id, status, message=message)
        self.queue.put((EventType.JOB_UPDATE, updated))
