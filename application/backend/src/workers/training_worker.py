# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from core.logging.utils import job_logging_ctx
from schemas import Job, Model, Snapshot
from schemas.base_job import JobStatus
from schemas.job import TrainJobPayload
from services import DatasetService, ModelService
from services.event_processor import EventType
from services.job_service import JobService
from services.snapshot_service import SnapshotService
from services.training_backends import TrainingCanceledError, TrainingContext, get_training_backend
from services.training_service import TrainingService, TrainingTrackingDispatcher
from settings import get_settings
from workers.base import BaseProcessWorker

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass

SCHEDULE_INTERVAL_SEC = 5


class TrainingWorker(BaseProcessWorker):
    ROLE = "TrainingWorker"

    def __init__(self, stop_event: EventClass, interrupt_event: EventClass, event_queue: mp.Queue):
        super().__init__(stop_event=stop_event)
        self.queue = event_queue
        self.interrupt_event = interrupt_event

    async def run_loop(self) -> None:
        job_service = JobService()
        logger.info("Training Worker is running")
        while not self.should_stop():
            settings = get_settings()

            job = await job_service.get_pending_train_job()
            if job is not None:
                with job_logging_ctx(job_id=str(job.id)):
                    payload = TrainJobPayload.model_validate(job.payload)
                    id = uuid4()

                    base_model = None
                    if payload.base_model_id is not None:
                        base_model = await ModelService.get_model_by_id(payload.base_model_id)

                    dataset = await DatasetService.get_dataset_by_id(payload.dataset_id)
                    model_dir = Path(str(settings.models_dir / str(id)))
                    snapshot_dir = settings.snapshot_dir / SnapshotService.generate_snapshot_folder_name()
                    snapshot = await SnapshotService.create_snapshot_for_dataset(dataset, destination=snapshot_dir)

                    model = Model(
                        id=id,
                        project_id=payload.project_id,
                        dataset_id=payload.dataset_id,
                        path=str(model_dir),
                        name=payload.model_name,
                        snapshot_id=snapshot.id,
                        policy=payload.policy,
                        properties={},
                        train_job_id=job.id,
                        parent_model_id=payload.base_model_id,
                        version=base_model.version + 1 if base_model else 1,
                        created_at=None,
                    )

                    self.interrupt_event.clear()
                    await asyncio.create_task(self._train_model(job, model, snapshot, payload, base_model))
            self.stop_aware_sleep(0.5)

    async def setup(self) -> None:
        await super().setup()
        with logger.contextualize(worker=self.__class__.__name__):
            await TrainingService.abort_orphan_jobs()

    async def teardown(self) -> None:
        await super().teardown()
        with logger.contextualize(worker=self.__class__.__name__):
            await TrainingService.abort_orphan_jobs()

    async def _train_model(
        self, job: Job, model: Model, snapshot: Snapshot, payload: TrainJobPayload, base_model: Model | None = None
    ) -> None:
        settings = get_settings()
        await JobService.update_job(
            job=job,
            update={
                "status": JobStatus.RUNNING,
                "message": "Training started",
                "start_time": datetime.datetime.now(tz=datetime.UTC),
            },
        )
        dispatcher = TrainingTrackingDispatcher(
            job_id=job.id,
            event_queue=self.queue,
            interrupt_event=self.interrupt_event,
        )
        interrupted = False
        error: Exception | None = None
        dispatcher.start()
        try:
            context = TrainingContext(
                job=job,
                model=model,
                snapshot=snapshot,
                payload=payload,
                base_model=base_model,
                output_dir=Path(model.path),
                cache_dir=settings.cache_dir / str(job.id),
                progress=dispatcher.report,
                should_stop=self._should_interrupt,
            )

            backend = get_training_backend()
            await backend.train(context)
            # The local backend stops cooperatively without raising; treat a
            # completed-but-interrupted run as a cancellation, not a success.
            interrupted = self._should_interrupt()
        except TrainingCanceledError:
            interrupted = True
        except Exception as e:  # surface any training failure as a FAILED job
            error = e
            logger.exception(f"Training failed: {e}")
        finally:
            # Stop the dispatcher and let it flush queued progress BEFORE writing
            # the terminal status. Otherwise a late RUNNING progress update can
            # land after the terminal write and revert the job (stuck at 95%).
            self.interrupt_event.set()
            if dispatcher.is_alive():
                dispatcher.join(timeout=10)

        if error is not None:
            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.FAILED, message=f"Training failed: {error}"
            )
        elif interrupted:
            logger.info("Training canceled")
            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.CANCELED, message="Training canceled"
            )
        else:
            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.COMPLETED, message="Training finished"
            )
            model = await ModelService.create_model(model)
            self.queue.put((EventType.MODEL_UPDATE, model))

        self.queue.put((EventType.JOB_UPDATE, job))

    def _should_interrupt(self) -> bool:
        """Stop training on global shutdown or an explicit interrupt request."""
        return self.should_stop() or self.interrupt_event.is_set()
