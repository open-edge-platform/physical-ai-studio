from __future__ import annotations
import traceback

import asyncio
from typing import TYPE_CHECKING

from uuid import uuid4, UUID
import multiprocessing as mp
from settings import get_settings

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass

from services import JobService, ModelService, TrainingService, DatasetService
from workers.base import BaseProcessWorker, BaseThreadWorker
from schemas import Model, Dataset, Job
from schemas.job import TrainJobPayload, JobStatus

SCHEDULE_INTERVAL_SEC = 5

import loguru
from loguru import logger

from getiaction.policies import ACTModel, ACT
from getiaction.train import Trainer
from getiaction.data import LeRobotDataModule

from lightning.pytorch.callbacks import ModelCheckpoint, Callback, ProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
from typing import Any
from services.event_processor import EventType

class TrainingTrackingDispatcher(BaseThreadWorker):
    """Dispatch events from the callback to a queue asynchronously."""
    def __init__(self, job_id: UUID, event_queue: mp.Queue, interrupt_event: mp.Event):
        super().__init__(stop_event=interrupt_event)
        self.job_id = job_id
        self.event_queue = event_queue
        self.queue = mp.Queue()
        self.interrupt_event = interrupt_event

    async def run_loop(self):
        while not self.interrupt_event.is_set():
            try:
                progress = self.queue.get_nowait()
                job = await JobService.update_job_status(self.job_id, JobStatus.RUNNING, progress=progress)
                self.event_queue.put((EventType.JOB_UPDATE, job))
            except mp.queues.Empty:
                await asyncio.sleep(0.05)

    def update_progress(self, progress: int):
        self.queue.put(progress)


class TrainingTrackingCallback(Callback):
    def __init__(
        self,
        shutdown_event: mp.Event,
        interrupt_event: mp.Event,
        dispatcher: TrainingTrackingDispatcher,
    ):
        super().__init__()
        self.shutdown_event = shutdown_event  # global stop event in case of shutdown
        self.interrupt_event = (
            interrupt_event  # event for interrupting training gracefully
        )
        self.dispatcher = dispatcher

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        progress = round((trainer.global_step) / trainer.max_steps * 100)
        self.dispatcher.update_progress(progress)
        if self.shutdown_event.is_set() or self.interrupt_event.is_set():
            trainer.should_stop = True

class CheckpointStorageCallback(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        filepath = trainer.checkpoint_callback.best_model_path
        print(filepath)


class TrainingCallback(Callback):
    def __init__(self, shutdown_event: mp.Event, interrupt_event: mp.Event):
        super().__init__()
        self._shutdown_event = shutdown_event  # global stop event in case of shutdown
        self._interrupt_event = (
            interrupt_event  # event for interrupting training gracefully
        )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._shutdown_event.is_set() or self._interrupt_event.is_set():
            trainer.should_stop = True


class TrainingWorker(BaseProcessWorker):
    ROLE = "Training"

    def __init__(self, stop_event: EventClass, interrupt_event: EventClass, event_queue: mp.Queue):
        super().__init__(stop_event=stop_event)
        self.queue = event_queue
        self.interrupt_event = interrupt_event

    async def run_loop(self) -> None:
        job_service = JobService()
        dataset_service = DatasetService()

        logger.info("Training Worker is running")
        while not self.should_stop():
            settings = get_settings()

            job = await job_service.get_pending_train_job()
            if job is not None:
                payload = TrainJobPayload.model_validate(job.payload)
                id = uuid4()
                model = Model(
                    id=id,
                    project_id=payload.project_id,
                    dataset_id=payload.dataset_id,
                    path=str(settings.models_dir / str(id)),
                    name=payload.model_name,
                    policy=payload.policy,
                    properties={},
                )

                dataset = await dataset_service.get_dataset_by_id(payload.dataset_id)
                self.interrupt_event.clear()
                await asyncio.create_task(self._train_model(job, dataset, model))
            await asyncio.sleep(0.5)

    def setup(self) -> None:
        super().setup()
        with logger.contextualize(worker=self.__class__.__name__):
            asyncio.run(TrainingService.abort_orphan_jobs())

    def teardown(self) -> None:
        super().teardown()
        with logger.contextualize(worker=self.__class__.__name__):
            asyncio.run(TrainingService.abort_orphan_jobs())

    async def _train_model(self, job, dataset, model):
        await JobService.update_job_status(
            job_id=job.id, status=JobStatus.RUNNING, message="Training started"
        )
        try:
            l_dm = LeRobotDataModule(
                repo_id=dataset.name,
                root=dataset.path,
                train_batch_size=48,
            )
            lib_model = ACTModel(
                input_features=l_dm.train_dataset.observation_features,
                output_features=l_dm.train_dataset.action_features,
            )

            policy = ACT(model=lib_model)

            checkpoint_callback = ModelCheckpoint(
                dirpath=model.path,
                filename="checkpoint_{step}",  # optional pattern
                save_top_k=1,  # only keep best checkpoint
                monitor="train/loss",  # metric to monitor
                mode="min",
            )

            dispatcher = TrainingTrackingDispatcher(
                job_id=job.id,
                event_queue=self.queue,
                interrupt_event=self.interrupt_event,
            )

            trainer = Trainer(
                callbacks=[
                    checkpoint_callback,
                    TrainingTrackingCallback(
                        shutdown_event=self._stop_event,
                        interrupt_event=self.interrupt_event,
                        dispatcher=dispatcher,
                    ),
                ],
                max_steps=10,
            )

            dispatcher.start()
            trainer.fit(model=policy, datamodule=l_dm)

            self.interrupt_event.set()
            dispatcher.join(timeout=10)

            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.COMPLETED, message="Training finished"
            )
            model = await ModelService.create_model(model)
            self.queue.put((EventType.MODEL_UPDATE, model))
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.FAILED, message=f"Training failed: {e}"
            )
        self.queue.put((EventType.JOB_UPDATE, job))
