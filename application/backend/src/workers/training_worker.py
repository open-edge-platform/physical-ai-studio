from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from uuid import uuid4, UUID
import multiprocessing as mp
from settings import get_settings

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass

from services.training_service import TrainingService
from services import JobService, ModelService
from workers.base import BaseProcessWorker
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

from enum import StrEnum


class EventType(StrEnum):
    CHECKPOINT = "CHECKPOINT"
    PROGRESS = "PROGRESS"
    DONE = "DONE"


class TrainingTrackingCallback(Callback):
    def __init__(
        self,
        shutdown_event: mp.Event,
        interrupt_event: mp.Event,
        queue: mp.Queue,
        model: Model,
        job_id: UUID,
    ):
        super().__init__()
        self.shutdown_event = shutdown_event  # global stop event in case of shutdown
        self.interrupt_event = (
            interrupt_event  # event for interrupting training gracefully
        )
        self.queue = queue
        self.model = model
        self.job_id = job_id

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        progress = round((trainer.global_step) / trainer.max_steps * 100)
        print("PROGRESS", progress)
        self.queue.put((EventType.PROGRESS, (self.job_id, progress)))

        if self.shutdown_event.is_set() or self.interrupt_event.is_set():
            trainer.should_stop = True

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        filepath = trainer.checkpoint_callback.best_model_path
        print("ON SAVE CHECKPOINT")
        self.queue.put((EventType.CHECKPOINT, (self.model, filepath)))


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

    def __init__(self, stop_event: EventClass, interrupt_event: EventClass):
        super().__init__(stop_event=stop_event)
        self.queue = mp.Queue()
        self.interrupt_event = interrupt_event

    async def run_loop(self) -> None:
        """Main training loop that polls for jobs and manages concurrent training tasks."""
        print("run loop")
        job_service = JobService()

        logger.info("Training Worker is running")
        while not self.should_stop():
            # Clean up completed tasks
            settings = get_settings()

            job = await job_service.get_pending_train_job()
            if job is not None:
                await JobService.update_job_status(
                    job_id=job.id, status=JobStatus.RUNNING, message="Training started"
                )
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

                dataset = Dataset(
                    id="b55bf695-8927-4ad9-96a1-88a6a0f33326",
                    name="block-placement",
                    path="/home/ronald/.cache/geti_action/datasets/block-placement",
                    project_id="cf8062f8-7354-4643-8899-9be8d57ab2ce",
                )
                self._train_model(job, dataset, model)
            await asyncio.sleep(0.5)

    async def event_processor(self):
        print("Starting event processor")
        try:
            while not self._stop_event.is_set():
                try:
                    event, payload = self.queue.get_nowait()
                    logger.info(f"{event} {payload}")
                    if event == EventType.CHECKPOINT:
                        model, path = payload
                        db_model = await ModelService.get_model_by_id(model.id)
                        if db_model is None:
                            model.path = path
                            await ModelService.create_model(model)
                        else:
                            await ModelService.update_model(model, {"path": path})
                    if event == EventType.DONE:
                        job_id, progress = payload
                        JobService.update_job_status(
                            job_id, JobStatus.COMPLETED, progress=progress
                        )
                    if event == EventType.PROGRESS:
                        job_id, progress = payload
                        if progress < 100:
                            JobService.update_job_status(
                                job_id, JobStatus.RUNNING, progress=progress
                            )
                except mp.queues.Empty:
                    await asyncio.sleep(0.05)
        except Exception as e:
            print(f"Outgoing task stopped: {e}")

    def setup(self) -> None:
        super().setup()
        with logger.contextualize(worker=self.__class__.__name__):
            asyncio.run(TrainingService.abort_orphan_jobs())

    def teardown(self) -> None:
        super().teardown()
        with logger.contextualize(worker=self.__class__.__name__):
            asyncio.run(TrainingService.abort_orphan_jobs())

    def _train_model(self, job, dataset, model):
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

        trainer = Trainer(
            callbacks=[
                checkpoint_callback,
                TrainingTrackingCallback(
                    shutdown_event=self._stop_event,
                    interrupt_event=mp.Event(),
                    queue=self.queue,
                    model=model,
                    job_id=job.id,
                ),  # placeholder event for interrupt
            ],
            max_steps=10,
        )

        trainer.fit(model=policy, datamodule=l_dm)
