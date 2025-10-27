import asyncio
import os

from pathlib import Path
from uuid import uuid4, UUID
#from utils.experiment_loggers import TrackioLogger
from loguru import logger

from schemas import Job, Model, Dataset
from schemas.job import JobStatus, JobType, TrainJobPayload

#from repositories.binary_repo import ImageBinaryRepository, ModelBinaryRepository
from services import ModelService, DatasetService
from services.job_service import JobService

from getiaction.policies import ACTModel, ACT
from getiaction.train import Trainer
from getiaction.data import LeRobotDataModule

from lightning.pytorch.callbacks import ModelCheckpoint, Callback, ProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
from typing import Any

from settings import get_settings
from multiprocessing.synchronize import Event

class TrainingCallback(Callback):
    def __init__(self, shutdown_event: Event, interrupt_event: Event):
        super().__init__()
        self._shutdown_event = shutdown_event # global stop event in case of shutdown
        self._interrupt_event = interrupt_event # event for interrupting training gracefully

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self._shutdown_event.is_set() or self._interrupt_event.is_set():
            trainer.should_stop = True

class ProgressCallback(ProgressBar):
    def __init__(self, job_id: UUID):
        super().__init__()
        self.job_id = job_id
        self.enable = True

    def on_fit_start(self, trainer, pl_module):
        """Pre‑compute total steps once training begins."""
        self.total_steps = trainer.max_steps

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        progress = round((trainer.global_step) / self.total_steps * 100)
        if progress < 100:
            asyncio.run(JobService().update_job_status(job_id=self.job_id, status=JobStatus.RUNNING, progress=progress))


class TrainingService:
    """
    Service for managing model training jobs.

    Handles the complete training pipeline including job fetching, model training,
    status updates, and error handling. Currently, using asyncio.to_thread for
    CPU-intensive training to maintain event loop responsiveness.

    Note: asyncio.to_thread is used assuming single concurrent training job.
    For true parallelism with multiple training jobs, consider ProcessPoolExecutor.
    """

    @classmethod
    async def train_pending_job(cls, stop_event: Event, interrupt_event: Event) -> Model | None:
        """
        Process the next pending training job from the queue.

        Fetches a pending job, executes training in a separate thread to maintain
        event loop responsiveness, and updates job status accordingly.

        Returns:
            Model: Trained model if successful, None if no pending jobs
        """
        job_service = JobService()
        dataset_service = DatasetService()
        job = await job_service.get_pending_train_job()
        if job is None:
            logger.trace("No pending training job")
            return None

        interrupt_event.clear() # Interrupt event cleared for new task
        return await cls._run_training_job(job, job_service, dataset_service, stop_event, interrupt_event)

    @classmethod
    async def _run_training_job(cls, job: Job, job_service: JobService, dataset_service: DatasetService, stop_event: Event, interrupt_event: Event) -> Model:
        # Mark job as running
        await job_service.update_job_status(job_id=job.id, status=JobStatus.RUNNING, message="Training started")
        project_id = job.project_id
        payload = TrainJobPayload.model_validate(job.payload)
        model_service = ModelService()

        settings = get_settings()
        settings.storage_dir
        id = uuid4()

        model = Model(
            id=id,
            project_id=project_id,
            dataset_id=payload.dataset_id,
            path=str(settings.models_dir / str(id)),
            name=payload.model_name,
            policy=payload.policy,
            properties={},
        )
        dataset = await dataset_service.get_dataset_by_id(payload.dataset_id)
        logger.info(f"Training model `{model.name}` for job `{job.id}`")

        try:
            # Use asyncio.to_thread to keep event loop responsive
            # TODO: Consider ProcessPoolExecutor for true parallelism with multiple jobs
            trained_model = await asyncio.to_thread(cls._train_model, model, dataset, stop_event, interrupt_event, job.id)
            if trained_model is None:
                raise ValueError("Training failed - model is None")

            await job_service.update_job_status(
                job_id=job.id, status=JobStatus.COMPLETED, message="Training completed successfully"
            )
            return await model_service.create_model(trained_model)
        except Exception as e:
            logger.exception("Failed to train pending training job: %s", e)
            await job_service.update_job_status(
                job_id=job.id, status=JobStatus.FAILED, message=f"Failed with exception: {str(e)}"
            )
            if model.path:
                logger.warning(f"Deleting partially created model with id: {model.id}")
                #TODO Remove partially created model
                #model_binary_repo = ModelBinaryRepository(project_id=project_id, model_id=model.id)
                #await model_binary_repo.delete_model_folder()
                #await model_service.delete_model(project_id=project_id, model_id=model.id)
            raise e

    @staticmethod
    def _train_model(model: Model, dataset: Dataset, shutdown_event: Event, interrupt_event: Event, job_id: UUID) -> Model | None:
        if model.policy != "act":
            raise ValueError(f"Unsupported policy type: {model.policy} -- Only ACT is supported for now.")

        l_dm = LeRobotDataModule(
            repo_id=dataset.name,
            root=dataset.path,
            train_batch_size=48,
        )
        lib_model = ACTModel(
            input_features=l_dm.train_dataset.observation_features,
            output_features=l_dm.train_dataset.action_features,
        )

        Path(model.path).mkdir(parents=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=model.path,
            filename=str(job_id),  # optional pattern
            save_top_k=1,  # only keep best checkpoint
            monitor="train/loss",  # metric to monitor
            mode="min",
        )

        policy = ACT(model=lib_model)

        trainer = Trainer(
            callbacks=[
                TrainingCallback(shutdown_event, interrupt_event),
                ProgressCallback(job_id),
                checkpoint_callback,
            ], 
            max_steps=200
        )
        trainer.fit(model=policy, datamodule=l_dm)

        return model


    @staticmethod
    async def abort_orphan_jobs() -> None:
        """
        Abort all running orphan training jobs (that do not belong to any worker).

        This method can be called during application shutdown/setup to ensure that
        any orphan in-progress training jobs are marked as failed.
        """
        query = {"status": JobStatus.RUNNING, "type": JobType.TRAINING}
        running_jobs = await JobService.get_job_list(extra_filters=query)
        for job in running_jobs:
            logger.warning(f"Aborting orphan training job with id: {job.id}")
            await JobService.update_job_status(
                job_id=job.id,
                status=JobStatus.FAILED,
                message="Job aborted due to application shutdown",
            )
