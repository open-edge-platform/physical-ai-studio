from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from getiaction.policies.base import Policy
from lightning.pytorch.callbacks import ModelCheckpoint

from services.snapshot_service import SnapshotService
from settings import get_settings

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass

from getiaction.data import DataModule, LeRobotDataModule
from getiaction.policies import ACT, ACTModel, Pi0
from getiaction.train import Trainer
from loguru import logger

from schemas import Job, Model, Snapshot
from schemas.job import JobStatus, TrainJobPayload
from services import DatasetService, JobService, ModelService
from services.event_processor import EventType
from services.training_service import TrainingService, TrainingTrackingCallback, TrainingTrackingDispatcher
from workers.base import BaseProcessWorker

SCHEDULE_INTERVAL_SEC = 5


class TrainingWorker(BaseProcessWorker):
    ROLE = "Training"

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
                payload = TrainJobPayload.model_validate(job.payload)
                id = uuid4()

                dataset = await DatasetService.get_dataset_by_id(payload.dataset_id)
                model_dir = Path(str(settings.models_dir / str(id)))
                model_dir.mkdir(parents=True)
                snapshot_dir = model_dir / SnapshotService.generate_snapshot_folder_name()
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
                    created_at=None,
                )

                self.interrupt_event.clear()
                await asyncio.create_task(self._train_model(job, model, snapshot))
            await asyncio.sleep(0.5)

    def setup(self) -> None:
        super().setup()
        with logger.contextualize(worker=self.__class__.__name__):
            asyncio.run(TrainingService.abort_orphan_jobs())

    def teardown(self) -> None:
        super().teardown()
        with logger.contextualize(worker=self.__class__.__name__):
            asyncio.run(TrainingService.abort_orphan_jobs())

    async def _train_model(self, job: Job, model: Model, snapshot: Snapshot):
        await JobService.update_job_status(job_id=job.id, status=JobStatus.RUNNING, message="Training started")
        dispatcher = TrainingTrackingDispatcher(
            job_id=job.id,
            event_queue=self.queue,
            interrupt_event=self.interrupt_event,
        )
        try:
            path = Path(model.path)

            l_dm = LeRobotDataModule(
                repo_id="snapshot",  # doesnt matter for loading the data.
                root=snapshot.path,
                train_batch_size=8,
            )
            policy = self._setup_policy(model, l_dm)

            checkpoint_callback = ModelCheckpoint(
                dirpath=path,
                filename="model",  # filename without suffix
                save_top_k=1,
                monitor="train/loss",
                mode="min",
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
                max_steps=10000,
            )

            dispatcher.start()
            # policy.export(path, backend="torch")
            trainer.fit(model=policy, datamodule=l_dm)
            # policy.export(path, backend="openvino")

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
        self.interrupt_event.set()
        dispatcher.join(timeout=10)
        self.queue.put((EventType.JOB_UPDATE, job))

    def _setup_policy(self, model: Model, l_dm: DataModule) -> Policy:
        if model.policy == "act":
            lib_model = ACTModel(
                input_features=l_dm.train_dataset.observation_features,
                output_features=l_dm.train_dataset.action_features,
            )

            return ACT(model=lib_model)
        if model.policy == "pi0":
            return Pi0(
                variant="pi0",
                chunk_size=50,
                learning_rate=2.5e-5,
            )

        raise ValueError(f"Policy not implemented yet: {model.policy}")
