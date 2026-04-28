# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from core.logging.utils import job_logging_ctx
from models.utils import load_policy, setup_policy
from schemas.job import TrainingPrecision
from services.snapshot_service import SnapshotService
from settings import get_settings

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass


from loguru import logger
from physicalai.data import LeRobotDataModule
from physicalai.export import ExportablePolicyMixin
from physicalai.train import Trainer

from schemas import Job, Model, Snapshot
from schemas.base_job import JobStatus
from schemas.job import TrainJobPayload
from services import DatasetService, ModelService
from services.event_processor import EventType
from services.job_service import JobService
from services.training_service import (
    TrainingLogCallback,
    TrainingService,
    TrainingTrackingCallback,
    TrainingTrackingDispatcher,
)
from utils.device import get_default_precision, get_lightning_strategy, get_torch_device
from workers.base import BaseProcessWorker

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
                        train_job_id=job.id,
                        parent_model_id=payload.base_model_id,
                        version=base_model.version + 1 if base_model else 1,
                        created_at=None,
                    )

                    self.interrupt_event.clear()
                    await asyncio.create_task(self._train_model(job, model, snapshot, payload, base_model))
            await asyncio.sleep(0.5)

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
    ):
        settings = get_settings()
        await JobService.update_job_status(job_id=job.id, status=JobStatus.RUNNING, message="Training started")
        dispatcher = TrainingTrackingDispatcher(
            job_id=job.id,
            event_queue=self.queue,
            interrupt_event=self.interrupt_event,
        )
        try:
            path = Path(model.path)

            # Resolve training device -- explicit from payload or auto-detected
            device_type = payload.device.type if payload.device else None
            device_index = payload.device.index if payload.device else None

            accelerator = get_torch_device(device_type)

            l_dm = LeRobotDataModule(
                repo_id="snapshot",  # doesnt matter for loading the data.
                root=snapshot.path,
                train_batch_size=payload.batch_size,
                num_workers=payload.num_workers,
                val_split=payload.val_split,
            )

            if base_model is not None:
                policy = load_policy(base_model, compile_model=payload.compile_model)
            else:
                policy = setup_policy(model, compile_model=payload.compile_model)

            if payload.precision != TrainingPrecision.DEFAULT:
                precision = str(payload.precision)
            else:
                precision = get_default_precision(accelerator)

            checkpoint_callback = ModelCheckpoint(
                dirpath=path,
                filename="model",  # filename without suffix
                save_top_k=1,
                monitor="val/loss",
                mode="min",
            )
            csv_logger = CSVLogger(path.parent, name=path.stem)

            def _create_trainer() -> Trainer:
                return Trainer(
                    logger=csv_logger,
                    callbacks=[
                        checkpoint_callback,
                        TrainingTrackingCallback(
                            shutdown_event=self._stop_event,
                            interrupt_event=self.interrupt_event,
                            dispatcher=dispatcher,
                        ),
                        TrainingLogCallback(),
                    ],
                    accelerator=accelerator,
                    strategy=get_lightning_strategy(device_type),
                    devices=[device_index] if device_index is not None else "auto",
                    max_steps=payload.max_steps,
                    auto_scale_batch_size=payload.auto_scale_batch_size,
                    precision=precision,
                    check_val_every_n_epoch=1,
                )

            trainer = _create_trainer()

            dispatcher.start()

            try:
                trainer.fit(model=policy, datamodule=l_dm)
            except Exception as fit_exc:
                if not payload.compile_model:
                    raise

                logger.exception(f"Training with compile failed: {fit_exc} — retrying without compilation")
                compile_fallback_msg = "Model compilation failed, falling back to non-compiled training."
                job = await JobService.update_job_status(
                    job_id=job.id, status=JobStatus.RUNNING, message=compile_fallback_msg
                )
                self.queue.put((EventType.JOB_UPDATE, job))

                policy = setup_policy(model, compile_model=False)
                trainer = _create_trainer()
                trainer.fit(model=policy, datamodule=l_dm)

            for backend in settings.supported_backends:
                export_dir = path / "exports" / backend
                if isinstance(policy, ExportablePolicyMixin):
                    policy.export(export_dir, backend=backend)

            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.COMPLETED, message="Training finished"
            )
            model = await ModelService.create_model(model)
            self.queue.put((EventType.MODEL_UPDATE, model))
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.FAILED, message=f"Training failed: {e}"
            )
        self.interrupt_event.set()
        if dispatcher.is_alive():
            dispatcher.join(timeout=10)
        self.queue.put((EventType.JOB_UPDATE, job))
