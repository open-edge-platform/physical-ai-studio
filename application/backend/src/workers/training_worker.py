# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from loguru import logger

from core.logging.utils import job_logging_ctx
from db import get_async_db_session_ctx
from schemas import Job, Model, Snapshot
from schemas.aws_auth import AwsAuthenticationIssue
from schemas.base_job import JobStatus
from schemas.job import RemoteTrainJobPayload, TrainingTarget, TrainJobPayload, TrainJobPayloadAdapter
from schemas.model import DORA_PROPERTY, LORA_PROPERTY, SNAPFLOW_PROPERTY
from services import DatasetService, ModelService
from services.event_processor import EventType
from services.job_service import JobService
from services.remote_trainer_service import RemoteTrainerService
from services.snapshot_service import SnapshotService
from services.training_backends import (
    TrainingCanceledError,
    TrainingContext,
    TrainingSuspendedError,
    get_training_backend,
)
from services.training_service import TrainingTrackingDispatcher
from settings import get_settings
from workers.base import BaseProcessWorker

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass


class TrainingWorker(BaseProcessWorker):
    ROLE = "TrainingWorker"

    def __init__(self, job_id: UUID, stop_event: EventClass, cancel_event: EventClass, event_queue: mp.Queue):
        super().__init__(stop_event=stop_event)
        self.job_id = job_id
        self.queue = event_queue
        self.cancel_event = cancel_event

    async def run_loop(self) -> None:
        """Execute only the job assigned by the scheduler, then exit."""
        if self.should_stop():
            return
        async with get_async_db_session_ctx() as session:
            job = await JobService(session, RemoteTrainerService(session)).claim_pending_train_job(self.job_id)
        if job is not None:
            await self._run_training_job(job, TrainJobPayloadAdapter.validate_python(job.payload))

    async def _run_training_job(self, job: Job, payload: TrainJobPayload) -> None:
        """Prepare and execute one job after its execution target has been reserved."""
        with job_logging_ctx(job_id=str(job.id)):
            settings = get_settings()
            model_id = uuid4()
            # Both remote kinds keep their trainer running independently of the
            # studio process, so either can carry a persisted remote_job_id to
            # reattach to across a restart; only local training never does.
            reattaching = payload.training_target is not TrainingTarget.LOCAL and bool(payload.remote_job_id)

            base_model = None
            if payload.base_model_id is not None:
                async with get_async_db_session_ctx() as session:
                    base_model = await ModelService(session).get_model_by_id(payload.base_model_id)

            model_dir = Path(str(settings.models_dir / str(model_id)))
            if reattaching:
                logger.info("Resuming in-flight remote training job (remote job {})", payload.remote_job_id)
                snapshot: Snapshot | None = None
                snapshot_id = payload.snapshot_id
            else:
                async with get_async_db_session_ctx() as session:
                    dataset = await DatasetService(session).get_dataset_by_id(payload.dataset_id)
                snapshot_dir = settings.snapshot_dir / SnapshotService.generate_snapshot_folder_name(job.id)
                async with get_async_db_session_ctx() as session:
                    snapshot = await SnapshotService(session).create_snapshot_for_dataset(
                        dataset, destination=snapshot_dir
                    )
                snapshot_id = snapshot.id
                payload.snapshot_id = snapshot_id

            model = Model(
                id=model_id,
                project_id=payload.project_id,
                dataset_id=payload.dataset_id,
                path=str(model_dir),
                name=payload.model_name,
                snapshot_id=snapshot_id,
                policy=payload.policy,
                properties={
                    LORA_PROPERTY: payload.lora_enabled,
                    DORA_PROPERTY: payload.lora_use_dora,
                    SNAPFLOW_PROPERTY: payload.snapflow_enabled,
                },
                train_job_id=job.id,
                parent_model_id=payload.base_model_id,
                version=base_model.version + 1 if base_model else 1,
                created_at=None,
            )
            await self._train_model(job, model, snapshot, payload, base_model)

    @staticmethod
    async def _update_training_progress(
        job_id: UUID, progress: int, message: str | None, extra_info: dict | None
    ) -> Job:
        async with get_async_db_session_ctx() as session:
            return await JobService(session, RemoteTrainerService(session)).update_job_status(
                job_id,
                JobStatus.RUNNING,
                message=message,
                progress=progress,
                extra_info=extra_info,
            )

    async def _train_model(
        self,
        job: Job,
        model: Model,
        snapshot: Snapshot | None,
        payload: TrainJobPayload,
        base_model: Model | None = None,
    ) -> None:
        settings = get_settings()
        dispatcher_stop_event = mp.Event()
        async with get_async_db_session_ctx() as session:
            await JobService(session, RemoteTrainerService(session)).update_job(
                job=job,
                update={
                    "status": JobStatus.RUNNING,
                    "message": "Training started",
                    "start_time": job.start_time or datetime.datetime.now(tz=datetime.UTC),
                },
            )
        dispatcher = TrainingTrackingDispatcher(
            job_id=job.id,
            event_queue=self.queue,
            interrupt_event=dispatcher_stop_event,
            update_progress=self._update_training_progress,
        )
        interrupted = False
        suspended = False
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
                remote_job_id=payload.remote_job_id,
                on_remote_job_id=lambda remote_job_id: self._persist_remote_job_id(job, payload, remote_job_id),
                should_suspend=self.should_stop,
                should_cancel_job=lambda: (
                    self.cancel_event.is_set() or bool(getattr(payload, "cancel_requested", False))
                ),
                on_authentication_issue=lambda issue: self._report_authentication_issue(job.id, payload, issue),
            )

            # A reattached remote job still needs its backend to cancel the remote execution.
            if self.cancel_event.is_set() and (
                payload.training_target is TrainingTarget.LOCAL or not payload.remote_job_id
            ):
                raise TrainingCanceledError("Training canceled during preparation")
            backend = await get_training_backend(payload)
            await backend.train(context)
            # The local backend stops cooperatively without raising; treat a
            # completed-but-interrupted run as a cancellation, not a success.
            interrupted = self._should_interrupt()
        except TrainingSuspendedError:
            # Leave the remote job running so a restart can reattach.
            suspended = True
            logger.info("Training suspended for restart; remote job left running")
        except TrainingCanceledError:
            interrupted = True
        except Exception as e:  # surface any training failure as a FAILED job
            error = e
            logger.exception(f"Training failed: {e}")
        finally:
            # Stop the dispatcher and let it flush queued progress BEFORE writing
            # the terminal status. Otherwise a late RUNNING progress update can
            # land after the terminal write and revert the job (stuck at 95%).
            dispatcher_stop_event.set()
            if dispatcher.is_alive():
                dispatcher.join(timeout=10)

        if suspended:
            # Requeue for reattachment after restart.
            async with get_async_db_session_ctx() as session:
                job = await JobService(session, RemoteTrainerService(session)).update_job_status(
                    job_id=job.id,
                    status=JobStatus.PENDING,
                    message="Reconnecting to remote training job after restart",
                )
            self.queue.put((EventType.JOB_UPDATE, job))
            return

        if error is not None:
            async with get_async_db_session_ctx() as session:
                job = await JobService(session, RemoteTrainerService(session)).update_job_status(
                    job_id=job.id, status=JobStatus.FAILED, message=f"Training failed: {error}"
                )
        elif interrupted:
            logger.info("Training canceled")
            async with get_async_db_session_ctx() as session:
                job = await JobService(session, RemoteTrainerService(session)).update_job_status(
                    job_id=job.id, status=JobStatus.CANCELED, message="Training canceled"
                )
        else:
            async with get_async_db_session_ctx() as session:
                job = await JobService(session, RemoteTrainerService(session)).update_job_status(
                    job_id=job.id, status=JobStatus.COMPLETED, message="Training finished"
                )
                model = await ModelService(session).create_model(model)
            self.queue.put((EventType.MODEL_UPDATE, model))

        self.queue.put((EventType.JOB_UPDATE, job))

    async def _persist_remote_job_id(self, job: Job, payload: TrainJobPayload, remote_job_id: UUID) -> None:
        """Persist the remote job id for restart recovery."""
        payload.remote_job_id = remote_job_id
        async with get_async_db_session_ctx() as session:
            service = JobService(session, RemoteTrainerService(session))
            if (
                isinstance(payload, RemoteTrainJobPayload)
                and payload.remote_trainer_url
                and payload.remote_trainer_url.startswith("aws-batch://")
            ):
                await service.update_remote_job_state(
                    job.id, remote_job_id=remote_job_id, snapshot_id=payload.snapshot_id
                )
            else:
                await service.update_job_payload(job.id, payload)
        logger.info("Persisted remote job id {} for restart recovery", remote_job_id)

    async def _report_authentication_issue(
        self, job_id: UUID, payload: TrainJobPayload, issue: AwsAuthenticationIssue | None
    ) -> None:
        if not isinstance(payload, RemoteTrainJobPayload):
            raise ValueError("AWS authentication requires a remote training payload")
        payload.authentication_issue = issue
        async with get_async_db_session_ctx() as session:
            job = await JobService(session, RemoteTrainerService(session)).update_remote_job_state(
                job_id, authentication_issue=issue
            )
        self.queue.put((EventType.JOB_UPDATE, job))

    def _should_interrupt(self) -> bool:
        """Stop training on global shutdown or this job's cancellation event."""
        return self.should_stop() or self.cancel_event.is_set()
