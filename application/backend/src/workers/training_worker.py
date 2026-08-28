# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import datetime
import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from loguru import logger

from core.logging.utils import job_logging_ctx
from db import get_async_db_session_ctx
from repositories.job_provisioning_repo import JobProvisioningRepository
from schemas import Job, Model, Snapshot
from schemas.base_job import JobStatus
from schemas.job import TrainingTarget, TrainJobPayload, TrainJobPayloadAdapter
from services import DatasetService, ModelService
from services.event_processor import EventType
from services.job_service import JobService
from services.remote_server_service import RemoteServerService
from services.remote_trainer_service import RemoteTrainerService
from services.snapshot_service import SnapshotService
from services.ssh.recovery import recover_ssh_jobs
from services.training_backends import (
    TrainingCanceledError,
    TrainingContext,
    TrainingSuspendedError,
    get_training_backend,
)
from services.training_service import TrainingService, TrainingTrackingDispatcher
from services.training_targets import target_key as training_target_key
from settings import get_settings
from workers.base import BaseProcessWorker

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.managers import DictProxy
    from multiprocessing.synchronize import Event as EventClass

SCHEDULE_INTERVAL_SEC = 5


class TrainingWorker(BaseProcessWorker):
    ROLE = "TrainingWorker"

    def __init__(self, stop_event: EventClass, job_interrupt_flags: DictProxy, event_queue: mp.Queue):
        super().__init__(stop_event=stop_event)
        self.queue = event_queue
        # Shared per-job interrupt flags (job id str -> True). Concurrent jobs on
        # different targets each read/clear only their own entry, so cancelling
        # one job cannot cross-cancel another running job.
        self.job_interrupt_flags = job_interrupt_flags
        self._active_training_tasks: dict[str, asyncio.Task[None]] = {}

    async def run_loop(self) -> None:
        logger.info("Training Worker is running")
        try:
            while not self.should_stop():
                async with get_async_db_session_ctx() as session:
                    pending_jobs = await JobService(session, RemoteTrainerService(session)).get_pending_train_jobs()
                for job in pending_jobs:
                    payload = TrainJobPayloadAdapter.validate_python(job.payload)
                    target = self._target_key(payload)
                    if target in self._active_training_tasks:
                        continue
                    task = asyncio.create_task(self._run_training_job(job, payload), name=f"training-{job.id}")
                    self._active_training_tasks[target] = task
                    task.add_done_callback(self._make_release_callback(target, job.id))
                self.stop_aware_sleep(0.5)
        finally:
            if self._active_training_tasks:
                await asyncio.gather(*self._active_training_tasks.values(), return_exceptions=True)

    def _release_target(self, target: str, job_id: UUID, completed_task: asyncio.Task[None]) -> None:
        """Release a target only when its currently registered task completes."""
        if self._active_training_tasks.get(target) is completed_task:
            self._active_training_tasks.pop(target)
        self.job_interrupt_flags.pop(str(job_id), None)

    def _make_release_callback(self, target: str, job_id: UUID) -> Callable[[asyncio.Task[None]], None]:
        """Build a done-callback bound to `target`/`job_id` for `add_done_callback`."""

        def _on_done(completed_task: asyncio.Task[None]) -> None:
            self._release_target(target, job_id, completed_task)

        return _on_done

    @staticmethod
    def _target_key(payload: TrainJobPayload) -> str:
        """Return the exclusive execution target for a training job.

        Delegates to `services.training_targets.target_key` so submission
        validation (`JobService`) and worker scheduling derive this key from
        the same per-target handler registry, instead of each keeping its own
        copy of the target-to-key mapping.
        """
        return training_target_key(payload)

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
                snapshot_dir = settings.snapshot_dir / SnapshotService.generate_snapshot_folder_name()
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
                properties={},
                train_job_id=job.id,
                parent_model_id=payload.base_model_id,
                version=base_model.version + 1 if base_model else 1,
                created_at=None,
            )
            await self._train_model(job, model, snapshot, payload, base_model)

    async def setup(self) -> None:
        await super().setup()
        with logger.contextualize(worker=self.__class__.__name__):
            # SSH recovery must run before the generic orphan abort: it confirms
            # or fails each SSH job's container explicitly, so the generic pass
            # only ever needs to catch a job this one somehow failed to reach.
            # Every job id it rendered a verdict for is excluded from the
            # generic pass, which otherwise judges solely on `remote_job_id`
            # and could re-fail a job SSH recovery just confirmed healthy.
            handled_job_ids = await self._recover_ssh_jobs()
            await self._abort_orphan_jobs(exclude_job_ids=handled_job_ids)

    async def teardown(self) -> None:
        await super().teardown()
        with logger.contextualize(worker=self.__class__.__name__):
            await self._abort_orphan_jobs()

    @staticmethod
    async def _abort_orphan_jobs(*, exclude_job_ids: frozenset[UUID] | None = None) -> None:
        async with get_async_db_session_ctx() as session:
            await TrainingService.abort_orphan_jobs(
                JobService(session, RemoteTrainerService(session)), exclude_job_ids=exclude_job_ids
            )

    @staticmethod
    async def _recover_ssh_jobs() -> frozenset[UUID]:
        """Reattach or fail every SSH-provisioned job left non-terminal by a restart.

        Returns:
            Every job id SSH recovery rendered a verdict for, so the caller can
            exclude them from the generic orphan abort that follows.
        """
        async with get_async_db_session_ctx() as session:
            provisioning_repo = JobProvisioningRepository(session)
            remote_server_service = RemoteServerService(session)
            job_service = JobService(session, RemoteTrainerService(session), remote_server_service)
            report = await recover_ssh_jobs(job_service, provisioning_repo, remote_server_service)
        logger.info(
            "SSH job recovery: {} confirmed, {} pending retry, {} failed, {} stale row(s) cleaned, "
            "{} orphan container(s) removed",
            report.confirmed,
            report.transient,
            report.failed,
            report.stale_rows_cleaned,
            report.orphans_removed,
        )
        return report.handled_job_ids

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
                    "start_time": datetime.datetime.now(tz=datetime.UTC),
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
                should_stop=lambda: self._should_interrupt(job.id),
                remote_job_id=payload.remote_job_id,
                on_remote_job_id=lambda remote_job_id: self._persist_remote_job_id(job, payload, remote_job_id),
                should_suspend=self.should_stop,
                should_cancel_job=lambda: bool(self.job_interrupt_flags.get(str(job.id), False)),
            )

            backend = await get_training_backend(payload, job.id)
            await backend.train(context)
            # The local backend stops cooperatively without raising; treat a
            # completed-but-interrupted run as a cancellation, not a success.
            interrupted = self._should_interrupt(job.id)
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
            await JobService(session, RemoteTrainerService(session)).update_job_payload(job.id, payload)
        logger.info("Persisted remote job id {} for restart recovery", remote_job_id)

    def _should_interrupt(self, job_id: UUID) -> bool:
        """Stop training on global shutdown or an interrupt requested for this job."""
        return self.should_stop() or bool(self.job_interrupt_flags.get(str(job_id), False))
