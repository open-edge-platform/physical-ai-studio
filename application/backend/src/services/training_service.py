import asyncio
import multiprocessing as mp
from collections.abc import Awaitable, Callable, Collection
from multiprocessing.synchronize import Event
from queue import Empty
from uuid import UUID

from loguru import logger

from schemas import Job
from schemas.base_job import JobStatus, JobType
from schemas.job import RemoteTrainJobPayload, SshTrainJobPayload, TrainingTarget
from services.event_processor import EventType
from services.job_service import JobService
from workers.base import BaseThreadWorker


class TrainingTrackingDispatcher(BaseThreadWorker):
    """Forward progress updates to the job store and event stream off the hot path.

    Backends call :meth:`report` (a `ProgressReporter`) from the training thread
    or event loop; the dispatcher thread drains the queue and performs the async
    DB writes so training is never blocked on I/O.
    """

    def __init__(
        self,
        job_id: UUID,
        event_queue: mp.Queue,
        interrupt_event: Event,
        update_progress: Callable[[UUID, int, str | None, dict | None], Awaitable[Job]],
    ):
        super().__init__(stop_event=interrupt_event)
        self.job_id = job_id
        self.event_queue = event_queue
        self.queue: mp.Queue = mp.Queue()
        self.interrupt_event = interrupt_event
        self.update_progress = update_progress

    async def run_loop(self) -> None:
        while not self.interrupt_event.is_set():
            if not await self._drain_one():
                await asyncio.sleep(0.05)
        while await self._drain_one():
            pass

    async def _drain_one(self) -> bool:
        """Apply one queued progress update. Return False when the queue is empty."""
        try:
            progress, message, extra_info = self.queue.get_nowait()
        except Empty:
            return False
        job = await self.update_progress(self.job_id, progress, message, extra_info)
        self.event_queue.put((EventType.JOB_UPDATE, job))
        return True

    def report(self, progress: int, *, message: str | None = None, extra_info: dict | None = None) -> None:
        """`ProgressReporter`-compatible entry point used by training backends."""
        self.queue.put((progress, message, extra_info))


class TrainingService:
    """
    Service for managing model training jobs.

    Handles the complete training pipeline including job fetching, model training,
    status updates, and error handling. Currently, using asyncio.to_thread for
    CPU-intensive training to maintain event loop responsiveness.

    Note: asyncio.to_thread is used assuming single concurrent training job.
    For true parallelism with multiple training jobs, consider ProcessPoolExecutor.
    """

    @staticmethod
    async def abort_orphan_jobs(job_service: JobService, *, exclude_job_ids: Collection[UUID] | None = None) -> None:
        """
        Reconcile RUNNING training jobs left behind by a previous process.

        Called on training-worker setup and teardown. A remote or SSH-provisioned
        trainer keeps running independently of the studio, so a RUNNING job that
        already recorded its ``remote_job_id`` is requeued (back to PENDING) to
        reattach and mirror progress on the next pickup -- this is what lets a run
        survive the studio restarting (e.g. the laptop was closed overnight). Any
        other orphaned RUNNING training job cannot resume and is marked FAILED.

        Args:
            job_service: Used to list and update RUNNING training jobs.
            exclude_job_ids: Job ids to skip entirely, because another
                recovery pass (e.g. `services.ssh.recovery.recover_ssh_jobs`)
                already rendered an explicit verdict for them. Without this,
                an SSH job that pass just confirmed healthy but that hasn't
                yet persisted its own ``remote_job_id`` (a crash between
                provisioning and the trainer job being submitted) would be
                re-judged here using only ``remote_job_id`` and incorrectly
                failed.
        """
        excluded = frozenset(exclude_job_ids) if exclude_job_ids is not None else frozenset()
        query = {"status": JobStatus.RUNNING, "type": JobType.TRAINING}
        running_jobs = await job_service.get_job_list(extra_filters=query)
        for job in running_jobs:
            if job.id in excluded:
                continue
            remote_job_id = TrainingService._reattachable_remote_job_id(job)
            if remote_job_id is not None:
                logger.info(
                    "Requeuing remote training job {} to reattach to trainer job {}",
                    job.id,
                    remote_job_id,
                )
                await job_service.update_job_status(
                    job_id=job.id,
                    status=JobStatus.PENDING,
                    message="Reconnecting to remote training job after restart",
                )
                continue
            logger.warning(f"Aborting orphan training job with id: {job.id}")
            await job_service.update_job_status(
                job_id=job.id,
                status=JobStatus.FAILED,
                message="Job aborted due to application shutdown",
            )

    # Targets whose trainer keeps running independently of the studio process,
    # and so can be reattached to after a restart via their remote_job_id.
    _REATTACHABLE_TARGETS = (TrainingTarget.REMOTE, TrainingTarget.SSH)

    @staticmethod
    def _reattachable_remote_job_id(job: object) -> UUID | None:
        """Return the persisted remote job id for a training job, if any.

        Only `RemoteTrainJobPayload` and `SshTrainJobPayload` carry a
        reattachable `remote_job_id`; class identity already encodes the
        target, so no separate read of `training_target` is needed here.
        """
        payload = getattr(job, "payload", None)
        if isinstance(payload, RemoteTrainJobPayload | SshTrainJobPayload):
            return payload.remote_job_id
        if isinstance(payload, dict):
            remote_job_id = payload.get("remote_job_id")
            if payload.get("training_target") not in {target.value for target in TrainingService._REATTACHABLE_TARGETS}:
                return None
            try:
                return UUID(str(remote_job_id))
            except (TypeError, ValueError, AttributeError):
                return None
        return None
