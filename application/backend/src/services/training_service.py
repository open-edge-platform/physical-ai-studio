import asyncio
import multiprocessing as mp
from multiprocessing.synchronize import Event
from queue import Empty
from uuid import UUID

from loguru import logger

from schemas.base_job import JobStatus, JobType
from services.event_processor import EventType
from services.job_service import JobService
from workers.base import BaseThreadWorker


class TrainingTrackingDispatcher(BaseThreadWorker):
    """Forward progress updates to the job store and event stream off the hot path.

    Backends call :meth:`report` (a `ProgressReporter`) from the training thread
    or event loop; the dispatcher thread drains the queue and performs the async
    DB writes so training is never blocked on I/O.
    """

    def __init__(self, job_id: UUID, event_queue: mp.Queue, interrupt_event: Event):
        super().__init__(stop_event=interrupt_event)
        self.job_id = job_id
        self.event_queue = event_queue
        self.queue: mp.Queue = mp.Queue()
        self.interrupt_event = interrupt_event

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
        job = await JobService.update_job_status(
            self.job_id,
            JobStatus.RUNNING,
            message=message,
            progress=progress,
            extra_info=extra_info,
        )
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
