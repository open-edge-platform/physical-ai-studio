import asyncio
import multiprocessing as mp
from multiprocessing.synchronize import Event
from queue import Empty
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger

from schemas.job import JobStatus, JobType
from services.event_processor import EventType
from services.job_service import JobService
from workers.base import BaseThreadWorker


class ProgressCallback(ProgressBar):
    def __init__(self, job_id: UUID):
        super().__init__()
        self.job_id = job_id

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # noqa ARG002
        """Pre-compute total steps once training begins."""
        self.total_steps = trainer.max_steps

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        progress = round((trainer.global_step) / self.total_steps * 100)
        if progress < 100:
            asyncio.run(JobService().update_job_status(job_id=self.job_id, status=JobStatus.RUNNING, progress=progress))


class TrainingTrackingDispatcher(BaseThreadWorker):
    """Dispatch events from the callback to a queue asynchronously."""

    def __init__(self, job_id: UUID, event_queue: mp.Queue, interrupt_event: Event):
        super().__init__(stop_event=interrupt_event)
        self.job_id = job_id
        self.event_queue = event_queue
        self.queue: mp.Queue = mp.Queue()
        self.interrupt_event = interrupt_event

    async def run_loop(self) -> None:
        while not self.interrupt_event.is_set():
            try:
                progress = self.queue.get_nowait()
                job = await JobService.update_job_status(self.job_id, JobStatus.RUNNING, progress=progress)
                self.event_queue.put((EventType.JOB_UPDATE, job))
            except Empty:
                await asyncio.sleep(0.05)

    def update_progress(self, progress: int) -> None:
        self.queue.put(progress)


class TrainingTrackingCallback(Callback):
    def __init__(
        self,
        shutdown_event: Event,
        interrupt_event: Event,
        dispatcher: TrainingTrackingDispatcher,
    ):
        super().__init__()
        self.shutdown_event = shutdown_event  # global stop event in case of shutdown
        self.interrupt_event = interrupt_event  # event for interrupting training gracefully
        self.dispatcher = dispatcher

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",  # noqa ARG002
        outputs: STEP_OUTPUT,  # noqa ARG002
        batch: Any,  # noqa ARG002
        batch_idx: int,  # noqa ARG002
    ) -> None:
        progress = round((trainer.global_step) / trainer.max_steps * 100)
        self.dispatcher.update_progress(progress)
        if self.shutdown_event.is_set() or self.interrupt_event.is_set():
            trainer.should_stop = True


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
