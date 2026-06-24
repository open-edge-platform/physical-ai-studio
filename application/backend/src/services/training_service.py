import asyncio
import multiprocessing as mp
import time
from collections.abc import Mapping
from multiprocessing.synchronize import Event
from queue import Empty
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger

from schemas.base_job import JobStatus, JobType
from services.event_processor import EventType
from services.job_service import JobService
from workers.base import BaseThreadWorker


def _safe_progress(global_step: int, max_steps: int) -> int:
    """Compute step-based training progress clamped to ``[0, 100]``.

    ``trainer.max_steps`` is ``-1`` when unset (epoch-driven runs) and may be ``0``;
    both would otherwise produce a negative value or a ``ZeroDivisionError``.

    Args:
        global_step: Current optimizer step.
        max_steps: Configured maximum number of steps.

    Returns:
        Progress percentage in ``[0, 100]``; ``0`` when ``max_steps`` is unset.
    """
    if max_steps <= 0:
        return 0
    return min(100, max(0, round(global_step / max_steps * 100)))


def _extract_loss(outputs: Any) -> float | None:
    """Best-effort scalar loss from a Lightning step output.

    Handles a ``{"loss": tensor}`` mapping (training) or a bare loss tensor
    (eval-loss validation). Returns ``None`` when no scalar loss is available.
    """
    candidate = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
    detach = getattr(candidate, "detach", None)
    if detach is None:
        return None
    try:
        return detach().cpu().item()
    except (RuntimeError, ValueError):
        return None


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
            asyncio.run(JobService.update_job_status(job_id=self.job_id, status=JobStatus.RUNNING, progress=progress))


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
                progress, extra_info = self.queue.get_nowait()
                job = await JobService.update_job_status(
                    self.job_id, JobStatus.RUNNING, progress=progress, extra_info=extra_info
                )
                self.event_queue.put((EventType.JOB_UPDATE, job))
            except Empty:
                await asyncio.sleep(0.05)

    def update_progress(self, progress: int, extra_info: dict) -> None:
        self.queue.put((progress, extra_info))


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
        self._last_val_progress: int | None = None

    def _should_stop(self, trainer: "pl.Trainer") -> None:
        if self.shutdown_event.is_set() or self.interrupt_event.is_set():
            trainer.should_stop = True

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",  # noqa ARG002
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa ARG002
        batch_idx: int,  # noqa ARG002
    ) -> None:
        loss_val = _extract_loss(outputs)
        progress = _safe_progress(trainer.global_step, trainer.max_steps)
        self.dispatcher.update_progress(progress, extra_info={"train/loss_step": loss_val})
        self._should_stop(trainer)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",  # noqa ARG002
        outputs: STEP_OUTPUT,  # noqa ARG002
        batch: Any,  # noqa ARG002
        batch_idx: int,  # noqa ARG002
        dataloader_idx: int = 0,  # noqa ARG002
    ) -> None:
        """Honor interrupts during validation and refresh job progress on change.

        ``global_step`` does not advance during validation, so the step-based
        progress is constant. Dispatch only when it changes to avoid redundant
        identical job updates.
        """
        progress = _safe_progress(trainer.global_step, trainer.max_steps)
        if progress != self._last_val_progress:
            self._last_val_progress = progress
            self.dispatcher.update_progress(progress, extra_info={"stage": "validation"})
        self._should_stop(trainer)


class TrainingLogCallback(Callback):
    """Mirror training progress/metrics to loguru as regular log lines."""

    def __init__(self):
        super().__init__()
        self.every_n_steps = 1
        self._val_start_t: float | None = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # noqa ARG002
        """Resolve logging interval once trainer values are available."""
        self.every_n_steps = self._auto_every_n_steps(trainer.max_steps)

        logger.info(
            f"Training log cadence configured: every_n_steps={self.every_n_steps}, max_steps={trainer.max_steps}"
        )

    @staticmethod
    def _auto_every_n_steps(total_steps: int) -> int:
        """Choose an interval that targets >=1000 logs and at least every 100 steps.

        Rules:
        - Never less frequent than every 100 steps.
        - Aim for at least 1000 progress log entries when possible.
        """
        if total_steps <= 0:
            return 1

        # Log at least once every 100 steps, otherwise make sure to log 1000 times
        return min(100, max(1, total_steps // 1000))

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",  # noqa ARG002
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa ARG002
        batch_idx: int,  # noqa ARG002
    ) -> None:
        global_step = trainer.global_step
        is_first_step = global_step <= 1
        if not is_first_step and global_step % self.every_n_steps != 0:
            return

        loss_val = _extract_loss(outputs)

        max_steps = max(1, trainer.max_steps)
        progress = _safe_progress(trainer.global_step, trainer.max_steps)
        logger.info(f"Training progress: step={global_step}/{max_steps} ({progress}%), train/loss_step={loss_val}")

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # noqa ARG002
        """Mark the start of validation."""
        self._val_start_t = time.monotonic()
        logger.info(f"Validation started at step={trainer.global_step}/{max(1, trainer.max_steps)}")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",  # noqa ARG002
        pl_module: "pl.LightningModule",  # noqa ARG002
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa ARG002
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa ARG002
    ) -> None:
        current = batch_idx + 1
        if current > 1 and current % self.every_n_steps != 0:
            return
        logger.info(f"Validation progress: batch={current}, val/loss_step={_extract_loss(outputs)}")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # noqa ARG002
        """Summarize validation with the aggregated loss and elapsed time."""
        val_loss = trainer.callback_metrics.get("val/loss")
        val_loss_val = val_loss.item() if val_loss is not None else None
        elapsed = time.monotonic() - self._val_start_t if self._val_start_t is not None else 0.0
        logger.info(
            f"Validation finished at step={trainer.global_step}, val/loss={val_loss_val}, elapsed={elapsed:.1f}s"
        )


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
