# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Background queue worker that dispatches training jobs.

A single asyncio loop polls the store for queued jobs and runs up to
``max_concurrent_jobs`` at a time, each in a worker thread (training is
blocking). Cancellation is cooperative via an in-memory request set checked by
the runner's ``should_stop`` callback.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from trainer.runner import JobCanceledError, TrainerRunner
from trainer.schemas import TrainerJobStatus
from trainer.settings import get_settings
from trainer.store import JobStore


class QueueManager:
    """Owns the job store and drives the dispatch loop."""

    def __init__(self) -> None:
        """Build the store, runner, and concurrency primitives."""
        settings = get_settings()
        self.store = JobStore(settings.db_path)
        self._runner = TrainerRunner()
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
        self._cancel_requested: set[str] = set()
        self._active: dict[str, asyncio.Task] = {}
        self._stopped = asyncio.Event()
        self._loop_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Reset orphaned jobs and begin dispatching."""
        self.store.reset_orphans()
        self._loop_task = asyncio.create_task(self._dispatch_loop())
        logger.info("Queue manager started")

    async def shutdown(self) -> None:
        """Stop dispatching and let in-flight jobs unwind."""
        self._stopped.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        for job_id in list(self._active):
            self._cancel_requested.add(job_id)
        await asyncio.gather(*self._active.values(), return_exceptions=True)
        logger.info("Queue manager stopped")

    def request_cancel(self, job_id: str) -> None:
        """Flag a job for cooperative cancellation."""
        self._cancel_requested.add(job_id)
        state = self.store.get(job_id)
        # A job not yet dispatched (queued, or still awaiting its dataset) is
        # canceled directly since no worker will observe the flag.
        if state is not None and state.status in {TrainerJobStatus.QUEUED, TrainerJobStatus.AWAITING_DATASET}:
            self.store.update(job_id, status=TrainerJobStatus.CANCELED, message="Canceled before start")

    async def _dispatch_loop(self) -> None:
        while not self._stopped.is_set():
            job_id = self.store.next_queued()
            if job_id is None:
                await asyncio.sleep(1.0)
                continue
            await self._semaphore.acquire()
            # Re-check: it may have been canceled while queued.
            state = self.store.get(job_id)
            if state is None or state.status != TrainerJobStatus.QUEUED:
                self._semaphore.release()
                continue
            self.store.update(job_id, status=TrainerJobStatus.RUNNING, progress=0, message="Starting")
            task = asyncio.create_task(self._run_job(job_id))
            self._active[job_id] = task

    async def _run_job(self, job_id: str) -> None:
        try:
            request = self.store.get_request(job_id)
            if request is None:
                self.store.update(job_id, status=TrainerJobStatus.FAILED, message="Missing job request")
                return

            def _report(progress: int, message: str | None, extra_info: dict | None) -> None:
                self.store.update(job_id, progress=progress, message=message, extra_info=extra_info)

            def _should_stop() -> bool:
                return job_id in self._cancel_requested or self._stopped.is_set()

            archive_path = await asyncio.to_thread(
                self._runner.run,
                job_id,
                request,
                should_stop=_should_stop,
                report=_report,
            )
            if job_id in self._cancel_requested:
                self.store.update(job_id, status=TrainerJobStatus.CANCELED, message="Canceled")
            else:
                self.store.update(
                    job_id,
                    status=TrainerJobStatus.COMPLETED,
                    progress=100,
                    message="Training finished",
                    artifact=str(archive_path),
                )
        except JobCanceledError:
            logger.info("Training job canceled")
            self.store.update(job_id, status=TrainerJobStatus.CANCELED, message="Canceled")
        except Exception as exc:  # noqa: BLE001  # surface any training failure as a FAILED job, never crash the loop
            logger.exception("Training job failed: {}", exc)
            status = TrainerJobStatus.CANCELED if job_id in self._cancel_requested else TrainerJobStatus.FAILED
            self.store.update(job_id, status=status, message=f"Training failed: {exc}")
        finally:
            self._cancel_requested.discard(job_id)
            self._active.pop(job_id, None)
            self._semaphore.release()
