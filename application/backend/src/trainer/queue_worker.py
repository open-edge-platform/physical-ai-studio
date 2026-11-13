# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Background queue worker that dispatches training jobs.

A single asyncio loop polls the store for queued jobs and reserves each GPU for
one job. Parallel training runs in separate processes so process-global state
(including HF_TOKEN) cannot leak between jobs. A multi-device job reserves every
GPU and runs as its own ``torchrun`` process group (see :mod:`trainer.ddp_entry`).
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import os
import shutil
import signal
from pathlib import Path
from queue import Empty
from typing import Any

from loguru import logger

from trainer import ddp_entry
from trainer.devices import get_training_devices, gpu_busy
from trainer.runner import JobCanceledError, TrainerRunner
from trainer.schemas import SubmitJobRequest, TrainerJobStatus
from trainer.settings import get_settings
from trainer.store import JobStore

_DDP_POLL_S = 0.5
"""How often a running multi-device job's events file is polled."""

_DDP_TERMINATE_GRACE_S = 30.0
"""How long a canceled multi-device job gets to exit after SIGTERM before SIGKILL."""


def _job_device(request: SubmitJobRequest | None) -> tuple[str, int] | None:
    """Return the single GPU a job trains on, or None when it may use (or uses) every device."""
    if request is None or request.spec.all_devices or request.spec.device_type not in {"cuda", "xpu"}:
        return None
    return request.spec.device_type, request.spec.device_index or 0


async def _terminate_process_group(process: asyncio.subprocess.Process, grace_s: float) -> None:
    """Stop a process started with ``start_new_session=True`` and everything it spawned."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), grace_s)
    except TimeoutError:
        logger.warning("Multi-device job did not exit within {}s of SIGTERM; killing it", grace_s)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


def _train_in_process(job_id: str, request: SubmitJobRequest, updates: Any, stop: Any) -> None:
    """Run one job in an isolated process; communicate progress over a queue."""
    try:
        archive = TrainerRunner().run(
            job_id,
            request,
            should_stop=stop.is_set,
            report=lambda progress, message, info: updates.put(("progress", progress, message, info)),
        )
        updates.put(("completed", str(archive)))
    except JobCanceledError:
        updates.put(("canceled",))
    except Exception as exc:
        logger.exception("Training job failed: {}", exc)
        updates.put(("failed", str(exc)))


class QueueManager:
    """Owns the job store and drives the dispatch loop."""

    def __init__(self) -> None:
        """Build the store, runner, and concurrency primitives."""
        settings = get_settings()
        self.store = JobStore(settings.db_path)
        self._runner = TrainerRunner()
        self._max_concurrent_jobs = settings.max_concurrent_jobs
        self._ddp_dir = settings.storage_dir / "ddp"
        self._cancel_requested: set[str] = set()
        self._active: dict[str, asyncio.Task] = {}
        self._active_devices: dict[str, tuple[str, int] | None] = {}
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
        # canceled directly since no worker will observe the flag, and its
        # cached HF token (if any) will never be consumed by `_run_job`.
        if state is not None and state.status in {TrainerJobStatus.QUEUED, TrainerJobStatus.AWAITING_DATASET}:
            self.store.update(job_id, status=TrainerJobStatus.CANCELED, message="Canceled before start")
            self.store.discard_secret(job_id)

    def _next_runnable(self) -> tuple[str, tuple[str, int] | None] | None:
        # Scan the small queue; index by device if queued job counts grow.
        next_job = None
        for job_id in self.store.queued():
            request = self.store.get_request(job_id)
            # Legacy auto-selection may use any GPU, and a multi-device job uses
            # every GPU: both reserve the entire trainer.
            device = _job_device(request)
            all_devices = request is not None and request.spec.all_devices
            reserved = (device is None and bool(self._active)) or any(
                active_device is None or active_device == device for active_device in self._active_devices.values()
            )
            # Unknown telemetry is not treated as busy; older setups still work.
            busy = reserved or (
                gpu_busy(*device) if device is not None else any(gpu.busy for gpu in get_training_devices())
            )
            message = (
                "Waiting for all GPUs to become available"
                if busy and all_devices
                else f"Waiting for {device[0].upper()} {device[1]} to become available"
                if busy and device is not None
                else "Waiting for a training slot to become available"
                if busy and reserved
                else "Waiting for a GPU to become available"
                if busy
                else "Queued"
            )
            state = self.store.get(job_id)
            if state is not None and state.status == TrainerJobStatus.QUEUED and state.message != message:
                self.store.update(job_id, message=message)
            if not busy and next_job is None:
                next_job = job_id, device
        return next_job

    async def _dispatch_loop(self) -> None:
        while not self._stopped.is_set():
            while len(self._active) < self._max_concurrent_jobs:
                next_job = self._next_runnable()
                if next_job is None:
                    break
                job_id, device = next_job
                state = self.store.get(job_id)
                if state is None or state.status != TrainerJobStatus.QUEUED:
                    continue
                self.store.update(job_id, status=TrainerJobStatus.RUNNING, progress=0, message="Starting")
                task = asyncio.create_task(self._run_job(job_id))
                self._active[job_id] = task
                self._active_devices[job_id] = device
            # Keep queued-job reasons current even when all training slots are occupied.
            if len(self._active) >= self._max_concurrent_jobs:
                self._next_runnable()
            await asyncio.sleep(0.5)

    async def _run_job(self, job_id: str) -> None:
        try:
            request = self.store.get_request(job_id)
            if request is None:
                self.store.update(job_id, status=TrainerJobStatus.FAILED, message="Missing job request")
                return

            # The HF token was never persisted with the request (see
            # `trainer.schemas.SubmitJobRequest.hf_token`); re-attach the
            # in-memory copy stashed at submission time before training runs.
            request.spec.run_options.hf_token = self.store.take_secret(job_id)

            def _report(progress: int, message: str | None, extra_info: dict | None) -> None:
                self.store.update(job_id, progress=progress, message=message, extra_info=extra_info)

            def _should_stop() -> bool:
                return job_id in self._cancel_requested or self._stopped.is_set()

            if request.spec.all_devices:
                archive_path = await self._run_ddp(job_id, request, _report, _should_stop)
            elif self._max_concurrent_jobs == 1:
                archive_path = await asyncio.to_thread(
                    self._runner.run, job_id, request, should_stop=_should_stop, report=_report
                )
            else:
                archive_path = await self._run_isolated(job_id, request, _report, _should_stop)
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
        except Exception as exc:  # surface any training failure as a FAILED job, never crash the loop
            logger.exception("Training job failed: {}", exc)
            status = TrainerJobStatus.CANCELED if job_id in self._cancel_requested else TrainerJobStatus.FAILED
            self.store.update(job_id, status=status, message=f"Training failed: {exc}")
        finally:
            self._cancel_requested.discard(job_id)
            self._active.pop(job_id, None)
            self._active_devices.pop(job_id, None)
            self._cleanup_if_not_completed(job_id)

    async def _run_isolated(self, job_id: str, request: SubmitJobRequest, report: Any, should_stop: Any) -> Path:
        context = mp.get_context("spawn")
        updates = context.Queue()
        stop = context.Event()
        process = context.Process(target=_train_in_process, args=(job_id, request, updates, stop))
        process.start()
        try:
            while True:
                if should_stop():
                    stop.set()
                try:
                    message = await asyncio.to_thread(updates.get, True, 0.2)
                except Empty:
                    if not process.is_alive():
                        raise RuntimeError(f"Trainer process exited unexpectedly ({process.exitcode})")
                    continue
                if message[0] == "progress":
                    report(message[1], message[2], message[3])
                elif message[0] == "completed":
                    return Path(message[1])
                elif message[0] == "canceled":
                    raise JobCanceledError("Training canceled")
                else:
                    raise RuntimeError(message[1])
        finally:
            stop.set()
            await asyncio.to_thread(process.join)
            updates.close()
            updates.join_thread()

    async def _run_ddp(self, job_id: str, request: SubmitJobRequest, report: Any, should_stop: Any) -> Path:
        """Run a multi-device job as a ``torchrun`` process group and mirror its events.

        See :mod:`trainer.ddp_entry` for why DDP cannot run in this process or
        in an ``mp`` child. The request goes to the job's DDP directory without
        its HF token, which travels in the child's environment instead.
        Cancellation stops the whole process group.

        Raises:
            JobCanceledError: If cancellation was requested.
            RuntimeError: If there is no CUDA device, or the job fails.
        """
        nproc = sum(1 for device in get_training_devices() if device.type == "cuda")
        if nproc == 0:
            msg = "Multi-device training needs CUDA devices, but the trainer reports none"
            raise RuntimeError(msg)

        job_dir = self._ddp_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        request_path, events_path = job_dir / "request.json", job_dir / "events.jsonl"
        request_path.write_text(request.model_dump_json(exclude={"spec": {"run_options"}}), encoding="utf-8")
        events_path.unlink(missing_ok=True)
        env = dict(os.environ)
        if request.spec.run_options.hf_token is not None:
            env["HF_TOKEN"] = request.spec.run_options.hf_token.get_secret_value()

        command = ddp_entry.build_command(job_id, request_path, events_path, nproc=nproc)
        logger.info("Starting multi-device job on {} CUDA devices", nproc)
        process = await asyncio.create_subprocess_exec(*command, env=env, start_new_session=True)
        events = ddp_entry.EventReader(events_path)
        archive: Path | None = None
        error: str | None = None

        def handle(new_events: list[dict[str, Any]]) -> None:
            nonlocal archive, error
            for event in new_events:
                if event.get("type") == "progress":
                    report(event.get("progress", 0), event.get("message"), event.get("extra_info"))
                elif event.get("type") == "completed":
                    archive = Path(event["archive"])
                elif event.get("type") == "failed":
                    error = event.get("error")

        try:
            while process.returncode is None:
                if should_stop():
                    await _terminate_process_group(process, _DDP_TERMINATE_GRACE_S)
                    raise JobCanceledError("Training canceled")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), _DDP_POLL_S)
                handle(events.read_new())
            handle(events.read_new())
            if process.returncode == 0 and archive is not None:
                return archive
            msg = error or f"Multi-device training exited with code {process.returncode}"
            raise RuntimeError(msg)
        finally:
            if process.returncode is None:
                await _terminate_process_group(process, _DDP_TERMINATE_GRACE_S)
            shutil.rmtree(job_dir, ignore_errors=True)
            # Rank 0 normally removes it; a killed or failed rank 0 may not have.
            self._runner.cleanup_uploaded_dataset(job_id)

    def _cleanup_if_not_completed(self, job_id: str) -> None:
        """Remove any leftover model/cache output for a job that didn't complete.

        A COMPLETED job's archive and model directory are kept until the studio
        backend explicitly deletes them after downloading the artifact (see
        the ``DELETE /jobs/{id}`` endpoint); every other terminal outcome has
        nothing worth keeping.
        """
        state = self.store.get(job_id)
        if state is not None and state.status != TrainerJobStatus.COMPLETED:
            self._runner.cleanup_job_outputs(job_id)
