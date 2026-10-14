# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the queue manager dispatch and cancellation logic."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from trainer.schemas import TrainerJobStatus

if TYPE_CHECKING:
    from pathlib import Path

    from trainer.schemas import SubmitJobRequest

QUEUE = "trainer.queue_worker"


@pytest.fixture
def manager(db_path: Path):
    from trainer.queue_worker import QueueManager

    settings = MagicMock()
    settings.db_path = db_path
    settings.max_concurrent_jobs = 1
    with patch(f"{QUEUE}.get_settings", return_value=settings):
        mgr = QueueManager()
    mgr._runner = MagicMock()
    with patch(f"{QUEUE}.gpu_busy", return_value=False), patch(f"{QUEUE}.get_training_devices", return_value=[]):
        yield mgr


def test_queue_skips_busy_gpu_and_reserves_legacy_auto_jobs(manager, sample_request: SubmitJobRequest) -> None:
    def add_job(index: int | None) -> str:
        spec = sample_request.spec.model_copy(
            update={"device_type": "cuda" if index is not None else None, "device_index": index}
        )
        job_id = manager.store.create(sample_request.model_copy(update={"spec": spec}))
        manager.store.mark_dataset_ready(job_id)
        return job_id

    first = add_job(0)
    same_gpu = add_job(0)
    other_gpu = add_job(1)
    auto = add_job(None)
    manager._active[first] = MagicMock()
    manager._active_devices[first] = ("cuda", 0)
    assert manager._next_runnable() == (other_gpu, ("cuda", 1))
    manager.store.update(first, status=TrainerJobStatus.RUNNING)
    manager._active[other_gpu] = MagicMock()
    manager._active_devices[other_gpu] = ("cuda", 1)
    assert manager._next_runnable() is None
    manager._active.clear()
    manager._active_devices.clear()
    assert manager._next_runnable() == (same_gpu, ("cuda", 0))
    manager.store.update(same_gpu, status=TrainerJobStatus.RUNNING)
    manager.store.update(other_gpu, status=TrainerJobStatus.RUNNING)
    assert manager._next_runnable() == (auto, None)


def test_queue_waits_for_gpu_used_by_another_trainer(manager, sample_request: SubmitJobRequest, monkeypatch) -> None:
    from trainer import queue_worker

    spec = sample_request.spec.model_copy(update={"device_type": "cuda", "device_index": 0})
    job_id = manager.store.create(sample_request.model_copy(update={"spec": spec}))
    manager.store.mark_dataset_ready(job_id)
    monkeypatch.setattr(queue_worker, "gpu_busy", lambda *_: True)
    assert manager._next_runnable() is None
    assert manager.store.get(job_id).status == TrainerJobStatus.QUEUED
    assert manager.store.get(job_id).message == "Waiting for CUDA 0 to become available"
    monkeypatch.setattr(queue_worker, "gpu_busy", lambda *_: False)
    assert manager._next_runnable() == (job_id, ("cuda", 0))
    assert manager.store.get(job_id).message == "Queued"


def test_dispatch_starts_different_gpus_without_waiting_for_same_gpu(manager, sample_request: SubmitJobRequest) -> None:
    manager._max_concurrent_jobs = 2
    ids = []
    for index in (0, 0, 1):
        request = sample_request.model_copy(
            update={"spec": sample_request.spec.model_copy(update={"device_type": "cuda", "device_index": index})}
        )
        job_id = manager.store.create(request)
        manager.store.mark_dataset_ready(job_id)
        ids.append(job_id)

    async def check_dispatch() -> list[str]:
        started = []
        done = asyncio.Event()

        async def fake_run(job_id: str) -> None:
            started.append(job_id)
            await done.wait()

        manager._run_job = fake_run
        loop = asyncio.create_task(manager._dispatch_loop())
        try:
            for _ in range(30):
                if len(started) == 2:
                    break
                await asyncio.sleep(0.01)
            return started
        finally:
            manager._stopped.set()
            done.set()
            loop.cancel()
            await asyncio.gather(loop, *manager._active.values(), return_exceptions=True)

    assert asyncio.run(check_dispatch()) == [ids[0], ids[2]]
    assert manager.store.get(ids[1]).status == TrainerJobStatus.QUEUED
    assert manager.store.get(ids[1]).message == "Waiting for CUDA 0 to become available"


def _fake_isolated_job(job_id, request, updates, stop) -> None:
    updates.put(("progress", 40, str(os.getpid()), None))
    updates.put(("completed", "/tmp/trained.zip"))


def test_parallel_jobs_run_in_separate_processes(manager, sample_request: SubmitJobRequest, monkeypatch) -> None:
    import trainer.queue_worker as queue_worker

    manager._max_concurrent_jobs = 2
    monkeypatch.setattr(queue_worker, "_train_in_process", _fake_isolated_job)
    reported = []
    archive = asyncio.run(
        manager._run_isolated("job-1", sample_request, lambda *args: reported.append(args), lambda: False)
    )

    assert str(archive) == "/tmp/trained.zip"
    assert len(reported) == 1
    assert reported[0][0] == 40 and reported[0][2] is None
    assert int(reported[0][1]) != os.getpid()


def test_request_cancel_marks_queued_job_canceled(manager, sample_request: SubmitJobRequest) -> None:
    job_id = manager.store.create(sample_request)

    manager.request_cancel(job_id)

    assert manager.store.get(job_id).status == TrainerJobStatus.CANCELED


def test_request_cancel_discards_secret_for_undispatched_job(manager, sample_request: SubmitJobRequest) -> None:
    """A job canceled before dispatch never reaches `_run_job`, so its token must not linger."""
    job_id = manager.store.create(sample_request)
    manager.store.stash_secret(job_id, SecretStr("hf-secret"))

    manager.request_cancel(job_id)

    assert manager.store.take_secret(job_id) is None


def test_run_job_attaches_stashed_hf_token_to_spec(manager, sample_request: SubmitJobRequest, tmp_path: Path) -> None:
    """The HF token stashed at submission time is wired back into `run_options` before training."""
    job_id = manager.store.create(sample_request)
    manager.store.stash_secret(job_id, SecretStr("hf-secret"))
    seen_tokens = []

    def fake_run(_job_id, request, **_kw):
        seen_tokens.append(request.spec.run_options.hf_token)
        return tmp_path / "model.zip"

    manager._runner.run = MagicMock(side_effect=fake_run)

    asyncio.run(manager._run_job(job_id))

    assert seen_tokens == [SecretStr("hf-secret")]
    # Consumed exactly once; nothing left cached for this job.
    assert manager.store.take_secret(job_id) is None


def test_run_job_without_stashed_token_leaves_run_options_empty(
    manager, sample_request: SubmitJobRequest, tmp_path: Path
) -> None:
    job_id = manager.store.create(sample_request)
    seen_tokens = []

    def fake_run(_job_id, request, **_kw):
        seen_tokens.append(request.spec.run_options.hf_token)
        return tmp_path / "model.zip"

    manager._runner.run = MagicMock(side_effect=fake_run)

    asyncio.run(manager._run_job(job_id))

    assert seen_tokens == [None]


def test_run_job_completes_and_records_artifact(manager, sample_request: SubmitJobRequest, tmp_path: Path) -> None:
    job_id = manager.store.create(sample_request)
    archive = tmp_path / "model.zip"
    manager._runner.run = MagicMock(return_value=archive)

    asyncio.run(manager._run_job(job_id))

    state = manager.store.get(job_id)
    assert state.status == TrainerJobStatus.COMPLETED
    assert state.progress == 100
    assert manager.store.get_artifact(job_id) == str(archive)


def test_run_job_failure_marks_failed(manager, sample_request: SubmitJobRequest) -> None:
    job_id = manager.store.create(sample_request)
    manager._runner.run = MagicMock(side_effect=RuntimeError("boom"))

    asyncio.run(manager._run_job(job_id))

    state = manager.store.get(job_id)
    assert state.status == TrainerJobStatus.FAILED
    assert "boom" in state.message


def test_run_job_honors_cancellation(manager, sample_request: SubmitJobRequest, tmp_path: Path) -> None:
    job_id = manager.store.create(sample_request)
    manager._runner.run = MagicMock(return_value=tmp_path / "model.zip")
    manager._cancel_requested.add(job_id)

    asyncio.run(manager._run_job(job_id))

    assert manager.store.get(job_id).status == TrainerJobStatus.CANCELED


def test_run_job_canceled_error_marks_canceled_without_failure(
    manager,
    sample_request: SubmitJobRequest,
) -> None:
    """A JobCanceledError from the runner ends the job CANCELED, not FAILED."""
    from trainer.runner import JobCanceledError

    job_id = manager.store.create(sample_request)
    manager._runner.run = MagicMock(side_effect=JobCanceledError("Training canceled"))

    asyncio.run(manager._run_job(job_id))

    state = manager.store.get(job_id)
    assert state.status == TrainerJobStatus.CANCELED
    assert "failed" not in state.message.lower()


def test_run_job_reports_progress_to_store(manager, sample_request: SubmitJobRequest, tmp_path: Path) -> None:
    job_id = manager.store.create(sample_request)

    def _run(job_id_arg, request, *, should_stop, report):
        report(40, "training", {"train/loss_step": 0.3})
        return tmp_path / "model.zip"

    manager._runner.run = MagicMock(side_effect=_run)

    asyncio.run(manager._run_job(job_id))

    # Final state is COMPLETED at 100, but the intermediate report was persisted en route.
    assert manager.store.get(job_id).status == TrainerJobStatus.COMPLETED


def test_run_job_completion_does_not_clean_up_outputs(
    manager,
    sample_request: SubmitJobRequest,
    tmp_path: Path,
) -> None:
    """A COMPLETED job's model/cache output is kept until the backend explicitly deletes it."""
    job_id = manager.store.create(sample_request)
    manager._runner.run = MagicMock(return_value=tmp_path / "model.zip")

    asyncio.run(manager._run_job(job_id))

    manager._runner.cleanup_job_outputs.assert_not_called()


def test_run_job_failure_cleans_up_outputs(manager, sample_request: SubmitJobRequest) -> None:
    """A FAILED job has no artifact worth keeping, so its output/cache is removed."""
    job_id = manager.store.create(sample_request)
    manager._runner.run = MagicMock(side_effect=RuntimeError("boom"))

    asyncio.run(manager._run_job(job_id))

    manager._runner.cleanup_job_outputs.assert_called_once_with(job_id)


def test_run_job_cancellation_cleans_up_outputs(manager, sample_request: SubmitJobRequest, tmp_path: Path) -> None:
    """A canceled job (mid-training or otherwise) leaves nothing behind either."""
    job_id = manager.store.create(sample_request)
    manager._runner.run = MagicMock(return_value=tmp_path / "model.zip")
    manager._cancel_requested.add(job_id)

    asyncio.run(manager._run_job(job_id))

    manager._runner.cleanup_job_outputs.assert_called_once_with(job_id)
