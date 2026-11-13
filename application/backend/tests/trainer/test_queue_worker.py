# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the queue manager dispatch and cancellation logic."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from trainer.schemas import TrainerJobStatus

if TYPE_CHECKING:
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


# ---------------------------------------------------------------------------
# Multi-device (DDP) jobs
# ---------------------------------------------------------------------------


def _add_job(manager, sample_request: SubmitJobRequest, **spec_update) -> str:
    spec = sample_request.spec.model_copy(update=spec_update)
    job_id = manager.store.create(sample_request.model_copy(update={"spec": spec}))
    manager.store.mark_dataset_ready(job_id)
    return job_id


def _cuda_devices(count: int) -> list:
    from trainer.schemas import DeviceInfo

    return [DeviceInfo(type="cuda", name=f"GPU {i}", index=i, busy=False) for i in range(count)]


class TestAllDevicesReservation:
    def test_waits_for_every_running_job_to_finish(self, manager, sample_request: SubmitJobRequest) -> None:
        running = _add_job(manager, sample_request, device_type="cuda", device_index=1)
        manager.store.update(running, status=TrainerJobStatus.RUNNING)
        manager._active[running] = MagicMock()
        manager._active_devices[running] = ("cuda", 1)
        ddp = _add_job(manager, sample_request, device_type="cuda", all_devices=True)

        assert manager._next_runnable() is None
        assert manager.store.get(ddp).message == "Waiting for all GPUs to become available"

        manager._active.clear()
        manager._active_devices.clear()
        assert manager._next_runnable() == (ddp, None)

    def test_blocks_every_single_gpu_job_while_it_runs(self, manager, sample_request: SubmitJobRequest) -> None:
        ddp = _add_job(manager, sample_request, device_type="cuda", all_devices=True)
        manager.store.update(ddp, status=TrainerJobStatus.RUNNING)
        manager._active[ddp] = MagicMock()
        manager._active_devices[ddp] = None
        single = _add_job(manager, sample_request, device_type="cuda", device_index=1)

        assert manager._next_runnable() is None
        assert manager.store.get(single).message == "Waiting for CUDA 1 to become available"

    def test_waits_while_another_workload_uses_any_gpu(self, manager, sample_request: SubmitJobRequest) -> None:
        ddp = _add_job(manager, sample_request, device_type="cuda", all_devices=True)
        busy = _cuda_devices(2)
        busy[1].busy = True

        with patch(f"{QUEUE}.get_training_devices", return_value=busy):
            assert manager._next_runnable() is None
        assert manager.store.get(ddp).message == "Waiting for all GPUs to become available"

    def test_run_job_sends_only_multi_device_jobs_to_torchrun(
        self, manager, sample_request: SubmitJobRequest, tmp_path: Path
    ) -> None:
        ddp = _add_job(manager, sample_request, device_type="cuda", all_devices=True)
        single = _add_job(manager, sample_request, device_type="cuda", device_index=0)
        seen = []

        async def fake_run_ddp(job_id, request, report, should_stop):
            seen.append(job_id)
            return tmp_path / "model.zip"

        manager._run_ddp = fake_run_ddp
        manager._runner.run = MagicMock(return_value=tmp_path / "model.zip")

        asyncio.run(manager._run_job(ddp))
        asyncio.run(manager._run_job(single))

        assert seen == [ddp]
        manager._runner.run.assert_called_once()
        assert manager.store.get(ddp).status == TrainerJobStatus.COMPLETED


# Plays rank 0 of a torchrun job: argv is [request_path, events_path, mode].
_FAKE_RANK_ZERO = r"""
import json, os, subprocess, sys, time
request_path, events_path, mode = sys.argv[1:4]
def emit(event):
    with open(events_path, "a") as f:
        f.write(json.dumps(event) + "\n")
if mode == "complete":
    request = json.load(open(request_path))
    emit({"type": "progress", "progress": 40, "message": os.environ.get("HF_TOKEN"), "extra_info": request["spec"]})
    emit({"type": "completed", "archive": "/tmp/job.zip"})
elif mode == "fail":
    emit({"type": "failed", "error": "CUDA out of memory"})
    sys.exit(1)
elif mode == "crash":
    sys.exit(3)
elif mode == "hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(os.path.join(os.path.dirname(os.path.dirname(events_path)), "child.pid"), "w") as f:
        f.write(str(child.pid))
    emit({"type": "progress", "progress": 5, "message": "started", "extra_info": None})
    time.sleep(600)
"""


def _process_running(pid: int) -> bool:
    """True if ``pid`` is alive; an exited-but-unreaped (zombie) process counts as stopped."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
        except (FileNotFoundError, ProcessLookupError):
            return False
        if state == "Z":
            return False
        time.sleep(0.05)
    return True


class TestRunDdp:
    @pytest.fixture
    def ddp(self, manager, tmp_path: Path, monkeypatch):
        from trainer import queue_worker

        manager._ddp_dir = tmp_path / "ddp"
        monkeypatch.setattr(queue_worker, "_DDP_POLL_S", 0.05)
        monkeypatch.setattr(queue_worker, "_DDP_TERMINATE_GRACE_S", 2.0)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        mode = {"value": "complete"}

        def fake_command(_job_id, request_path, events_path, *, nproc):
            mode["nproc"] = nproc
            return [sys.executable, "-c", _FAKE_RANK_ZERO, str(request_path), str(events_path), mode["value"]]

        with (
            patch(f"{QUEUE}.ddp_entry.build_command", side_effect=fake_command),
            patch(f"{QUEUE}.get_training_devices", return_value=_cuda_devices(2)),
        ):
            yield manager, mode

    @staticmethod
    def _request(sample_request: SubmitJobRequest, token: str | None = None) -> SubmitJobRequest:
        spec = sample_request.spec.model_copy(update={"device_type": "cuda", "all_devices": True})
        spec.run_options.hf_token = SecretStr(token) if token else None
        return sample_request.model_copy(update={"spec": spec})

    def test_completed_job_returns_its_archive_and_mirrors_progress(
        self, ddp, sample_request: SubmitJobRequest, tmp_path: Path
    ) -> None:
        manager, mode = ddp
        reported = []

        archive = asyncio.run(
            manager._run_ddp(
                "job-1", self._request(sample_request, token="hf-secret"), lambda *a: reported.append(a), lambda: False
            )
        )

        assert str(archive) == "/tmp/job.zip"
        assert mode["nproc"] == 2
        (progress, token_seen, spec_on_disk) = reported[0]
        assert progress == 40
        assert token_seen == "hf-secret"  # via the environment
        assert "run_options" not in spec_on_disk  # never on disk
        assert spec_on_disk["all_devices"] is True
        assert not (tmp_path / "ddp" / "job-1").exists()
        manager._runner.cleanup_uploaded_dataset.assert_called_once_with("job-1")

    def test_rank_zeros_error_becomes_the_failure_message(self, ddp, sample_request: SubmitJobRequest) -> None:
        manager, mode = ddp
        mode["value"] = "fail"

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            asyncio.run(manager._run_ddp("job-1", self._request(sample_request), MagicMock(), lambda: False))

    def test_an_exit_without_an_event_reports_the_exit_code(self, ddp, sample_request: SubmitJobRequest) -> None:
        manager, mode = ddp
        mode["value"] = "crash"

        with pytest.raises(RuntimeError, match="exited with code 3"):
            asyncio.run(manager._run_ddp("job-1", self._request(sample_request), MagicMock(), lambda: False))

    def test_cancel_stops_the_whole_process_group(self, ddp, sample_request: SubmitJobRequest, tmp_path: Path) -> None:
        """Dataloader workers and every rank must go, not just the torchrun parent."""
        from trainer.runner import JobCanceledError

        manager, mode = ddp
        mode["value"] = "hang"
        reported = []

        with pytest.raises(JobCanceledError):
            asyncio.run(
                manager._run_ddp(
                    "job-1", self._request(sample_request), lambda *a: reported.append(a), lambda: bool(reported)
                )
            )

        assert reported[0][:2] == (5, "started")
        assert not (tmp_path / "ddp" / "job-1").exists()
        child_pid = int((tmp_path / "ddp" / "child.pid").read_text())
        assert not _process_running(child_pid)

    def test_no_cuda_device_fails_before_launching(self, ddp, sample_request: SubmitJobRequest) -> None:
        manager, _mode = ddp

        with (
            patch(f"{QUEUE}.get_training_devices", return_value=[]),
            pytest.raises(RuntimeError, match="needs CUDA devices"),
        ):
            asyncio.run(manager._run_ddp("job-1", self._request(sample_request), MagicMock(), lambda: False))
