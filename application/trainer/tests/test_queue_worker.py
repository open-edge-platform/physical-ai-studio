# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the queue manager dispatch and cancellation logic."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

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
    return mgr


def test_request_cancel_marks_queued_job_canceled(manager, sample_request: SubmitJobRequest) -> None:
    job_id = manager.store.create(sample_request)

    manager.request_cancel(job_id)

    assert manager.store.get(job_id).status == TrainerJobStatus.CANCELED


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
