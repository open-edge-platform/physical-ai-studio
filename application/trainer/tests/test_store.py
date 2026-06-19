# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SQLite-backed job store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trainer.schemas import TrainerJobStatus
from trainer.store import JobStore

if TYPE_CHECKING:
    from pathlib import Path

    from trainer.schemas import SubmitJobRequest


def test_create_persists_queued_job(db_path: Path, sample_request: SubmitJobRequest) -> None:
    store = JobStore(db_path)

    job_id = store.create(sample_request)
    state = store.get(job_id)

    assert state is not None
    assert state.status == TrainerJobStatus.QUEUED
    assert state.progress == 0


def test_get_request_round_trips(db_path: Path, sample_request: SubmitJobRequest) -> None:
    store = JobStore(db_path)
    job_id = store.create(sample_request)

    restored = store.get_request(job_id)

    assert restored is not None
    assert restored.repo_id == sample_request.repo_id
    assert restored.revision == sample_request.revision


def test_next_queued_is_fifo(db_path: Path, sample_request: SubmitJobRequest) -> None:
    store = JobStore(db_path)
    first = store.create(sample_request)
    store.create(sample_request)

    assert store.next_queued() == first


def test_update_progress_and_status(db_path: Path, sample_request: SubmitJobRequest) -> None:
    store = JobStore(db_path)
    job_id = store.create(sample_request)

    store.update(job_id, status=TrainerJobStatus.RUNNING, progress=55, message="halfway")
    state = store.get(job_id)

    assert state is not None
    assert state.status == TrainerJobStatus.RUNNING
    assert state.progress == 55
    assert state.message == "halfway"


def test_progress_is_clamped(db_path: Path, sample_request: SubmitJobRequest) -> None:
    store = JobStore(db_path)
    job_id = store.create(sample_request)

    store.update(job_id, progress=250)
    state = store.get(job_id)

    assert state is not None
    assert state.progress == 100


def test_running_count_reflects_status(db_path: Path, sample_request: SubmitJobRequest) -> None:
    store = JobStore(db_path)
    job_id = store.create(sample_request)
    assert store.running_count() == 0

    store.update(job_id, status=TrainerJobStatus.RUNNING)
    assert store.running_count() == 1


def test_artifact_only_returned_when_set(db_path: Path, sample_request: SubmitJobRequest) -> None:
    store = JobStore(db_path)
    job_id = store.create(sample_request)
    assert store.get_artifact(job_id) is None

    store.update(job_id, artifact="/models/job.zip")
    assert store.get_artifact(job_id) == "/models/job.zip"


def test_reset_orphans_fails_running_jobs(db_path: Path, sample_request: SubmitJobRequest) -> None:
    store = JobStore(db_path)
    job_id = store.create(sample_request)
    store.update(job_id, status=TrainerJobStatus.RUNNING)

    store.reset_orphans()
    state = store.get(job_id)

    assert state is not None
    assert state.status == TrainerJobStatus.FAILED


def test_get_unknown_job_returns_none(db_path: Path) -> None:
    store = JobStore(db_path)
    assert store.get("does-not-exist") is None
