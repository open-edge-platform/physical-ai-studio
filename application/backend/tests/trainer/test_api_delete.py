# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trainer's job-artifact cleanup endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer import api as api_module
from trainer.schemas import JobState, TrainerJobStatus

if TYPE_CHECKING:
    from pathlib import Path

_JOB_UUID = uuid4()
_JOB_ID = str(_JOB_UUID)


class _FakeStore:
    def __init__(self, state: JobState | None) -> None:
        self._state = state
        self.deleted: list[str] = []

    def get(self, job_id: str) -> JobState | None:
        return self._state

    def delete(self, job_id: str) -> None:
        self.deleted.append(job_id)


def _app(tmp_path: Path, status_value: TrainerJobStatus, monkeypatch) -> tuple[TestClient, _FakeStore]:
    settings = MagicMock()
    settings.models_dir = tmp_path / "models"
    settings.storage_dir = tmp_path / "storage"
    settings.datasets_dir = tmp_path / "datasets"
    settings.archives_dir = tmp_path / "archives"
    monkeypatch.setattr(api_module, "get_settings", lambda: settings)

    state = JobState(remote_job_id=_JOB_UUID, status=status_value)
    store = _FakeStore(state)

    app = FastAPI()
    app.include_router(api_module.router)
    app.state.queue_manager = SimpleNamespace(store=store)
    return TestClient(app), store


def test_delete_removes_model_cache_dataset_and_archive(tmp_path: Path, monkeypatch) -> None:
    test_client, store = _app(tmp_path, TrainerJobStatus.COMPLETED, monkeypatch)
    model_dir = tmp_path / "models" / _JOB_ID
    cache_dir = tmp_path / "storage" / "cache" / _JOB_ID
    dataset_dir = tmp_path / "datasets" / _JOB_ID
    archive = tmp_path / "archives" / f"{_JOB_ID}.zip"
    for path in (model_dir, cache_dir, dataset_dir):
        path.mkdir(parents=True)
        (path / "file.bin").write_bytes(b"x")
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"zip")

    response = test_client.delete(f"/jobs/{_JOB_ID}")

    assert response.status_code == 204
    assert not model_dir.exists()
    assert not cache_dir.exists()
    assert not dataset_dir.exists()
    assert not archive.exists()
    assert store.deleted == [_JOB_ID]


def test_delete_is_a_no_op_when_nothing_exists(tmp_path: Path, monkeypatch) -> None:
    """Missing files/directories must not raise."""
    test_client, store = _app(tmp_path, TrainerJobStatus.CANCELED, monkeypatch)

    response = test_client.delete(f"/jobs/{_JOB_ID}")

    assert response.status_code == 204
    assert store.deleted == [_JOB_ID]


@pytest.mark.parametrize(
    "status_value",
    [TrainerJobStatus.QUEUED, TrainerJobStatus.RUNNING, TrainerJobStatus.AWAITING_DATASET],
)
def test_delete_rejects_in_progress_job(tmp_path: Path, monkeypatch, status_value) -> None:
    test_client, store = _app(tmp_path, status_value, monkeypatch)

    response = test_client.delete(f"/jobs/{_JOB_ID}")

    assert response.status_code == 409
    assert store.deleted == []
