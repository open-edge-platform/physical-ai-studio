# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trainer's HTTP dataset-upload endpoint."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer import api as api_module
from trainer.schemas import DatasetTransfer, JobState, TrainerJobStatus


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


_JOB_UUID = uuid4()
_JOB_ID = str(_JOB_UUID)


def test_job_id_dependency_returns_canonical_storage_id() -> None:
    assert api_module._get_job_id(UUID(_JOB_UUID.hex)) == _JOB_ID


def test_get_job_resolves_job_dependency(client) -> None:
    test_client, _ = client

    response = test_client.get(f"/jobs/{_JOB_ID}")

    assert response.status_code == 200
    assert response.json()["remote_job_id"] == _JOB_ID


class _FakeStore:
    def __init__(self, state: JobState | None, *, transfer: DatasetTransfer) -> None:
        self._state = state
        self._transfer = transfer
        self.ready_called = False

    def get(self, job_id: str) -> JobState | None:
        return self._state

    def get_request(self, job_id: str):
        return SimpleNamespace(dataset_transfer=self._transfer)

    def mark_dataset_ready(self, job_id: str) -> None:
        self.ready_called = True
        if self._state is not None:
            self._state = self._state.model_copy(update={"status": TrainerJobStatus.QUEUED})


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> tuple[TestClient, _FakeStore]:
    settings = MagicMock()
    settings.datasets_dir = tmp_path / "datasets"
    settings.max_uncompressed_bytes = 100 * 1024 * 1024
    settings.min_free_bytes = 0
    monkeypatch.setattr(api_module, "get_settings", lambda: settings)

    state = JobState(remote_job_id=_JOB_UUID, status=TrainerJobStatus.AWAITING_DATASET)
    store = _FakeStore(state, transfer=DatasetTransfer.HTTP)

    app = FastAPI()
    app.include_router(api_module.router)
    app.state.queue_manager = SimpleNamespace(store=store)
    return TestClient(app), store


_ZIP_HEADERS = {"Content-Type": "application/zip"}


def test_upload_extracts_and_queues_job(client) -> None:
    test_client, store = client
    payload = _zip_bytes({"meta/info.json": b"{}"})

    response = test_client.put(f"/jobs/{_JOB_ID}/dataset", content=payload, headers=_ZIP_HEADERS)

    assert response.status_code == 202
    assert store.ready_called is True
    assert response.json()["status"] == TrainerJobStatus.QUEUED


def test_upload_reserves_space_for_staged_zip_not_extraction_limit(client, monkeypatch) -> None:
    """The upload preflight uses the ZIP request size; extraction checks its own expanded size."""
    test_client, _ = client
    payload = _zip_bytes({"meta/info.json": b"{}"})
    required_bytes: list[int] = []
    monkeypatch.setattr(
        api_module,
        "check_disk_headroom",
        lambda _directory, required, _min_free: required_bytes.append(required),
    )

    response = test_client.put(f"/jobs/{_JOB_ID}/dataset", content=payload, headers=_ZIP_HEADERS)

    assert response.status_code == 202
    assert required_bytes == [len(payload)]


def test_upload_resumes_from_staged_offset(client) -> None:
    """A partial request remains staged until the final byte range arrives."""
    test_client, store = client
    payload = _zip_bytes({"meta/info.json": b"{}"})
    split = len(payload) // 2

    first = test_client.put(
        f"/jobs/{_JOB_ID}/dataset",
        content=payload[:split],
        headers={**_ZIP_HEADERS, "Content-Range": f"bytes 0-{split - 1}/{len(payload)}"},
    )

    assert first.status_code == 202
    assert first.headers["upload-offset"] == str(split)
    assert store.ready_called is False
    offset = test_client.head(f"/jobs/{_JOB_ID}/dataset")
    assert offset.status_code == 204
    assert offset.headers["upload-offset"] == str(split)

    final = test_client.put(
        f"/jobs/{_JOB_ID}/dataset",
        content=payload[split:],
        headers={**_ZIP_HEADERS, "Content-Range": f"bytes {split}-{len(payload) - 1}/{len(payload)}"},
    )

    assert final.status_code == 202
    assert final.headers["upload-offset"] == str(len(payload))
    assert store.ready_called is True


def test_upload_rejects_non_zip_content_type(client) -> None:
    test_client, _ = client

    response = test_client.put(
        f"/jobs/{_JOB_ID}/dataset",
        content=b"not a zip",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 415


def test_upload_rejects_unsafe_zip(client) -> None:
    test_client, store = client
    payload = _zip_bytes({"../escape.txt": b"x"})

    response = test_client.put(f"/jobs/{_JOB_ID}/dataset", content=payload, headers=_ZIP_HEADERS)

    assert response.status_code == 400
    assert store.ready_called is False


def test_upload_conflicts_when_not_awaiting_dataset(tmp_path: Path, monkeypatch) -> None:
    settings = MagicMock()
    settings.datasets_dir = tmp_path / "datasets"
    settings.max_uncompressed_bytes = 100 * 1024 * 1024
    settings.min_free_bytes = 0
    monkeypatch.setattr(api_module, "get_settings", lambda: settings)

    state = JobState(remote_job_id=_JOB_UUID, status=TrainerJobStatus.RUNNING)
    store = _FakeStore(state, transfer=DatasetTransfer.HTTP)

    app = FastAPI()
    app.include_router(api_module.router)
    app.state.queue_manager = SimpleNamespace(store=store)

    response = TestClient(app).put(
        f"/jobs/{_JOB_ID}/dataset",
        content=_zip_bytes({"a": b"b"}),
        headers=_ZIP_HEADERS,
    )

    assert response.status_code == 409


def test_upload_rejects_malformed_job_id(client) -> None:
    test_client, _ = client

    response = test_client.put("/jobs/not-a-uuid/dataset", content=b"", headers=_ZIP_HEADERS)

    assert response.status_code == 422
