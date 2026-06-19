# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for RemoteTrainingBackend.

Network and HuggingFace boundaries are mocked; the orchestration in
``RemoteTrainingBackend.train`` (push -> submit -> poll -> download -> cleanup)
runs for real.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from schemas.dataset import Snapshot
from schemas.job import TrainJobPayload
from schemas.model import Model
from services.training_backends.base import TrainingContext

if TYPE_CHECKING:
    from pathlib import Path

REMOTE = "services.training_backends.remote"
_SHA = "a" * 40
# create_repo resolves the bare name to a namespaced id; the backend must use this
# resolved id for the upload and cleanup, not the requested (possibly bare) id.
_RESOLVED_REPO_ID = "acme/pais-snapshot-resolved"


# ---------------------------------------------------------------------------
# Fakes for the httpx boundary
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, *, json_data: dict | None = None, chunks: list[bytes] | None = None) -> None:
        self._json = json_data
        self._chunks = chunks or []

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._json or {}

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamCtx:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *_args: object) -> bool:
        return False


class _Controller:
    """Drives fake HTTP responses for a single training run."""

    def __init__(self, states: list[dict], *, remote_job_id: str = "rj-123") -> None:
        self.remote_job_id = remote_job_id
        self._states = states
        self.artifact_chunks = [b"zip-bytes"]
        self.posted_urls: list[str] = []
        self.cancelled = False

    def next_state(self) -> dict:
        if len(self._states) > 1:
            return self._states.pop(0)
        return self._states[0]


class _FakeClient:
    def __init__(self, controller: _Controller, **_kwargs: object) -> None:
        self._c = controller

    async def __aenter__(self) -> _FakeClient:  # noqa: PYI034
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def post(self, url: str, json: dict | None = None) -> _FakeResponse:
        self._c.posted_urls.append(url)
        if url.endswith("/cancel"):
            self._c.cancelled = True
            return _FakeResponse(json_data={})
        return _FakeResponse(json_data={"remote_job_id": self._c.remote_job_id})

    async def get(self, url: str) -> _FakeResponse:
        return _FakeResponse(json_data=self._c.next_state())

    def stream(self, method: str, url: str) -> _FakeStreamCtx:
        return _FakeStreamCtx(_FakeResponse(chunks=self._c.artifact_chunks))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.trainer_url = "https://trainer.test"
    settings.trainer_hf_namespace = "acme"
    settings.trainer_request_timeout_s = 5.0
    settings.data_import_max_uncompressed_bytes = 10 * 1024 * 1024
    settings.data_import_min_free_bytes = 0
    return settings


def _context(tmp_path: Path, *, should_stop: bool = False) -> TrainingContext:
    snap = tmp_path / "snap"
    snap.mkdir()
    model = Model(
        id=uuid4(),
        project_id=uuid4(),
        dataset_id=uuid4(),
        path=str(tmp_path / "model"),
        name="m",
        snapshot_id=uuid4(),
        policy="act",
        properties={},
        train_job_id=uuid4(),
        version=1,
        created_at=None,
    )
    payload = TrainJobPayload(project_id=uuid4(), dataset_id=uuid4(), policy="act", model_name="m")
    return TrainingContext(
        job=MagicMock(),
        model=model,
        snapshot=Snapshot(id=uuid4(), dataset_id=uuid4(), path=str(snap)),
        payload=payload,
        base_model=None,
        output_dir=tmp_path / "model",
        cache_dir=tmp_path / "cache",
        progress=MagicMock(),
        should_stop=lambda: should_stop,
    )


def _hf_api_mock() -> MagicMock:
    api_instance = MagicMock()
    api_instance.create_repo = MagicMock(return_value=MagicMock(repo_id=_RESOLVED_REPO_ID))
    api_instance.upload_folder = MagicMock(return_value=MagicMock(oid=_SHA))
    api_instance.delete_repo = MagicMock()
    return MagicMock(return_value=api_instance)


def _backend(settings: MagicMock):
    from services.training_backends.remote import RemoteTrainingBackend

    with patch(f"{REMOTE}.get_settings", return_value=settings):
        return RemoteTrainingBackend()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRemoteTrainingBackend:
    @pytest.mark.anyio
    async def test_happy_path_pushes_polls_downloads_and_cleans_up(self, tmp_path):
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(
            states=[
                {"status": "running", "progress": 50, "message": "Training", "extra_info": {"train/loss_step": 0.2}},
                {"status": "completed", "progress": 100, "message": "Done"},
            ]
        )
        api_cls = _hf_api_mock()
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch("huggingface_hub.HfApi", api_cls),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=safe_zip),
            patch(f"{REMOTE}._POLL_INTERVAL_S", 0),
        ):
            backend = _backend(settings)
            await backend.train(context)

        # Snapshot pushed to an ephemeral repo and submitted with the pinned SHA.
        api_cls.return_value.create_repo.assert_called_once()
        api_cls.return_value.upload_folder.assert_called_once()
        # Upload and cleanup use the resolved repo id, not the requested bare name.
        assert api_cls.return_value.upload_folder.call_args.kwargs["repo_id"] == _RESOLVED_REPO_ID
        assert api_cls.return_value.delete_repo.call_args.kwargs["repo_id"] == _RESOLVED_REPO_ID
        assert any(url.endswith("/jobs") for url in controller.posted_urls)

        # Artifact extracted safely and ephemeral repo deleted.
        safe_zip.validate.assert_called_once()
        safe_zip.extract_to.assert_called_once()
        api_cls.return_value.delete_repo.assert_called_once()

        # Progress reached the final 99% before the worker marks completion.
        final_progress = [call.args[0] for call in context.progress.call_args_list]
        assert max(final_progress) == 99

    @pytest.mark.anyio
    async def test_cancellation_requests_remote_cancel_and_cleans_up(self, tmp_path):
        settings = _settings()
        context = _context(tmp_path, should_stop=True)
        controller = _Controller(states=[{"status": "running", "progress": 10}])
        api_cls = _hf_api_mock()

        from services.training_backends.base import TrainingCanceledError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch("huggingface_hub.HfApi", api_cls),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}._POLL_INTERVAL_S", 0),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingCanceledError, match="canceled"):
                await backend.train(context)

        assert controller.cancelled is True
        api_cls.return_value.delete_repo.assert_called_once()

    @pytest.mark.anyio
    async def test_remote_failure_raises_and_cleans_up(self, tmp_path):
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[{"status": "failed", "progress": 30, "message": "OOM"}])
        api_cls = _hf_api_mock()

        from services.training_backends.remote import RemoteTrainingError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch("huggingface_hub.HfApi", api_cls),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}._POLL_INTERVAL_S", 0),
        ):
            backend = _backend(settings)
            with pytest.raises(RemoteTrainingError, match="OOM"):
                await backend.train(context)

        api_cls.return_value.delete_repo.assert_called_once()

    @pytest.mark.anyio
    async def test_remote_canceled_status_raises_cancellation(self, tmp_path):
        """A remote terminal 'canceled' state surfaces as cancellation, not a failure."""
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[{"status": "canceled", "progress": 40, "message": "stopped"}])
        api_cls = _hf_api_mock()

        from services.training_backends.base import TrainingCanceledError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch("huggingface_hub.HfApi", api_cls),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}._POLL_INTERVAL_S", 0),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingCanceledError):
                await backend.train(context)

        api_cls.return_value.delete_repo.assert_called_once()

    @pytest.mark.anyio
    async def test_missing_config_raises_on_construction(self):
        settings = _settings()
        settings.trainer_url = None
        from services.training_backends.remote import RemoteTrainingError

        with patch(f"{REMOTE}.get_settings", return_value=settings), pytest.raises(RemoteTrainingError):
            from services.training_backends.remote import RemoteTrainingBackend

            RemoteTrainingBackend()
