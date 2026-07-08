# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for RemoteTrainingBackend.

Network and HuggingFace boundaries are mocked; the orchestration in
``RemoteTrainingBackend.train`` (push -> submit -> stream -> download -> cleanup)
runs for real.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from schemas.dataset import Snapshot
from schemas.job import TrainJobPayload
from schemas.model import Model
from services.training_backends.base import TrainingContext
from services.training_backends.remote import SNAPSHOT_UPLOAD_PROGRESS, TRAINING_PROGRESS_END

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


def _sse_lines(states: list[dict]) -> list[str]:
    """Render job states as Server-Sent Events frames, matching the trainer."""
    lines: list[str] = []
    for state in states:
        lines.append("event: state")
        lines.append(f"data: {json.dumps(state)}")
        lines.append("")
    return lines


class _FakeResponse:
    def __init__(
        self,
        *,
        json_data: dict | list | None = None,
        chunks: list[bytes] | None = None,
        lines: list[str] | None = None,
        headers: dict | None = None,
    ) -> None:
        self._json = json_data
        self._chunks = chunks or []
        self._lines = lines or []
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict | list:
        return self._json if self._json is not None else {}

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aiter_lines(self):
        for line in self._lines:
            yield line


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
        # Each entry is one connection's worth of frames. A reconnect pops the next
        # entry; the last entry repeats. Defaults to a single batch of all states.
        self._event_batches: list[list[dict]] = [list(states)]
        self.artifact_chunks = [b"zip-bytes"]
        self.artifact_headers: dict = {}
        self.posted_urls: list[str] = []
        self.put_urls: list[str] = []
        self.cancelled = False
        self.event_stream_opens = 0
        # Payload served by GET /devices; tests override as needed.
        self.devices_response: dict | list = [{"type": "cpu", "name": "CPU", "memory": None, "index": None}]

    def set_event_batches(self, batches: list[list[dict]]) -> None:
        """Serve a distinct set of frames per connection to exercise reconnection."""
        self._event_batches = batches

    def event_lines(self) -> list[str]:
        self.event_stream_opens += 1
        batch = self._event_batches.pop(0) if len(self._event_batches) > 1 else self._event_batches[0]
        return _sse_lines(batch)


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

    async def put(self, url: str, content: object = None, headers: dict | None = None) -> _FakeResponse:
        self._c.put_urls.append(url)
        # Drain the streamed upload body so the client-side progress generator runs.
        if hasattr(content, "__aiter__"):
            async for _ in content:  # type: ignore[union-attr]
                pass
        elif hasattr(content, "__iter__"):
            for _ in content:  # type: ignore[union-attr]
                pass
        return _FakeResponse(json_data={})

    async def get(self, url: str) -> _FakeResponse:
        # /health is the proxy probe; /devices reports trainer hardware; job state arrives via SSE.
        if url.endswith("/devices"):
            return _FakeResponse(json_data=self._c.devices_response)
        return _FakeResponse(json_data={})

    def stream(self, method: str, url: str, headers: dict | None = None) -> _FakeStreamCtx:
        if url.endswith("/events"):
            return _FakeStreamCtx(_FakeResponse(lines=self._c.event_lines()))
        return _FakeStreamCtx(_FakeResponse(chunks=self._c.artifact_chunks, headers=self._c.artifact_headers))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.trainer_url = "https://trainer.test"
    settings.trainer_hf_namespace = "acme"
    # Existing tests exercise the HF push/pull path; http tests override this.
    settings.trainer_dataset_transfer = "hf"
    settings.trainer_request_timeout_s = 5.0
    settings.trainer_download_read_timeout_s = 120.0
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
    async def test_happy_path_pushes_streams_downloads_and_cleans_up(self, tmp_path):
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
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
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

        # Streamed states drove progress: the mid-run 50% maps into the training window.
        span = TRAINING_PROGRESS_END - SNAPSHOT_UPLOAD_PROGRESS
        reported = [call.args[0] for call in context.progress.call_args_list]
        assert SNAPSHOT_UPLOAD_PROGRESS + round(50 * span / 100) in reported
        # Progress reached 100% before the worker marks completion.
        assert max(reported) == 100

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
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
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
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
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
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingCanceledError):
                await backend.train(context)

        api_cls.return_value.delete_repo.assert_called_once()

    @pytest.mark.anyio
    async def test_truncated_artifact_download_raises(self, tmp_path):
        """A short read versus Content-Length must fail loudly, not hang or extract garbage."""
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[{"status": "completed", "progress": 100, "message": "Done"}])
        # Server advertises more bytes than it streams (connection dropped mid-transfer).
        controller.artifact_chunks = [b"partial"]
        controller.artifact_headers = {"content-length": "999"}
        api_cls = _hf_api_mock()

        from services.training_backends.remote import RemoteTrainingError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch("huggingface_hub.HfApi", api_cls),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=MagicMock()),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            with pytest.raises(RemoteTrainingError, match="truncated"):
                await backend.train(context)

        # Ephemeral repo is still cleaned up on the failure path.
        api_cls.return_value.delete_repo.assert_called_once()

    @pytest.mark.anyio
    async def test_event_stream_reconnects_when_closed_before_terminal(self, tmp_path):
        """A stream that drops before a terminal state reconnects and finishes the job."""
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[])
        # First connection ends after a non-terminal state (drop); second completes.
        controller.set_event_batches(
            [
                [{"status": "running", "progress": 20, "message": "Training"}],
                [{"status": "completed", "progress": 100, "message": "Done"}],
            ]
        )
        api_cls = _hf_api_mock()
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch("huggingface_hub.HfApi", api_cls),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=safe_zip),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
            patch(f"{REMOTE}._RECONNECT_BACKOFF_S", 0),
        ):
            backend = _backend(settings)
            await backend.train(context)

        # The backend opened the stream twice: initial connection plus one reconnect.
        assert controller.event_stream_opens == 2
        safe_zip.extract_to.assert_called_once()
        api_cls.return_value.delete_repo.assert_called_once()

    @pytest.mark.anyio
    async def test_get_training_devices_returns_remote_hardware(self):
        """The backend parses the trainer's /devices report into DeviceInfo."""
        settings = _settings()
        controller = _Controller(states=[])
        controller.devices_response = [
            {"type": "cpu", "name": "CPU", "memory": None, "index": None},
            {"type": "cuda", "name": "NVIDIA A100", "memory": 42949672960, "index": 0},
        ]

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
        ):
            backend = _backend(settings)
            devices = await backend.get_training_devices()

        assert [d.type for d in devices] == ["cpu", "cuda"]
        gpu = devices[1]
        assert gpu.name == "NVIDIA A100"
        assert gpu.memory == 42949672960
        assert gpu.index == 0

    @pytest.mark.anyio
    async def test_get_training_devices_raises_on_invalid_payload(self):
        """A non-list devices payload surfaces as RemoteTrainingError."""
        settings = _settings()
        controller = _Controller(states=[])
        controller.devices_response = {"unexpected": "shape"}

        from services.training_backends.remote import RemoteTrainingError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
        ):
            backend = _backend(settings)
            with pytest.raises(RemoteTrainingError):
                await backend.get_training_devices()

    @pytest.mark.anyio
    async def test_missing_config_raises_on_construction(self):
        settings = _settings()
        settings.trainer_url = None
        from services.training_backends.remote import RemoteTrainingError

        with patch(f"{REMOTE}.get_settings", return_value=settings), pytest.raises(RemoteTrainingError):
            from services.training_backends.remote import RemoteTrainingBackend

            RemoteTrainingBackend()


class TestHttpDatasetTransfer:
    """HTTP transfer submits first, streams the ZIP, then runs the job (no HF repo)."""

    @pytest.mark.anyio
    async def test_uploads_zip_over_http_and_skips_hf(self, tmp_path):
        settings = _settings()
        settings.trainer_dataset_transfer = "http"
        # A file in the snapshot dir gives the archive real bytes to stream.
        context = _context(tmp_path)
        (tmp_path / "snap" / "info.json").write_text("{}")
        controller = _Controller(
            states=[{"status": "completed", "progress": 100, "message": "Done"}],
        )
        api_cls = _hf_api_mock()
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch("huggingface_hub.HfApi", api_cls),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=safe_zip),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            await backend.train(context)

        # Dataset streamed to the trainer's upload endpoint; no HF repo created.
        assert any(url.endswith(f"/jobs/{controller.remote_job_id}/dataset") for url in controller.put_urls)
        assert any(url.endswith("/jobs") for url in controller.posted_urls)
        api_cls.return_value.create_repo.assert_not_called()
        api_cls.return_value.upload_folder.assert_not_called()
        api_cls.return_value.delete_repo.assert_not_called()

        # Model still ingested via the shared download/extract path.
        safe_zip.validate.assert_called_once()
        safe_zip.extract_to.assert_called_once()
        reported = [call.args[0] for call in context.progress.call_args_list]
        assert max(reported) == 100

    @pytest.mark.anyio
    async def test_submit_body_omits_repo_fields_for_http(self, tmp_path):
        settings = _settings()
        settings.trainer_dataset_transfer = "http"
        context = _context(tmp_path)
        (tmp_path / "snap" / "info.json").write_text("{}")

        captured: dict = {}

        async def _fake_submit(_ctx, *, dataset_transfer, repo_id=None, revision=None):
            captured.update(dataset_transfer=dataset_transfer, repo_id=repo_id, revision=revision)
            return "rj-123"

        controller = _Controller(states=[{"status": "completed", "progress": 100}])
        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=MagicMock()),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            with patch.object(backend, "_submit_job", _fake_submit):
                await backend.train(context)

        assert captured == {"dataset_transfer": "http", "repo_id": None, "revision": None}


class TestModelDownloadProgress:
    """The artifact download mirrors bytes received into the model-download window."""

    @pytest.mark.anyio
    async def test_download_bytes_drive_progress_within_reserved_window(self, tmp_path):
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[{"status": "completed", "progress": 100, "message": "Done"}])
        # Two equal chunks against a known total lets us assert exact intermediate percentages.
        controller.artifact_chunks = [b"a" * 500, b"b" * 500]
        controller.artifact_headers = {"content-length": "1000"}
        api_cls = _hf_api_mock()
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch("huggingface_hub.HfApi", api_cls),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=safe_zip),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            await backend.train(context)

        span = 100 - TRAINING_PROGRESS_END
        reported = [call.args[0] for call in context.progress.call_args_list]
        # Halfway through the artifact lands inside the window, capped below 100 (still streaming).
        assert TRAINING_PROGRESS_END + round(0.5 * span) in reported
        # The explicit "Model downloaded" step owns the final 100% mark, not the byte mirror.
        assert max(reported) == 100

    @pytest.mark.anyio
    async def test_missing_content_length_holds_at_window_start(self, tmp_path):
        """Without Content-Length there is no denominator, so progress just holds at the window start."""
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[{"status": "completed", "progress": 100, "message": "Done"}])
        controller.artifact_chunks = [b"chunk-1", b"chunk-2"]
        controller.artifact_headers = {}  # no content-length
        api_cls = _hf_api_mock()
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch("huggingface_hub.HfApi", api_cls),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=safe_zip),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            await backend.train(context)

        reported = [call.args[0] for call in context.progress.call_args_list]
        # No intermediate download-window percentage other than the explicit start/end marks.
        assert TRAINING_PROGRESS_END in reported
        assert max(reported) == 100


class TestSnapshotUploadProgress:
    """The upload mirrors huggingface_hub's byte progress into the snapshot-upload window."""

    class _BaseTqdm:
        """Minimal stand-in for the aggregate "Processing Files" tqdm bar."""

        def __init__(self, *_args, desc: str = "", total: float = 0, initial: float = 0, **_kwargs):
            self.desc = desc
            self.total = total
            self.n = initial

        def update(self, n: float = 1) -> None:
            self.n += n

    def test_processing_bar_drives_progress_and_caps_below_window(self, tmp_path):
        import huggingface_hub.utils._xet_progress_reporting as xet_mod

        backend = _backend(_settings())
        context = _context(tmp_path)
        context.progress = MagicMock()

        cap = SNAPSHOT_UPLOAD_PROGRESS - 1  # the explicit "Snapshot uploaded" step owns the mark
        with patch.object(xet_mod, "tqdm", self._BaseTqdm):
            with backend._mirror_upload_progress(context):
                patched_tqdm = xet_mod.tqdm  # replaced with the forwarding subclass
                bar = patched_tqdm(desc="Processing Files (0 / 4)", total=1000, initial=0)
                bar.update(500)  # 50% -> round(0.5 * window)
                bar.update(450)  # 95% -> rounds up to the window, capped just below it
                bar.update(50)  # 100% -> still capped (the reserved mark is the explicit step)
            # The base class is restored on exit so later uploads are unaffected.
            assert xet_mod.tqdm is self._BaseTqdm

        reported = [call.args[0] for call in context.progress.call_args_list]
        # Duplicates suppressed; never reaches the reserved upload mark.
        assert reported == [round(0.5 * SNAPSHOT_UPLOAD_PROGRESS), cap]

    def test_per_file_bars_do_not_drive_progress(self, tmp_path):
        import huggingface_hub.utils._xet_progress_reporting as xet_mod

        backend = _backend(_settings())
        context = _context(tmp_path)
        context.progress = MagicMock()

        with patch.object(xet_mod, "tqdm", self._BaseTqdm), backend._mirror_upload_progress(context):
            # A per-file bar is keyed by filename, not the aggregate label.
            bar = xet_mod.tqdm(desc="data/chunk-000/file-000.mp4", total=1000, initial=0)
            bar.update(1000)

        context.progress.assert_not_called()

    def test_degrades_gracefully_when_internals_change(self, tmp_path):
        import huggingface_hub.utils._xet_progress_reporting as xet_mod

        backend = _backend(_settings())
        context = _context(tmp_path)
        context.progress = MagicMock()

        # huggingface_hub no longer exposes a tqdm bar to wrap: upload still runs.
        with patch.object(xet_mod, "tqdm", None), backend._mirror_upload_progress(context):
            pass

        context.progress.assert_not_called()
