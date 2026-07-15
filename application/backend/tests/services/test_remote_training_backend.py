# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for RemoteTrainingBackend.

The network boundary is mocked; the orchestration in
``RemoteTrainingBackend.train`` (archive -> submit -> upload -> stream ->
download) runs for real.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest

from schemas.dataset import Snapshot
from schemas.job import TrainJobPayload
from schemas.model import Model
from services.training_backends._transfer_progress import TransferProgressLogger, format_bytes, format_throughput
from services.training_backends.base import TrainingContext
from services.training_backends.remote import SNAPSHOT_UPLOAD_PROGRESS, TRAINING_PROGRESS_END, RemoteTrainingError

if TYPE_CHECKING:
    from pathlib import Path

REMOTE = "services.training_backends.remote"
TRANSFER = "services.training_backends._transfer_progress"


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

    async def aiter_bytes(self, chunk_size: int | None = None):
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

    def __init__(self, states: list[dict], *, remote_job_id: UUID | None = None) -> None:
        remote_job_id = remote_job_id or uuid4()
        self.remote_job_id = remote_job_id
        # Each entry is one connection's worth of frames. A reconnect pops the next
        # entry; the last entry repeats. Defaults to a single batch of all states.
        self._event_batches: list[list[dict]] = [list(states)]
        self.artifact_chunks = [b"zip-bytes"]
        self.artifact_headers: dict = {}
        self.artifact_batches: list[tuple[list[bytes], dict]] | None = None
        self.artifact_range_headers: list[str | None] = []
        self.upload_offset = 0
        self.upload_ranges: list[str] = []
        self.posted_urls: list[str] = []
        self.posted_bodies: list[dict | None] = []
        self.put_urls: list[str] = []
        self.cancelled = False
        self.event_stream_opens = 0
        # Payload served by GET /devices; tests override as needed.
        self.devices_response: dict | list = [{"type": "cpu", "name": "CPU", "memory": None, "index": None}]
        # Payload served by GET /jobs/{id} (the stream-independent poll fallback).
        self.poll_state: dict = {}
        # When True, opening the event stream and polling the job endpoint raise a
        # connection error, simulating the trainer being transiently unreachable.
        self.raise_connection_error = False
        # Number of job-state polls served (GET /jobs/{id}).
        self.poll_count = 0

    def set_event_batches(self, batches: list[list[dict]]) -> None:
        """Serve a distinct set of frames per connection to exercise reconnection."""
        self._event_batches = batches

    def event_lines(self) -> list[str]:
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
        self._c.posted_bodies.append(json)
        if url.endswith("/cancel"):
            self._c.cancelled = True
            return _FakeResponse(json_data={})
        return _FakeResponse(json_data={"remote_job_id": str(self._c.remote_job_id)})

    async def put(self, url: str, content: object = None, headers: dict | None = None) -> _FakeResponse:
        self._c.put_urls.append(url)
        if headers and "Content-Range" in headers:
            self._c.upload_ranges.append(headers["Content-Range"])
        # Drain the streamed upload body so the client-side progress generator runs.
        if hasattr(content, "__aiter__"):
            async for _ in content:  # type: ignore[union-attr]
                pass
        elif hasattr(content, "__iter__"):
            for _ in content:  # type: ignore[union-attr]
                pass
        if headers and "Content-Range" in headers:
            self._c.upload_offset = int(headers["Content-Range"].split(" ")[1].split("-")[1].split("/")[0]) + 1
        return _FakeResponse(json_data={}, headers={"upload-offset": str(self._c.upload_offset)})

    async def head(self, url: str) -> _FakeResponse:
        return _FakeResponse(headers={"upload-offset": str(self._c.upload_offset)})

    async def get(self, url: str) -> _FakeResponse:
        # /health is the proxy probe; /devices reports trainer hardware; job state
        # arrives via SSE, with GET /jobs/{id} as the reconnect poll fallback.
        if url.endswith("/devices"):
            return _FakeResponse(json_data=self._c.devices_response)
        if url.endswith(f"/jobs/{self._c.remote_job_id}"):
            self._c.poll_count += 1
            if self._c.raise_connection_error:
                raise httpx.ConnectError("All connection attempts failed")
            return _FakeResponse(json_data=self._c.poll_state)
        return _FakeResponse(json_data={})

    def stream(self, method: str, url: str, headers: dict | None = None) -> _FakeStreamCtx:
        if url.endswith("/events"):
            self._c.event_stream_opens += 1
            if self._c.raise_connection_error:
                raise httpx.ConnectError("All connection attempts failed")
            return _FakeStreamCtx(_FakeResponse(lines=self._c.event_lines()))
        self._c.artifact_range_headers.append(headers.get("Range") if headers else None)
        if self._c.artifact_batches:
            chunks, artifact_headers = self._c.artifact_batches.pop(0)
            return _FakeStreamCtx(_FakeResponse(chunks=chunks, headers=artifact_headers))
        return _FakeStreamCtx(_FakeResponse(chunks=self._c.artifact_chunks, headers=self._c.artifact_headers))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.trainer_url = "https://trainer.test"
    settings.trainer_request_timeout_s = 5.0
    settings.trainer_download_read_timeout_s = 120.0
    settings.trainer_stream_reconnect_max_s = 900.0
    settings.trainer_stream_reconnect_backoff_max_s = 30.0
    settings.data_import_max_uncompressed_bytes = 10 * 1024 * 1024
    settings.data_import_min_free_bytes = 0
    return settings


def _context(tmp_path: Path, *, should_stop: bool = False, remote_job_id: UUID | None = None) -> TrainingContext:
    snap = tmp_path / "snap"
    snap.mkdir()
    # A file in the snapshot dir gives the ZIP archive real bytes to stream.
    (snap / "info.json").write_text("{}")
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
        remote_job_id=remote_job_id,
    )


def _backend(settings: MagicMock):
    from services.training_backends.remote import RemoteTrainingBackend

    with patch(f"{REMOTE}.get_settings", return_value=settings):
        return RemoteTrainingBackend()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRemoteTrainingBackend:
    @pytest.mark.anyio
    async def test_happy_path_uploads_streams_and_downloads(self, tmp_path):
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(
            states=[
                {"status": "running", "progress": 50, "message": "Training", "extra_info": {"train/loss_step": 0.2}},
                {"status": "completed", "progress": 100, "message": "Done"},
            ]
        )
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=safe_zip),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            await backend.train(context)

        assert any(url.endswith("/jobs") for url in controller.posted_urls)

        safe_zip.validate.assert_called_once()
        safe_zip.extract_to.assert_called_once()

        # Streamed states drove progress: the mid-run 50% maps into the training window.
        span = TRAINING_PROGRESS_END - SNAPSHOT_UPLOAD_PROGRESS
        reported = [call.args[0] for call in context.progress.call_args_list]
        assert SNAPSHOT_UPLOAD_PROGRESS + round(50 * span / 100) in reported
        # Progress reached 100% before the worker marks completion.
        assert max(reported) == 100

    @pytest.mark.anyio
    async def test_cancellation_requests_remote_cancel(self, tmp_path):
        settings = _settings()
        context = _context(tmp_path, should_stop=True)
        controller = _Controller(states=[{"status": "running", "progress": 10}])

        from services.training_backends.base import TrainingCanceledError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingCanceledError, match="canceled"):
                await backend.train(context)

        assert controller.cancelled is True

    @pytest.mark.anyio
    async def test_remote_failure_raises(self, tmp_path):
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[{"status": "failed", "progress": 30, "message": "OOM"}])

        from services.training_backends.remote import RemoteTrainingError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            with pytest.raises(RemoteTrainingError, match="OOM"):
                await backend.train(context)

    @pytest.mark.anyio
    async def test_remote_canceled_status_raises_cancellation(self, tmp_path):
        """A remote terminal 'canceled' state surfaces as cancellation, not a failure."""
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[{"status": "canceled", "progress": 40, "message": "stopped"}])

        from services.training_backends.base import TrainingCanceledError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingCanceledError):
                await backend.train(context)

    @pytest.mark.anyio
    async def test_truncated_artifact_download_raises(self, tmp_path):
        """A short read versus Content-Length must fail loudly, not hang or extract garbage."""
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[{"status": "completed", "progress": 100, "message": "Done"}])
        # Server advertises more bytes than it streams (connection dropped mid-transfer).
        controller.artifact_chunks = [b"partial"]
        controller.artifact_headers = {"content-length": "999"}

        from services.training_backends.remote import RemoteTrainingError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=MagicMock()),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            with pytest.raises(RemoteTrainingError, match="truncated"):
                await backend.train(context)

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
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
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

    @pytest.mark.anyio
    async def test_completion_detected_via_poll_when_stream_unreachable(self, tmp_path):
        """If the stream drops before a terminal state, the poll fallback ingests a finished job.

        The trainer keeps training while the event stream is down, so a terminal
        state reached during the outage must be picked up by the poll fallback
        rather than requiring a fresh terminal event on the stream.
        """
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[])
        # The stream emits a non-terminal state and then drops.
        controller.set_event_batches([[{"status": "running", "progress": 20, "message": "Training"}]])
        # The stream-independent poll reports the job finished during the outage.
        controller.poll_state = {"status": "completed", "progress": 100, "message": "Done"}
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=safe_zip),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
            patch(f"{REMOTE}._RECONNECT_BACKOFF_S", 0),
        ):
            backend = _backend(settings)
            await backend.train(context)

        # Completion came from the poll, and the artifact was still ingested.
        assert controller.poll_count >= 1
        safe_zip.extract_to.assert_called_once()

    @pytest.mark.anyio
    async def test_abandons_job_when_trainer_unreachable_past_budget(self, tmp_path):
        """Persistent unreachability past the reconnect budget aborts the job."""
        settings = _settings()
        # Zero budget: the first failed reconnect+poll cycle exhausts it.
        settings.trainer_stream_reconnect_max_s = 0.0
        settings.trainer_stream_reconnect_backoff_max_s = 0.0
        context = _context(tmp_path)
        controller = _Controller(states=[])
        controller.raise_connection_error = True

        from services.training_backends.remote import RemoteTrainingError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
            patch(f"{REMOTE}._RECONNECT_BACKOFF_S", 0),
        ):
            backend = _backend(settings)
            with pytest.raises(RemoteTrainingError, match="unreachable"):
                await backend.train(context)

        # Both the stream and the poll fallback were attempted before giving up.
        assert controller.event_stream_opens >= 1
        assert controller.poll_count >= 1

    @pytest.mark.anyio
    async def test_reattaches_to_running_job_without_resubmitting(self, tmp_path):
        """Resuming after a restart streams progress and downloads the model, but never resubmits."""
        settings = _settings()
        # Reattach: the remote job id is already known from a prior (pre-restart) run.
        remote_job_id = uuid4()
        context = _context(tmp_path, remote_job_id=remote_job_id)
        controller = _Controller(
            states=[{"status": "completed", "progress": 100, "message": "Done"}],
            remote_job_id=remote_job_id,
        )
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=safe_zip),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            await backend.train(context)

        assert not any(url.endswith("/jobs") for url in controller.posted_urls)
        # The finished model was still streamed and extracted.
        safe_zip.extract_to.assert_called_once()

    @pytest.mark.anyio
    async def test_submit_rejects_malformed_remote_job_id(self, tmp_path):
        """The trainer response is validated before an ID reaches URL construction."""
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[])
        controller.remote_job_id = "valid-job/../unexpected"  # type: ignore[assignment]

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
        ):
            backend = _backend(settings)
            with pytest.raises(RemoteTrainingError, match="Trainer did not return a valid remote_job_id"):
                await backend.submit_job(context)

    @pytest.mark.anyio
    async def test_shutdown_suspends_without_canceling_remote_job(self, tmp_path):
        """On studio shutdown the remote job is left running and TrainingSuspendedError is raised."""
        settings = _settings()
        # should_stop True (a stop is requested) and should_suspend True (it is a
        # shutdown, not a user cancel).
        context = _context(tmp_path, should_stop=True)
        context.should_suspend = lambda: True
        controller = _Controller(states=[{"status": "running", "progress": 30}])

        from services.training_backends import TrainingSuspendedError

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingSuspendedError):
                await backend.train(context)

        # The remote job must NOT be canceled on shutdown so it can be reattached.
        assert controller.cancelled is False

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
    """HTTP transfer submits first, streams the ZIP, then runs the job."""

    @pytest.mark.anyio
    async def test_uploads_zip_over_http(self, tmp_path):
        settings = _settings()
        # A file in the snapshot dir gives the archive real bytes to stream.
        context = _context(tmp_path)
        (tmp_path / "snap" / "info.json").write_text("{}")
        controller = _Controller(
            states=[{"status": "completed", "progress": 100, "message": "Done"}],
        )
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=safe_zip),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            await backend.train(context)

        assert any(url.endswith(f"/jobs/{controller.remote_job_id}/dataset") for url in controller.put_urls)
        assert any(url.endswith("/jobs") for url in controller.posted_urls)

        # Model still ingested via the shared download/extract path.
        safe_zip.validate.assert_called_once()
        safe_zip.extract_to.assert_called_once()
        reported = [call.args[0] for call in context.progress.call_args_list]
        assert max(reported) == 100

    @pytest.mark.anyio
    async def test_submit_body_uses_http_transfer_without_repo_fields(self, tmp_path):
        settings = _settings()
        context = _context(tmp_path)

        controller = _Controller(states=[{"status": "completed", "progress": 100}])
        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=MagicMock()),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            await backend.train(context)

        body = next(
            body
            for url, body in zip(controller.posted_urls, controller.posted_bodies, strict=False)
            if url.endswith("/jobs") and body is not None
        )
        assert body["dataset_transfer"] == "http"
        assert "repo_id" not in body
        assert "revision" not in body

    @pytest.mark.anyio
    async def test_http_persists_remote_job_id_after_upload(self, tmp_path):
        """The HTTP path records the remote job id so a restart can reattach.

        Regression guard: without this the default (http) transfer never persists
        the id and a restart re-submits and re-uploads the whole dataset.
        """
        settings = _settings()
        context = _context(tmp_path)
        (tmp_path / "snap" / "info.json").write_text("{}")

        persisted: list[UUID] = []

        async def _on_remote_job_id(remote_job_id: UUID) -> None:
            persisted.append(remote_job_id)

        context.on_remote_job_id = _on_remote_job_id
        controller = _Controller(states=[{"status": "completed", "progress": 100, "message": "Done"}])

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=MagicMock()),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            await backend.train(context)

        # The id was persisted, and it matches the id the dataset was uploaded to.
        assert persisted == [controller.remote_job_id]
        assert any(url.endswith(f"/jobs/{controller.remote_job_id}/dataset") for url in controller.put_urls)

    @pytest.mark.anyio
    async def test_http_upload_starts_at_trainer_staged_offset(self, tmp_path):
        """Reattachment only streams the portion of a snapshot absent from the trainer."""
        settings = _settings()
        context = _context(tmp_path)
        archive = tmp_path / "snapshot.zip"
        archive.write_bytes(b"abcdefghij")
        controller = _Controller(states=[])
        controller.upload_offset = 4

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
        ):
            backend = _backend(settings)
            await backend.upload_snapshot_http(context, controller.remote_job_id, archive)

        assert controller.upload_ranges == ["bytes 4-9/10"]


class TestSnapshotUploadCancellation:
    """Cancelling / suspending during the HTTP dataset upload reacts promptly.

    Regression guard: the upload loop must observe ``should_stop`` instead of
    streaming the whole (potentially multi-GB) snapshot before the wait loop
    finally notices the cancel.
    """

    @pytest.mark.anyio
    async def test_cancel_before_upload_cancels_remote_job(self, tmp_path):
        from services.training_backends.base import TrainingCanceledError

        settings = _settings()
        context = _context(tmp_path, should_stop=True)
        archive = tmp_path / "snapshot.zip"
        archive.write_bytes(b"abcdefghij")
        controller = _Controller(states=[])

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingCanceledError):
                await backend.upload_snapshot_http(context, controller.remote_job_id, archive)

        assert controller.cancelled is True
        assert not controller.put_urls

    @pytest.mark.anyio
    async def test_cancel_mid_stream_aborts_and_cancels_remote_job(self, tmp_path):
        from services.training_backends.base import TrainingCanceledError

        settings = _settings()
        context = _context(tmp_path)
        archive = tmp_path / "snapshot.zip"
        archive.write_bytes(b"abcdefghij")
        controller = _Controller(states=[])

        # False for the pre-attempt guard, then True once streaming has started so
        # the abort happens from inside the upload body generator.
        calls = {"n": 0}

        def _stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        context.should_stop = _stop

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}._UPLOAD_CHUNK_SIZE", 4),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingCanceledError):
                await backend.upload_snapshot_http(context, controller.remote_job_id, archive)

        assert controller.cancelled is True

    @pytest.mark.anyio
    async def test_shutdown_suspend_leaves_remote_job_reattachable(self, tmp_path):
        from services.training_backends.base import TrainingSuspendedError

        settings = _settings()
        context = _context(tmp_path, should_stop=True)
        # A shutdown-driven stop must not cancel the remote job; it stays awaiting
        # its dataset so a restart can resume the upload.
        context.should_suspend = lambda: True
        archive = tmp_path / "snapshot.zip"
        archive.write_bytes(b"abcdefghij")
        controller = _Controller(states=[])

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingSuspendedError):
                await backend.upload_snapshot_http(context, controller.remote_job_id, archive)

        assert controller.cancelled is False


class TestModelDownloadCancellation:
    """Cancelling / suspending during the model artifact download reacts promptly.

    Regression guard: the download loop must observe ``should_stop`` instead of
    streaming the whole (potentially multi-GB) artifact before the caller finally
    notices the cancel.
    """

    @pytest.mark.anyio
    async def test_cancel_before_download_cancels_remote_job(self, tmp_path):
        from services.training_backends.base import TrainingCanceledError

        settings = _settings()
        context = _context(tmp_path, should_stop=True)
        controller = _Controller(states=[])
        controller.artifact_chunks = [b"a" * 500, b"b" * 500]
        controller.artifact_headers = {"content-length": "1000"}
        stream_timeout = httpx.Timeout(5.0)

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingCanceledError):
                await backend._stream_archive(context, controller.remote_job_id, tmp_path / "model.zip", stream_timeout)

        # The remote job was canceled rather than the artifact fully downloaded.
        assert controller.cancelled is True
        assert controller.artifact_range_headers == []

    @pytest.mark.anyio
    async def test_cancel_mid_stream_aborts_and_cancels_remote_job(self, tmp_path):
        from services.training_backends.base import TrainingCanceledError

        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[])
        controller.artifact_chunks = [b"a" * 500, b"b" * 500]
        controller.artifact_headers = {"content-length": "1000"}
        stream_timeout = httpx.Timeout(5.0)

        # False for the pre-attempt guard, then True once streaming has started so
        # the abort happens from inside the artifact chunk loop.
        calls = {"n": 0}

        def _stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        context.should_stop = _stop

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingCanceledError):
                await backend._stream_archive(context, controller.remote_job_id, tmp_path / "model.zip", stream_timeout)

        assert controller.cancelled is True

    @pytest.mark.anyio
    async def test_shutdown_suspend_leaves_remote_job_reattachable(self, tmp_path):
        from services.training_backends.base import TrainingSuspendedError

        settings = _settings()
        context = _context(tmp_path, should_stop=True)
        # A shutdown-driven stop must not cancel the remote job; the artifact stays
        # available so a restart can resume the download.
        context.should_suspend = lambda: True
        controller = _Controller(states=[])
        controller.artifact_chunks = [b"a" * 500, b"b" * 500]
        controller.artifact_headers = {"content-length": "1000"}
        stream_timeout = httpx.Timeout(5.0)

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
        ):
            backend = _backend(settings)
            with pytest.raises(TrainingSuspendedError):
                await backend._stream_archive(context, controller.remote_job_id, tmp_path / "model.zip", stream_timeout)

        assert controller.cancelled is False


class TestSnapshotUploadHeartbeat:
    """The HTTP upload loop emits throttled byte heartbeats for large snapshots."""

    @pytest.mark.anyio
    async def test_http_upload_emits_transfer_heartbeat_logs(self, tmp_path):
        settings = _settings()
        context = _context(tmp_path)
        # Give the archive real bytes so the streaming read loop runs and updates the heartbeat.
        (tmp_path / "snap" / "info.json").write_text("x" * 4096)
        controller = _Controller(states=[{"status": "completed", "progress": 100, "message": "Done"}])

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=MagicMock()),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
            # A negative interval forces a heartbeat on the first streamed chunk.
            patch(f"{TRANSFER}._TRANSFER_LOG_INTERVAL_S", -1.0),
            patch(f"{TRANSFER}.logger") as mock_logger,
        ):
            backend = _backend(settings)
            await backend.train(context)

        templates = [call for call in mock_logger.info.call_args_list if call.args and "progress:" in call.args[0]]
        assert any(call.args[1] == "Dataset upload" for call in templates)


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
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
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
        safe_zip = MagicMock()

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
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

    @pytest.mark.anyio
    async def test_download_emits_transfer_heartbeat_logs(self, tmp_path):
        """Long transfers log throttled heartbeats; a zero interval forces one per chunk."""
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[{"status": "completed", "progress": 100, "message": "Done"}])
        controller.artifact_chunks = [b"a" * 500, b"b" * 500]
        controller.artifact_headers = {"content-length": "1000"}

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=MagicMock()),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
            # A negative interval makes every update cross the throttle threshold.
            patch(f"{TRANSFER}._TRANSFER_LOG_INTERVAL_S", -1.0),
            patch(f"{TRANSFER}.logger") as mock_logger,
        ):
            backend = _backend(settings)
            await backend.train(context)

        templates = [call for call in mock_logger.info.call_args_list if call.args and "progress:" in call.args[0]]
        assert any(call.args[1] == "Model download" for call in templates)

    @pytest.mark.anyio
    async def test_download_resumes_with_range_after_short_response(self, tmp_path):
        """The second artifact request starts exactly after the persisted partial file."""
        settings = _settings()
        context = _context(tmp_path)
        controller = _Controller(states=[{"status": "completed", "progress": 100, "message": "Done"}])
        controller.artifact_batches = [
            ([b"part"], {"content-length": "10"}),
            ([b"ial-data"], {"content-range": "bytes 4-11/12", "content-length": "8"}),
        ]

        with (
            patch(f"{REMOTE}.get_settings", return_value=settings),
            patch(f"{REMOTE}.httpx.AsyncClient", lambda **kw: _FakeClient(controller, **kw)),
            patch(f"{REMOTE}.SafeZipArchive", return_value=MagicMock()),
            patch(f"{REMOTE}._EVENT_WAIT_TIMEOUT_S", 0.01),
        ):
            backend = _backend(settings)
            await backend.train(context)

        assert controller.artifact_range_headers == [None, "bytes=4-"]


class TestByteFormatting:
    """Human-readable byte/throughput helpers used in transfer log lines."""

    @pytest.mark.parametrize(
        ("num_bytes", "expected"),
        [
            (0, "0 B"),
            (512, "512 B"),
            (1024, "1.0 KiB"),
            (1536, "1.5 KiB"),
            (1024**2, "1.0 MiB"),
            (1024**3, "1.0 GiB"),
            (50 * 1024**3, "50.0 GiB"),
            (1024**4, "1.0 TiB"),
            # Beyond TiB the largest unit is reused rather than overflowing the table.
            (1024**5, "1024.0 TiB"),
        ],
    )
    def testformat_bytes(self, num_bytes, expected):
        assert format_bytes(num_bytes) == expected

    def testformat_throughput_reports_rate_per_second(self):
        assert format_throughput(1024, 1.0) == "1.0 KiB/s"
        assert format_throughput(50 * 1024**2, 2.0) == "25.0 MiB/s"

    def testformat_throughput_guards_zero_elapsed(self):
        # A zero/negative window has no meaningful rate and must not divide by zero.
        assert format_throughput(1000, 0) == "n/a"
        assert format_throughput(1000, -1.0) == "n/a"


class TestTransferProgressLogger:
    """Throttled heartbeats reassure operators during multi-GB transfers."""

    def test_suppresses_heartbeats_within_the_interval(self):
        # init at t=0; both updates fall inside the 15s window, so nothing is logged.
        clock = iter([0.0, 5.0, 10.0])
        with (
            patch(f"{TRANSFER}.time.monotonic", side_effect=lambda: next(clock)),
            patch(f"{TRANSFER}.logger") as mock_logger,
        ):
            heartbeat = TransferProgressLogger("Model download", 1000)
            heartbeat.update(200)
            heartbeat.update(400)

        mock_logger.info.assert_not_called()

    def test_logs_percent_and_eta_when_total_known(self):
        # init at t=0; first update inside the window is quiet, second crosses 15s and logs.
        clock = iter([0.0, 5.0, 16.0])
        with (
            patch(f"{TRANSFER}.time.monotonic", side_effect=lambda: next(clock)),
            patch(f"{TRANSFER}.logger") as mock_logger,
        ):
            heartbeat = TransferProgressLogger("Model download", 1000)
            heartbeat.update(200)
            heartbeat.update(500)

        mock_logger.info.assert_called_once()
        args = mock_logger.info.call_args.args
        # verb, transferred, total, percent, rate, eta are forwarded to the template.
        assert args[1] == "Model download"
        assert args[2] == "500 B"
        assert args[3] == "1000 B"
        assert args[4] == 50  # round(500 / 1000 * 100)
        assert "ETA" in args[0]

    def test_logs_transferred_bytes_when_total_unknown(self):
        # A chunked transfer without Content-Length still emits a heartbeat, just no percent/ETA.
        clock = iter([0.0, 20.0])
        with (
            patch(f"{TRANSFER}.time.monotonic", side_effect=lambda: next(clock)),
            patch(f"{TRANSFER}.logger") as mock_logger,
        ):
            heartbeat = TransferProgressLogger("Dataset upload", None)
            heartbeat.update(2048)

        mock_logger.info.assert_called_once()
        args = mock_logger.info.call_args.args
        assert "transferred" in args[0]
        assert "ETA" not in args[0]
        assert args[1] == "Dataset upload"
        assert args[2] == "2.0 KiB"

    def test_resets_window_after_each_heartbeat(self):
        # init t=0; log at t=16 (>=15s), stay quiet at t=20 (<15s since last log), log again at t=32.
        clock = iter([0.0, 16.0, 20.0, 32.0])
        with (
            patch(f"{TRANSFER}.time.monotonic", side_effect=lambda: next(clock)),
            patch(f"{TRANSFER}.logger") as mock_logger,
        ):
            heartbeat = TransferProgressLogger("Model download", 10_000)
            heartbeat.update(1000)
            heartbeat.update(2000)
            heartbeat.update(3000)

        assert mock_logger.info.call_count == 2

    def test_treats_zero_total_as_unknown(self):
        # A zero total must not divide by zero; it falls back to the bytes-transferred form.
        clock = iter([0.0, 20.0])
        with (
            patch(f"{TRANSFER}.time.monotonic", side_effect=lambda: next(clock)),
            patch(f"{TRANSFER}.logger") as mock_logger,
        ):
            heartbeat = TransferProgressLogger("Model download", 0)
            heartbeat.update(500)

        mock_logger.info.assert_called_once()
        assert "transferred" in mock_logger.info.call_args.args[0]
