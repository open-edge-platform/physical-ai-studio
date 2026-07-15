# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Remote training backend.

Offloads training to a trainer service. The dataset snapshot is streamed as a
ZIP straight to the trainer over HTTP; the trained model is returned over HTTP
and extracted into the model directory.

This module avoids importing torch/`physicalai` so it stays usable in a
recording-only install.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
from loguru import logger
from physicalai.data.archive_safety import SafeZipArchive
from pydantic import ValidationError

from schemas.hardware import DeviceInfo
from services.dataset_download_service import DatasetDownloadService
from services.staged_archive import cleanup_staged_archive
from services.training_backends._log_format import render_progress_log
from services.training_backends._transfer_progress import TransferProgressLogger, format_bytes, format_throughput
from services.training_backends.base import TrainingCanceledError, TrainingSuspendedError
from settings import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from services.training_backends._training_methods import TrainingMethod
    from services.training_backends.base import TrainingContext

_EVENT_WAIT_TIMEOUT_S = 3.0
_RECONNECT_BACKOFF_S = 2.0
_TERMINAL_STATES = {"completed", "failed", "canceled"}
_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
_TRANSFER_RETRY_LIMIT = 3

# Cap the trainer-supplied telemetry blob.
_MAX_EXTRA_INFO_BYTES = 16 * 1024


# Job progress (0-100) is partitioned across three sub-steps. These boundaries
# separate them and are the single place to retune how much of the bar each
# phase owns:
#   - snapshot upload: 0 .. SNAPSHOT_UPLOAD_PROGRESS
#   - remote training: SNAPSHOT_UPLOAD_PROGRESS .. TRAINING_PROGRESS_END
#   - model download:  TRAINING_PROGRESS_END .. 100
SNAPSHOT_UPLOAD_PROGRESS = 10
TRAINING_PROGRESS_END = 95


class RemoteTrainingError(RuntimeError):
    """Raised when the trainer service reports a failure."""


class _UploadStopRequested(Exception):
    """Internal signal that a stop was requested while streaming the dataset."""


class RemoteTrainingBackend:
    """Submit training to a trainer service and ingest the returned model."""

    _last_progress_log: str | None = None

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.trainer_url:
            raise RemoteTrainingError("Remote training requires TRAINER_URL")
        self._base_url = settings.trainer_url.rstrip("/")
        self._timeout = settings.trainer_request_timeout_s
        # Resolved once by _resolve_trust_env(): True honors proxy env vars,
        # False bypasses them. None means "not yet probed".
        self._trust_env: bool | None = None
        self._trust_env_lock = asyncio.Lock()
        # Suppress duplicate consecutive progress lines (e.g. the trainer
        # re-emitting the final training state while it optimizes/exports).
        self._last_progress_log: str | None = None

    async def _resolve_trust_env(self) -> bool:
        """Decide once whether proxy env vars should be honored for trainer calls.

        The trainer is an internal endpoint. An outbound proxy usually rejects
        it (403), so the safe default is to bypass proxies. Some deployments do
        route the trainer through the proxy, so probe /health once with proxies
        enabled and cache the verdict for all later clients.
        """
        cached = self._trust_env
        if cached is not None:
            return cached
        async with self._trust_env_lock:
            # Re-check: another coroutine may have resolved it while we waited.
            cached = self._trust_env
            if cached is None:
                cached = await self._probe_proxy()
                self._trust_env = cached
        return cached

    async def _probe_proxy(self) -> bool:
        """Return True if /health is reachable with proxy env vars honored."""
        try:
            # AsyncClient honors HTTP_PROXY/HTTPS_PROXY by default (trust_env).
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self._base_url}/health")
                response.raise_for_status()
        except httpx.HTTPError:
            logger.debug("Trainer not reachable via proxy; bypassing proxy for trainer calls")
            return False
        logger.debug("Trainer reachable via proxy; honoring proxy settings for trainer calls")
        return True

    async def _client(self, client_timeout: httpx.Timeout | float | None = None) -> httpx.AsyncClient:
        """Build a client for direct trainer calls."""
        trust_env = await self._resolve_trust_env()
        return httpx.AsyncClient(
            timeout=client_timeout if client_timeout is not None else self._timeout,
            trust_env=trust_env,
        )

    async def get_training_devices(self) -> list[DeviceInfo]:
        """Fetch the compute devices available on the trainer service.

        Lets the studio surface the remote server's real hardware (GPU/XPU) instead of the studio host's local device.
        Raises RemoteTrainingError on any transport or parsing failure so callers can fall back.
        """
        try:
            async with await self._client() as client:
                response = await client.get(f"{self._base_url}/devices")
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise RemoteTrainingError(f"Failed to query trainer devices: {exc}") from exc

        if not isinstance(data, list):
            raise RemoteTrainingError("Trainer returned an invalid devices payload")

        try:
            return [DeviceInfo.model_validate(item) for item in data]
        except ValidationError as exc:
            raise RemoteTrainingError(f"Trainer returned malformed device info: {exc}") from exc

    async def train(self, context: TrainingContext) -> None:
        """Deliver the snapshot, submit the job, mirror progress, ingest the model.

        When ``context.remote_job_id`` is set the studio is resuming after a
        restart: the snapshot is already on the trainer and the job is running
        (or finished), so we skip straight to reattaching and ingesting the
        model rather than submitting a new job.
        """
        if context.remote_job_id:
            logger.info("Reattaching to in-flight remote training job")
            await self._resume_pending_http_upload(context, context.remote_job_id)
            await self.await_and_ingest(context, context.remote_job_id)
            return
        await self._training_method().train(context)

    def _training_method(self) -> TrainingMethod:
        """Pick the dataset-transfer strategy for this backend."""
        from services.training_backends._training_methods import HttpTrainingMethod

        return HttpTrainingMethod(self)

    async def _resume_pending_http_upload(self, context: TrainingContext, remote_job_id: uuid.UUID) -> None:
        """Resume an HTTP dataset upload when the trainer still awaits its ZIP."""
        if context.snapshot is None:
            return
        try:
            async with await self._client() as client:
                response = await client.get(f"{self._base_url}/jobs/{remote_job_id}")
                response.raise_for_status()
                state = response.json()
        except httpx.HTTPError as exc:
            raise RemoteTrainingError("Unable to inspect remote training job for upload resume") from exc
        if not isinstance(state, dict) or state.get("status") != "awaiting_dataset":
            return
        archive_path = await asyncio.to_thread(
            DatasetDownloadService().create_dataset_archive, Path(context.snapshot.path)
        )
        try:
            await self.upload_snapshot_http(context, remote_job_id, archive_path)
            context.progress(SNAPSHOT_UPLOAD_PROGRESS, message="Dataset uploaded, starting training")
        finally:
            cleanup_staged_archive(archive_path)

    async def await_and_ingest(self, context: TrainingContext, remote_job_id: uuid.UUID) -> None:
        """Wait for the remote job, then download and extract its model."""
        await self._wait_for_completion(context, remote_job_id)
        context.progress(TRAINING_PROGRESS_END, message="Downloading trained model")
        await self._download_and_extract(context, remote_job_id)
        context.progress(100, message="Model downloaded")

    async def upload_snapshot_http(
        self, context: TrainingContext, remote_job_id: uuid.UUID, archive_path: Path
    ) -> None:
        """Stream the snapshot ZIP to the trainer, mirroring bytes into the 0-10% window."""
        total = archive_path.stat().st_size
        report = context.progress
        to_percent = self._upload_progress
        logger.info(
            "Uploading dataset snapshot to trainer: {} ({}) -> job {}",
            archive_path.name,
            format_bytes(total),
            remote_job_id,
        )
        started = time.monotonic()

        async def _read_chunks(offset: int) -> AsyncIterator[bytes]:
            sent = offset
            last_percent = -1
            heartbeat = TransferProgressLogger("Dataset upload", total)
            with archive_path.open("rb") as fobj:
                fobj.seek(offset)
                while chunk := await asyncio.to_thread(fobj.read, _UPLOAD_CHUNK_SIZE):
                    if context.should_stop():
                        raise _UploadStopRequested
                    sent += len(chunk)
                    percent = to_percent(sent, total)
                    if percent != last_percent:
                        last_percent = percent
                        report(percent, message="Uploading dataset snapshot")
                    heartbeat.update(sent)
                    yield chunk

        settings = get_settings()
        # A large upload must not trip the short request timeout; guard stalls with a per-write gap.
        stream_timeout = httpx.Timeout(self._timeout, write=settings.trainer_download_read_timeout_s)
        offset = await self._get_upload_offset(remote_job_id, total, stream_timeout)
        for attempt in range(_TRANSFER_RETRY_LIMIT + 1):
            if offset == total:
                break
            if context.should_stop():
                await self._handle_stop_request(context, remote_job_id)
            headers = {
                "Content-Type": "application/zip",
                "Content-Range": f"bytes {offset}-{total - 1}/{total}",
            }
            try:
                client = await self._client(stream_timeout)
                async with client:
                    response = await client.put(
                        f"{self._base_url}/jobs/{remote_job_id}/dataset",
                        content=_read_chunks(offset),
                        headers=headers,
                    )
                    response.raise_for_status()
                    reported_offset = response.headers.get("upload-offset")
                    offset = int(reported_offset) if reported_offset is not None else total
            except _UploadStopRequested:
                await self._handle_stop_request(context, remote_job_id)
            except (httpx.HTTPError, ValueError) as exc:
                if attempt == _TRANSFER_RETRY_LIMIT:
                    raise RemoteTrainingError("Dataset upload could not be resumed") from exc
                logger.warning("Dataset upload interrupted; resuming from trainer offset")
                offset = await self._get_upload_offset(remote_job_id, total, stream_timeout)
        if offset != total:
            raise RemoteTrainingError(f"Trainer accepted an incomplete dataset upload ({offset} of {total} bytes)")
        elapsed = time.monotonic() - started
        logger.info(
            "Snapshot uploaded to trainer: {} in {:.1f}s ({})",
            format_bytes(total),
            elapsed,
            format_throughput(total, elapsed),
        )

    async def _get_upload_offset(self, remote_job_id: uuid.UUID, total: int, client_timeout: httpx.Timeout) -> int:
        """Read and validate the trainer's staged dataset-upload offset."""
        try:
            async with await self._client(client_timeout) as client:
                response = await client.head(f"{self._base_url}/jobs/{remote_job_id}/dataset")
                response.raise_for_status()
                offset = int(response.headers.get("upload-offset", "0"))
        except (httpx.HTTPError, ValueError) as exc:
            raise RemoteTrainingError("Unable to determine dataset upload resume offset") from exc
        if not 0 <= offset <= total:
            raise RemoteTrainingError(f"Trainer returned invalid dataset upload offset {offset}")
        return offset

    @staticmethod
    def _upload_progress(uploaded_bytes: int, total_bytes: int) -> int:
        """Map uploaded bytes into the reserved snapshot-upload window.

        Capped one below SNAPSHOT_UPLOAD_PROGRESS so the explicit "Snapshot
        uploaded" step owns that mark.
        """
        if total_bytes <= 0:
            return 0
        return min(
            SNAPSHOT_UPLOAD_PROGRESS - 1,
            round(uploaded_bytes / total_bytes * SNAPSHOT_UPLOAD_PROGRESS),
        )

    async def submit_job(self, context: TrainingContext) -> uuid.UUID:
        """Submit the training job and return the remote job id."""
        body: dict[str, Any] = {
            "payload": context.payload.model_dump(mode="json"),
            "policy": context.model.policy,
            "dataset_transfer": "http",
        }
        async with await self._client() as client:
            response = await client.post(f"{self._base_url}/jobs", json=body)
            response.raise_for_status()
            data = response.json()

        remote_job_id = data.get("remote_job_id")
        if not isinstance(remote_job_id, str):
            raise RemoteTrainingError("Trainer did not return a valid remote_job_id")
        try:
            remote_job_uuid = uuid.UUID(remote_job_id)
        except ValueError as exc:
            raise RemoteTrainingError("Trainer did not return a valid remote_job_id") from exc
        logger.info("Remote training job submitted")
        return remote_job_uuid

    async def _wait_for_completion(self, context: TrainingContext, remote_job_id: uuid.UUID) -> None:
        """Consume the trainer's SSE event stream, mirroring progress into the local job.

        The job keeps running on the trainer regardless of this connection, so a
        dropped stream is recoverable: we reconnect with exponential backoff, and
        also poll the plain job endpoint as a fallback for middleboxes that break
        long-lived SSE (see :meth:`_poll_state`). We only abandon the job once the
        trainer has been continuously unreachable for ``trainer_stream_reconnect_max_s``
        seconds.
        """
        settings = get_settings()
        unreachable_budget_s = settings.trainer_stream_reconnect_max_s
        max_backoff_s = settings.trainer_stream_reconnect_backoff_max_s
        backoff_s = _RECONNECT_BACKOFF_S
        # Monotonic timestamp of the last successful contact with the trainer
        # (an event received, or a successful poll). Resets the outage budget.
        last_contact = time.monotonic()
        while True:
            try:
                completed, received_event = await self._consume_event_stream(context, remote_job_id)
            except httpx.HTTPError as exc:
                # Connection-level failure while opening or reading the stream.
                logger.warning("Trainer event stream connection failed, reconnecting: {}", exc)
                completed, received_event = False, False

            if completed:
                return

            if context.should_stop():
                await self._handle_stop_request(context, remote_job_id)

            # Stream dropped before a terminal state; fall back to a plain GET,
            # which also catches middleboxes that break long-lived SSE.
            reachable, completed = await self._poll_state(context, remote_job_id)
            if completed:
                return

            if received_event or reachable:
                last_contact = time.monotonic()
                backoff_s = _RECONNECT_BACKOFF_S
            elif time.monotonic() - last_contact > unreachable_budget_s:
                raise RemoteTrainingError(
                    f"Trainer unreachable for over {unreachable_budget_s:.0f}s; abandoning progress tracking"
                )

            await asyncio.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, max_backoff_s)

    async def _poll_state(self, context: TrainingContext, remote_job_id: uuid.UUID) -> tuple[bool, bool]:
        """Read job state via a plain GET, as a transport fallback for broken SSE.

        Returns ``(reachable, completed)``; ``reachable`` resets the outage budget
        even if the job isn't done yet. Propagates ``TrainingCanceledError`` /
        ``RemoteTrainingError`` via :meth:`_apply_state`.
        """
        try:
            async with await self._client() as client:
                response = await client.get(f"{self._base_url}/jobs/{remote_job_id}")
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError:
            return False, False
        if not isinstance(data, dict):
            return True, False
        return True, self._apply_state(context, data)

    async def _consume_event_stream(self, context: TrainingContext, remote_job_id: uuid.UUID) -> tuple[bool, bool]:
        """Open one SSE connection and mirror state until it closes.

        Returns ``(completed, received_event)``. ``completed`` is True only when
        the job reached the ``completed`` terminal state. Raises
        ``TrainingCanceledError`` on local/remote cancellation and
        ``RemoteTrainingError`` on remote failure.
        """
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        received_event = False
        result: tuple[bool, bool] | None = None
        url = f"{self._base_url}/jobs/{remote_job_id}/events"
        client = await self._client()
        async with (
            client,
            client.stream("GET", url, headers={"Accept": "text/event-stream"}) as response,
        ):
            response.raise_for_status()
            reader = asyncio.create_task(self._read_sse_events(response, queue))
            try:
                while result is None:
                    if context.should_stop():
                        await self._handle_stop_request(context, remote_job_id)

                    try:
                        kind, payload = await asyncio.wait_for(queue.get(), timeout=_EVENT_WAIT_TIMEOUT_S)
                    except TimeoutError:
                        # No event yet; loop back to re-check cancellation.
                        continue

                    if kind == "end":
                        result = False, received_event
                        continue
                    if kind == "error":
                        # Transient read error: surface to the reconnect loop.
                        logger.warning("Trainer event stream read error: {}", payload)
                        result = False, received_event
                        continue

                    received_event = True
                    if self._apply_state(context, self._parse_state(payload)):
                        result = True, received_event
            finally:
                reader.cancel()
                with contextlib.suppress(BaseException):
                    await reader
        if result is None:
            raise RemoteTrainingError("Trainer event stream ended unexpectedly")
        return cast("tuple[bool, bool]", result)

    @staticmethod
    async def _read_sse_events(response: httpx.Response, queue: asyncio.Queue[tuple[str, object]]) -> None:
        """Parse SSE frames from ``response`` and push them onto ``queue``.

        Emits ``("event", data)`` per complete frame, ``("end", None)`` when the
        stream closes, and ``("error", exc)`` on a read failure. Comment lines
        (keep-alive pings) and non-``state`` events are dropped.
        """
        event: str | None = None
        data_lines: list[str] = []
        try:
            async for line in response.aiter_lines():
                if line == "":
                    if data_lines and event == "state":
                        await queue.put(("event", "\n".join(data_lines)))
                    event, data_lines = None, []
                    continue
                if line.startswith(":"):
                    continue  # Comment / keep-alive ping.
                field, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if field == "event":
                    event = value
                elif field == "data":
                    data_lines.append(value)
            await queue.put(("end", None))
        except httpx.HTTPError as exc:
            await queue.put(("error", exc))

    def _apply_state(self, context: TrainingContext, state: dict[str, Any]) -> bool:
        """Mirror a job state into the local job; return True if completed.

        Raises ``TrainingCanceledError`` / ``RemoteTrainingError`` on terminal
        cancellation / failure.
        """
        status = state.get("status")
        remote_progress = self._coerce_progress(state.get("progress"))
        raw_extra = state.get("extra_info")
        extra_info = self._sanitize_extra_info(raw_extra)
        context.progress(
            self._to_local_progress(remote_progress),
            message=state.get("message"),
            extra_info=extra_info,
        )
        if extra_info is not None:
            line = render_progress_log(extra_info)
            if line is not None and line != self._last_progress_log:
                logger.info(line)
                self._last_progress_log = line

        if status in _TERMINAL_STATES:
            if status == "completed":
                return True
            if status == "canceled":
                raise TrainingCanceledError("Remote training canceled")
            raise RemoteTrainingError(f"Remote training {status}: {state.get('message')}")
        return False

    @staticmethod
    def _sanitize_extra_info(raw_extra: object) -> dict[str, Any] | None:
        """Return the trainer's telemetry dict, or None if absent or too large.

        extra_info is untrusted and persisted verbatim, so reject any blob whose
        serialized size exceeds ``_MAX_EXTRA_INFO_BYTES`` rather than storing it.
        """
        if not isinstance(raw_extra, dict):
            return None
        try:
            size = len(json.dumps(raw_extra).encode())
        except (TypeError, ValueError, RecursionError, OverflowError):
            logger.warning("Dropping non-serializable extra_info from trainer state")
            return None
        if size > _MAX_EXTRA_INFO_BYTES:
            logger.warning("Dropping oversized extra_info from trainer state ({} bytes)", size)
            return None
        return raw_extra

    @staticmethod
    def _parse_state(payload: object) -> dict[str, Any]:
        """Parse an SSE ``state`` data payload into a job-state dict."""
        if not isinstance(payload, str) or not payload:
            raise RemoteTrainingError("Trainer event stream sent an empty state payload")
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RemoteTrainingError("Trainer event stream sent malformed JSON") from exc
        if not isinstance(parsed, dict):
            raise RemoteTrainingError("Trainer returned a malformed job state")
        return parsed

    async def _download_and_extract(self, context: TrainingContext, remote_job_id: uuid.UUID) -> None:
        """Stream the model archive and extract it into the model directory."""
        settings = get_settings()
        tmp_archive = Path(tempfile.gettempdir()) / f"remote-model-{uuid.uuid4().hex}.zip"
        stream_timeout = httpx.Timeout(self._timeout, read=settings.trainer_download_read_timeout_s)
        logger.info("Downloading trained model artifact from trainer job {}", remote_job_id)
        started = time.monotonic()
        try:
            received = await self._stream_archive(context, remote_job_id, tmp_archive, stream_timeout)
            elapsed = time.monotonic() - started
            logger.info(
                "Downloaded model artifact: {} in {:.1f}s ({})",
                format_bytes(received),
                elapsed,
                format_throughput(received, elapsed),
            )

            logger.info("Extracting model artifact into {}", context.output_dir)
            await asyncio.to_thread(
                self._extract_archive,
                tmp_archive,
                context.output_dir,
                settings.data_import_max_uncompressed_bytes,
                settings.data_import_min_free_bytes,
            )
            logger.info("Model artifact extracted into {}", context.output_dir)
        finally:
            tmp_archive.unlink(missing_ok=True)

    async def _stream_archive(
        self,
        context: TrainingContext,
        remote_job_id: uuid.UUID,
        tmp_archive: Path,
        stream_timeout: httpx.Timeout,
    ) -> int:
        """Stream the artifact to ``tmp_archive`` and reject truncated downloads."""
        report = context.progress
        to_percent = self._download_progress
        received = tmp_archive.stat().st_size if tmp_archive.exists() else 0
        expected_bytes: int | None = None
        last_percent = -1
        heartbeat: TransferProgressLogger | None = None
        for attempt in range(_TRANSFER_RETRY_LIMIT + 1):
            if context.should_stop():
                await self._handle_stop_request(context, remote_job_id)
            headers = {"Range": f"bytes={received}-"} if received else None
            try:
                client = await self._client(stream_timeout)
                async with (
                    client,
                    client.stream(
                        "GET", f"{self._base_url}/jobs/{remote_job_id}/artifact", headers=headers
                    ) as response,
                ):
                    response.raise_for_status()
                    expected_bytes = self._artifact_total_bytes(response.headers, received)
                    if heartbeat is None:
                        heartbeat = TransferProgressLogger("Model download", expected_bytes)
                    with tmp_archive.open("ab") as fobj:
                        async for chunk in response.aiter_bytes(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                            if context.should_stop():
                                await self._handle_stop_request(context, remote_job_id)
                            fobj.write(chunk)
                            received += len(chunk)
                            if expected_bytes is not None:
                                percent = to_percent(received, expected_bytes)
                                if percent != last_percent:
                                    last_percent = percent
                                    report(percent, message="Downloading trained model")
                            heartbeat.update(received)
                if expected_bytes is None or received == expected_bytes:
                    return received
                raise RemoteTrainingError(f"Artifact download truncated: received {received} of {expected_bytes} bytes")
            except (httpx.HTTPError, RemoteTrainingError) as exc:
                if attempt == _TRANSFER_RETRY_LIMIT:
                    raise RemoteTrainingError(str(exc) or "Model download could not be resumed") from exc
                logger.warning("Model download interrupted; resuming from {} bytes", received)
        raise RemoteTrainingError("Model download could not be resumed")

    @staticmethod
    def _artifact_total_bytes(headers: Any, offset: int) -> int | None:
        """Return complete artifact length from a range or regular response."""
        content_range = headers.get("content-range")
        if isinstance(content_range, str) and "/" in content_range:
            total = content_range.rsplit("/", maxsplit=1)[1]
            if total.isdigit():
                return int(total)
        content_length = headers.get("content-length")
        if isinstance(content_length, str) and content_length.isdigit():
            return offset + int(content_length)
        return None

    @staticmethod
    def _download_progress(received_bytes: int, total_bytes: int) -> int:
        """Map artifact bytes into the download window, reserving 100 for extraction."""
        if total_bytes <= 0:
            return TRAINING_PROGRESS_END
        span = 100 - TRAINING_PROGRESS_END
        return min(99, TRAINING_PROGRESS_END + round(received_bytes / total_bytes * span))

    @staticmethod
    def _extract_archive(
        tmp_archive: Path,
        output_dir: Path,
        max_uncompressed_bytes: int,
        min_free_bytes: int,
    ) -> None:
        """Validate and extract the archive into ``output_dir`` (blocking)."""
        archive = SafeZipArchive(tmp_archive, max_uncompressed_bytes=max_uncompressed_bytes)
        archive.validate()
        output_dir.mkdir(parents=True, exist_ok=True)
        archive.extract_to(output_dir, min_free_bytes=min_free_bytes)

    async def _handle_stop_request(self, context: TrainingContext, remote_job_id: uuid.UUID) -> None:
        """Suspend on shutdown; cancel the remote job for a user stop request."""
        if context.should_suspend():
            raise TrainingSuspendedError("Studio shutting down; leaving remote training job running for reattach")
        await self._cancel(remote_job_id)
        raise TrainingCanceledError("Training canceled")

    async def _cancel(self, remote_job_id: uuid.UUID) -> None:
        """Request remote cancellation; best effort."""
        try:
            async with await self._client() as client:
                await client.post(f"{self._base_url}/jobs/{remote_job_id}/cancel")
        except httpx.HTTPError as exc:
            logger.warning("Failed to cancel remote job: {}", exc)

    @staticmethod
    def _coerce_progress(value: object) -> int:
        if isinstance(value, int | float):
            return max(0, min(100, int(value)))
        return 0

    @staticmethod
    def _to_local_progress(remote_progress: int) -> int:
        """Map raw trainer progress into the reserved local training window."""
        span = TRAINING_PROGRESS_END - SNAPSHOT_UPLOAD_PROGRESS
        return min(TRAINING_PROGRESS_END, SNAPSHOT_UPLOAD_PROGRESS + round(remote_progress * span / 100))
