# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Remote training backend.

Offloads training to a trainer service. The dataset snapshot is transferred via
an ephemeral private HuggingFace dataset repo (pushed here, pulled there). The
trained model is returned over HTTP and extracted into the model directory.

This module avoids importing torch/`physicalai` so it stays usable in a
recording-only install.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger
from pydantic import ValidationError

from schemas.hardware import DeviceInfo
from services.archive_safety import SafeZipArchive
from services.training_backends._log_format import render_progress_log
from services.training_backends.base import TrainingCanceledError
from settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from services.training_backends.base import TrainingContext

# Only these patterns are pulled by the trainer; mirrors snapshot_download allowlists.
_SNAPSHOT_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.md", "*.parquet", "*.mp4", "*.png", "*.jpg"]
_EVENT_WAIT_TIMEOUT_S = 3.0
_RECONNECT_BACKOFF_S = 2.0
_MAX_STREAM_RECONNECTS = 5
_TERMINAL_STATES = {"completed", "failed", "canceled"}

_REMOTE_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

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


class RemoteTrainingBackend:
    """Submit training to a trainer service and ingest the returned model."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.trainer_url:
            raise RemoteTrainingError("Remote training requires TRAINER_URL")
        self._base_url = settings.trainer_url.rstrip("/")
        self._namespace = settings.trainer_hf_namespace
        self._timeout = settings.trainer_request_timeout_s
        # Token from env only; never logged.
        self._hf_token = os.environ.get("HF_TOKEN")
        # Resolved once by _resolve_trust_env(): True honors proxy env vars,
        # False bypasses them. None means "not yet probed".
        self._trust_env: bool | None = None
        self._trust_env_lock = asyncio.Lock()

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
        """Push snapshot, submit job, mirror progress, and ingest the model."""
        repo_id: str | None = None
        try:
            # Sub-step 1: push the snapshot to an ephemeral private dataset repo (0-10%).
            context.progress(0, message="Uploading dataset snapshot")
            repo_id, revision = await self._push_snapshot(context)
            context.progress(SNAPSHOT_UPLOAD_PROGRESS, message="Snapshot uploaded")

            # Sub-step 2: submit and wait for the remote job (10-95%).
            remote_job_id = await self._submit_job(context, repo_id=repo_id, revision=revision)
            await self._wait_for_completion(context, remote_job_id)

            # Sub-step 3: download and extract the trained model (95-100%).
            context.progress(TRAINING_PROGRESS_END, message="Downloading trained model")
            await self._download_and_extract(context, remote_job_id)
            context.progress(100, message="Model downloaded")
        finally:
            if repo_id is not None:
                await self._delete_repo(repo_id)

    async def _push_snapshot(self, context: TrainingContext) -> tuple[str, str]:
        """Create an ephemeral private dataset repo and upload the snapshot.

        Real byte progress is mirrored into the job's reserved progress window by
        :meth:`_mirror_upload_progress`. Returns the repo id and the commit SHA.
        """
        from huggingface_hub import HfApi

        api = HfApi(token=self._hf_token)
        repo_name = f"pais-snapshot-{uuid.uuid4().hex[:12]}"
        requested_repo_id = f"{self._namespace}/{repo_name}" if self._namespace else repo_name

        def _upload() -> tuple[str, str]:
            repo_url = api.create_repo(repo_id=requested_repo_id, repo_type="dataset", private=True)
            resolved_repo_id = repo_url.repo_id
            with self._mirror_upload_progress(context):
                commit = api.upload_folder(
                    repo_id=resolved_repo_id,
                    repo_type="dataset",
                    folder_path=str(Path(context.snapshot.path)),
                    allow_patterns=_SNAPSHOT_ALLOW_PATTERNS,
                )
            # upload_folder returns a CommitInfo; oid is the concrete commit SHA.
            if not commit.oid:
                raise RemoteTrainingError("Snapshot upload did not return a commit SHA")
            return resolved_repo_id, str(commit.oid)

        repo_id, revision = await asyncio.to_thread(_upload)
        logger.info("Snapshot pushed to ephemeral dataset repo (revision pinned)")
        return repo_id, revision

    @contextlib.contextmanager
    def _mirror_upload_progress(self, context: TrainingContext) -> Iterator[None]:
        """Mirror huggingface_hub's internal upload bytes into the 0-10% window.

        huggingface_hub renders upload progress with its own tqdm bars. The
        aggregate "Processing Files" bar tracks total bytes processed, so we
        subclass it to forward byte progress to the job without slowing the
        upload. Best-effort: if huggingface_hub reshapes its progress internals,
        the upload still runs and progress simply stays coarse (0 then 10).
        """
        try:
            from huggingface_hub.utils import _xet_progress_reporting as xet_mod
        except ImportError:
            yield
            return

        base_tqdm = getattr(xet_mod, "tqdm", None)
        if base_tqdm is None:
            yield
            return

        report = context.progress
        to_percent = self._upload_progress
        last_percent = -1

        class _ProgressTqdm(base_tqdm):  # type: ignore[valid-type, misc]
            """tqdm that forwards the aggregate processing bar's bytes to the job."""

            def update(self, n: float = 1) -> bool | None:
                result = super().update(n)
                nonlocal last_percent
                desc = self.desc or ""
                if "Processing Files" in desc and self.total:
                    percent = to_percent(int(self.n), int(self.total))
                    if percent != last_percent:
                        last_percent = percent
                        report(percent, message="Uploading dataset snapshot")
                return result

        setattr(xet_mod, "tqdm", _ProgressTqdm)
        try:
            yield
        finally:
            setattr(xet_mod, "tqdm", base_tqdm)

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

    async def _submit_job(self, context: TrainingContext, *, repo_id: str, revision: str) -> str:
        """Submit the training job and return the remote job id."""
        body = {
            "payload": context.payload.model_dump(mode="json"),
            "repo_id": repo_id,
            "revision": revision,
            "policy": context.model.policy,
        }
        async with await self._client() as client:
            response = await client.post(f"{self._base_url}/jobs", json=body)
            response.raise_for_status()
            data = response.json()

        remote_job_id = data.get("remote_job_id")
        if not isinstance(remote_job_id, str) or not _REMOTE_JOB_ID_PATTERN.fullmatch(remote_job_id):
            raise RemoteTrainingError("Trainer did not return a valid remote_job_id")
        logger.info("Remote training job submitted")
        return remote_job_id

    async def _wait_for_completion(self, context: TrainingContext, remote_job_id: str) -> None:
        """Consume the trainer's SSE event stream, mirroring progress into the local job.

        The trainer streams a ``state`` event on every change at
        ``/jobs/{id}/events`` and closes the stream on a terminal state. A dropped
        stream before a terminal state (idle timeout, network blip) is transient:
        reconnect, and the trainer re-emits the current state on the fresh
        connection. A run of reconnects that never delivers an event aborts the
        job rather than looping forever.
        """
        stalled_reconnects = 0
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
                await self._cancel(remote_job_id)
                raise TrainingCanceledError("Training canceled")

            stalled_reconnects = 0 if received_event else stalled_reconnects + 1
            if stalled_reconnects > _MAX_STREAM_RECONNECTS:
                raise RemoteTrainingError("Trainer event stream closed repeatedly without a terminal state")
            await asyncio.sleep(_RECONNECT_BACKOFF_S)

    async def _consume_event_stream(self, context: TrainingContext, remote_job_id: str) -> tuple[bool, bool]:
        """Open one SSE connection and mirror state until it closes.

        Returns ``(completed, received_event)``. ``completed`` is True only when
        the job reached the ``completed`` terminal state. Raises
        ``TrainingCanceledError`` on local/remote cancellation and
        ``RemoteTrainingError`` on remote failure.
        """
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        received_event = False
        url = f"{self._base_url}/jobs/{remote_job_id}/events"
        client = await self._client()
        async with (
            client,
            client.stream("GET", url, headers={"Accept": "text/event-stream"}) as response,
        ):
            response.raise_for_status()
            reader = asyncio.create_task(self._read_sse_events(response, queue))
            try:
                while True:
                    if context.should_stop():
                        await self._cancel(remote_job_id)
                        raise TrainingCanceledError("Training canceled")

                    try:
                        kind, payload = await asyncio.wait_for(queue.get(), timeout=_EVENT_WAIT_TIMEOUT_S)
                    except TimeoutError:
                        # No event yet; loop back to re-check cancellation.
                        continue

                    if kind == "end":
                        return False, received_event
                    if kind == "error":
                        # Transient read error: surface to the reconnect loop.
                        logger.warning("Trainer event stream read error: {}", payload)
                        return False, received_event

                    received_event = True
                    if self._apply_state(context, self._parse_state(payload)):
                        return True, received_event
            finally:
                reader.cancel()
                with contextlib.suppress(BaseException):
                    await reader

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
            if line is not None:
                logger.info(line)

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
        except (TypeError, ValueError):
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

    async def _download_and_extract(self, context: TrainingContext, remote_job_id: str) -> None:
        """Stream the model archive and extract it into the model directory."""
        settings = get_settings()
        tmp_archive = Path(tempfile.gettempdir()) / f"remote-model-{uuid.uuid4().hex}.zip"
        stream_timeout = httpx.Timeout(self._timeout, read=settings.trainer_download_read_timeout_s)
        try:
            received = await self._stream_archive(remote_job_id, tmp_archive, stream_timeout)
            logger.info("Downloaded model artifact ({} bytes)", received)

            await asyncio.to_thread(
                self._extract_archive,
                tmp_archive,
                context.output_dir,
                settings.data_import_max_uncompressed_bytes,
                settings.data_import_min_free_bytes,
            )
        finally:
            tmp_archive.unlink(missing_ok=True)

    async def _stream_archive(self, remote_job_id: str, tmp_archive: Path, stream_timeout: httpx.Timeout) -> int:
        """Stream the artifact to ``tmp_archive``; return the byte count.

        Verifies the transfer against ``Content-Length`` so a connection dropped
        mid-stream surfaces as an error instead of a silently truncated archive.
        """
        client = await self._client(stream_timeout)
        async with (
            client,
            client.stream("GET", f"{self._base_url}/jobs/{remote_job_id}/artifact") as response,
        ):
            response.raise_for_status()
            expected = response.headers.get("content-length")
            expected_bytes = int(expected) if expected is not None and expected.isdigit() else None

            received = 0
            with tmp_archive.open("wb") as fobj:
                async for chunk in response.aiter_bytes():
                    fobj.write(chunk)
                    received += len(chunk)

        if expected_bytes is not None and received != expected_bytes:
            raise RemoteTrainingError(f"Artifact download truncated: received {received} of {expected_bytes} bytes")
        return received

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

    async def _cancel(self, remote_job_id: str) -> None:
        """Request remote cancellation; best effort."""
        try:
            async with await self._client() as client:
                await client.post(f"{self._base_url}/jobs/{remote_job_id}/cancel")
        except httpx.HTTPError as exc:
            logger.warning("Failed to cancel remote job: {}", exc)

    async def _delete_repo(self, repo_id: str) -> None:
        """Delete the ephemeral snapshot repo; best effort."""
        from huggingface_hub import HfApi

        def _delete() -> None:
            HfApi(token=self._hf_token).delete_repo(repo_id=repo_id, repo_type="dataset", missing_ok=True)

        try:
            await asyncio.to_thread(_delete)
            logger.info("Deleted ephemeral snapshot repo")
        except Exception as exc:
            logger.warning("Failed to delete ephemeral snapshot repo: {}", exc)

    @staticmethod
    def _coerce_progress(value: object) -> int:
        if isinstance(value, int | float):
            return max(0, min(100, int(value)))
        return 0

    @staticmethod
    def _to_local_progress(remote_progress: int) -> int:
        """Map the trainer's raw 0-100 progress into the local training window.

        Snapshot upload reserves 0..SNAPSHOT_UPLOAD_PROGRESS and model download
        reserves TRAINING_PROGRESS_END..100, so training occupies the span
        between them. Applied once here; the trainer reports raw.
        """
        span = TRAINING_PROGRESS_END - SNAPSHOT_UPLOAD_PROGRESS
        return min(TRAINING_PROGRESS_END, SNAPSHOT_UPLOAD_PROGRESS + round(remote_progress * span / 100))
