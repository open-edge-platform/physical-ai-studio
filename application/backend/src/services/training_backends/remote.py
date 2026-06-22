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
import os
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from services.archive_safety import SafeZipArchive
from services.training_backends.base import TrainingCanceledError
from settings import get_settings

if TYPE_CHECKING:
    from services.training_backends.base import TrainingContext

# Only these patterns are pulled by the trainer; mirrors snapshot_download allowlists.
_SNAPSHOT_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.md", "*.parquet", "*.mp4", "*.png", "*.jpg"]
_POLL_INTERVAL_S = 3.0
_TERMINAL_STATES = {"completed", "failed", "canceled"}


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
            logger.info("Trainer not reachable via proxy; bypassing proxy for trainer calls")
            return False
        logger.info("Trainer reachable via proxy; honoring proxy settings for trainer calls")
        return True

    async def _client(self, client_timeout: httpx.Timeout | float | None = None) -> httpx.AsyncClient:
        """Build a client for direct trainer calls.

        Whether HTTP_PROXY/HTTPS_PROXY are honored is decided once by
        _resolve_trust_env(). The trainer is an explicitly configured internal
        endpoint; routing it through an outbound web proxy commonly makes the
        proxy reject the internal host (403), so proxies are bypassed unless the
        one-time probe confirms they work. Outbound proxies remain in effect for
        HuggingFace uploads, which use their own client.
        """
        trust_env = await self._resolve_trust_env()
        return httpx.AsyncClient(
            timeout=client_timeout if client_timeout is not None else self._timeout,
            trust_env=trust_env,
        )

    async def train(self, context: TrainingContext) -> None:
        """Push snapshot, submit job, mirror progress, and ingest the model."""
        repo_id: str | None = None
        try:
            # Sub-step 1: push the snapshot to an ephemeral private dataset repo (0-10%).
            context.progress(0, message="Uploading dataset snapshot")
            repo_id, revision = await self._push_snapshot(context)
            context.progress(10, message="Snapshot uploaded")

            # Sub-step 2: submit and wait for the remote job (10-95%).
            remote_job_id = await self._submit_job(context, repo_id=repo_id, revision=revision)
            await self._wait_for_completion(context, remote_job_id)

            # Sub-step 3: download and extract the trained model (95-100%).
            context.progress(95, message="Downloading trained model")
            await self._download_and_extract(context, remote_job_id)
            context.progress(99, message="Model downloaded")
        finally:
            if repo_id is not None:
                await self._delete_repo(repo_id)

    async def _push_snapshot(self, context: TrainingContext) -> tuple[str, str]:
        """Create an ephemeral private dataset repo and upload the snapshot.

        Returns the repo id and the concrete commit SHA to pin on the server.
        """
        from huggingface_hub import HfApi

        api = HfApi(token=self._hf_token)
        repo_name = f"pais-snapshot-{uuid.uuid4().hex[:12]}"
        requested_repo_id = f"{self._namespace}/{repo_name}" if self._namespace else repo_name

        def _upload() -> tuple[str, str]:
            repo_url = api.create_repo(repo_id=requested_repo_id, repo_type="dataset", private=True)
            resolved_repo_id = repo_url.repo_id
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
        if not isinstance(remote_job_id, str) or not remote_job_id:
            raise RemoteTrainingError("Trainer did not return a valid remote_job_id")
        logger.info("Remote training job submitted")
        return remote_job_id

    async def _wait_for_completion(self, context: TrainingContext, remote_job_id: str) -> None:
        """Poll the remote job, mirroring progress into the local job."""
        async with await self._client() as client:
            while True:
                if context.should_stop():
                    await self._cancel(client, remote_job_id)
                    raise TrainingCanceledError("Training canceled")

                state = await self._fetch_state(client, remote_job_id)
                status = state.get("status")
                remote_progress = self._coerce_progress(state.get("progress"))
                context.progress(
                    self._to_local_progress(remote_progress),
                    message=state.get("message"),
                    extra_info=state.get("extra_info") if isinstance(state.get("extra_info"), dict) else None,
                )

                if status in _TERMINAL_STATES:
                    if status == "completed":
                        return
                    if status == "canceled":
                        raise TrainingCanceledError("Remote training canceled")
                    raise RemoteTrainingError(f"Remote training {status}: {state.get('message')}")

                await asyncio.sleep(_POLL_INTERVAL_S)

    async def _fetch_state(self, client: httpx.AsyncClient, remote_job_id: str) -> dict[str, Any]:
        response = await client.get(f"{self._base_url}/jobs/{remote_job_id}")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RemoteTrainingError("Trainer returned a malformed job state")
        return data

    async def _download_and_extract(self, context: TrainingContext, remote_job_id: str) -> None:
        """Stream the model archive and extract it into the model directory."""
        settings = get_settings()
        tmp_archive = Path(tempfile.gettempdir()) / f"remote-model-{remote_job_id}.zip"
        # Finite per-read timeout, not a total cap: artifacts may be large and
        # take a while, but a stalled transfer (proxy/firewall holding the
        # connection open) must fail instead of hanging the job at 95% forever.
        stream_timeout = httpx.Timeout(self._timeout, read=settings.trainer_download_read_timeout_s)
        try:
            received = await self._stream_archive(remote_job_id, tmp_archive, stream_timeout)
            logger.info("Downloaded model artifact ({} bytes)", received)

            # Validation and extraction are blocking (zip CRC + disk writes); keep
            # them off the worker event loop so progress/cancellation stay live.
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

    async def _cancel(self, client: httpx.AsyncClient, remote_job_id: str) -> None:
        try:
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
        """Map the trainer's raw 0-100 progress into the local 10-95 window.

        0-10 is reserved for snapshot upload and 95-100 for model download, so
        training occupies 10-95. Applied once here; the trainer reports raw.
        """
        return min(95, 10 + round(remote_progress * 0.85))
