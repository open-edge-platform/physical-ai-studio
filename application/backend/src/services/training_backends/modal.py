# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run single-GPU training on a deployed Modal function."""

from __future__ import annotations

import asyncio
import io
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from modal.exception import AuthError, InternalError, PermissionDeniedError, ServiceError
from modal.exception import ConnectionError as ModalConnectionError
from modal.exception import Error as ModalError

from schemas.job import RemoteTrainJobPayload
from services.dataset_download_service import DatasetDownloadService
from services.modal_training import modal_resources, read_volume_file
from services.staged_archive import cleanup_staged_archive
from services.training_backends._transfer_progress import TransferProgressLogger, format_bytes, format_throughput
from services.training_backends.artifacts import extract_training_artifact
from services.training_backends.base import TrainingCanceledError, TrainingSuspendedError
from services.training_backends.local import build_spec, resolve_hf_token
from settings import get_settings
from trainer.batch.status import BatchJobStatus
from training.log_format import render_progress_log

if TYPE_CHECKING:
    import modal

    from schemas.remote_trainer import ModalConnection
    from services.training_backends.base import TrainingContext


_CONNECTION_ERRORS = (AuthError, ModalConnectionError, InternalError, PermissionDeniedError, ServiceError)


class ModalTrainingBackend:
    """Stage files on a Volume, persist the function call ID, and ingest the export."""

    def __init__(self, connection: ModalConnection) -> None:
        self._connection = connection

    async def train(self, context: TrainingContext) -> None:
        """Submit once, or reattach after Studio restarts without canceling the GPU job."""
        import modal

        function, volume = modal_resources(self._connection)
        call = modal.FunctionCall.from_id(str(context.remote_job_id)) if context.remote_job_id else None
        await self._check_stop(context, call)
        prefix = f"jobs/{context.job.id}"
        if call is None:
            if context.snapshot is None:
                raise ValueError("Modal training requires a dataset snapshot")
            function = await asyncio.to_thread(self._with_hf_token, function)
            await self._check_stop(context, None)
            context.progress(0, message="Preparing dataset snapshot")
            archive = await asyncio.to_thread(
                DatasetDownloadService().create_dataset_archive, Path(context.snapshot.path)
            )
            try:
                await self._check_stop(context, None)
                context.progress(5, message="Uploading dataset to Modal Volume")
                await asyncio.to_thread(self._upload, context, volume, prefix, archive)
            finally:
                cleanup_staged_archive(archive)
            await self._check_stop(context, None)
            context.progress(10, message="Starting Modal GPU function")
            if isinstance(context.payload, RemoteTrainJobPayload) and context.payload.modal_gpu is not None:
                function = function.with_options(gpu=context.payload.modal_gpu)
            call = await asyncio.to_thread(function.spawn, str(context.job.id))
            if call is None:
                raise RuntimeError("Modal did not return a training call")
            context.remote_job_id = call.object_id
            try:
                if context.on_remote_job_id is not None:
                    await context.on_remote_job_id(call.object_id)
            except Exception:
                await asyncio.to_thread(call.cancel)
                raise
        await self._wait(context, call, volume, prefix)
        await self._check_stop(context, call)
        context.progress(95, message="Downloading trained model from Modal")
        with tempfile.TemporaryDirectory(prefix="modal-model-") as directory:
            archive = Path(directory) / "artifact.zip"
            try:
                await asyncio.to_thread(self._download, context, volume, f"{prefix}/artifact.zip", archive)
            except _TransferStopped:
                await self._check_stop(context, call)
                raise
            await self._check_stop(context, call)
            settings = get_settings()
            await asyncio.to_thread(
                extract_training_artifact,
                archive,
                context.output_dir,
                settings.data_import_max_uncompressed_bytes,
                settings.data_import_min_free_bytes,
            )
        await self._check_stop(context, call)
        context.progress(99, message="Model downloaded")

    def _with_hf_token(self, function: modal.Function) -> modal.Function:
        import modal

        token = resolve_hf_token()
        if token is None or not token.get_secret_value():
            return function
        # with_options replaces secrets, so retain every existing binding before adding HF_TOKEN.
        secret_ids = function.info().secrets
        available = (
            {
                secret.object_id: secret
                for secret in modal.Secret.objects.list(environment_name=self._connection.environment_name)
            }
            if secret_ids
            else {}
        )
        if any(secret_id not in available for secret_id in secret_ids):
            raise ValueError(
                "Cannot forward Studio's HF token while preserving unnamed deployment secrets. "
                "Use named Modal secrets in the deployed function."
            )
        secrets = [available[secret_id] for secret_id in secret_ids]
        secrets.append(modal.Secret.from_dict({"HF_TOKEN": token.get_secret_value()}))
        return function.with_options(secrets=secrets)

    @staticmethod
    def _upload(context: TrainingContext, volume: modal.Volume, prefix: str, archive: Path) -> None:
        spec = build_spec(context).model_copy(update={"device_type": "cuda", "device_index": 0})
        total = archive.stat().st_size
        logger.info("Uploading dataset snapshot to Modal Volume: {} ({})", archive.name, format_bytes(total))
        started = time.monotonic()
        # HF credentials are injected through Modal Secrets, never stored in the Volume.
        with volume.batch_upload(force=True) as batch:
            batch.put_file(archive, f"/{prefix}/dataset.zip")
            batch.put_file(io.BytesIO(spec.model_dump_json(exclude={"run_options"}).encode()), f"/{prefix}/spec.json")
        elapsed = time.monotonic() - started
        logger.info(
            "Snapshot uploaded to Modal Volume: {} in {:.1f}s ({})",
            format_bytes(total),
            elapsed,
            format_throughput(total, elapsed),
        )

    @staticmethod
    async def _check_stop(context: TrainingContext, call: modal.FunctionCall | None) -> None:
        while context.should_cancel_job() or (context.should_stop() and not context.should_suspend()):
            if call is not None:
                try:
                    await asyncio.to_thread(call.cancel)
                except _CONNECTION_ERRORS as error:
                    context.progress(
                        context.job.progress or 0,
                        message="Stop pending — Modal connection unavailable; remote job not confirmed stopped",
                    )
                    if context.should_suspend():
                        raise TrainingSuspendedError("Modal cancellation pending; reconnect after restart") from error
                    await asyncio.sleep(5)
                    continue
            raise TrainingCanceledError("Modal training canceled")
        if context.should_suspend():
            raise TrainingSuspendedError("Modal job left running for reattachment")

    async def _wait(
        self, context: TrainingContext, call: modal.FunctionCall, volume: modal.Volume, prefix: str
    ) -> None:
        import modal

        progress = max(10, context.job.progress or 0)
        last_progress_log = None
        while True:
            await self._check_stop(context, call)
            try:
                contents = await asyncio.to_thread(read_volume_file, volume, f"{prefix}/status.json", max_bytes=65536)
                status = BatchJobStatus.model_validate_json(contents) if contents else None
                if status:
                    progress = max(progress, min(95, 10 + round(status.progress * 0.85)))
                    context.progress(progress, message=status.message, extra_info=status.extra_info)
                    line = render_progress_log(status.extra_info) if status.extra_info else None
                    line = line or status.message or f"Remote training progress: {status.progress}%"
                    if line != last_progress_log:
                        logger.info("Modal: {} (call {})", line, context.remote_job_id)
                        last_progress_log = line
                    if status.status == "completed":
                        return
                    if status.status in {"failed", "canceled"}:
                        await asyncio.to_thread(call.cancel)
                        raise RuntimeError(status.message or "Modal training failed")
                try:
                    code = await asyncio.to_thread(call.get, timeout=0)
                except (modal.exception.FunctionTimeoutError, modal.exception.OutputExpiredError):
                    raise
                except (TimeoutError, modal.exception.TimeoutError):
                    if status is None:
                        context.progress(progress, message="Modal: waiting for GPU or building the trainer image")
                else:
                    if code != 0:
                        raise RuntimeError(f"Modal trainer exited with code {code}; see Modal application logs")
                    return
            except _CONNECTION_ERRORS as error:
                logger.warning("Modal monitoring unavailable for {}: {}", context.remote_job_id, error)
                context.progress(
                    progress, message="Modal connection unavailable; remote job may still run — reconnecting"
                )
            await asyncio.sleep(5)

    @staticmethod
    def _download(context: TrainingContext, volume: modal.Volume, remote_path: str, archive: Path) -> None:
        total = None
        try:
            entries = volume.listdir(remote_path)
            total = entries[0].size if entries else None
        except (ModalError, OSError) as error:
            logger.warning("Could not read Modal artifact size ({}); downloading without a total", type(error).__name__)
        logger.info(
            "Downloading trained model from Modal Volume: {} ({})",
            remote_path,
            format_bytes(total) if total is not None else "unknown size",
        )
        started = time.monotonic()
        received = 0
        heartbeat = TransferProgressLogger("Model download", total, logger_=logger)
        with archive.open("wb") as destination:
            for chunk in volume.read_file(remote_path):
                if context.should_stop():
                    raise _TransferStopped
                destination.write(chunk)
                received += len(chunk)
                heartbeat.update(received)
        elapsed = time.monotonic() - started
        logger.info(
            "Model downloaded from Modal Volume: {} in {:.1f}s ({})",
            format_bytes(received),
            elapsed,
            format_throughput(received, elapsed),
        )


class _TransferStopped(Exception):
    pass
