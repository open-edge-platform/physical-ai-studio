"""Submit one-off training jobs through AWS Batch and S3."""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger
from physicalai.data.archive_safety import SafeZipArchive

from schemas.aws_auth import AwsAuthenticationIssue
from schemas.remote_trainer import AwsBatchConnection
from services.aws_auth_service import authentication_issue
from services.aws_batch import POLICY_INSTANCE_TYPES, _make_session
from services.dataset_download_service import DatasetDownloadService
from services.staged_archive import cleanup_staged_archive
from services.training_backends.base import TrainingCanceledError, TrainingSuspendedError
from services.training_backends.local import build_spec, resolve_hf_token
from settings import get_settings
from trainer.batch.status import BatchJobStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from services.training_backends.base import TrainingContext

T = TypeVar("T")


class _TransferStopped(Exception):
    pass


class AwsBatchTrainingBackend:
    """Stage inputs in S3, submit to the policy's queue, and ingest its artifacts."""

    def __init__(self, connection: AwsBatchConnection) -> None:
        self._connection = connection
        self._session: Any = None
        self._session_created = 0.0
        self._log_token: str | None = None
        self._log_warning_shown = False
        self._task_warning_shown = False
        self._authentication_issue: AwsAuthenticationIssue | None = None
        self._progress = 0

    async def _with_authentication(
        self, context: TrainingContext, operation: Callable[[], T], *, phase: str, stopping: bool = False
    ) -> T:
        while True:
            if not stopping:
                await self._check_stop(context, str(context.remote_job_id) if context.remote_job_id else None)
            try:
                result = await asyncio.to_thread(operation)
            except Exception as error:
                issue = authentication_issue(error, self._connection.profile_name)
                if issue is None:
                    raise
                self._session = None
                if issue != self._authentication_issue:
                    self._authentication_issue = issue
                    if context.on_authentication_issue:
                        await context.on_authentication_issue(issue)
                message = (
                    "Stop pending — AWS authentication required; remote job not confirmed stopped"
                    if stopping
                    else f"Awaiting AWS authentication — {phase}"
                )
                context.progress(max(self._progress, context.job.progress or 0), message=message)
                if context.should_suspend():
                    raise TrainingSuspendedError("Awaiting AWS authentication; reconnect after restart") from error
                await asyncio.sleep(5)
                continue
            if self._authentication_issue is not None or getattr(context.payload, "authentication_issue", None):
                self._authentication_issue = None
                if context.on_authentication_issue:
                    await context.on_authentication_issue(None)
                context.progress(max(self._progress, context.job.progress or 0), message="AWS connection restored")
            return result

    def _client(self, service: str) -> Any:
        if self._session is None or time.monotonic() - self._session_created > 600:
            self._session = _make_session(self._connection)
            self._session_created = time.monotonic()
        return self._session.client(service)

    async def train(self, context: TrainingContext) -> None:
        """Run a fresh job or reattach to its persisted Batch job ID."""
        prefix = f"jobs/{context.job.id}"
        job_id = str(context.remote_job_id) if context.remote_job_id else None
        await self._check_stop(context, job_id)
        if job_id is None:
            if context.snapshot is None:
                raise ValueError("AWS Batch training requires a dataset snapshot")
            context.progress(0, message="Preparing dataset snapshot")
            archive = await asyncio.to_thread(
                DatasetDownloadService().create_dataset_archive, Path(context.snapshot.path)
            )
            try:
                await self._check_stop(context, None)
                context.progress(5, message="Uploading dataset snapshot to S3")
                try:
                    await self._with_authentication(
                        context, lambda: self._upload_inputs(context, prefix, archive), phase="uploading dataset"
                    )
                except _TransferStopped:
                    await self._check_stop(context, None)
                    raise
            finally:
                cleanup_staged_archive(archive)
            await self._check_stop(context, None)
            context.progress(10, message="Submitting AWS Batch job")
            job_id = await self._with_authentication(
                context, lambda: self._submit(context, prefix), phase="submitting training job"
            )
            context.remote_job_id = UUID(job_id)
            if context.on_remote_job_id is not None:
                try:
                    await context.on_remote_job_id(UUID(job_id))
                except Exception:
                    await asyncio.to_thread(self._terminate, job_id)
                    raise
        await self._wait(context, job_id, prefix)
        await self._check_stop(context, job_id)
        context.progress(95, message="Downloading trained model")
        with tempfile.TemporaryDirectory(prefix="batch-model-") as directory:
            archive = Path(directory) / "artifact.zip"
            try:
                self._progress = 95
                await self._with_authentication(
                    context, lambda: self._download(context, prefix, archive), phase="downloading trained model"
                )
            except _TransferStopped:
                await self._check_stop(context, job_id)
                raise
            await self._check_stop(context, job_id)
            await asyncio.to_thread(self._extract, archive, context.output_dir)
        await self._check_stop(context, job_id)
        context.progress(99, message="Model downloaded")

    def _upload_inputs(self, context: TrainingContext, prefix: str, archive: Path) -> None:
        spec = build_spec(context).model_copy(update={"device_type": "cuda", "device_index": 0})
        s3 = self._client("s3")
        s3.upload_file(
            str(archive),
            self._connection.bucket,
            f"{prefix}/dataset.zip",
            Callback=lambda _: self._check_transfer_stop(context),
        )
        s3.put_object(
            Bucket=self._connection.bucket,
            Key=f"{prefix}/spec.json",
            Body=spec.model_dump_json(exclude={"run_options"}).encode(),
            ContentType="application/json",
        )

    def _download(self, context: TrainingContext, prefix: str, archive: Path) -> None:
        self._client("s3").download_file(
            self._connection.bucket,
            f"{prefix}/artifact.zip",
            str(archive),
            Callback=lambda _: self._check_transfer_stop(context),
        )

    @staticmethod
    def _check_transfer_stop(context: TrainingContext) -> None:
        if context.should_stop():
            raise _TransferStopped

    def _terminate(self, job_id: str, reason: str = "Canceled from Physical AI Studio") -> None:
        self._client("batch").terminate_job(jobId=job_id, reason=reason)

    def _submit(self, context: TrainingContext, prefix: str) -> str:
        target = self._connection.targets[POLICY_INSTANCE_TYPES[context.payload.policy]]
        token = resolve_hf_token()
        overrides = {"environment": [{"name": "HF_TOKEN", "value": token.get_secret_value()}]} if token else {}
        response = self._client("batch").submit_job(
            jobName=f"physicalai-{context.job.id}",
            jobQueue=target.queue,
            jobDefinition=target.job_definition,
            parameters={"job_prefix": prefix},
            containerOverrides=overrides,
        )
        return response["jobId"]

    def _poll(self, job_id: str, prefix: str) -> tuple[dict[str, Any], BatchJobStatus | None]:
        jobs = self._client("batch").describe_jobs(jobs=[job_id])["jobs"]
        if not jobs:
            raise RuntimeError(f"AWS Batch job {job_id} was not found")
        try:
            body = self._client("s3").get_object(Bucket=self._connection.bucket, Key=f"{prefix}/status.json")["Body"]
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"NoSuchKey", "404"}:
                raise
            return jobs[0], None
        try:
            status = BatchJobStatus.model_validate_json(body.read())
        finally:
            body.close()
        return jobs[0], status

    async def _wait(self, context: TrainingContext, job_id: str, prefix: str) -> None:
        last_contact = time.monotonic()
        last_phase: tuple[str, str] | None = None
        extra_info: dict[str, Any] = {}
        progress = max(10, context.job.progress or 0)
        while True:
            await self._check_stop(context, job_id)
            try:
                self._progress = progress
                job, status = await self._with_authentication(
                    context,
                    lambda: self._poll(job_id, prefix),
                    phase="job status unavailable; remote job may still run",
                )
            except (BotoCoreError, ClientError):
                if time.monotonic() - last_contact > 300:
                    raise
                context.progress(progress, message="AWS Batch status unavailable; retrying", extra_info=extra_info)
                await asyncio.sleep(5)
                continue
            last_contact = time.monotonic()
            await self._with_authentication(context, lambda: self._relay_logs(job), phase="reading training logs")
            if status is not None:
                progress = max(progress, min(95, 10 + round(status.progress * 0.85)))
            task = (
                await self._with_authentication(
                    context, lambda: self._starting_task(job), phase="reading startup diagnostics"
                )
                if job["status"] == "STARTING"
                else {}
            )
            message, details = self._execution_progress(job, status, task)
            details["job_id"] = job_id
            extra_info = dict(status.extra_info or {}) if status else {}
            extra_info["aws_batch"] = details
            context.progress(progress, message=message, extra_info=extra_info)
            phase = (job["status"], details["phase"])
            if phase != last_phase:
                logger.info("{} (job {})", message, job_id)
                last_phase = phase
            if status and status.status == "failed":
                if job["status"] not in {"FAILED", "SUCCEEDED"}:
                    await asyncio.to_thread(self._terminate, job_id, reason="Trainer reported failure")
                raise RuntimeError(status.message or "AWS Batch training failed")
            if job["status"] == "FAILED":
                reason = status.message if status and status.status == "failed" else job.get("statusReason")
                raise RuntimeError(reason or "AWS Batch training failed")
            if job["status"] == "SUCCEEDED":
                if status and status.status in {"failed", "canceled"}:
                    raise RuntimeError(status.message or "AWS Batch training failed")
                return
            await asyncio.sleep(5)

    def _starting_task(self, job: dict[str, Any]) -> dict[str, Any]:
        task_arn = job.get("container", {}).get("taskArn")
        if not task_arn or ":task/" not in task_arn:
            return {}
        resource = task_arn.split(":task/", 1)[1]
        if "/" not in resource:
            return {}
        try:
            response = self._client("ecs").describe_tasks(cluster=resource.rsplit("/", 1)[0], tasks=[task_arn])
        except (BotoCoreError, ClientError) as error:
            if authentication_issue(error, self._connection.profile_name):
                raise
            if not self._task_warning_shown:
                logger.warning("Cannot read AWS Batch startup diagnostics: {}", error)
                self._task_warning_shown = True
            return {}
        tasks = response.get("tasks", [])
        return tasks[0] if tasks else {}

    @staticmethod
    def _execution_progress(
        job: dict[str, Any], status: BatchJobStatus | None, task: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        state = job["status"]
        phase = {
            "SUBMITTED": "Job submitted",
            "PENDING": "Waiting for job dependencies",
            "RUNNABLE": "Waiting for compute capacity and scheduling",
            "STARTING": "Starting container; image download and setup may take several minutes",
            "RUNNING": "Container running; waiting for trainer progress",
            "SUCCEEDED": "Container completed",
            "FAILED": "Container failed",
        }.get(state, state)
        if state == "STARTING" and task.get("pullStartedAt"):
            phase = (
                "Container image downloaded; preparing startup"
                if task.get("pullStoppedAt")
                else "Pulling container image"
            )
        if state == "RUNNING" and status is not None:
            phase = status.message or "Training model"
        details: dict[str, Any] = {"state": state, "phase": phase}
        if image := job.get("container", {}).get("image"):
            details["image"] = image
        message = f"AWS Batch {state}: {phase}"
        if job.get("createdAt"):
            elapsed = max(0, int(time.time() - job["createdAt"] / 1000))
            details["elapsed_seconds"] = elapsed
            message += f" · {elapsed // 60}m {elapsed % 60:02d}s elapsed"
        if reason := job.get("statusReason"):
            details["reason"] = reason
            message += f" · {reason}"
        if task.get("lastStatus"):
            details["container_state"] = task["lastStatus"]
        if status is not None:
            age = max(0, int(time.time() - status.updated_at.timestamp()))
            details["trainer_update_age_seconds"] = age
            if state == "RUNNING" and age >= 60:
                message += f" · last trainer update {age // 60}m {age % 60:02d}s ago"
        return message, details

    def _relay_logs(self, job: dict[str, Any]) -> None:
        container = job.get("container", {})
        stream = container.get("logStreamName")
        group = container.get("logConfiguration", {}).get("options", {}).get("awslogs-group")
        if not stream or not group:
            return
        try:
            response = self._client("logs").get_log_events(
                logGroupName=group,
                logStreamName=stream,
                startFromHead=True,
                **({"nextToken": self._log_token} if self._log_token else {}),
            )
        except (BotoCoreError, ClientError) as error:
            if authentication_issue(error, self._connection.profile_name):
                raise
            if isinstance(error, ClientError) and error.response["Error"]["Code"] == "ResourceNotFoundException":
                return
            if not self._log_warning_shown:
                logger.warning("Cannot read AWS Batch container logs: {}", error)
                self._log_warning_shown = True
            return
        for event in response["events"]:
            logger.info("[AWS Batch] {}", event["message"])
        self._log_token = response.get("nextForwardToken")

    async def _check_stop(self, context: TrainingContext, job_id: str | None) -> None:
        if not context.should_stop() and not context.should_cancel_job():
            return
        if not context.should_cancel_job() and context.should_suspend():
            raise TrainingSuspendedError("Studio shutting down; leaving AWS Batch job running for reattach")
        if job_id is not None:
            await self._with_authentication(
                context, lambda: self._terminate(job_id), phase="stopping training", stopping=True
            )
            logger.info("Terminated AWS Batch job {}", job_id)
        raise TrainingCanceledError("Training canceled")

    @staticmethod
    def _extract(archive: Path, output_dir: Path) -> None:
        settings = get_settings()
        safe = SafeZipArchive(archive, max_uncompressed_bytes=settings.data_import_max_uncompressed_bytes)
        safe.validate()
        output_dir.mkdir(parents=True, exist_ok=True)
        safe.extract_to(output_dir, min_free_bytes=settings.data_import_min_free_bytes)
