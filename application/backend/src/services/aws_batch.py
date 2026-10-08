"""AWS Batch trainer health probe."""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import TYPE_CHECKING, Any

from loguru import logger

from schemas.hardware import DeviceInfo, DeviceType
from schemas.remote_trainer import HealthStatus, RemoteTrainerHealth

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from schemas.remote_trainer import AwsBatchConnection

_PROBE_TIMEOUT_S = 10.0


def _make_session(connection: AwsBatchConnection) -> Any:
    """Assume the stack's Studio role with the ambient credential chain."""
    import boto3

    sts = boto3.client("sts", region_name=connection.region)
    creds = sts.assume_role(RoleArn=connection.studio_role_arn, RoleSessionName="physicalai-studio")["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=connection.region,
    )


def _probe_sync(connection: AwsBatchConnection) -> tuple[HealthStatus, str | None]:
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        session = _make_session(connection)
        batch = session.client("batch")
        queues = batch.describe_job_queues(jobQueues=[t.queue for t in connection.targets.values()])["jobQueues"]
    except (BotoCoreError, ClientError) as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
        logger.debug("AWS Batch probe failed: {}", exc)
        return "unreachable", code or "aws_unavailable"
    if len(queues) != len(connection.targets):
        return "degraded", "job_queue_missing"
    if any(q.get("state") != "ENABLED" or q.get("status") != "VALID" for q in queues):
        return "degraded", "job_queue_disabled"
    return "healthy", None


async def probe_aws_batch(
    remote_trainer_id: UUID, connection: AwsBatchConnection, checked_at: datetime
) -> RemoteTrainerHealth:
    """Check credentials and job queues; report one CUDA device per provisioned instance type."""
    started = perf_counter()
    try:
        status, reason_code = await asyncio.wait_for(asyncio.to_thread(_probe_sync, connection), _PROBE_TIMEOUT_S)
    except TimeoutError:
        status, reason_code = "unreachable", "timeout"
    devices = (
        [
            DeviceInfo(type=DeviceType.CUDA, name=instance_type, index=index)
            for index, instance_type in enumerate(connection.targets)
        ]
        if status == "healthy"
        else []
    )
    return RemoteTrainerHealth(
        remote_trainer_id=remote_trainer_id,
        status=status,
        checked_at=checked_at,
        latency_ms=round((perf_counter() - started) * 1000),
        devices=devices,
        storage=None,
        reason_code=reason_code,
    )
