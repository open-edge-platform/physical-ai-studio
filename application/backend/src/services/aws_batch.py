"""AWS Batch trainer health probe."""

from __future__ import annotations

import asyncio
import json
from time import perf_counter
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlsplit

from loguru import logger

from core.security import require_aws_feature_active
from schemas.aws_auth import AwsAuthenticationIssue
from schemas.remote_trainer import AwsBatchConnection, AwsBatchProviderConfiguration, HealthStatus, RemoteTrainerHealth
from training.policy_catalog import PolicyCatalog

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

_PROBE_TIMEOUT_S = 10.0

POLICY_INSTANCE_TYPES = {
    "act": "g4dn.2xlarge",
    "pi05": "g6e.2xlarge",
    "smolvla": "g4dn.2xlarge",
    "molmoact2": "g7e.8xlarge",
}


def aws_batch_trainer_url(connection: AwsBatchConnection) -> str:
    """Encode resolved resources in a restart-stable trainer URL."""
    query = urlencode(
        {
            "studio_role_arn": connection.studio_role_arn,
            "targets": json.dumps({key: target.model_dump() for key, target in connection.targets.items()}),
            **({"stack_id": connection.stack_id} if connection.stack_id else {}),
            "profile_name": connection.profile_name,
        }
    )
    return f"aws-batch://{connection.region}/{connection.bucket}?{query}"


def aws_batch_connection_from_url(url: str) -> AwsBatchConnection:
    """Restore pinned Batch resources without consulting mutable trainer configuration."""
    location = urlsplit(url)
    if location.scheme != "aws-batch":
        raise ValueError("Expected an aws-batch trainer URL")
    query = parse_qs(location.query, strict_parsing=True)
    try:
        return AwsBatchConnection.model_validate(
            {
                "region": location.netloc,
                "bucket": location.path.removeprefix("/"),
                "studio_role_arn": query["studio_role_arn"][0],
                "targets": json.loads(query["targets"][0]),
                "stack_id": query.get("stack_id", [None])[0],
                "profile_name": query.get("profile_name", ["default"])[0],
            }
        )
    except KeyError as exc:
        raise ValueError(f"Batch trainer URL is missing {exc.args[0]}") from exc


def batch_policy_catalog(connection: AwsBatchConnection) -> PolicyCatalog:
    """Declare capabilities of the provisioned CUDA trainer image."""
    from training.policy_catalog import declared_policy_catalog

    return declared_policy_catalog(
        {
            policy: ["torch", "openvino", "onnx", "executorch"] if policy == "act" else ["torch", "openvino"]
            for policy, instance_type in POLICY_INSTANCE_TYPES.items()
            if instance_type in connection.targets
        }
    )


def _make_session(connection: AwsBatchConnection) -> Any:
    """Create a session scoped to the stack's Studio role using the configured credential provider."""
    from services.aws_auth_service import get_credential_provider

    return get_credential_provider(connection).create_session(
        region=connection.region,
        role_arn=connection.studio_role_arn,
    )


def resolve_aws_batch_configuration(configuration: AwsBatchProviderConfiguration) -> AwsBatchConnection:
    """Derive stack resources from stack_id and verify Batch queues using the selected credential provider."""
    require_aws_feature_active()
    connection = AwsBatchConnection.model_validate(
        {
            "stack_id": configuration.stack_id,
            "profile_name": configuration.profile_name,
        }
    )
    session = _make_session(connection)
    queue_names = list(dict.fromkeys(t.queue for t in connection.targets.values()))
    queues = session.client("batch").describe_job_queues(jobQueues=queue_names)["jobQueues"]
    if len(queues) != len(queue_names) or any(
        q.get("state") != "ENABLED" or q.get("status") != "VALID" for q in queues
    ):
        raise ValueError("AWS Batch resources for the specified stack ID are missing or disabled.")
    return connection


def _probe_sync(connection: AwsBatchConnection) -> tuple[HealthStatus, str | None, AwsAuthenticationIssue | None]:
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        session = _make_session(connection)
        batch = session.client("batch")
        queue_names = list(dict.fromkeys(t.queue for t in connection.targets.values()))
        queues = batch.describe_job_queues(jobQueues=queue_names)["jobQueues"]
    except (BotoCoreError, ClientError) as exc:
        from services.aws_auth_service import credential_error_reason, get_credential_provider

        if reason := credential_error_reason(exc):
            return (
                "degraded",
                "access_denied" if reason == "aws_access_denied" else "authentication_required",
                get_credential_provider(connection).diagnose_error(exc),
            )
        logger.debug("AWS Batch probe failed: {}", exc)
        return "unreachable", "service_error" if isinstance(exc, ClientError) else "connection_failed", None
    if len(queues) != len(queue_names):
        return "degraded", "resource_missing", None
    if any(q.get("state") != "ENABLED" or q.get("status") != "VALID" for q in queues):
        return "degraded", "resource_disabled", None
    return "healthy", None, None


async def probe_aws_batch(
    remote_trainer_id: UUID, connection: AwsBatchConnection, checked_at: datetime
) -> RemoteTrainerHealth:
    """Check credentials and job queues without representing queues as physical GPUs."""
    from services.aws_auth_service import operational_authentication_issue

    started = perf_counter()
    try:
        status, reason_code, issue = await asyncio.wait_for(
            asyncio.to_thread(_probe_sync, connection), _PROBE_TIMEOUT_S
        )
    except TimeoutError:
        status, reason_code = "unreachable", "timeout"
        issue = None
    return RemoteTrainerHealth(
        remote_trainer_id=remote_trainer_id,
        status=status,
        checked_at=checked_at,
        latency_ms=round((perf_counter() - started) * 1000),
        devices=[],
        storage=None,
        reason_code=reason_code,
        issues=[operational_authentication_issue(issue)]
        if issue
        else ([] if status in {"healthy", "degraded"} else None),
    )
