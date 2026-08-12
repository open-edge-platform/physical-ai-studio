# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Startup recovery for SSH-provisioned training jobs.

Runs once at studio startup, before the generic orphan-job abort in
`services.training_service.TrainingService.abort_orphan_jobs`. For every
non-terminal job with a persisted `JobProvisioningDB` row, `recover_ssh_jobs`:

1. Reconciles stale rows - a row whose job already reached a terminal state,
   left behind by a crash between stopping a container and deleting its row.
2. Verifies every remaining row's container with
   `SshProvisioningService.verify_reattach`, and requeues (`PENDING`) or fails
   the job depending on the outcome (see `ReattachFailureReason`).
3. Sweeps orphan containers per configured server, excluding every job this
   pass just confirmed or left pending - so a container reattach is about to
   claim is never swept out from under it.

Step order matters: sweeping runs last, and only after every non-terminal
job's container has had a chance to be recognized as still claimed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from core.logging.utils import job_logging_ctx
from exceptions import ResourceNotFoundError
from schemas.base_job import JobStatus
from services.ssh.provisioning import ReattachFailureReason, SshProvisioningService

if TYPE_CHECKING:
    from uuid import UUID

    from repositories.job_provisioning_repo import JobProvisioningRepository
    from services.job_service import JobService
    from services.remote_server_service import RemoteServerService
    from settings import Settings

# Reasons whose container is provably this installation's own and safe to
# reclaim once the job is failed - the report never counts a foreign or
# unreachable container as "cleaned up".
_ACTIONABLE_MESSAGES: dict[ReattachFailureReason, str] = {
    ReattachFailureReason.CONTAINER_GONE: "its trainer container is no longer running on the remote server",
    ReattachFailureReason.OWNERSHIP_MISMATCH: (
        "a container with its name exists but is not owned by this installation"
    ),
    ReattachFailureReason.DIGEST_MISMATCH: "its trainer container is running an unexpected image",
    ReattachFailureReason.HOST_KEY_FAILURE: (
        "the remote server's SSH host key could not be verified; accept its current fingerprint, then resubmit the job"
    ),
    ReattachFailureReason.ALIAS_MISSING: ("its remote server's SSH host alias is no longer present in the SSH config"),
}

# Outcomes left for the normal per-job reattach path to retry, rather than
# failed outright: each is either transient (a slow-starting trainer, a flaky
# network path) or an outcome this pass could not evaluate at all.
_TRANSIENT_REASONS = frozenset({ReattachFailureReason.HEALTH_NEVER_READY, ReattachFailureReason.PORT_UNREACHABLE})


@dataclass(frozen=True, slots=True)
class SshRecoveryReport:
    """Summary of one startup recovery pass, for a single structured log line."""

    confirmed: int = 0
    transient: int = 0
    failed: int = 0
    stale_rows_cleaned: int = 0
    orphans_removed: int = 0
    failures_by_reason: dict[str, int] = field(default_factory=dict)


async def _reconcile_stale_rows(
    provisioning_service: SshProvisioningService,
    provisioning_repo: JobProvisioningRepository,
    remote_server_service: RemoteServerService,
) -> int:
    """Tear down and drop provisioning rows whose job already finished."""
    stale_rows = await provisioning_repo.list_stale()
    cleaned = 0
    for row in stale_rows:
        with job_logging_ctx(job_id=str(row.job_id)):
            try:
                server = await remote_server_service.get_remote_server(row.remote_server_id)
            except ResourceNotFoundError:
                # The server itself is gone; there is nothing left to dial.
                logger.warning(
                    "Dropping stale provisioning row for job {}: remote server {} no longer registered",
                    row.job_id,
                    row.remote_server_id,
                )
                await provisioning_repo.delete_by_job_id(row.job_id)
                cleaned += 1
                continue
            try:
                await provisioning_service.teardown(row.job_id, server)
            except Exception as error:  # one server's failure must not abort the pass
                logger.warning(
                    "Could not reconcile stale provisioning row for job {} on server '{}': {}",
                    row.job_id,
                    server.name,
                    error,
                )
                continue
            logger.info("Reconciled stale provisioning row for finished job {}", row.job_id)
            cleaned += 1
    return cleaned


async def recover_ssh_jobs(
    job_service: JobService,
    provisioning_repo: JobProvisioningRepository,
    remote_server_service: RemoteServerService,
    settings: Settings | None = None,
) -> SshRecoveryReport:
    """Reattach or fail every SSH-provisioned job left non-terminal by a restart.

    Safe to call with no SSH servers registered and no provisioning rows at
    all: every step degrades to a no-op.
    """
    provisioning_service = SshProvisioningService(provisioning_repo, settings)
    active_rows = await provisioning_repo.list_active()

    stale_cleaned = await _reconcile_stale_rows(provisioning_service, provisioning_repo, remote_server_service)

    confirmed = 0
    transient = 0
    failed = 0
    failures_by_reason: dict[str, int] = defaultdict(int)
    active_job_ids_by_server: dict[UUID, set[UUID]] = defaultdict(set)

    for row in active_rows:
        with job_logging_ctx(job_id=str(row.job_id)):
            if row.container_name is None:
                # Never got far enough to launch a container - nothing here for
                # this pass to verify; leave it to the normal pickup path.
                active_job_ids_by_server[row.remote_server_id].add(row.job_id)
                transient += 1
                continue

            try:
                server = await remote_server_service.get_remote_server(row.remote_server_id)
            except ResourceNotFoundError:
                logger.error("Failing job {}: remote server {} no longer registered", row.job_id, row.remote_server_id)
                await job_service.update_job_status(
                    job_id=row.job_id,
                    status=JobStatus.FAILED,
                    message="Training job failed: its remote server is no longer registered",
                )
                failed += 1
                failures_by_reason["server_not_registered"] += 1
                continue

            outcome = await provisioning_service.verify_reattach(row, server)

            if outcome.ok:
                logger.info("Confirmed reattach for job {} on server '{}'", row.job_id, server.name)
                active_job_ids_by_server[server.id].add(row.job_id)
                confirmed += 1
                continue

            if outcome.reason in _TRANSIENT_REASONS:
                logger.warning(
                    "Reattach check for job {} on server '{}' was inconclusive ({}); leaving pending for retry: {}",
                    row.job_id,
                    server.name,
                    outcome.reason,
                    outcome.detail,
                )
                active_job_ids_by_server[server.id].add(row.job_id)
                transient += 1
                continue

            message = (
                _ACTIONABLE_MESSAGES.get(outcome.reason, "its trainer container could not be verified")
                if outcome.reason is not None
                else "its trainer container could not be verified"
            )
            logger.error("Failing job {} on server '{}': {} ({})", row.job_id, server.name, message, outcome.detail)
            await job_service.update_job_status(
                job_id=row.job_id,
                status=JobStatus.FAILED,
                message=f"Training job failed: {message}",
            )
            # A failed job is never added back to active_job_ids_by_server, so
            # the sweep below reclaims its container - but only when ownership
            # was actually established; a foreign container is never listed by
            # the sweep in the first place (it filters on this installation's
            # own backend_instance_id label).
            failed += 1
            if outcome.reason is not None:
                failures_by_reason[outcome.reason.value] += 1

    orphans_removed = 0
    for server in await remote_server_service.list_remote_servers():
        try:
            removed = await provisioning_service.sweep_orphans(server, active_job_ids_by_server.get(server.id, set()))
        except Exception as error:  # one server's failure must not abort the sweep of others
            logger.warning("Could not sweep orphan containers on server '{}': {}", server.name, error)
            continue
        if removed:
            logger.info("Swept {} orphan container(s) on server '{}'", len(removed), server.name)
        orphans_removed += len(removed)

    return SshRecoveryReport(
        confirmed=confirmed,
        transient=transient,
        failed=failed,
        stale_rows_cleaned=stale_cleaned,
        orphans_removed=orphans_removed,
        failures_by_reason=dict(failures_by_reason),
    )


__all__ = ["SshRecoveryReport", "recover_ssh_jobs"]
