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

Every job with a provisioning row is explicitly resolved here - confirmed,
left pending for retry, or failed - so the caller must skip all of them
(`SshRecoveryReport.handled_job_ids`) when it runs the generic orphan abort
right after. That generic pass only inspects a job's own persisted
``remote_job_id``, which an SSH job may not have recorded yet even though its
container was just confirmed healthy (e.g. a crash between provisioning and
the trainer job actually being submitted); without the exclusion it would
re-judge and incorrectly fail a job this module already decided to keep.
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
    from schemas.job_provisioning import JobProvisioning
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
_TRANSIENT_REASONS = frozenset(
    {
        ReattachFailureReason.HEALTH_NEVER_READY,
        ReattachFailureReason.PORT_UNREACHABLE,
        ReattachFailureReason.INSPECTION_FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class _RowVerdict:
    """What one active row's processing decided, for the caller to tally and act on."""

    outcome: str  # "confirmed" | "transient" | "failed"
    # Server whose active set this job's id should be added to, so the
    # per-server sweep does not reclaim its container. None only when the
    # job was failed outright (its container, if any, is left for the sweep).
    active_on_server: UUID | None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SshRecoveryReport:
    """Summary of one startup recovery pass, for a single structured log line."""

    confirmed: int = 0
    transient: int = 0
    failed: int = 0
    stale_rows_cleaned: int = 0
    orphans_removed: int = 0
    failures_by_reason: dict[str, int] = field(default_factory=dict)
    # Every job id this pass rendered a verdict for (confirmed, left pending
    # for retry, or explicitly failed). The caller must exclude these from the
    # generic orphan abort that follows - see the module docstring.
    handled_job_ids: frozenset[UUID] = field(default_factory=frozenset)


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


async def _requeue_pending(job_service: JobService, job_id: UUID) -> None:
    """Requeue a job to PENDING so `run_loop` picks it back up to retry.

    Shared by every active-row branch that decides a job is still worth
    trying (container never launched, confirmed healthy, or transiently
    inconclusive): none of them is itself watched by anything - only a
    PENDING pickup opens the real tunnel and resumes progress streaming, or
    tries the reattach again. Leaving the job at whatever non-terminal
    status a hard crash left it in (usually RUNNING) would strand it forever.
    """
    await job_service.update_job_status(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Reconnecting to remote training job after restart",
    )


async def _process_active_row(
    row: JobProvisioning,
    job_service: JobService,
    provisioning_service: SshProvisioningService,
    remote_server_service: RemoteServerService,
) -> _RowVerdict:
    """Decide one active row's fate: requeued (confirmed/transient) or failed."""
    if row.container_name is None:
        # Never got far enough to launch a container - nothing here for this
        # pass to verify; requeue so it is retried later.
        await _requeue_pending(job_service, row.job_id)
        return _RowVerdict(outcome="transient", active_on_server=row.remote_server_id)

    try:
        server = await remote_server_service.get_remote_server(row.remote_server_id)
    except ResourceNotFoundError:
        logger.error("Failing job {}: remote server {} no longer registered", row.job_id, row.remote_server_id)
        await job_service.update_job_status(
            job_id=row.job_id,
            status=JobStatus.FAILED,
            message="Training job failed: its remote server is no longer registered",
        )
        return _RowVerdict(outcome="failed", active_on_server=None, failure_reason="server_not_registered")

    outcome = await provisioning_service.verify_reattach(row, server)

    if outcome.ok:
        logger.info("Confirmed reattach for job {} on server '{}'", row.job_id, server.name)
        # A confirmed container only proves it is still safe to trust, not
        # that anything is watching it: only a PENDING pickup opens the real
        # (long-lived) tunnel and resumes SSE streaming (see `_requeue_pending`).
        await _requeue_pending(job_service, row.job_id)
        return _RowVerdict(outcome="confirmed", active_on_server=server.id)

    if outcome.reason in _TRANSIENT_REASONS:
        logger.warning(
            "Reattach check for job {} on server '{}' was inconclusive ({}); leaving pending for retry: {}",
            row.job_id,
            server.name,
            outcome.reason,
            outcome.detail,
        )
        await _requeue_pending(job_service, row.job_id)
        return _RowVerdict(outcome="transient", active_on_server=server.id)

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
    # A failed job is never added back to active_job_ids_by_server, so the
    # sweep below reclaims its container - but only when ownership was
    # actually established; a foreign container is never listed by the sweep
    # in the first place (it filters on this installation's own
    # backend_instance_id label).
    return _RowVerdict(
        outcome="failed",
        active_on_server=None,
        failure_reason=outcome.reason.value if outcome.reason is not None else None,
    )


async def _sweep_all_servers(
    provisioning_service: SshProvisioningService,
    remote_server_service: RemoteServerService,
    active_job_ids_by_server: dict[UUID, set[UUID]],
) -> int:
    """Sweep orphan containers on every registered server; one server's failure never aborts the rest."""
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
    return orphans_removed


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
            verdict = await _process_active_row(row, job_service, provisioning_service, remote_server_service)

        if verdict.active_on_server is not None:
            active_job_ids_by_server[verdict.active_on_server].add(row.job_id)
        if verdict.outcome == "confirmed":
            confirmed += 1
        elif verdict.outcome == "transient":
            transient += 1
        else:
            failed += 1
            if verdict.failure_reason is not None:
                failures_by_reason[verdict.failure_reason] += 1

    orphans_removed = await _sweep_all_servers(provisioning_service, remote_server_service, active_job_ids_by_server)

    return SshRecoveryReport(
        confirmed=confirmed,
        transient=transient,
        failed=failed,
        stale_rows_cleaned=stale_cleaned,
        orphans_removed=orphans_removed,
        failures_by_reason=dict(failures_by_reason),
        handled_job_ids=frozenset(row.job_id for row in active_rows),
    )


__all__ = ["SshRecoveryReport", "recover_ssh_jobs"]
