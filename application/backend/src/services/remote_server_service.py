# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Persistence for SSH-provisioned training servers.

Create and update here only persist the row. Tier-1-preflight-gated saves and
resolved-host display are layered on top by the API; those two operations
never dial SSH. `get_status`, however, does: it runs a throttled, shared
Tier 1 preflight probe (see `_tier1_status`) so the status endpoint can
report a server's live health.
"""

import asyncio
from time import monotonic
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from repositories.job_provisioning_repo import JobProvisioningRepository
from repositories.job_repo import JobRepository
from repositories.remote_server_repo import RemoteServerRepository
from schemas.remote_server import RemoteServer, RemoteServerCreate, RemoteServerUpdate
from schemas.ssh_preflight import PreflightResult, RemoteServerStatus
from services.ssh.preflight import DEFAULT_PROTOCOL_VERSION, run_tier1_preflight, run_tier2_preflight
from services.training_backends.phase import PhaseState
from settings import Settings, get_settings

# Per-server Tier 1 status: at most one in-flight SSH probe shared across
# concurrent pollers, plus a cached result for `settings.ssh_preflight_throttle_s`
# so repeated UI polling doesn't pile SSH connections onto a busy server.
_status_checks: dict[UUID, asyncio.Task[PreflightResult]] = {}
_status_cache: dict[UUID, tuple[float, PreflightResult]] = {}


def reset_status_cache() -> None:
    """Drop every cached/in-flight Tier 1 status probe. Test-support only."""
    _status_checks.clear()
    _status_cache.clear()


class RemoteServerService:
    """Manage SSH-provisioned training server registrations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = RemoteServerRepository(session)

    async def list_remote_servers(self) -> list[RemoteServer]:
        """Return registered servers in stable creation order."""
        return await self.repo.list_ordered()

    async def get_remote_server(self, remote_server_id: UUID) -> RemoteServer:
        """Return one registered server or raise a not-found error."""
        remote_server = await self.repo.get_by_id(remote_server_id)
        if remote_server is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))
        return remote_server

    async def create_remote_server(self, config: RemoteServerCreate) -> RemoteServer:
        """Persist an SSH-provisioned training server."""
        remote_server = RemoteServer(id=uuid4(), **config.model_dump())
        try:
            return await self.repo.save(remote_server)
        except IntegrityError as error:
            await self.session.rollback()
            raise ResourceAlreadyExistsError(
                "Remote server",
                "A server with this SSH host alias is already configured.",
            ) from error

    async def update_remote_server(self, remote_server_id: UUID, update: RemoteServerUpdate) -> RemoteServer:
        """Update a registered server's mutable fields."""
        remote_server = await self.repo.get_by_id(remote_server_id)
        if remote_server is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))
        try:
            return await self.repo.update(remote_server, update.model_dump(exclude_none=True, exclude_unset=True))
        except IntegrityError as error:
            await self.session.rollback()
            raise ResourceAlreadyExistsError(
                "Remote server",
                "A server with this SSH host alias is already configured.",
            ) from error

    async def delete_remote_server(self, remote_server_id: UUID) -> None:
        """Delete a registered server."""
        if await self.repo.get_by_id(remote_server_id) is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))
        await self.repo.delete_by_id(remote_server_id)

    async def record_check_result(self, remote_server_id: UUID, result: PreflightResult) -> RemoteServer:
        """Persist the outcome of a Tier 2 verification (explicit ``/check``, or `ensure_verified`).

        This is the only path that moves ``last_check_status`` off ``"unknown"``.
        A server whose check already failed still requires the user to
        re-verify deliberately - neither this nor `ensure_verified` retries
        automatically. Also persists the per-check detail (``last_check_checks``)
        so it survives a page refresh.
        """
        remote_server = await self.repo.get_by_id(remote_server_id)
        if remote_server is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))

        reason_code = result.blocking_failures[0].reason_code if result.blocking_failures else None
        partial_update = {
            "last_check_status": "healthy" if result.passed else "degraded",
            "last_check_at": result.checked_at,
            "last_check_latency_ms": result.latency_ms,
            "last_check_reason_code": reason_code,
            "last_check_checks": [check.model_dump(mode="json") for check in result.checks],
        }
        return await self.repo.update(remote_server, partial_update)

    async def ensure_verified(
        self, remote_server_id: UUID, *, protocol_version: int = DEFAULT_PROTOCOL_VERSION
    ) -> RemoteServer:
        """Run Tier 2 preflight automatically the first time a server is used, then return it.

        Job submission trusts the persisted ``last_check_status``: a server
        whose last explicit ``/check`` failed still requires the user to
        re-run it deliberately. But ``"unknown"`` just means nobody has run
        the one-time verification yet, so this runs it inline instead of
        rejecting the job.
        """
        remote_server = await self.get_remote_server(remote_server_id)
        if remote_server.last_check_status != "unknown":
            return remote_server
        result = await run_tier2_preflight(remote_server, protocol_version=protocol_version)
        return await self.record_check_result(remote_server_id, result)

    async def get_status(self, remote_server_id: UUID, settings: Settings | None = None) -> RemoteServerStatus:
        """Return one server's live status: a throttled Tier 1 probe plus in-use/GPU-wait state.

        The Tier 1 probe is shared and cached per `settings.ssh_preflight_throttle_s`
        so concurrent UI polling never dials out more than once per window.
        ``in_use_by_job_id``/``waiting_for_gpu`` are cheap DB reads and are
        never cached, so they reflect the current job the moment it changes.
        """
        settings = settings or get_settings()
        server = await self.get_remote_server(remote_server_id)

        active = await JobProvisioningRepository(self.session).get_active_for_server(remote_server_id)
        in_use_by_job_id = active.job_id if active is not None else None
        waiting_for_gpu = False
        if active is not None:
            job = await JobRepository(self.session).get_by_id(active.job_id)
            waiting_for_gpu = job is not None and self._is_waiting_for_gpu(job)

        result = await self._tier1_status(server, settings)
        status_value = "healthy" if result.passed else "degraded"
        reason_code = result.blocking_failures[0].reason_code if result.blocking_failures else None

        return RemoteServerStatus(
            remote_server_id=remote_server_id,
            status=status_value,
            device_type=server.device_type.value,
            checks=result.checks,
            checked_at=result.checked_at,
            latency_ms=result.latency_ms,
            reason_code=reason_code,
            in_use_by_job_id=in_use_by_job_id,
            waiting_for_gpu=waiting_for_gpu,
        )

    @staticmethod
    def _is_waiting_for_gpu(job: object) -> bool:
        """True when a job's last reported phase is waiting on a busy remote GPU.

        Reads the structured phase descriptor at `extra_info["phase"]`; per
        the `extra_info` contract, this is display-only and never gates a
        workflow decision.
        """
        extra_info = getattr(job, "extra_info", None)
        if not isinstance(extra_info, dict):
            return False
        phase = extra_info.get("phase")
        if not isinstance(phase, dict):
            return False
        return phase.get("state") == PhaseState.WAITING.value

    async def _tier1_status(self, server: RemoteServer, settings: Settings) -> PreflightResult:
        """Return a Tier 1 preflight result, coalesced and throttled per server."""
        cached = _status_cache.get(server.id)
        if cached is not None:
            cached_at, cached_result = cached
            if monotonic() - cached_at < settings.ssh_preflight_throttle_s:
                return cached_result

        task = _status_checks.get(server.id)
        if task is None:
            task = asyncio.ensure_future(
                asyncio.wait_for(run_tier1_preflight(server), timeout=settings.ssh_preflight_timeout_s)
            )
            _status_checks[server.id] = task

            def _clear_inflight(done: asyncio.Task[PreflightResult], server_id: UUID = server.id) -> None:
                if _status_checks.get(server_id) is done:
                    del _status_checks[server_id]

            task.add_done_callback(_clear_inflight)

        # Shield so a cancelled caller never cancels the shared probe for
        # other concurrent pollers.
        result = await asyncio.shield(task)
        _status_cache[server.id] = (monotonic(), result)
        return result
