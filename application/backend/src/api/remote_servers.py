# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Remote-server administration and verify-on-save API.

Tier 1 preflight gates create/update with a bounded timeout, run against a
throwaway candidate that is never persisted if a blocking check fails. Tier 2
(registry pull, signature policy, in-container device probe) is only ever
triggered by the explicit ``/check`` action - never inline in a save - because
it can pull multiple gigabytes.
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, status

from api.dependencies import RemoteServerServiceDep, SettingsDep, get_remote_server_id
from exceptions import BaseException as StudioBaseException
from exceptions import (
    RemoteServerPreflightError,
    SshAgentRequiredError,
    SshAuthenticationError,
    SshConnectionError,
    SshHostAliasNotFoundError,
    SshHostKeyMismatchError,
    SshHostKeyUnknownError,
)
from schemas.remote_server import RemoteServer, RemoteServerCreate, RemoteServerUpdate, SshHostAliasOption
from schemas.ssh_preflight import PreflightCheck, PreflightResult, RemoteServerStatus
from services import ssh_config_reader
from services.ssh.preflight import run_tier1_preflight, run_tier2_preflight

router = APIRouter(prefix="/api/remote-servers", tags=["Remote servers"])

# `run_tier1_preflight` never raises: every SSH failure it hits becomes a FAILED
# check carrying one of these reason codes (see `services/ssh/preflight.py`).
# Mapping the credential-adjacent ones back to their dedicated exception here
# is what gives each one its own `error_code` in the API response instead of
# every save failure collapsing onto the generic `remote_server_preflight_failed`.
_REASON_CODE_TO_ERROR: dict[str, Callable[[str], StudioBaseException]] = {
    "alias_not_found": SshHostAliasNotFoundError,
    "host_key_unknown": SshHostKeyUnknownError,
    "host_key_mismatch": SshHostKeyMismatchError,
    "ssh_agent_required": SshAgentRequiredError,
    "authentication_failed": SshAuthenticationError,
    "unreachable": lambda alias: SshConnectionError(alias, reason="unreachable"),
}


def _actionable_error(alias: str, blocking_failures: list[PreflightCheck]) -> StudioBaseException | None:
    """Return the dedicated exception for the first credential-adjacent failure.

    Only alias/connection/host-key/auth/agent failures get their own exception;
    everything else (Docker, disk, driver, registry) stays the generic
    `RemoteServerPreflightError` below, since those have no separate error code.
    """
    for check in blocking_failures:
        if check.reason_code and (factory := _REASON_CODE_TO_ERROR.get(check.reason_code)):
            return factory(alias)
    return None


async def _gate_on_tier1(candidate: RemoteServer, settings: SettingsDep) -> None:
    """Run Tier 1 preflight against a throwaway candidate and raise if it fails.

    Never persists anything: the caller passes an in-memory ``RemoteServer``
    that is only saved once this returns without raising.
    """
    result = await asyncio.wait_for(run_tier1_preflight(candidate), timeout=settings.ssh_preflight_timeout_s)
    if result.passed:
        return

    if actionable := _actionable_error(candidate.ssh_host_alias, result.blocking_failures):
        raise actionable

    failures = [f"{check.key.value}: {check.reason_code}" for check in result.blocking_failures]
    raise RemoteServerPreflightError("Remote server failed required checks", failures=failures)


@router.get("/aliases")
async def list_ssh_host_aliases(settings: SettingsDep) -> list[SshHostAliasOption]:
    """Return every selectable SSH host alias for the create/edit form."""
    return ssh_config_reader.list_host_aliases(settings.ssh_config_path)


@router.get("")
async def list_remote_servers(remote_server_service: RemoteServerServiceDep) -> list[RemoteServer]:
    """Return every registered SSH-provisioned training server."""
    return await remote_server_service.list_remote_servers()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_remote_server(
    config: RemoteServerCreate,
    remote_server_service: RemoteServerServiceDep,
    settings: SettingsDep,
) -> RemoteServer:
    """Persist a new SSH-provisioned training server.

    Tier 1 preflight runs first, against a throwaway candidate that is never
    saved. A blocking failure never touches the database.
    """
    candidate = RemoteServer(id=uuid4(), **config.model_dump())
    await _gate_on_tier1(candidate, settings)
    return await remote_server_service.create_remote_server(config)


@router.patch("/{remote_server_id}")
async def update_remote_server(
    remote_server_id: Annotated[UUID, Depends(get_remote_server_id)],
    update: RemoteServerUpdate,
    remote_server_service: RemoteServerServiceDep,
    settings: SettingsDep,
) -> RemoteServer:
    """Update a registered server's mutable fields.

    Tier 1 preflight runs against the merged candidate before anything is
    persisted, same as create.
    """
    existing = await remote_server_service.get_remote_server(remote_server_id)
    merged = existing.model_copy(update=update.model_dump(exclude_none=True, exclude_unset=True))
    await _gate_on_tier1(merged, settings)
    return await remote_server_service.update_remote_server(remote_server_id, update)


@router.delete("/{remote_server_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_remote_server(
    remote_server_id: Annotated[UUID, Depends(get_remote_server_id)],
    remote_server_service: RemoteServerServiceDep,
) -> None:
    """Delete a registered server."""
    await remote_server_service.delete_remote_server(remote_server_id)


@router.post("/{remote_server_id}/check")
async def check_remote_server(
    remote_server_id: Annotated[UUID, Depends(get_remote_server_id)],
    remote_server_service: RemoteServerServiceDep,
) -> PreflightResult:
    """Explicitly run Tier 2 preflight against a registered server.

    Never runs inline on save: this is the only path that can trigger a
    registry pull and one-shot GPU container probe. Its outcome is also the
    only thing that ever moves the server's persisted ``last_check_status``
    off ``"unknown"`` - a live Tier 1 read from ``/status`` never does.
    """
    server = await remote_server_service.get_remote_server(remote_server_id)
    result = await run_tier2_preflight(server)
    await remote_server_service.record_check_result(remote_server_id, result)
    return result


@router.get("/{remote_server_id}/status")
async def get_remote_server_status(
    remote_server_id: Annotated[UUID, Depends(get_remote_server_id)],
    remote_server_service: RemoteServerServiceDep,
    settings: SettingsDep,
) -> RemoteServerStatus:
    """Return structured status for one registered server.

    A live Tier 1 check always runs (per-connection throttling coordinated
    with the GPU-busy re-check is not yet wired up standalone here - see the
    TODO below); ``in_use_by_job_id``/``waiting_for_gpu`` are not-yet-implemented
    placeholders until job provisioning exists.

    TODO: throttle the live Tier 1 call using ``settings.ssh_preflight_throttle_s``
    against the server's ``last_check_at`` once connection-level coordination
    with the GPU-busy re-check (also throttled) is implemented; for now this
    always dials out, which is a known simplification.
    """
    server = await remote_server_service.get_remote_server(remote_server_id)
    try:
        result = await asyncio.wait_for(run_tier1_preflight(server), timeout=settings.ssh_preflight_timeout_s)
    except TimeoutError as exc:
        raise SshConnectionError(server.ssh_host_alias, reason="timed_out") from exc

    status_value = "healthy" if result.passed else "degraded"
    reason_code = None
    if result.blocking_failures:
        reason_code = result.blocking_failures[0].reason_code

    return RemoteServerStatus(
        remote_server_id=remote_server_id,
        status=status_value,
        device_type=server.device_type.value,
        checks=result.checks,
        checked_at=result.checked_at or datetime.now(UTC),
        latency_ms=result.latency_ms,
        reason_code=reason_code,
        in_use_by_job_id=None,
        waiting_for_gpu=False,
    )
