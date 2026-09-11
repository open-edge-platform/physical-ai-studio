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
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Path, status

from api.dependencies import RemoteServerServiceDep, SettingsDep, get_remote_server_id, require_ssh_feature_active
from core.security import SshFeatureAvailability, get_ssh_feature_availability
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
from schemas.remote_server import (
    SSH_HOST_ALIAS_PATTERN,
    DeviceTypeDetection,
    RemoteServer,
    RemoteServerCreate,
    RemoteServerUpdate,
    SshHostAliasOption,
)
from schemas.ssh_preflight import CheckKey, PreflightCheck, PreflightResult, RemoteServerStatus
from services import ssh_config_reader
from services.ssh import preflight
from services.ssh.preflight import run_tier1_preflight, run_tier2_preflight

# The whole administration surface fails closed behind `require_ssh_feature_active`
# except `/feature-status` itself, which must stay reachable to explain *why*
# everything else is unavailable (see its docstring below), so that dependency
# is applied per-route below rather than at the router level.
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


# Human-readable labels for Tier 1 checks, used to describe *which* check
# failed rather than only reporting that the save was rejected. Kept in sync
# with the UI's own `checkLabel` map (`remote-server-status-utils.ts`), which
# labels the same `CheckKey` values for the status/verification cards.
_CHECK_LABELS: dict[CheckKey, str] = {
    CheckKey.ALIAS_RESOLVED: "SSH host alias resolves",
    CheckKey.REACHABLE: "Reachable",
    CheckKey.AUTHENTICATED: "Authenticated",
    CheckKey.HOST_KEY_VERIFIED: "Host key verified",
    CheckKey.DOCKER_USABLE: "Docker available",
    CheckKey.DISK_SPACE: "Storage available",
    CheckKey.DRIVER_PRESENT: "GPU driver present",
    CheckKey.REGISTRY_REACHABLE: "Registry reachable",
    CheckKey.GPU_FREE: "GPU free",
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


def _describe_failure(check: PreflightCheck) -> str:
    """Human-readable summary of one failed check, e.g. ``"Storage available: 3.2 GiB free, 20.0 GiB required"``."""
    label = _CHECK_LABELS.get(check.key, check.key.value.replace("_", " "))
    return f"{label}: {check.detail}" if check.detail else label


async def _gate_on_tier1(candidate: RemoteServer, settings: SettingsDep) -> None:
    """Run Tier 1 preflight against a throwaway candidate and raise if it fails.

    Never persists anything: the caller passes an in-memory ``RemoteServer``
    that is only saved once this returns without raising.
    """
    try:
        result = await asyncio.wait_for(run_tier1_preflight(candidate), timeout=settings.ssh_preflight_timeout_s)
    except TimeoutError as exc:
        # `run_tier1_preflight` itself never raises, so this only fires if the
        # whole probe outlives its own timeout budget (e.g. a wedged transport).
        # Mapped the same way `/status` maps it, so create/update never surfaces
        # a bare 500 for a failure mode the rest of this API already has an
        # actionable error for.
        raise SshConnectionError(candidate.ssh_host_alias, reason="timed_out") from exc
    if result.passed:
        return

    if actionable := _actionable_error(candidate.ssh_host_alias, result.blocking_failures):
        raise actionable

    # Names each failed check by its human label plus its own detail (e.g. free
    # disk space, or the raw Docker/driver error) instead of just the reason code.
    failures = [_describe_failure(check) for check in result.blocking_failures]
    message = "Could not save: " + "; ".join(failures)
    raise RemoteServerPreflightError(message, failures=failures)


@router.get("/aliases", dependencies=[Depends(require_ssh_feature_active)])
async def list_ssh_host_aliases(settings: SettingsDep) -> list[SshHostAliasOption]:
    """Return every selectable SSH host alias for the create/edit form."""
    return ssh_config_reader.list_host_aliases(settings.ssh_config_path)


@router.get("/aliases/{alias}/device-type", dependencies=[Depends(require_ssh_feature_active)])
async def detect_device_type(
    alias: Annotated[str, Path(min_length=1, max_length=255, pattern=SSH_HOST_ALIAS_PATTERN)],
    settings: SettingsDep,
) -> DeviceTypeDetection:
    """Best-effort device-type autodetection, to prefill the add-target form.

    Never raises for a driverless or unreachable host: it reports
    ``device_type=None`` with a ``reason_code`` instead, so the form falls back
    to asking the user to pick manually rather than blocking on this call.
    """
    try:
        device_type, method, reason_code = await asyncio.wait_for(
            preflight.detect_device_type(alias), timeout=settings.ssh_preflight_timeout_s
        )
    except TimeoutError:
        return DeviceTypeDetection(reason_code="timed_out")
    return DeviceTypeDetection(device_type=device_type, method=method, reason_code=reason_code)


@router.get("/feature-status")
async def get_feature_status() -> SshFeatureAvailability:
    """Report whether the SSH remote-trainer feature is currently active.

    Unauthenticated by design (no `require_ssh_feature_active` dependency):
    the UI needs this to explain *why* the feature is unavailable, which would
    be circular if reading the status itself required the feature to be
    active. Safe to expose: `reason`, when set, already names no host alias,
    container, or other registered-server detail.
    """
    return get_ssh_feature_availability()


@router.get("", dependencies=[Depends(require_ssh_feature_active)])
async def list_remote_servers(remote_server_service: RemoteServerServiceDep) -> list[RemoteServer]:
    """Return every registered SSH-provisioned training server."""
    return await remote_server_service.list_remote_servers()


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_ssh_feature_active)])
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


@router.patch("/{remote_server_id}", dependencies=[Depends(require_ssh_feature_active)])
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


@router.delete(
    "/{remote_server_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_ssh_feature_active)],
)
async def delete_remote_server(
    remote_server_id: Annotated[UUID, Depends(get_remote_server_id)],
    remote_server_service: RemoteServerServiceDep,
) -> None:
    """Delete a registered server."""
    await remote_server_service.delete_remote_server(remote_server_id)


@router.post("/{remote_server_id}/check", dependencies=[Depends(require_ssh_feature_active)])
async def check_remote_server(
    remote_server_id: Annotated[UUID, Depends(get_remote_server_id)],
    remote_server_service: RemoteServerServiceDep,
) -> PreflightResult:
    """Explicitly run Tier 2 preflight against a registered server.

    The only other path that can trigger a registry pull and one-shot GPU
    container probe is `RemoteServerService.ensure_verified`, called once,
    automatically, the first time a server with ``last_check_status ==
    "unknown"`` is selected for a job - so submitting a job never has to
    reject a server for the sole reason that nobody happened to click
    "Test connection" first. This endpoint is what a user reaches for
    afterward to re-verify a server, or to verify one before ever submitting
    a job against it.
    """
    server = await remote_server_service.get_remote_server(remote_server_id)
    result = await run_tier2_preflight(server)
    await remote_server_service.record_check_result(remote_server_id, result)
    return result


@router.get("/{remote_server_id}/status", dependencies=[Depends(require_ssh_feature_active)])
async def get_remote_server_status(
    remote_server_id: Annotated[UUID, Depends(get_remote_server_id)],
    remote_server_service: RemoteServerServiceDep,
) -> RemoteServerStatus:
    """Return structured status for one registered server.

    The Tier 1 check, and ``in_use_by_job_id``/``waiting_for_gpu``, are computed
    by `RemoteServerService.get_status`: the Tier 1 probe is coalesced and
    throttled per server (shared across concurrent pollers, at most one dial per
    `settings.ssh_preflight_throttle_s`), while the in-use/GPU-wait fields are
    always-fresh DB reads of the server's currently provisioning/training job,
    if any.
    """
    try:
        return await remote_server_service.get_status(remote_server_id)
    except TimeoutError as exc:
        server = await remote_server_service.get_remote_server(remote_server_id)
        raise SshConnectionError(server.ssh_host_alias, reason="timed_out") from exc
