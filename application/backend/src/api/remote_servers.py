# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Read and add SSH host aliases for managed remote trainers.

Aliases are ``Host`` entries in the user's ``~/.ssh/config`` and are
selectable in the remote-trainer form.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Header

from api.dependencies import SettingsDep, require_ssh_feature_active
from core.security import SshFeatureAvailability, get_ssh_feature_availability
from schemas.remote_server import SshHostAliasCreate, SshHostAliasOption
from services import ssh_config_reader, ssh_config_writer

# The whole administration surface fails closed behind `require_ssh_feature_active`
# except `/feature-status` itself, which must stay reachable to explain *why*
# everything else is unavailable (see its docstring below), so that dependency
# is applied per-route below rather than at the router level.
router = APIRouter(prefix="/api/remote-servers", tags=["SSH hosts"])


@router.get("/aliases", dependencies=[Depends(require_ssh_feature_active)])
async def list_ssh_host_aliases(settings: SettingsDep) -> list[SshHostAliasOption]:
    """Return every selectable SSH host alias for the remote-trainer form."""
    return ssh_config_reader.list_host_aliases(settings.ssh_config_path)


@router.post("/aliases", status_code=201, dependencies=[Depends(require_ssh_feature_active)])
async def create_ssh_host_alias(
    config: SshHostAliasCreate,
    settings: SettingsDep,
    accepted_host_key_fingerprint: Annotated[str | None, Header()] = None,
) -> SshHostAliasOption:
    """Append a new Host entry to the user's ``~/.ssh/config`` and return it as a selectable alias.

    For a user who wants to point Studio at a host without hand-editing their
    SSH config first. Verifies the new host is actually reachable before
    returning: a failed verification removes the entry it just wrote rather
    than leaving an unreachable entry in the user's real config. Also rejects
    an alias that already exists rather than editing it - see
    `services.ssh_config_writer` for why.

    ``accepted_host_key_fingerprint`` mirrors the same header on remote-trainer
    create/update: a genuinely new host has no ``known_hosts`` entry yet, so the
    first attempt raises `exceptions.SshHostKeyUnknownError` and the caller
    re-submits once the user confirms the fingerprint it surfaced.
    """
    return await ssh_config_writer.add_verified_host_alias(
        settings.ssh_config_path,
        config,
        settings,
        accepted_host_key_fingerprint=accepted_host_key_fingerprint,
    )


@router.get("/feature-status")
async def get_feature_status() -> SshFeatureAvailability:
    """Report whether the SSH remote-trainer feature is currently active.

    Unauthenticated by design (no `require_ssh_feature_active` dependency):
    the UI needs this to explain *why* the feature is unavailable, which would
    be circular if reading the status itself required the feature to be
    active.
    """
    return get_ssh_feature_availability()
