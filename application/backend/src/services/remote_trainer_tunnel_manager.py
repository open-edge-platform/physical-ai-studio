# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standing SSH port-forward tunnels for direct-URL remote trainers.

A direct trainer's URL is dialed directly by every caller (health checks,
`RemoteTrainingBackend`) - nothing in this codepath rewrites it. When a
trainer is configured with an ``ssh_host_alias``, this module keeps one
`SshTunnel` open per trainer for as long as Studio runs, forwarding
`ssh_local_port` on the studio host to `ssh_remote_port` on the SSH host's own
loopback interface. The trainer's `url` is expected to already point at that
local port (e.g. ``http://127.0.0.1:<ssh_local_port>``), so every other
codepath keeps working unmodified once the tunnel is up.

Gated the same way as every other SSH capability: `sync_tunnel` is a no-op
whenever `get_ssh_feature_availability().active` is False, so a trainer saved
with tunnel config while the feature was on never dials SSH once it's off
(e.g. after a restart with a changed bind host).

Module-level rather than a class: tunnels are a process-wide resource (one
per studio process, not one per caller), so there is nothing a second
instance would ever mean.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING

from loguru import logger

from core.security import get_ssh_feature_availability
from services.ssh.connection import DirectTarget
from services.ssh.transport import SshTransport, open_transport
from services.ssh.tunnel import SshTunnel
from settings import get_settings

if TYPE_CHECKING:
    from uuid import UUID

    from schemas.remote_trainer import RemoteTrainer

_tunnels: dict[UUID, SshTunnel] = {}
_lock = asyncio.Lock()


async def sync_tunnel(remote_trainer: RemoteTrainer, accepted_host_key_fingerprint: str | None = None) -> None:
    """Open, replace, or close this trainer's tunnel to match its current config.

    Connection failures propagate to create and update requests so they cannot
    report success without opening the configured local port.
    """
    async with _lock:
        await _close_locked(remote_trainer.id)
        if remote_trainer.ssh_host_alias is None and remote_trainer.ssh_connection is None:
            return
        if not get_ssh_feature_availability().active:
            logger.warning(
                "Not opening SSH tunnel for trainer '{}': the SSH remote-trainer feature is unavailable",
                remote_trainer.name,
            )
            return
        await _open_locked(remote_trainer, accepted_host_key_fingerprint)


async def stop_tunnel(remote_trainer_id: UUID) -> None:
    """Close and drop a trainer's tunnel, e.g. because it was deleted."""
    async with _lock:
        await _close_locked(remote_trainer_id)


async def start_all(remote_trainers: list[RemoteTrainer]) -> None:
    """Open a tunnel for every configured trainer that wants one.

    Called once at studio startup. Best-effort per trainer: one trainer's
    SSH host being unreachable must never keep another trainer's tunnel from
    opening. Feature-gating is left entirely to `sync_tunnel`.
    """
    for remote_trainer in remote_trainers:
        if remote_trainer.ssh_host_alias is not None or remote_trainer.ssh_connection is not None:
            try:
                await sync_tunnel(remote_trainer)
            except Exception as error:
                if remote_trainer.ssh_host_alias is not None:
                    connection_name = remote_trainer.ssh_host_alias
                elif remote_trainer.ssh_connection is not None:
                    connection_name = remote_trainer.ssh_connection.hostname
                else:
                    continue
                logger.warning(
                    "Failed to restore SSH tunnel for trainer '{}' via '{}': {}",
                    remote_trainer.name,
                    connection_name,
                    error,
                )


async def stop_all() -> None:
    """Close every open tunnel. Called once at studio shutdown."""
    async with _lock:
        for remote_trainer_id in list(_tunnels):
            await _close_locked(remote_trainer_id)


async def _open_locked(remote_trainer: RemoteTrainer, accepted_host_key_fingerprint: str | None = None) -> None:
    settings = get_settings()
    alias = remote_trainer.ssh_host_alias
    connection = remote_trainer.ssh_connection
    remote_port = remote_trainer.ssh_remote_port
    local_port = remote_trainer.ssh_local_port
    if (alias is None and connection is None) or remote_port is None:
        return
    if alias is not None:
        connection_name = alias
        open_ssh_transport = partial(
            open_transport,
            alias,
            settings,
            accepted_host_key_fingerprint=accepted_host_key_fingerprint,
        )
    elif connection is not None:
        connection_name = connection.hostname
        open_ssh_transport = partial(
            SshTransport,
            DirectTarget(
                hostname=connection.hostname,
                port=connection.port,
                username=connection.user,
                identity_file=connection.identity_file,
            ),
            settings,
            accepted_host_key_fingerprint=accepted_host_key_fingerprint,
        )
    else:
        return
    tunnel = SshTunnel(
        open_ssh_transport,
        "127.0.0.1",
        remote_port,
        settings,
        local_port=local_port,
    )
    await tunnel.open()
    _tunnels[remote_trainer.id] = tunnel
    logger.info(
        "SSH tunnel open for trainer '{}': 127.0.0.1:{} -> {} (127.0.0.1:{})",
        remote_trainer.name,
        tunnel.local_port,
        connection_name,
        remote_port,
    )


async def _close_locked(remote_trainer_id: UUID) -> None:
    tunnel = _tunnels.pop(remote_trainer_id, None)
    if tunnel is not None:
        await tunnel.close()
