# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enforces the SSH remote-trainer feature's fail-closed network-exposure policy.

The SSH remote-trainer feature has no authentication model: anyone who can
reach the backend's API can execute code as root on every registered server,
and a compromised backend process can reach every identity in the user's SSH
agent, not just the registered servers. It is safe only on a single-user
localhost workstation.

This module is the runtime enforcement of that boundary, on top of - not a
replacement for - the operator turning the feature on in the first place
(`Settings.ssh_remote_trainer_enabled`, off by default) and the documentation
in `docs/ssh-remote-trainer.md`. Even when explicitly enabled, the feature
fails closed if the backend is bound to anything but a loopback address.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from settings import get_settings

if TYPE_CHECKING:
    from settings import Settings

_LOOPBACK_HOSTNAMES = frozenset({"localhost"})


def is_loopback_host(host: str) -> bool:
    """True when `host`, as an ASGI server bind address, reaches only this machine.

    `host` must be the literal bind argument the server was (or will be) told
    to listen on - not a value resolved during a request, and not the
    documented default. `0.0.0.0`/`::` (bind to every interface) are never
    loopback, regardless of what a DNS name for the same machine might also
    resolve to.

    A hostname is loopback only when *every* address it resolves to is
    loopback: a name that resolves to a loopback address on one interface and
    a routable one on another is not safely restricted to this machine.

    Args:
        host: The literal bind host/interface string.

    Returns:
        True when every address `host` can be reached at is loopback-only.
        False for an unresolvable hostname - failing closed, since a bind
        target that cannot be verified is not the same as one verified safe.
    """
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass

    if host in _LOOPBACK_HOSTNAMES:
        return True

    try:
        resolved = socket.getaddrinfo(host, None)
    except OSError:
        return False
    if not resolved:
        return False
    return all(ipaddress.ip_address(info[4][0]).is_loopback for info in resolved)


@dataclass(frozen=True, slots=True)
class SshFeatureAvailability:
    """Whether the SSH remote-trainer feature is safe to serve right now, and why not if it isn't.

    Attributes:
        enabled: The operator turned the feature on via
            `Settings.ssh_remote_trainer_enabled`.
        network_exposed: The backend is bound to a non-loopback address.
            Meaningless (always False) when `enabled` is False - an unreachable
            feature exposes nothing regardless of the bind address.
        reason: A user-facing explanation for why the feature is unavailable
            despite being enabled. Names no host alias, container, or other
            registered-server detail - this can reach a pre-auth log line or
            an unauthenticated status endpoint.
    """

    enabled: bool
    network_exposed: bool
    reason: str | None = None

    @property
    def active(self) -> bool:
        """True only when the feature is both turned on and safe to expose."""
        return self.enabled and not self.network_exposed


def evaluate_ssh_feature_availability(settings: Settings, *, bind_host: str | None = None) -> SshFeatureAvailability:
    """Evaluate the SSH feature's availability from settings and the actual bind host.

    Args:
        settings: Application settings.
        bind_host: The literal host argument the ASGI server was actually
            started with, when a caller knows it directly (e.g. an explicit
            `--host` CLI flag). Falls back to `settings.host` - the two can
            disagree if a caller overrides the bind address without also
            updating `Settings`.

    Returns:
        The evaluated availability. `network_exposed` is always False when
        the feature is off: there is nothing to expose.
    """
    if not settings.ssh_remote_trainer_enabled:
        return SshFeatureAvailability(enabled=False, network_exposed=False)

    host = bind_host if bind_host is not None else settings.host
    if is_loopback_host(host):
        return SshFeatureAvailability(enabled=True, network_exposed=False)

    return SshFeatureAvailability(
        enabled=True,
        network_exposed=True,
        reason=(
            "the backend is bound to a non-loopback address and the SSH remote-trainer feature has no "
            "authentication model; it is only safe on a single-user localhost workstation"
        ),
    )


@lru_cache
def get_ssh_feature_availability() -> SshFeatureAvailability:
    """Return the cached SSH feature availability for this process."""
    return evaluate_ssh_feature_availability(get_settings())


__all__ = [
    "SshFeatureAvailability",
    "evaluate_ssh_feature_availability",
    "get_ssh_feature_availability",
    "is_loopback_host",
]
