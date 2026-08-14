# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration test: real SSH transport/preflight against a containerized ``sshd``.

Every other SSH test in this suite fakes the transport (`services.ssh.transport`
is reached through `transport_factory`/`open_transport` specifically so it can
be substituted). That is the right default - it keeps the unit suite hermetic
and fast - but it also means nothing in the suite proves that
`SshTransport.connect()` actually negotiates a real SSH handshake, verifies a
real host key, and authenticates with a real key pair against a real server.

This module closes that gap: it starts a throwaway Alpine container running a
real ``sshd``, generates a fresh ed25519 key pair, writes a purpose-built
``ssh_config``/``known_hosts`` pointed at the container, and drives the real
`SshTransport` and `run_tier1_preflight` against it - no mocks. Skipped
(not failed) when Docker is unavailable, since this is the only test in the
backend suite with that dependency.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import asyncssh
import pytest

from exceptions import SshHostKeyUnknownError
from schemas.hardware import DeviceType
from schemas.remote_server import RemoteServer
from schemas.ssh_preflight import CheckKey, CheckOutcome
from services.ssh import preflight as preflight_module
from services.ssh import transport as transport_module
from services.ssh.preflight import run_tier1_preflight
from services.ssh.transport import SshTransport
from settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_ALIAS = "physicalai-containerized-sshd-test"
_CONTAINER_USER = "trainer"
_CONTAINER_IMAGE = "alpine:3.20"

# Installs and starts a real `sshd`, key-only for `_CONTAINER_USER`. The account
# is given a password (never used - `PasswordAuthentication no` is set below)
# only because Alpine's `adduser -D` otherwise leaves it "locked", which `sshd`
# refuses to authenticate at all regardless of key.
_SSHD_ENTRYPOINT = (
    "set -e && apk add --no-cache openssh >/dev/null 2>&1 && "
    "ssh-keygen -A >/dev/null 2>&1 && "
    f"adduser -D -s /bin/sh {_CONTAINER_USER} && "
    f"echo '{_CONTAINER_USER}:{_CONTAINER_USER}' | chpasswd && "
    f"mkdir -p /home/{_CONTAINER_USER}/.ssh && "
    f"cp /keys/authorized_keys /home/{_CONTAINER_USER}/.ssh/authorized_keys && "
    f"chown -R {_CONTAINER_USER}:{_CONTAINER_USER} /home/{_CONTAINER_USER}/.ssh && "
    f"chmod 700 /home/{_CONTAINER_USER}/.ssh && "
    f"chmod 600 /home/{_CONTAINER_USER}/.ssh/authorized_keys && "
    "exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no -o PermitRootLogin=no"
)


@dataclass
class _ContainerizedSshd:
    """A running ``sshd`` container plus the `Settings` that reach it."""

    container_id: str
    port: int
    settings: Settings


def _read_banner(sock: socket.socket, size: int = 4) -> bytes:
    """Read exactly *size* bytes (or less, on EOF) from *sock*."""
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _wait_for_ssh_banner(port: int, timeout_s: float = 30.0) -> None:
    """Block until the container's ``sshd`` is actually speaking the protocol.

    Docker's published port accepts TCP the moment the container starts - well
    before ``apk add`` finishes installing `openssh` inside it - so a bare TCP
    connect is not a readiness signal. Only the SSH version banner is.
    """
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
                sock.settimeout(2)
                if _read_banner(sock).startswith(b"SSH-"):
                    return
        except OSError as error:
            last_error = error
        time.sleep(0.5)
    raise TimeoutError(f"sshd on port {port} never presented an SSH banner: {last_error}")


@pytest.fixture
def containerized_sshd(tmp_path: Path) -> Iterator[_ContainerizedSshd]:
    """Start a throwaway container running a real ``sshd`` and wire `Settings` at it.

    Yields `Settings` with `ssh_config_path`/`ssh_known_hosts_path` pointed at a
    generated config and a `known_hosts` pre-seeded with the container's real
    host key, so `SshTransport` resolves and connects to it exactly as it would
    a real remote server - through the same alias-resolution and
    known-hosts-matching code path, not a shortcut around it.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker is not available in this environment")

    private_key_path = tmp_path / "id_ed25519"
    authorized_keys_path = tmp_path / "authorized_keys"
    key = asyncssh.generate_private_key("ssh-ed25519")
    key.write_private_key(str(private_key_path))
    private_key_path.chmod(0o600)
    authorized_keys_path.write_text(key.export_public_key().decode().strip() + "\n")

    run_result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            "127.0.0.1::22",
            "-v",
            f"{authorized_keys_path}:/keys/authorized_keys:ro",
            _CONTAINER_IMAGE,
            "sh",
            "-c",
            _SSHD_ENTRYPOINT,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if run_result.returncode != 0:
        pytest.skip(f"could not start the containerized sshd fixture: {run_result.stderr.strip()}")
    container_id = run_result.stdout.strip()

    try:
        port_result = subprocess.run(
            ["docker", "port", container_id, "22/tcp"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        port = int(port_result.stdout.strip().rsplit(":", 1)[1])

        _wait_for_ssh_banner(port)

        host_key_result = subprocess.run(
            ["docker", "exec", container_id, "cat", "/etc/ssh/ssh_host_ed25519_key.pub"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        host_public_key = host_key_result.stdout.strip()

        ssh_config_path = tmp_path / "ssh_config"
        ssh_config_path.write_text(
            f"Host {_ALIAS}\n"
            "    HostName 127.0.0.1\n"
            f"    Port {port}\n"
            f"    User {_CONTAINER_USER}\n"
            f"    IdentityFile {private_key_path}\n"
        )

        # Every plausible pattern `asyncssh`'s known_hosts matcher might look
        # up the same key under (bare address, bracketed address:port, and the
        # alias itself), rather than assuming which one it is handed.
        known_hosts_path = tmp_path / "known_hosts"
        known_hosts_path.write_text(
            f"127.0.0.1 {host_public_key}\n[127.0.0.1]:{port} {host_public_key}\n{_ALIAS} {host_public_key}\n"
        )

        settings = Settings(
            SSH_CONFIG_PATH=str(ssh_config_path),
            SSH_KNOWN_HOSTS_PATH=str(known_hosts_path),
            SSH_CONNECT_TIMEOUT_S=10.0,
            SSH_COMMAND_TIMEOUT_S=10.0,
            SSH_PREFLIGHT_TIMEOUT_S=30.0,
        )

        yield _ContainerizedSshd(container_id=container_id, port=port, settings=settings)
    finally:
        subprocess.run(["docker", "stop", "-t", "2", container_id], capture_output=True, timeout=15, check=False)


def _make_remote_server() -> RemoteServer:
    return RemoteServer(id=uuid4(), name="containerized-sshd-test", ssh_host_alias=_ALIAS, device_type=DeviceType.CUDA)


@pytest.mark.anyio
async def test_transport_connects_and_runs_a_command_against_real_sshd(
    containerized_sshd: _ContainerizedSshd,
) -> None:
    """`SshTransport` performs a real connect, host-key check, auth, and exec."""
    async with SshTransport(_ALIAS, containerized_sshd.settings) as transport:
        result = await transport.run_command(["echo", "hello-from-container"])

    assert result.ok
    assert result.first_line() == "hello-from-container"


@pytest.mark.anyio
async def test_transport_rejects_an_untrusted_host_key(containerized_sshd: _ContainerizedSshd, tmp_path: Path) -> None:
    """An empty `known_hosts` must fail closed as unknown, never connect anyway."""
    empty_known_hosts = tmp_path / "empty_known_hosts"
    empty_known_hosts.write_text("")
    settings = containerized_sshd.settings.model_copy(update={"ssh_known_hosts_path": empty_known_hosts})

    with pytest.raises(SshHostKeyUnknownError):
        async with SshTransport(_ALIAS, settings):
            pass


@pytest.mark.anyio
async def test_tier1_preflight_passes_ssh_layer_checks_against_real_sshd(
    containerized_sshd: _ContainerizedSshd, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real Tier 1 preflight resolves, connects, verifies, and authenticates.

    `DOCKER_USABLE`/`DRIVER_PRESENT`/`GPU_FREE` are expected to fail or skip:
    the container has no Docker or GPU stack, and asserting those bodies is
    the job of `tests/services/ssh/test_preflight.py`'s fake-transport unit
    tests. This test's only claim is that the SSH layer itself - alias
    resolution, the real handshake, host-key verification, and
    authentication - works end to end against a real server.
    """
    monkeypatch.setattr(preflight_module, "get_settings", lambda: containerized_sshd.settings)
    monkeypatch.setattr(transport_module, "get_settings", lambda: containerized_sshd.settings)

    result = await run_tier1_preflight(_make_remote_server())

    checks_by_key = {check.key: check for check in result.checks}
    for key in (CheckKey.ALIAS_RESOLVED, CheckKey.REACHABLE, CheckKey.HOST_KEY_VERIFIED, CheckKey.AUTHENTICATED):
        assert checks_by_key[key].outcome == CheckOutcome.PASSED, checks_by_key[key]
