# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SSH local-forward tunnel and its reconnect behavior."""

from __future__ import annotations

import asyncio

import pytest

from services.ssh.tunnel import SshTunnel
from settings import Settings


class FakeListener:
    """A minimal stand-in for `asyncssh.SSHListener`."""

    def __init__(self, port: int) -> None:
        self._port = port
        self._closed_event = asyncio.Event()
        self.closed = False

    def get_port(self) -> int:
        return self._port

    def close(self) -> None:
        self.closed = True
        self._closed_event.set()

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    def drop(self) -> None:
        """Simulate the remote side dropping the connection (not a deliberate close)."""
        self._closed_event.set()


class FakeTransport:
    """A minimal stand-in for `SshTransport`, tracking connect/close and one listener."""

    def __init__(self, listener: FakeListener, *, fail_connect: bool = False) -> None:
        self.listener = listener
        self.fail_connect = fail_connect
        self.connected = False
        self.closed = False
        self.requested_local_ports: list[int] = []

    async def connect(self) -> None:
        if self.fail_connect:
            raise ConnectionError("simulated connect failure")
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    async def forward_local_port(self, remote_host: str, remote_port: int, local_port: int = 0):
        self.requested_local_ports.append(local_port)
        return self.listener


@pytest.fixture
def settings() -> Settings:
    return Settings(
        SSH_TUNNEL_RECONNECT_BUDGET_S=5,
        SSH_TUNNEL_RECONNECT_BACKOFF_MAX_S=0.01,
    )


async def test_open_forwards_to_an_ephemeral_local_port(settings) -> None:
    listener = FakeListener(port=54321)
    transport = FakeTransport(listener)

    tunnel = SshTunnel(lambda: transport, "127.0.0.1", 8080, settings)
    await tunnel.open()

    assert tunnel.local_port == 54321
    assert transport.connected

    await tunnel.close()
    assert transport.closed
    assert listener.closed


async def test_local_port_raises_before_open(settings) -> None:
    tunnel = SshTunnel(lambda: FakeTransport(FakeListener(1)), "127.0.0.1", 8080, settings)
    with pytest.raises(RuntimeError):
        _ = tunnel.local_port


async def test_dropped_tunnel_reconnects_without_failing(settings) -> None:
    """A dropped tunnel reconnects and resumes, rather than raising to the caller."""
    first_listener = FakeListener(port=1111)
    second_listener = FakeListener(port=2222)
    transports = [FakeTransport(first_listener), FakeTransport(second_listener)]
    calls = iter(transports)

    tunnel = SshTunnel(lambda: next(calls), "127.0.0.1", 8080, settings)
    await tunnel.open()
    assert tunnel.local_port == 1111

    # Simulate the network drop; the watchdog should reconnect on a new transport.
    first_listener.drop()
    for _ in range(50):
        if tunnel.local_port == 2222:
            break
        await asyncio.sleep(0.01)

    assert tunnel.local_port == 2222
    await tunnel.close()


async def test_reconnect_gives_up_after_budget_exhausted() -> None:
    settings = Settings(SSH_TUNNEL_RECONNECT_BUDGET_S=0.05, SSH_TUNNEL_RECONNECT_BACKOFF_MAX_S=0.01)
    listener = FakeListener(port=1111)
    transport = FakeTransport(listener)

    def open_transport():
        return transport if not transport.connected else FakeTransport(listener, fail_connect=True)

    tunnel = SshTunnel(open_transport, "127.0.0.1", 8080, settings)
    await tunnel.open()

    listener.drop()
    await asyncio.sleep(0.2)  # let the watchdog exhaust its reconnect budget

    # The tunnel gave up quietly (logged, not raised) - closing it must still be safe.
    await tunnel.close()


async def test_reconnect_closes_the_previous_transport_it_replaces(settings) -> None:
    """A reconnect must never leak the connection it is replacing."""
    first_listener = FakeListener(port=1111)
    second_listener = FakeListener(port=2222)
    first_transport = FakeTransport(first_listener)
    second_transport = FakeTransport(second_listener)
    transports = iter([first_transport, second_transport])

    tunnel = SshTunnel(lambda: next(transports), "127.0.0.1", 8080, settings)
    await tunnel.open()
    assert tunnel.local_port == 1111

    first_listener.drop()
    for _ in range(50):
        if tunnel.local_port == 2222:
            break
        await asyncio.sleep(0.01)

    assert tunnel.local_port == 2222
    # The dropped connection's transport must have been closed, not left open.
    assert first_transport.closed

    await tunnel.close()


async def test_reconnect_requests_the_same_local_port(settings) -> None:
    """A reconnect asks to re-bind the previously-assigned local port."""
    first_listener = FakeListener(port=1111)
    second_listener = FakeListener(port=1111)
    transports = [FakeTransport(first_listener), FakeTransport(second_listener)]
    calls = iter(transports)

    tunnel = SshTunnel(lambda: next(calls), "127.0.0.1", 8080, settings)
    await tunnel.open()
    assert tunnel.local_port == 1111
    assert transports[0].requested_local_ports == [0]  # first connect: no preference yet

    first_listener.drop()
    for _ in range(50):
        if transports[1].requested_local_ports:
            break
        await asyncio.sleep(0.01)

    # The reconnect requested the same port the caller's cached base URL relies on.
    assert transports[1].requested_local_ports == [1111]
    assert tunnel.local_port == 1111
    await tunnel.close()


async def test_reconnect_falls_back_to_a_fresh_port_if_the_old_one_is_unavailable(settings) -> None:
    """If re-binding the previous port fails, the tunnel falls back rather than giving up."""
    first_listener = FakeListener(port=1111)
    second_listener = FakeListener(port=2222)

    class _PortUnavailableTransport(FakeTransport):
        async def forward_local_port(self, remote_host: str, remote_port: int, local_port: int = 0):
            self.requested_local_ports.append(local_port)
            if local_port != 0:
                raise OSError("simulated address already in use")
            return self.listener

    transports = [FakeTransport(first_listener), _PortUnavailableTransport(second_listener)]
    calls = iter(transports)

    tunnel = SshTunnel(lambda: next(calls), "127.0.0.1", 8080, settings)
    await tunnel.open()
    assert tunnel.local_port == 1111

    first_listener.drop()
    for _ in range(50):
        if tunnel.local_port == 2222:
            break
        await asyncio.sleep(0.01)

    assert tunnel.local_port == 2222
    assert transports[1].requested_local_ports == [1111, 0]
    await tunnel.close()


async def test_forward_failure_closes_the_newly_connected_transport(settings) -> None:
    """A transport that connects but fails to forward must not be leaked."""

    class _FailingForwardTransport(FakeTransport):
        async def forward_local_port(self, remote_host: str, remote_port: int, local_port: int = 0):
            raise RuntimeError("simulated forward failure")

    transport = _FailingForwardTransport(FakeListener(port=1111))

    tunnel = SshTunnel(lambda: transport, "127.0.0.1", 8080, settings)
    with pytest.raises(RuntimeError):
        await tunnel.open()

    assert transport.connected
    assert transport.closed
