# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SSH local-forward tunnel to an SSH-provisioned trainer container.

The tunnel is the only path a studio process ever reaches a provisioned
trainer through: the container publishes on its host's loopback interface
only, and :class:`SshTunnel` forwards a local loopback port to it over the
same SSH connection class the rest of :mod:`services.ssh` uses. A dropped
tunnel reconnects and re-forwards against the still-running container within a
bounded retry budget, so a flaky network path never fails a job that is
otherwise progressing fine.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from time import monotonic
from typing import TYPE_CHECKING, Self

from loguru import logger

from services.ssh.transport import SshTransport
from settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    import asyncssh


class TunnelReconnectExhaustedError(RuntimeError):
    """Raised when a dropped tunnel could not be re-established within its budget."""


class SshTunnel:
    """A local-forward tunnel to one remote `host:port`, with reconnect.

    Use as an async context manager::

        async with SshTunnel(open_transport, "127.0.0.1", 54321, settings) as tunnel:
            ...  # tunnel.local_port is the loopback port to talk to
    """

    def __init__(
        self,
        open_transport: Callable[[], SshTransport],
        remote_host: str,
        remote_port: int,
        settings: Settings,
    ) -> None:
        self._open_transport = open_transport
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._settings = settings
        self._transport: SshTransport | None = None
        self._listener: asyncssh.SSHListener | None = None
        self._local_port: int | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def local_port(self) -> int:
        """The loopback port forwarding to the remote trainer."""
        if self._local_port is None:
            raise RuntimeError("SshTunnel is not open")
        return self._local_port

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def open(self) -> None:
        """Connect and establish the forward. Starts the reconnect watchdog."""
        await self._connect_and_forward()
        self._watchdog_task = asyncio.create_task(self._watch())

    async def _connect_and_forward(self) -> None:
        """Connect a fresh transport and forward, closing whatever this replaces.

        Tears down any previously-open transport/listener before assigning the
        new ones (a reconnect must never leak the connection it is replacing),
        and closes the newly-connected transport itself if `forward_local_port`
        fails (a half-open connect must never leak either).

        On a reconnect (`self._local_port` already set), re-binds to that same
        local port so callers that cached a base URL derived from it keep
        working. Falls back to a fresh ephemeral port only if the previous one
        is not immediately available for re-bind (e.g. still in the OS's
        `TIME_WAIT`) - a best effort, not a guarantee: a caller that needs a
        hard guarantee should still re-read `local_port` after a reconnect.
        """
        await self._close_current_connection()

        transport = self._open_transport()
        preferred_port = self._local_port
        try:
            await transport.connect()
            try:
                listener = await transport.forward_local_port(
                    self._remote_host, self._remote_port, local_port=preferred_port or 0
                )
            except OSError:
                if preferred_port is None:
                    raise
                logger.warning(
                    "SSH tunnel could not re-bind local port {}; a new port will be assigned", preferred_port
                )
                listener = await transport.forward_local_port(self._remote_host, self._remote_port)
        except BaseException:
            await transport.close()
            raise

        self._transport = transport
        self._listener = listener
        self._local_port = listener.get_port()

    async def _close_current_connection(self) -> None:
        """Close and clear whatever transport/listener are currently held."""
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.close()
            with suppress(Exception):
                await listener.wait_closed()

        transport, self._transport = self._transport, None
        if transport is not None:
            await transport.close()

    async def _watch(self) -> None:
        """Reconnect the tunnel if the underlying connection drops.

        Runs for the tunnel's lifetime. A dropped connection is detected by
        the listener's wait_closed() resolving; this never happens on a
        deliberate `close()`, since that cancels this task first.
        """
        while True:
            listener = self._listener
            if listener is None:
                return
            try:
                await listener.wait_closed()
            except asyncio.CancelledError:
                return
            if self._closed:
                return
            logger.warning("SSH tunnel to remote trainer dropped; attempting to reconnect")
            try:
                await self._reconnect_with_backoff()
            except TunnelReconnectExhaustedError:
                logger.error("SSH tunnel reconnect budget exhausted; giving up")
                return

    async def _reconnect_with_backoff(self) -> None:
        """Retry `_connect_and_forward` with exponential backoff, within budget.

        `_connect_and_forward` re-binds to the same `local_port` on success, so
        a caller's cached base URL keeps working across a reconnect in the
        common case. That re-bind is best effort, not guaranteed (see its
        docstring), so a caller that needs a hard guarantee should still read
        `local_port` again after a successful reconnect rather than caching it
        once.
        """
        settings = self._settings
        started = monotonic()
        backoff = 1.0
        last_error: Exception | None = None
        while monotonic() - started < settings.ssh_tunnel_reconnect_budget_s:
            try:
                await self._connect_and_forward()
                logger.info("SSH tunnel reconnected on local port {}", self._local_port)
                return
            except Exception as error:
                last_error = error
                await asyncio.sleep(min(backoff, settings.ssh_tunnel_reconnect_backoff_max_s))
                backoff = min(backoff * 2, settings.ssh_tunnel_reconnect_backoff_max_s)
        raise TunnelReconnectExhaustedError(
            f"Could not reconnect the SSH tunnel within {settings.ssh_tunnel_reconnect_budget_s:.0f}s"
        ) from last_error

    async def close(self) -> None:
        """Cancel the reconnect watchdog and tear down the tunnel and connection."""
        self._closed = True
        watchdog, self._watchdog_task = self._watchdog_task, None
        if watchdog is not None:
            watchdog.cancel()
            try:
                await watchdog
            except asyncio.CancelledError:
                logger.debug("SSH tunnel watchdog task canceled")
            except Exception as error:
                logger.debug("SSH tunnel watchdog task ended with {}: {}", type(error).__name__, error)

        listener, self._listener = self._listener, None
        if listener is not None:
            listener.close()
            try:
                await listener.wait_closed()
            except Exception as error:
                logger.debug("SSH tunnel listener close raised {}: {}", type(error).__name__, error)

        transport, self._transport = self._transport, None
        if transport is not None:
            await transport.close()
