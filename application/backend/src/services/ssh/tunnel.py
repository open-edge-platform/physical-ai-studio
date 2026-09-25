# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SSH local-forward tunnel to a managed remote trainer.

Trainer containers publish on the SSH host's loopback interface.
:class:`SshTunnel` forwards a local loopback port to the container and
reconnects until the SSH connection returns or the tunnel is closed.
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
        *,
        local_port: int | None = None,
    ) -> None:
        self._open_transport = open_transport
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._settings = settings
        self._transport: SshTransport | None = None
        self._listener: asyncssh.SSHListener | None = None
        self._local_port = local_port
        self._fixed_port = local_port is not None
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

    async def open(self, *, retry_on_failure: bool = False) -> None:
        """Connect and establish the forward. Starts the reconnect watchdog.

        At startup, an offline VPN can delay the first connection without
        preventing remote jobs from reattaching once it returns.
        """
        try:
            await self._connect_and_forward()
        except Exception:
            if not retry_on_failure:
                raise
            logger.warning("SSH tunnel unavailable at startup; retrying in background")
        self._watchdog_task = asyncio.create_task(self._watch())

    async def _connect_and_forward(self) -> None:
        """Connect a fresh transport and forward, closing whatever this replaces.

        Tears down any previously-open transport/listener before assigning the
        new ones (a reconnect must never leak the connection it is replacing),
        and closes the newly-connected transport itself if `forward_local_port`
        fails (a half-open connect must never leak either).

        On a reconnect (`self._local_port` already set), re-binds to that same
        local port so callers that cached a base URL derived from it keep
        working. A configured port must not change: persisted job URLs point
        at it. Only ephemeral tunnels may fall back to a fresh port.
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
                if preferred_port is None or self._fixed_port:
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
        while not self._closed:
            listener = self._listener
            if listener is not None:
                try:
                    await listener.wait_closed()
                except asyncio.CancelledError:
                    return
                if self._closed:
                    return
                logger.warning("SSH tunnel to remote trainer dropped; attempting to reconnect")
            await self._reconnect_with_backoff()

    async def _reconnect_with_backoff(self) -> None:
        """Retry `_connect_and_forward` with exponential backoff until closed.

        A configured local port stays fixed across reconnects. The reconnect
        budget is a warning threshold, not a reason to abandon remote jobs.
        """
        settings = self._settings
        started = monotonic()
        backoff = 1.0
        warned = False
        while not self._closed:
            try:
                await self._connect_and_forward()
                logger.info("SSH tunnel reconnected on local port {}", self._local_port)
                return
            except Exception as error:
                if not warned and monotonic() - started >= settings.ssh_tunnel_reconnect_budget_s:
                    logger.warning("SSH tunnel still unavailable; retrying: {}", error)
                    warned = True
                await asyncio.sleep(min(backoff, settings.ssh_tunnel_reconnect_backoff_max_s))
                backoff = min(backoff * 2, settings.ssh_tunnel_reconnect_backoff_max_s)

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
