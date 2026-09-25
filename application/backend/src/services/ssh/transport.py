# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Async SSH transport for managed remote trainers.

This module is the remote-execution trust boundary. Everything crossing it is
constrained here rather than at the call sites:

* **Credentials stay with the user.** ``asyncssh`` is handed the user's own
  ``~/.ssh/config`` and ``~/.ssh/known_hosts`` and resolves the alias itself.
  Studio never reads, stores, or transports key material.
* **Commands are built from argument lists.** :meth:`SshTransport.run_command`
  takes ``argv`` and shell-quotes it with :func:`shlex.join`. SSH's ``exec``
  channel is string-based at the protocol level, so quoting each argument is
  what makes an argument unable to break out of its position.
* **Failures map to actionable errors.** Every ``asyncssh`` connect/auth failure
  becomes one of the ``Ssh*Error`` classes in :mod:`exceptions`, and none of them
  carry raw ``asyncssh`` exception text: a raw SSH error can contain the resolved
  hostname or an identity path, neither of which belongs in an API response.
* **Output is sanitized.** ``stdout``/``stderr`` on a :class:`CommandResult` have
  already been through :func:`services.ssh.sanitize.sanitize_output`, so remote
  output cannot carry escape sequences or unbounded length into a job message.
* **Everything is bounded.** Connect, command, per-alias concurrency, and
  per-alias connect rate all have caps from :class:`settings.Settings`.

Host-key trust on first use
---------------------------
``asyncssh`` Raises the same :class:`asyncssh.HostKeyNotVerifiable` for a host
  absent from ``known_hosts`` and for a host whose key changed - both arrive as
``ValueError('Host key is not trusted')`` inside
``SSHClientConnection.validate_server_host_key``. To tell them apart, this module
installs a callable ``known_hosts`` matcher that records whether the host has an
existing trust entry. If none exists, a client validation callback atomically
persists and accepts the first key presented. If an entry exists, verification
still fails closed as a mismatch. An ambiguous case also fails closed.
"""

import asyncio
import shlex
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from types import TracebackType
from typing import Final, Self

import asyncssh

from exceptions import SshConnectionError
from services.ssh.connection import AliasTarget, SshConnectionTarget
from services.ssh.sanitize import sanitize_output
from settings import Settings, get_settings

# Exit status reported for a command that never produced one. 124 is the
# conventional timeout status; -1 marks a channel that never ran.
COMMAND_TIMEOUT_EXIT_STATUS: Final = 124
COMMAND_FAILED_EXIT_STATUS: Final = -1

_REASON_TIMEOUT: Final = "timeout"
_REASON_CONNECTION_LOST: Final = "connection_lost"


class CommandFailure(StrEnum):
    """Why a command produced no exit status of its own."""

    TIMEOUT = "timeout"
    # The server refused to open a session channel (e.g. a forced command, or a
    # shell-less account).
    CHANNEL_REFUSED = "channel_refused"
    # The remote process died on a signal.
    SIGNALED = "signaled"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of one remote command.

    ``stdout``/``stderr`` are already sanitized and length-capped: remote output
    is environment-influenced, not trusted text, so no raw copy is kept.

    Attributes:
        argv: The argument list as supplied by the caller.
        command: The shell-quoted string actually sent over the exec channel.
        exit_status: The remote exit status, or a synthetic status when the
            command produced none.
        stdout: Sanitized standard output.
        stderr: Sanitized standard error.
        duration_ms: Wall-clock duration of the command.
        failure: Set when the command produced no exit status of its own.
    """

    argv: tuple[str, ...]
    command: str
    exit_status: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    failure: CommandFailure | None = None

    @property
    def ok(self) -> bool:
        """True when the command ran to completion and exited zero."""
        return self.failure is None and self.exit_status == 0

    def first_line(self) -> str:
        """Return the first non-empty stdout line, for a short check detail."""
        for line in self.stdout.splitlines():
            if line.strip():
                return line.strip()
        return ""


@dataclass(slots=True)
class _AliasGate:
    """Per-alias concurrency cap and connect-rate throttle."""

    semaphore: asyncio.Semaphore
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_connect_at: float | None = None


class _AliasGateRegistry:
    """In-process registry of per-alias gates.

    Studio is a single-process, single-user application, so an in-memory
    registry is the whole coordination requirement - there is no second worker
    to synchronize with.
    """

    def __init__(self) -> None:
        self._gates: dict[str, _AliasGate] = {}

    def get(self, alias: str, max_connections: int) -> _AliasGate:
        """Return the gate for one alias, creating it on first use."""
        gate = self._gates.get(alias)
        if gate is None:
            gate = _AliasGate(semaphore=asyncio.Semaphore(max(1, max_connections)))
            self._gates[alias] = gate
        return gate

    def clear(self) -> None:
        """Drop every gate. Test-support only."""
        self._gates.clear()


_GATES: Final = _AliasGateRegistry()


class SshTransport:
    """One bounded SSH connection to an explicit target.

    Use as an async context manager so the connection, the per-alias
    concurrency slot, and the throttle are all released on every path::

        async with SshTransport(AliasTarget("gpu-box")) as transport:
            result = await transport.run_command(["docker", "version"])

    Attributes:
        alias: The SSH config alias this transport dials.
    """

    def __init__(
        self,
        target: SshConnectionTarget,
        settings: Settings | None = None,
        *,
        accepted_host_key_fingerprint: str | None = None,
    ) -> None:
        self._target = target
        self.alias = target.name
        self._settings = settings or get_settings()
        self._connection: asyncssh.SSHClientConnection | None = None
        self._gate: _AliasGate | None = None
        self._accepted_host_key_fingerprint = accepted_host_key_fingerprint

    async def __aenter__(self) -> Self:
        """Open the connection."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the connection and release the per-alias slot."""
        await self.close()

    @property
    def connected(self) -> bool:
        """True while a connection is open."""
        return self._connection is not None

    async def _acquire_gate(self) -> _AliasGate:
        """Take a per-alias connection slot, honoring the connect throttle."""
        settings = self._settings
        gate = _GATES.get(self.alias, settings.ssh_max_connections_per_server)
        await gate.semaphore.acquire()
        try:
            async with gate.lock:
                # Limit connection bursts to an SSH host during health checks
                # and trainer management.
                if gate.last_connect_at is not None:
                    elapsed = perf_counter() - gate.last_connect_at
                    remaining = settings.ssh_preflight_throttle_s - elapsed
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                gate.last_connect_at = perf_counter()
        except BaseException:
            gate.semaphore.release()
            raise
        return gate

    async def connect(self) -> None:
        """Dial the alias and authenticate.

        Raises:
            SshHostAliasNotFoundError: The alias is absent from the SSH config,
                or matches only a wildcard entry.
            SshHostKeyUnknownError: The host is absent from ``known_hosts``.
            SshHostKeyMismatchError: The host key differs from the accepted one.
            SshAgentRequiredError: The identity is passphrase-protected and no
                agent can unlock it.
            SshAuthenticationError: Every offered identity was rejected.
            SshConnectionError: The host could not be reached.
        """
        if self._connection is not None:
            return

        gate = await self._acquire_gate()
        try:
            self._connection = await self._target.connect(
                self._settings,
                self._accepted_host_key_fingerprint,
            )
        except BaseException:
            gate.semaphore.release()
            raise
        self._gate = gate

    # ASYNC109: an explicit `timeout` is part of this method's contract - a caller
    # gets a CommandResult carrying a TIMEOUT failure rather than a raised
    # CancelledError, which `asyncio.timeout` at the call site cannot express.
    async def run_command(self, argv: Sequence[str], timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
        """Run one command on the remote host and return its sanitized output.

        ``argv`` is shell-quoted with :func:`shlex.join` before it reaches the
        exec channel, so no element can break out of its argument position. An
        SSH ``exec`` request carries a command *string* at the protocol level;
        quoting each element is what makes building the command from a list safe.

        A command that times out, is refused a channel, or dies on a signal
        returns a :class:`CommandResult` carrying a ``failure`` rather than
        raising, so callers can handle individual command failures.

        Args:
            argv: Program and arguments. Every element comes from an application
                constant or an already-validated identifier.
            timeout: Per-command budget. Defaults to ``ssh_command_timeout_s``.

        Returns:
            The command's exit status and sanitized output.

        Raises:
            SshConnectionError: The connection dropped while the command ran.
            RuntimeError: The transport is not connected.
        """
        if self._connection is None:
            raise RuntimeError("SshTransport.run_command requires an open connection")
        if not argv:
            raise ValueError("argv must not be empty")

        settings = self._settings
        # shlex.join shell-quotes each argument, so metacharacters in any element
        # are passed through as literal text instead of being interpreted.
        command = shlex.join(argv)
        budget = settings.ssh_command_timeout_s if timeout is None else timeout
        started = perf_counter()

        try:
            completed = await self._connection.run(
                command,
                check=False,
                timeout=budget,
                encoding="utf-8",
                errors="replace",
            )
        except asyncssh.TimeoutError as error:
            return self._result(
                argv,
                command,
                COMMAND_TIMEOUT_EXIT_STATUS,
                error.stdout,
                error.stderr,
                started,
                CommandFailure.TIMEOUT,
            )
        except asyncssh.ChannelOpenError:
            return self._result(
                argv,
                command,
                COMMAND_FAILED_EXIT_STATUS,
                "",
                "",
                started,
                CommandFailure.CHANNEL_REFUSED,
            )
        except asyncssh.ProcessError as error:
            return self._result(
                argv,
                command,
                error.exit_status if error.exit_status is not None else COMMAND_FAILED_EXIT_STATUS,
                error.stdout,
                error.stderr,
                started,
                CommandFailure.SIGNALED if error.exit_signal else None,
            )
        except (asyncssh.ConnectionLost, asyncssh.DisconnectError) as error:
            raise SshConnectionError(self.alias, reason=_REASON_CONNECTION_LOST) from error
        except TimeoutError as error:
            raise SshConnectionError(self.alias, reason=_REASON_TIMEOUT) from error

        if completed.exit_signal:
            return self._result(
                argv,
                command,
                COMMAND_FAILED_EXIT_STATUS,
                completed.stdout,
                completed.stderr,
                started,
                CommandFailure.SIGNALED,
            )
        return self._result(
            argv,
            command,
            completed.exit_status if completed.exit_status is not None else COMMAND_FAILED_EXIT_STATUS,
            completed.stdout,
            completed.stderr,
            started,
            None,
        )

    def _result(
        self,
        argv: Sequence[str],
        command: str,
        exit_status: int,
        stdout: object,
        stderr: object,
        started: float,
        failure: CommandFailure | None,
    ) -> CommandResult:
        """Build a result with both output streams sanitized."""
        return CommandResult(
            argv=tuple(argv),
            command=command,
            exit_status=exit_status,
            stdout=self._sanitize(stdout),
            stderr=self._sanitize(stderr),
            duration_ms=round((perf_counter() - started) * 1000),
            failure=failure,
        )

    def _sanitize(self, stream: object) -> str:
        """Sanitize and cap one output stream."""
        if stream is None:
            return ""
        text = stream.decode("utf-8", errors="replace") if isinstance(stream, bytes) else str(stream)
        return sanitize_output(
            text,
            max_line_chars=self._settings.ssh_output_max_line_chars,
            max_total_chars=self._settings.ssh_output_max_total_chars,
        )

    async def close(self) -> None:
        """Close the connection and release the per-alias slot.

        The semaphore release happens in a ``finally`` so it runs even if this
        coroutine is cancelled while awaiting ``wait_closed()``:
        ``asyncio.CancelledError`` is a ``BaseException`` and is not caught by
        ``suppress(Exception)``, so without the ``finally`` a cancellation here
        would skip the release and wedge the alias's concurrency slot for the
        rest of the process.
        """
        connection, self._connection = self._connection, None
        gate, self._gate = self._gate, None

        try:
            if connection is not None:
                connection.close()
                with suppress(Exception):
                    await connection.wait_closed()
        finally:
            if gate is not None:
                gate.semaphore.release()

    async def forward_local_port(self, remote_host: str, remote_port: int, local_port: int = 0) -> asyncssh.SSHListener:
        """Open a local-forward tunnel to ``remote_host:remote_port`` over this connection.

        Binds the local end to ``127.0.0.1``, so a managed SSH trainer is
        reachable only through the tunnel, never from another host on the
        network.

        Args:
            remote_host: Host to connect to from the remote end (typically
                ``127.0.0.1``, since the trainer itself publishes on its
                container host's loopback interface).
            remote_port: Port to connect to on ``remote_host``.
            local_port: Local port to bind to, or ``0`` (the default) for an
                OS-assigned ephemeral port. Passed by :class:`~services.ssh.tunnel.SshTunnel`
                on reconnect to try to keep the tunnel's address stable for its
                caller.

        Returns:
            The listener. ``listener.get_port()`` reports the bound local
            port; ``listener.close()`` followed by ``await
            listener.wait_closed()`` tears the forward down.

        Raises:
            RuntimeError: The transport is not connected.
            OSError: ``local_port`` is nonzero and unavailable to bind.
        """
        if self._connection is None:
            raise RuntimeError("SshTransport.forward_local_port requires an open connection")
        return await self._connection.forward_local_port("127.0.0.1", local_port, remote_host, remote_port)


def open_transport(
    alias: str,
    settings: Settings | None = None,
    *,
    accepted_host_key_fingerprint: str | None = None,
) -> SshTransport:
    """Return a transport for one alias.

    Callers can substitute a fake transport in tests without patching
    ``asyncssh`` itself.

    Args:
        alias: SSH config alias to dial.
        settings: Settings override, for tests.

    Returns:
        An unconnected transport, usable as an async context manager.
    """
    return SshTransport(
        AliasTarget(alias),
        settings,
        accepted_host_key_fingerprint=accepted_host_key_fingerprint,
    )


def reset_alias_gates() -> None:
    """Drop every per-alias concurrency gate. Test-support only."""
    _GATES.clear()
