# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Async SSH transport for SSH-provisioned remote training servers.

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

Host-key unknown vs. changed
----------------------------
``asyncssh`` Raises the same :class:`asyncssh.HostKeyNotVerifiable` for a host
  absent from ``known_hosts`` and for a host whose key changed - both arrive as
``ValueError('Host key is not trusted')`` inside
``SSHClientConnection.validate_server_host_key``. To tell them apart, this module
installs a callable ``known_hosts`` matcher that wraps
:func:`asyncssh.match_known_hosts` and records how many entries matched the host
before verification ran. No matching entry means the host was never accepted
(unknown); entries that matched while verification still failed means the
presented key differs from the accepted one (mismatch). An ambiguous case fails
closed as a mismatch, the more suspicious interpretation.
"""

import asyncio
import shlex
import socket
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Final, Self

import asyncssh
from loguru import logger

from exceptions import (
    SshAgentRequiredError,
    SshAuthenticationError,
    SshConnectionError,
    SshHostAliasNotFoundError,
    SshHostKeyMismatchError,
    SshHostKeyUnknownError,
)
from services.ssh.sanitize import sanitize_output
from services.ssh_config_reader import resolve_alias
from settings import Settings, get_settings

# Exit status reported for a command that never produced one. 124 is the
# conventional timeout status; -1 marks a channel that never ran.
COMMAND_TIMEOUT_EXIT_STATUS: Final = 124
COMMAND_FAILED_EXIT_STATUS: Final = -1

# Pre-approved `SshConnectionError.reason` categories. The reason reaches an API
# response, so it is chosen from this set and never derived from exception text.
_REASON_TIMEOUT: Final = "timeout"
_REASON_UNREACHABLE: Final = "unreachable"
_REASON_PROTOCOL: Final = "protocol_error"
_REASON_CONNECTION_LOST: Final = "connection_lost"

# `asyncssh` reports a passphrase-protected key it cannot decrypt with a
# KeyImportError whose message starts with this word.
# S105: this is the first word of an asyncssh error message, not a credential.
_PASSPHRASE_ERROR_PREFIX: Final = "Passphrase"  # noqa: S105


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
class _HostKeyMatch:
    """What ``known_hosts`` held for this host before verification ran."""

    consulted: bool = False
    trusted_keys: int = 0
    ca_keys: int = 0
    revoked_keys: int = 0

    @property
    def has_entry(self) -> bool:
        """True when ``known_hosts`` already held something for this host."""
        return bool(self.trusted_keys or self.ca_keys or self.revoked_keys)


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


def _existing_config_paths(config_path: Path) -> list[str]:
    """Return the SSH config paths to hand ``asyncssh``.

    A missing file is dropped rather than passed through: ``asyncssh`` raises
    ``FileNotFoundError`` for a config path that does not exist, and an absent
    SSH config must surface as "alias not found", not as an unhandled OS error.
    """
    return [str(config_path)] if config_path.is_file() else []


def _identity_files(options: asyncssh.SSHClientConnectionOptions) -> list[str]:
    """Return the ``IdentityFile`` entries the resolved config names."""
    configured = options.config.get("IdentityFile")
    if isinstance(configured, str):
        return [configured]
    if isinstance(configured, Sequence):
        return [str(entry) for entry in configured]
    return []


def _is_passphrase_protected(path: Path) -> bool:
    """True when importing this private key needs a passphrase.

    Only the exception is inspected. A key that imports successfully is dropped
    immediately, and no key material is retained or logged.
    """
    try:
        asyncssh.read_private_key(str(path))
    except asyncssh.KeyImportError as error:
        return str(error).startswith(_PASSPHRASE_ERROR_PREFIX)
    except (OSError, asyncssh.KeyEncryptionError, ValueError):
        return False
    return False


async def _agent_has_keys(agent_path: str | None) -> bool:
    """True when an SSH agent is reachable and holds at least one identity."""
    try:
        agent = await asyncssh.connect_agent(agent_path)
    except (OSError, ValueError, asyncssh.Error):
        return False
    try:
        return bool(await agent.get_keys())
    except (OSError, ValueError, asyncssh.Error):
        return False
    finally:
        agent.close()


async def _needs_agent(options: asyncssh.SSHClientConnectionOptions) -> bool:
    """True when the resolved identity is encrypted and no agent can unlock it.

    Studio never prompts for or stores a passphrase, so an agent is the only way
    a protected key can be used. Checked only after authentication already
    failed, to turn a generic "permission denied" into the actionable cause.
    """
    encrypted = [
        path
        for path in (Path(entry).expanduser() for entry in _identity_files(options))
        if path.is_file() and await asyncio.to_thread(_is_passphrase_protected, path)
    ]
    if not encrypted:
        return False
    agent_path = options.agent_path if isinstance(options.agent_path, str) else None
    return not await _agent_has_keys(agent_path)


class SshTransport:
    """One bounded SSH connection to a configured host alias.

    Use as an async context manager so the connection, the per-alias
    concurrency slot, and the throttle are all released on every path::

        async with SshTransport("gpu-box") as transport:
            result = await transport.run_command(["docker", "version"])

    Attributes:
        alias: The SSH config alias this transport dials.
    """

    def __init__(self, alias: str, settings: Settings | None = None) -> None:
        self.alias = alias
        self._settings = settings or get_settings()
        self._connection: asyncssh.SSHClientConnection | None = None
        self._gate: _AliasGate | None = None
        self._host_key_match = _HostKeyMatch()

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

    def _build_options(self) -> asyncssh.SSHClientConnectionOptions:
        """Build connect options from the user's SSH config.

        The alias is passed as ``host`` together with the user's config, so
        ``asyncssh`` performs the ``Host`` stanza resolution itself. Studio does
        not reimplement hostname/port/user/identity resolution.
        """
        settings = self._settings
        self._host_key_match = _HostKeyMatch()
        return asyncssh.SSHClientConnectionOptions(
            host=self.alias,
            config=_existing_config_paths(settings.ssh_config_path),
            known_hosts=self._match_known_hosts,
            connect_timeout=settings.ssh_connect_timeout_s,
            keepalive_interval=settings.ssh_keepalive_interval_s,
            keepalive_count_max=settings.ssh_keepalive_count_max,
        )

    def _match_known_hosts(
        self,
        host: str,
        addr: str,
        port: int | None,
    ) -> tuple[Sequence[object], ...]:
        """Look up the host in ``known_hosts``, recording what matched.

        The recorded counts are the only way to tell an unknown host from a
        changed key: ``asyncssh`` collapses both into one exception. A missing
        ``known_hosts`` file is treated as an empty one, which lands on the
        unknown-host branch - the correct actionable outcome for a user who has
        never accepted any fingerprint.
        """
        known_hosts_path = self._settings.ssh_known_hosts_path
        source: str | bytes = str(known_hosts_path) if known_hosts_path.is_file() else b""
        result = asyncssh.match_known_hosts(source, host, addr, port)
        self._host_key_match = _HostKeyMatch(
            consulted=True,
            trusted_keys=len(result[0]),
            ca_keys=len(result[1]),
            revoked_keys=len(result[2]),
        )
        return result

    def _host_key_error(self) -> SshHostKeyUnknownError | SshHostKeyMismatchError:
        """Classify a host-key verification failure.

        Fails closed: when the matcher never ran, or ran and found an entry, the
        presented key is treated as a mismatch. Only a matcher that ran and
        found nothing yields the "never accepted this host" error, because that
        error tells the user to accept a fingerprint - advice that must never be
        given for a key that actually changed.
        """
        match = self._host_key_match
        if match.consulted and not match.has_entry:
            return SshHostKeyUnknownError(self.alias)
        return SshHostKeyMismatchError(self.alias)

    def _key_error(self, error: Exception) -> SshAgentRequiredError | SshAuthenticationError:
        """Classify a private-key load failure.

        A key that could not be decrypted needs an agent; a key that is malformed
        is not an agent problem, and saying so would send the user to ``ssh-add``
        for a file that will never load. Only the exception's *category* is
        inspected - its text can name the identity path.
        """
        if str(error).startswith(_PASSPHRASE_ERROR_PREFIX):
            return SshAgentRequiredError(self.alias)
        if isinstance(error, asyncssh.KeyEncryptionError):
            return SshAgentRequiredError(self.alias)
        return SshAuthenticationError(self.alias)

    async def _acquire_gate(self) -> _AliasGate:
        """Take a per-alias connection slot, honoring the connect throttle."""
        settings = self._settings
        gate = _GATES.get(self.alias, settings.ssh_max_connections_per_server)
        await gate.semaphore.acquire()
        try:
            async with gate.lock:
                # Status polling and the GPU-busy re-check share this throttle so
                # UI polling cannot pile connections onto a server running a job.
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
                or matches only a wildcard stanza.
            SshHostKeyUnknownError: The host is absent from ``known_hosts``.
            SshHostKeyMismatchError: The host key differs from the accepted one.
            SshAgentRequiredError: The identity is passphrase-protected and no
                agent can unlock it.
            SshAuthenticationError: Every offered identity was rejected.
            SshConnectionError: The host could not be reached.
        """
        if self._connection is not None:
            return

        # Pre-validated against the same config asyncssh will read, so an absent
        # alias is an actionable 400 instead of a connection attempt against a
        # hostname that is really an unresolved alias.
        resolved = resolve_alias(self._settings.ssh_config_path, self.alias)
        if not resolved.found:
            raise SshHostAliasNotFoundError(self.alias)

        # asyncssh loads the configured identities while building options, so a
        # missing or corrupt IdentityFile fails here rather than on the wire. It
        # still has to arrive as an actionable Ssh* error, not a raw OSError.
        try:
            options = self._build_options()
        except (asyncssh.KeyImportError, asyncssh.KeyEncryptionError) as error:
            raise self._key_error(error) from None
        except OSError:
            # A configured identity that cannot be read at all.
            raise SshAuthenticationError(self.alias) from None

        gate = await self._acquire_gate()
        try:
            self._connection = await asyncssh.connect(options=options)
        except BaseException as error:
            gate.semaphore.release()
            raise await self._map_connect_error(error, options) from None
        self._gate = gate

    # PLR0911: one return per failure category. The isinstance order is load-bearing
    # (subclasses first), which a lookup table would obscure.
    async def _map_connect_error(  # noqa: PLR0911
        self,
        error: BaseException,
        options: asyncssh.SSHClientConnectionOptions,
    ) -> BaseException:
        """Translate a connect failure into an actionable Studio exception.

        Deliberately drops the original exception text: it can contain the
        resolved hostname and identity paths, and it reaches an API response.
        """
        if isinstance(error, asyncio.CancelledError):
            return error

        if isinstance(error, asyncssh.HostKeyNotVerifiable):
            logger.warning("SSH host key verification failed for alias '{}'", self.alias)
            return self._host_key_error()

        if isinstance(error, asyncssh.PermissionDenied):
            if await _needs_agent(options):
                return SshAgentRequiredError(self.alias)
            return SshAuthenticationError(self.alias)

        # A key asyncssh could not load. An encrypted one needs an agent; a
        # malformed one does not.
        if isinstance(error, asyncssh.KeyImportError | asyncssh.KeyEncryptionError):
            return self._key_error(error)

        if isinstance(error, TimeoutError):
            return SshConnectionError(self.alias, reason=_REASON_TIMEOUT)

        if isinstance(error, asyncssh.ConnectionLost):
            return SshConnectionError(self.alias, reason=_REASON_CONNECTION_LOST)

        if isinstance(error, asyncssh.ProtocolError | asyncssh.KeyExchangeFailed):
            return SshConnectionError(self.alias, reason=_REASON_PROTOCOL)

        if isinstance(error, OSError | socket.gaierror):
            return SshConnectionError(self.alias, reason=_REASON_UNREACHABLE)

        if isinstance(error, asyncssh.Error):
            return SshConnectionError(self.alias, reason=_REASON_PROTOCOL)

        if isinstance(error, Exception):
            logger.warning("Unexpected SSH failure for alias '{}': {}", self.alias, type(error).__name__)
            return SshConnectionError(self.alias, reason=_REASON_UNREACHABLE)

        return error

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
        raising, so one failed probe never aborts a whole preflight tier.

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

        Binds the local end to ``127.0.0.1``, so an SSH-provisioned trainer is
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


def open_transport(alias: str, settings: Settings | None = None) -> SshTransport:
    """Return a transport for one alias.

    The seam preflight and provisioning go through, so a test can substitute a
    fake transport without patching ``asyncssh`` itself.

    Args:
        alias: SSH config alias to dial.
        settings: Settings override, for tests.

    Returns:
        An unconnected transport, usable as an async context manager.
    """
    return SshTransport(alias, settings)


def reset_alias_gates() -> None:
    """Drop every per-alias concurrency gate. Test-support only."""
    _GATES.clear()
