# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SSH transport - the remote-execution trust boundary.

Three properties matter most here and are tested against real ``asyncssh``
exception objects rather than sentinels, so an upstream signature change fails
these tests instead of silently changing behaviour:

* ``test_run_command_quotes_*`` prove that argument text cannot break out of its
  position: the exact string handed to the exec channel is asserted.
* ``test_map_*`` prove every ``asyncssh`` connect failure becomes one of the
  actionable ``Ssh*Error`` classes, and that none of the raw text - hostnames,
  key paths, exception detail - reaches the resulting message.
* ``test_host_key_*`` prove the unknown-vs-changed distinction, which
  ``asyncssh`` itself does not make: both raise the same
  ``HostKeyNotVerifiable``.
"""

import asyncio
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncssh
import pytest

from exceptions import BaseException as StudioBaseException
from exceptions import (
    SshAgentRequiredError,
    SshAuthenticationError,
    SshConnectionError,
    SshHostAliasNotFoundError,
    SshHostKeyMismatchError,
    SshHostKeyUnknownError,
)
from services.ssh import transport as transport_module
from services.ssh.transport import (
    COMMAND_FAILED_EXIT_STATUS,
    COMMAND_TIMEOUT_EXIT_STATUS,
    CommandFailure,
    CommandResult,
    SshTransport,
    open_transport,
    reset_alias_gates,
)
from settings import Settings

ALIAS = "gpu-box"
_HOSTNAME = "gpu-box.internal.example.com"
_IDENTITY_PATH = "/home/tester/.ssh/id_secret_test_key"


@pytest.fixture(autouse=True)
def _clear_gates():
    reset_alias_gates()
    yield
    reset_alias_gates()


@pytest.fixture
def ssh_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config"
    config_path.write_text(f"Host {ALIAS}\n  HostName {_HOSTNAME}\n  User tester\n")
    return config_path


@pytest.fixture
def known_hosts(tmp_path: Path) -> Path:
    return tmp_path / "known_hosts"


@pytest.fixture
def settings(ssh_config: Path, known_hosts: Path) -> Settings:
    return Settings(
        SSH_CONFIG_PATH=ssh_config,
        SSH_KNOWN_HOSTS_PATH=known_hosts,
        # No throttle delay: the throttle is tested explicitly, and every other
        # test would otherwise pay for it.
        SSH_PREFLIGHT_THROTTLE_S=0.0,
        SSH_OUTPUT_MAX_LINE_CHARS=512,
        SSH_OUTPUT_MAX_TOTAL_CHARS=4096,
    )


@pytest.fixture
def encrypted_identity_settings(tmp_path: Path) -> Settings:
    """Settings whose alias resolves to a genuinely passphrase-protected key.

    A real encrypted key, not a stub, so the passphrase detection is exercised
    rather than mocked. PKCS#8 PEM is used because OpenSSH-format encryption needs
    ``bcrypt``, which is not a dependency here.
    """
    identity = tmp_path / "id_encrypted"
    key = asyncssh.generate_private_key("ssh-ed25519")
    identity.write_bytes(key.export_private_key("pkcs8-pem", passphrase="test-passphrase"))
    config_path = tmp_path / "config"
    config_path.write_text(f"Host {ALIAS}\n  HostName {_HOSTNAME}\n  IdentityFile {identity}\n")
    return Settings(
        SSH_CONFIG_PATH=config_path,
        SSH_KNOWN_HOSTS_PATH=tmp_path / "known_hosts",
        SSH_PREFLIGHT_THROTTLE_S=0.0,
    )


def _completed(
    stdout: str | bytes = "", stderr: str | bytes = "", exit_status: int | None = 0, exit_signal: Any = None
) -> MagicMock:
    completed = MagicMock()
    completed.stdout = stdout
    completed.stderr = stderr
    completed.exit_status = exit_status
    completed.exit_signal = exit_signal
    return completed


def _await_args(mock: AsyncMock) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    """Return one AsyncMock's last call args/kwargs, asserting it was awaited.

    ``await_args`` is typed ``_Call | None`` because an un-awaited mock has none;
    every caller here has already awaited it, so this narrows that away instead
    of repeating the same assert at each call site.
    """
    call = mock.await_args
    assert call is not None
    return call.args, call.kwargs


def _connected_transport(settings: Settings, connection: MagicMock) -> SshTransport:
    """Return a transport with a fake open connection, skipping the dial."""
    transport = SshTransport(ALIAS, settings)
    transport._connection = connection
    return transport


def _process_error(*, exit_status: int | None, stdout: str = "", stderr: str = "", exit_signal=None):
    return asyncssh.ProcessError(
        env=None,
        command=None,
        subsystem=None,
        exit_status=exit_status,
        exit_signal=exit_signal,
        returncode=exit_status,
        stdout=stdout,
        stderr=stderr,
    )


def _timeout_error(stdout: str = "", stderr: str = ""):
    return asyncssh.TimeoutError(
        env=None,
        command=None,
        subsystem=None,
        exit_status=None,
        exit_signal=None,
        returncode=None,
        stdout=stdout,
        stderr=stderr,
    )


# --------------------------------------------------------------------------- #
# Command construction: injection safety                                      #
# --------------------------------------------------------------------------- #


async def test_run_command_sends_a_shell_quoted_string(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(return_value=_completed(stdout="27.3.1\n"))
    transport = _connected_transport(settings, connection)

    result = await transport.run_command(["docker", "version", "--format", "{{.Server.Version}}"])

    sent = _await_args(connection.run)[0][0]
    assert sent == "docker version --format '{{.Server.Version}}'"
    assert result.command == sent
    assert result.ok


async def test_run_command_quotes_shell_metacharacters_verbatim(settings: Settings) -> None:
    # The exact string is asserted because this is the injection boundary: every
    # metacharacter must arrive as literal text inside a quoted argument.
    connection = MagicMock()
    connection.run = AsyncMock(return_value=_completed())
    transport = _connected_transport(settings, connection)

    hostile = ["docker", "run", "; rm -rf /", "$(whoami)", "`id`", "a && b", "x | y", "n\nm"]
    result = await transport.run_command(hostile)

    sent = _await_args(connection.run)[0][0]
    assert sent == "docker run '; rm -rf /' '$(whoami)' '`id`' 'a && b' 'x | y' 'n\nm'"
    assert result.argv == tuple(hostile)


async def test_run_command_quotes_embedded_single_quotes(settings: Settings) -> None:
    # The classic escape: a bare "'" would otherwise close the quoting shlex
    # opened and let the rest of the element be interpreted.
    connection = MagicMock()
    connection.run = AsyncMock(return_value=_completed())
    transport = _connected_transport(settings, connection)

    await transport.run_command(["echo", "it's; rm -rf /"])

    assert _await_args(connection.run)[0][0] == """echo 'it'"'"'s; rm -rf /'"""


async def test_run_command_rejects_an_empty_argv(settings: Settings) -> None:
    transport = _connected_transport(settings, MagicMock())

    with pytest.raises(ValueError, match="argv"):
        await transport.run_command([])


async def test_run_command_without_a_connection_raises(settings: Settings) -> None:
    transport = SshTransport(ALIAS, settings)

    with pytest.raises(RuntimeError):
        await transport.run_command(["true"])


async def test_run_command_passes_the_configured_command_timeout(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(return_value=_completed())
    transport = _connected_transport(settings, connection)

    await transport.run_command(["true"])

    assert _await_args(connection.run)[1]["timeout"] == settings.ssh_command_timeout_s


async def test_run_command_honours_an_explicit_timeout(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(return_value=_completed())
    transport = _connected_transport(settings, connection)

    await transport.run_command(["true"], timeout=2.5)

    assert _await_args(connection.run)[1]["timeout"] == 2.5


# --------------------------------------------------------------------------- #
# Command results and output sanitization                                     #
# --------------------------------------------------------------------------- #


async def test_command_output_is_sanitized(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(
        return_value=_completed(stdout="\x1b[31mfail\x1b[0m\x00", stderr="\x1b]8;;https://evil.example.com\x07link")
    )
    transport = _connected_transport(settings, connection)

    result = await transport.run_command(["docker", "version"])

    assert result.stdout == "fail"
    assert result.stderr == "link"


async def test_command_output_is_length_capped(ssh_config: Path, known_hosts: Path) -> None:
    settings = Settings(
        SSH_CONFIG_PATH=ssh_config,
        SSH_KNOWN_HOSTS_PATH=known_hosts,
        SSH_OUTPUT_MAX_LINE_CHARS=16,
        SSH_OUTPUT_MAX_TOTAL_CHARS=32,
    )
    connection = MagicMock()
    connection.run = AsyncMock(return_value=_completed(stdout="\n".join(["x" * 200] * 20)))
    transport = _connected_transport(settings, connection)

    result = await transport.run_command(["true"])

    assert len(result.stdout) <= 32


async def test_byte_output_is_decoded(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(return_value=_completed(stdout=b"bytes out", stderr=b"\xff\xfe bad"))
    transport = _connected_transport(settings, connection)

    result = await transport.run_command(["true"])

    assert result.stdout == "bytes out"
    assert "bad" in result.stderr


async def test_nonzero_exit_status_is_not_ok(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(return_value=_completed(stderr="no such file", exit_status=127))
    transport = _connected_transport(settings, connection)

    result = await transport.run_command(["nvidia-smi"])

    assert result.ok is False
    assert result.exit_status == 127
    assert result.stderr == "no such file"


async def test_process_error_becomes_a_result_not_an_exception(settings: Settings) -> None:
    # One failing probe must never abort a whole preflight tier.
    connection = MagicMock()
    connection.run = AsyncMock(side_effect=_process_error(exit_status=2, stderr="denied"))
    transport = _connected_transport(settings, connection)

    result = await transport.run_command(["docker", "version"])

    assert result.exit_status == 2
    assert result.failure is None
    assert result.stderr == "denied"


async def test_command_timeout_becomes_a_timeout_result(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(side_effect=_timeout_error(stdout="partial"))
    transport = _connected_transport(settings, connection)

    result = await transport.run_command(["sleep", "600"])

    assert result.failure is CommandFailure.TIMEOUT
    assert result.exit_status == COMMAND_TIMEOUT_EXIT_STATUS
    assert result.ok is False
    assert result.stdout == "partial"


async def test_channel_open_error_becomes_a_channel_refused_result(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(side_effect=asyncssh.ChannelOpenError(4, "administratively prohibited"))
    transport = _connected_transport(settings, connection)

    result = await transport.run_command(["true"])

    assert result.failure is CommandFailure.CHANNEL_REFUSED
    assert result.exit_status == COMMAND_FAILED_EXIT_STATUS


async def test_signalled_process_reports_a_signal_failure(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(return_value=_completed(exit_status=None, exit_signal=("KILL", False, "", "en-US")))
    transport = _connected_transport(settings, connection)

    result = await transport.run_command(["python", "-c", "pass"])

    assert result.failure is CommandFailure.SIGNALED
    assert result.ok is False


async def test_connection_lost_during_a_command_raises_a_connection_error(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(side_effect=asyncssh.ConnectionLost("connection lost"))
    transport = _connected_transport(settings, connection)

    with pytest.raises(SshConnectionError) as error:
        await transport.run_command(["true"])

    assert "connection_lost" in error.value.message


async def test_builtin_timeout_during_a_command_raises_a_connection_error(settings: Settings) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(side_effect=TimeoutError)
    transport = _connected_transport(settings, connection)

    with pytest.raises(SshConnectionError) as error:
        await transport.run_command(["true"])

    assert "timeout" in error.value.message


def test_command_result_first_line_skips_blank_lines() -> None:
    result = CommandResult(argv=("true",), command="true", exit_status=0, stdout="\n  \nDocker 27.3.1\nmore\n")

    assert result.first_line() == "Docker 27.3.1"


def test_command_result_first_line_of_empty_output_is_empty() -> None:
    result = CommandResult(argv=("true",), command="true", exit_status=0)

    assert result.first_line() == ""


# --------------------------------------------------------------------------- #
# Alias resolution before dialing                                             #
# --------------------------------------------------------------------------- #


async def test_connect_rejects_an_unknown_alias_without_dialing(settings: Settings, monkeypatch) -> None:
    connect = AsyncMock()
    monkeypatch.setattr(asyncssh, "connect", connect)
    transport = SshTransport("not-in-config", settings)

    with pytest.raises(SshHostAliasNotFoundError):
        await transport.connect()

    connect.assert_not_awaited()


async def test_connect_rejects_a_wildcard_only_alias(tmp_path: Path, monkeypatch) -> None:
    # A pattern stanza is not a usable target, so it must not be dialed.
    config_path = tmp_path / "config"
    config_path.write_text("Host *\n  User tester\n")
    settings = Settings(SSH_CONFIG_PATH=config_path, SSH_KNOWN_HOSTS_PATH=tmp_path / "known_hosts")
    connect = AsyncMock()
    monkeypatch.setattr(asyncssh, "connect", connect)

    with pytest.raises(SshHostAliasNotFoundError):
        await SshTransport(ALIAS, settings).connect()

    connect.assert_not_awaited()


async def test_connect_is_idempotent(settings: Settings, monkeypatch) -> None:
    connect = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(asyncssh, "connect", connect)
    transport = SshTransport(ALIAS, settings)

    await transport.connect()
    await transport.connect()

    assert connect.await_count == 1
    await transport.close()


async def test_context_manager_connects_and_closes(settings: Settings, monkeypatch) -> None:
    connection = MagicMock()
    connection.wait_closed = AsyncMock()
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(return_value=connection))

    async with SshTransport(ALIAS, settings) as transport:
        assert transport.connected is True

    connection.close.assert_called_once()


async def test_connect_reads_the_users_config_and_alias(settings: Settings, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_connect(*, options, **_kwargs):
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(asyncssh, "connect", fake_connect)
    transport = SshTransport(ALIAS, settings)

    await transport.connect()

    # asyncssh resolved the alias itself, from the user's own config file.
    assert captured["options"].host == _HOSTNAME
    await transport.close()


# --------------------------------------------------------------------------- #
# Host-key classification                                                     #
# --------------------------------------------------------------------------- #
#
# asyncssh raises the same HostKeyNotVerifiable for an unknown host and for a
# changed key, so the transport records what known_hosts held before verification
# ran. These tests pin that classification, including its fail-closed default.


def _match(trusted: int = 0, ca: int = 0, revoked: int = 0):
    return ([object()] * trusted, [object()] * ca, [object()] * revoked, [], [], [], [])


def _verifying_connect(monkeypatch, match_result) -> None:
    """Fake ``asyncssh.connect`` that consults the matcher, then rejects the key.

    Mirrors the real handshake order - asyncssh looks the host up in
    ``known_hosts`` and only then raises ``HostKeyNotVerifiable`` - because the
    lookup result is the only thing that distinguishes unknown from changed.
    """
    monkeypatch.setattr(asyncssh, "match_known_hosts", lambda *_args: match_result)

    async def fake_connect(*, options, **_kwargs):
        options.known_hosts(_HOSTNAME, "10.0.0.1", 22)
        raise asyncssh.HostKeyNotVerifiable(f"Host key is not trusted for host {_HOSTNAME}")

    monkeypatch.setattr(asyncssh, "connect", fake_connect)


async def test_no_known_hosts_entry_is_reported_as_an_unknown_host(settings: Settings, monkeypatch) -> None:
    _verifying_connect(monkeypatch, _match())

    with pytest.raises(SshHostKeyUnknownError):
        await SshTransport(ALIAS, settings).connect()


async def test_an_existing_known_hosts_entry_is_reported_as_a_mismatch(settings: Settings, monkeypatch) -> None:
    # An entry existed and verification still failed: the presented key differs
    # from the accepted one. Telling the user to accept a fingerprint here would
    # be exactly the wrong advice.
    _verifying_connect(monkeypatch, _match(trusted=1))

    with pytest.raises(SshHostKeyMismatchError):
        await SshTransport(ALIAS, settings).connect()


async def test_a_revoked_key_is_reported_as_a_mismatch(settings: Settings, monkeypatch) -> None:
    _verifying_connect(monkeypatch, _match(revoked=1))

    with pytest.raises(SshHostKeyMismatchError):
        await SshTransport(ALIAS, settings).connect()


async def test_a_ca_only_entry_is_reported_as_a_mismatch(settings: Settings, monkeypatch) -> None:
    _verifying_connect(monkeypatch, _match(ca=1))

    with pytest.raises(SshHostKeyMismatchError):
        await SshTransport(ALIAS, settings).connect()


async def test_the_matcher_state_does_not_leak_between_connects(settings: Settings, monkeypatch) -> None:
    # A mismatch recorded on one attempt must not make the next attempt - which
    # never consulted known_hosts - look like a mismatch for that reason.
    _verifying_connect(monkeypatch, _match(trusted=1))
    transport = SshTransport(ALIAS, settings)
    with pytest.raises(SshHostKeyMismatchError):
        await transport.connect()

    _verifying_connect(monkeypatch, _match())

    with pytest.raises(SshHostKeyUnknownError):
        await transport.connect()


async def test_an_unconsulted_matcher_fails_closed_as_a_mismatch(settings: Settings, monkeypatch) -> None:
    # asyncssh rejected the key without ever calling the matcher. Ambiguous, so
    # the more suspicious interpretation wins: never tell a user to accept a
    # fingerprint for a key that may have changed.
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=asyncssh.HostKeyNotVerifiable("not trusted")))

    with pytest.raises(SshHostKeyMismatchError):
        await SshTransport(ALIAS, settings).connect()


async def test_missing_known_hosts_file_is_treated_as_empty(settings: Settings, monkeypatch) -> None:
    # A nonexistent known_hosts file would make asyncssh raise FileNotFoundError;
    # the actionable outcome is "you have never accepted this host".
    recorded: dict[str, Any] = {}

    def fake_match(source, host, addr, port):
        recorded["source"] = source
        return _match()

    monkeypatch.setattr(asyncssh, "match_known_hosts", fake_match)
    transport = SshTransport(ALIAS, settings)

    transport._match_known_hosts(_HOSTNAME, "10.0.0.1", 22)

    assert recorded["source"] == b""


async def test_existing_known_hosts_file_is_passed_by_path(settings: Settings, known_hosts: Path, monkeypatch) -> None:
    known_hosts.write_text(f"{_HOSTNAME} ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIexample\n")
    recorded: dict[str, Any] = {}

    def fake_match(source, host, addr, port):
        recorded["source"] = source
        return _match(trusted=1)

    monkeypatch.setattr(asyncssh, "match_known_hosts", fake_match)
    transport = SshTransport(ALIAS, settings)

    transport._match_known_hosts(_HOSTNAME, "10.0.0.1", 22)

    assert recorded["source"] == str(known_hosts)


# --------------------------------------------------------------------------- #
# Connect-failure mapping                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("error", "expected", "reason_fragment"),
    [
        (TimeoutError(), SshConnectionError, "timeout"),
        (asyncssh.ConnectionLost("lost"), SshConnectionError, "connection_lost"),
        (asyncssh.ProtocolError("bad packet"), SshConnectionError, "protocol_error"),
        (asyncssh.KeyExchangeFailed("no kex"), SshConnectionError, "protocol_error"),
        (ConnectionRefusedError(), SshConnectionError, "unreachable"),
        (socket.gaierror("name resolution failed"), SshConnectionError, "unreachable"),
        (OSError("network unreachable"), SshConnectionError, "unreachable"),
        (asyncssh.DisconnectError(2, "protocol error"), SshConnectionError, "protocol_error"),
    ],
)
async def test_map_connect_failures_to_a_connection_error(
    settings: Settings,
    monkeypatch,
    error: Exception,
    expected: type[SshConnectionError],
    reason_fragment: str,
) -> None:
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=error))

    with pytest.raises(expected) as raised:
        await SshTransport(ALIAS, settings).connect()

    assert reason_fragment in raised.value.message


async def test_map_permission_denied_to_an_authentication_error(settings: Settings, monkeypatch) -> None:
    # No encrypted identity is configured, so this is a plain rejection.
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=asyncssh.PermissionDenied("denied")))

    with pytest.raises(SshAuthenticationError):
        await SshTransport(ALIAS, settings).connect()


async def test_map_permission_denied_with_a_locked_key_to_an_agent_error(
    encrypted_identity_settings: Settings,
    monkeypatch,
) -> None:
    # A passphrase-protected key with no usable agent arrives from asyncssh as a
    # plain PermissionDenied, so the actionable cause has to be diagnosed after
    # the fact - otherwise the user is told to check their key instead of to run
    # ssh-add. The key here is genuinely encrypted, so the detection is real.
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=asyncssh.PermissionDenied("denied")))
    monkeypatch.setattr(transport_module, "_agent_has_keys", AsyncMock(return_value=False))

    with pytest.raises(SshAgentRequiredError):
        await SshTransport(ALIAS, encrypted_identity_settings).connect()


async def test_a_locked_key_with_a_loaded_agent_is_an_authentication_error(
    encrypted_identity_settings: Settings,
    monkeypatch,
) -> None:
    # The agent holds keys, so the passphrase is not the problem: the server
    # rejected the identity.
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=asyncssh.PermissionDenied("denied")))
    monkeypatch.setattr(transport_module, "_agent_has_keys", AsyncMock(return_value=True))

    with pytest.raises(SshAuthenticationError):
        await SshTransport(ALIAS, encrypted_identity_settings).connect()


async def test_a_plaintext_key_rejection_is_an_authentication_error(tmp_path: Path, monkeypatch) -> None:
    # No passphrase involved, so an agent would not help.
    identity = tmp_path / "id_plain"
    identity.write_bytes(asyncssh.generate_private_key("ssh-ed25519").export_private_key("openssh"))
    config_path = tmp_path / "config"
    config_path.write_text(f"Host {ALIAS}\n  HostName {_HOSTNAME}\n  IdentityFile {identity}\n")
    settings = Settings(SSH_CONFIG_PATH=config_path, SSH_KNOWN_HOSTS_PATH=tmp_path / "known_hosts")
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=asyncssh.PermissionDenied("denied")))

    with pytest.raises(SshAuthenticationError):
        await SshTransport(ALIAS, settings).connect()


@pytest.mark.parametrize(
    "error",
    [
        asyncssh.KeyImportError("Passphrase must be specified to import encrypted private keys"),
        asyncssh.KeyEncryptionError("bad passphrase"),
    ],
)
async def test_map_encrypted_key_load_failures_to_an_agent_error(
    settings: Settings,
    monkeypatch,
    error: Exception,
) -> None:
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=error))

    with pytest.raises(SshAgentRequiredError):
        await SshTransport(ALIAS, settings).connect()


async def test_a_malformed_identity_file_is_an_authentication_error(tmp_path: Path, monkeypatch) -> None:
    # asyncssh loads identities while building options, so this fails before the
    # dial. It must still arrive as an actionable Ssh* error - and not as an agent
    # error, because ssh-add will never fix a corrupt file.
    identity = tmp_path / "id_broken"
    identity.write_text("not-a-real-key")
    config_path = tmp_path / "config"
    config_path.write_text(f"Host {ALIAS}\n  HostName {_HOSTNAME}\n  IdentityFile {identity}\n")
    settings = Settings(SSH_CONFIG_PATH=config_path, SSH_KNOWN_HOSTS_PATH=tmp_path / "known_hosts")
    connect = AsyncMock()
    monkeypatch.setattr(asyncssh, "connect", connect)

    with pytest.raises(SshAuthenticationError):
        await SshTransport(ALIAS, settings).connect()

    connect.assert_not_awaited()


async def test_a_missing_identity_file_is_an_authentication_error(tmp_path: Path, monkeypatch) -> None:
    # A configured IdentityFile that does not exist makes asyncssh raise
    # FileNotFoundError; an unhandled OSError must not reach the API layer.
    config_path = tmp_path / "config"
    config_path.write_text(f"Host {ALIAS}\n  HostName {_HOSTNAME}\n  IdentityFile {tmp_path / 'absent'}\n")
    settings = Settings(SSH_CONFIG_PATH=config_path, SSH_KNOWN_HOSTS_PATH=tmp_path / "known_hosts")
    monkeypatch.setattr(asyncssh, "connect", AsyncMock())

    with pytest.raises(SshAuthenticationError) as raised:
        await SshTransport(ALIAS, settings).connect()

    assert str(tmp_path) not in raised.value.message


async def test_map_an_unexpected_exception_to_a_connection_error(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=RuntimeError("something odd")))

    with pytest.raises(SshConnectionError):
        await SshTransport(ALIAS, settings).connect()


async def test_cancellation_is_not_swallowed(settings: Settings, monkeypatch) -> None:
    # A cancelled request must cancel, not surface as a server error.
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=asyncio.CancelledError))

    with pytest.raises(asyncio.CancelledError):
        await SshTransport(ALIAS, settings).connect()


@pytest.mark.parametrize(
    "error",
    [
        asyncssh.PermissionDenied(f"Permission denied for key {_IDENTITY_PATH} on {_HOSTNAME}"),
        asyncssh.ProtocolError(f"Protocol error talking to {_HOSTNAME}"),
        asyncssh.HostKeyNotVerifiable(f"Host key is not trusted for host {_HOSTNAME}"),
        OSError(f"[Errno 113] No route to host: {_HOSTNAME}"),
    ],
)
async def test_mapped_errors_never_leak_hostnames_or_key_paths(
    settings: Settings,
    monkeypatch,
    error: Exception,
) -> None:
    # A raw SSH error can name the resolved hostname or an identity path, and the
    # mapped message reaches an API response. The alias is the only host-ish
    # identifier allowed through, because the user chose it and it is not secret.
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=error))

    with pytest.raises(StudioBaseException) as raised:
        await SshTransport(ALIAS, settings).connect()

    message = raised.value.message
    assert _HOSTNAME not in message
    assert _IDENTITY_PATH not in message
    assert str(error) not in message
    assert ALIAS in message


async def test_mapped_errors_drop_the_original_exception_chain(settings: Settings, monkeypatch) -> None:
    # `raise ... from None`: a traceback rendered into a log must not re-expose
    # the raw text the mapping deliberately dropped.
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=asyncssh.ProtocolError(f"talking to {_HOSTNAME}")))

    with pytest.raises(SshConnectionError) as raised:
        await SshTransport(ALIAS, settings).connect()

    assert raised.value.__cause__ is None


# --------------------------------------------------------------------------- #
# Bounded concurrency                                                         #
# --------------------------------------------------------------------------- #


async def test_the_per_alias_connection_slot_is_released_on_failure(settings: Settings, monkeypatch) -> None:
    # A leaked semaphore slot would wedge the alias for the rest of the process,
    # so every failing path must release it.
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=OSError("refused")))

    for _ in range(settings.ssh_max_connections_per_server + 2):
        with pytest.raises(SshConnectionError):
            await SshTransport(ALIAS, settings).connect()


async def test_concurrent_connections_to_one_alias_are_capped(
    ssh_config: Path,
    known_hosts: Path,
    monkeypatch,
) -> None:
    # UI polling plus a status check plus a preflight must not pile connections
    # onto one server, so the cap is enforced per alias.
    settings = Settings(
        SSH_CONFIG_PATH=ssh_config,
        SSH_KNOWN_HOSTS_PATH=known_hosts,
        SSH_MAX_CONNECTIONS_PER_SERVER=2,
        SSH_PREFLIGHT_THROTTLE_S=0.0,
    )
    in_flight = 0
    peak = 0

    async def fake_connect(**_kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        # Yield, so a slot is genuinely held while the other tasks get a turn.
        await asyncio.sleep(0.01)
        in_flight -= 1
        connection = MagicMock()
        connection.wait_closed = AsyncMock()
        return connection

    monkeypatch.setattr(asyncssh, "connect", fake_connect)
    transports = [SshTransport(ALIAS, settings) for _ in range(5)]

    async def dial(transport: SshTransport) -> None:
        await transport.connect()
        await transport.close()

    await asyncio.gather(*(dial(transport) for transport in transports))

    assert peak <= 2


async def test_close_is_safe_without_a_connection(settings: Settings) -> None:
    transport = SshTransport(ALIAS, settings)

    await transport.close()

    assert transport.connected is False


async def test_close_releases_the_slot_even_if_wait_closed_fails(settings: Settings, monkeypatch) -> None:
    connection = MagicMock()
    connection.wait_closed = AsyncMock(side_effect=asyncssh.ConnectionLost("already gone"))
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(return_value=connection))
    settings_with_room = settings

    for _ in range(settings_with_room.ssh_max_connections_per_server + 2):
        transport = SshTransport(ALIAS, settings_with_room)
        await transport.connect()
        await transport.close()


def test_open_transport_returns_an_unconnected_transport(settings: Settings) -> None:
    transport = open_transport(ALIAS, settings)

    assert isinstance(transport, SshTransport)
    assert transport.alias == ALIAS
    assert transport.connected is False
