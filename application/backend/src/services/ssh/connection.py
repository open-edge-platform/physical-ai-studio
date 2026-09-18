# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SSH connection targets and AsyncSSH connection establishment."""

import asyncio
import socket
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import asyncssh
from loguru import logger

from exceptions import SshAgentRequiredError, SshAuthenticationError, SshConnectionError, SshHostAliasNotFoundError
from services.ssh.host_keys import AsyncSshHostKeyVerifier
from services.ssh.http_proxy import open_http_connect_socket, resolve_http_connect_proxy
from services.ssh_config_reader import resolve_alias
from settings import Settings

_REASON_TIMEOUT: Final = "timeout"
_REASON_UNREACHABLE: Final = "unreachable"
_REASON_PROTOCOL: Final = "protocol_error"
_REASON_CONNECTION_LOST: Final = "connection_lost"

# S105: this is the first word of an AsyncSSH error message, not a credential.
_PASSPHRASE_ERROR_PREFIX: Final = "Passphrase"  # noqa: S105


class SshConnectionTarget(ABC):
    """Connect to one SSH target using target-specific configuration."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the non-secret target name used in errors and connection gates."""

    @property
    @abstractmethod
    def host(self) -> str:
        """Return the host argument passed to AsyncSSH."""

    @abstractmethod
    def _validate(self, settings: Settings) -> None:
        """Validate the target before dialing."""

    @abstractmethod
    def _build_options(
        self,
        settings: Settings,
        host_keys: AsyncSshHostKeyVerifier,
    ) -> asyncssh.SSHClientConnectionOptions:
        """Build target-specific AsyncSSH options."""

    async def _open_socket(self, _settings: Settings) -> socket.socket | None:
        return None

    async def connect(
        self,
        settings: Settings,
        accepted_host_key_fingerprint: str | None = None,
    ) -> asyncssh.SSHClientConnection:
        """Validate, configure, and connect to this target."""
        self._validate(settings)
        host_keys = AsyncSshHostKeyVerifier(
            self.name,
            settings.ssh_known_hosts_path,
            accepted_host_key_fingerprint,
        )
        try:
            options = self._build_options(settings, host_keys)
        except (asyncssh.KeyImportError, asyncssh.KeyEncryptionError) as error:
            raise _key_error(self.name, error) from None
        except OSError:
            raise SshAuthenticationError(self.name) from None

        proxy_socket = None
        try:
            proxy_socket = await self._open_socket(settings)
            return await asyncssh.connect(host=self.host, options=options, sock=proxy_socket)
        except BaseException as error:
            if proxy_socket is not None:
                proxy_socket.close()
            raise await _map_connect_error(self.name, error, options, host_keys) from None


@dataclass(frozen=True, slots=True)
class AliasTarget(SshConnectionTarget):
    """An SSH target resolved through the user's SSH config."""

    alias: str

    @property
    def name(self) -> str:
        return self.alias

    @property
    def host(self) -> str:
        return self.alias

    def _validate(self, settings: Settings) -> None:
        if not resolve_alias(settings.ssh_config_path, self.alias).found:
            raise SshHostAliasNotFoundError(self.alias)

    def _build_options(
        self,
        settings: Settings,
        host_keys: AsyncSshHostKeyVerifier,
    ) -> asyncssh.SSHClientConnectionOptions:
        return asyncssh.SSHClientConnectionOptions(
            host=self.alias,
            client_factory=host_keys.create_client,
            config=_existing_config_paths(settings.ssh_config_path),
            known_hosts=host_keys.match_known_hosts,
            connect_timeout=settings.ssh_connect_timeout_s,
            keepalive_interval=settings.ssh_keepalive_interval_s,
            keepalive_count_max=settings.ssh_keepalive_count_max,
        )


@dataclass(frozen=True, slots=True)
class DirectTarget(SshConnectionTarget):
    """An SSH target described without an SSH config alias."""

    hostname: str
    port: int = 22
    username: str | None = None
    identity_file: str | None = None

    @property
    def name(self) -> str:
        return self.hostname

    @property
    def host(self) -> str:
        return self.hostname

    def _validate(self, _settings: Settings) -> None:
        return None

    def _build_options(
        self,
        settings: Settings,
        host_keys: AsyncSshHostKeyVerifier,
    ) -> asyncssh.SSHClientConnectionOptions:
        return asyncssh.SSHClientConnectionOptions(
            host=self.hostname,
            port=self.port,
            username=self.username,
            client_keys=[str(Path(self.identity_file).expanduser())] if self.identity_file else None,
            client_factory=host_keys.create_client,
            config=[],
            known_hosts=host_keys.match_known_hosts,
            connect_timeout=settings.ssh_connect_timeout_s,
            keepalive_interval=settings.ssh_keepalive_interval_s,
            keepalive_count_max=settings.ssh_keepalive_count_max,
        )

    async def _open_socket(self, settings: Settings) -> socket.socket | None:
        proxy = resolve_http_connect_proxy(self.hostname, self.port)
        if proxy is None:
            return None
        return await open_http_connect_socket(proxy, self.hostname, self.port, settings.ssh_connect_timeout_s)


def _existing_config_paths(config_path: Path) -> list[str]:
    return [str(config_path)] if config_path.is_file() else []


def _identity_files(options: asyncssh.SSHClientConnectionOptions) -> list[str]:
    configured = options.config.get("IdentityFile")
    if isinstance(configured, str):
        return [configured]
    if isinstance(configured, Sequence):
        return [str(entry) for entry in configured]
    return []


def _is_passphrase_protected(path: Path) -> bool:
    try:
        asyncssh.read_private_key(str(path))
    except asyncssh.KeyImportError as error:
        return str(error).startswith(_PASSPHRASE_ERROR_PREFIX)
    except (OSError, asyncssh.KeyEncryptionError, ValueError):
        return False
    return False


def _passphrase_protected_identity_files(options: asyncssh.SSHClientConnectionOptions) -> list[Path]:
    paths = (Path(entry).expanduser() for entry in _identity_files(options))
    return [path for path in paths if path.is_file() and _is_passphrase_protected(path)]


async def _agent_has_keys(agent_path: str | None) -> bool:
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
    encrypted = await asyncio.to_thread(_passphrase_protected_identity_files, options)
    if not encrypted:
        return False
    agent_path = options.agent_path if isinstance(options.agent_path, str) else None
    return not await _agent_has_keys(agent_path)


def _key_error(alias: str, error: BaseException) -> SshAgentRequiredError | SshAuthenticationError:
    if str(error).startswith(_PASSPHRASE_ERROR_PREFIX) or isinstance(error, asyncssh.KeyEncryptionError):
        return SshAgentRequiredError(alias)
    return SshAuthenticationError(alias)


async def _map_connect_error(  # noqa: PLR0911
    alias: str,
    error: BaseException,
    options: asyncssh.SSHClientConnectionOptions,
    host_keys: AsyncSshHostKeyVerifier,
) -> BaseException:
    if isinstance(error, asyncio.CancelledError):
        return error

    if isinstance(error, asyncssh.HostKeyNotVerifiable):
        logger.warning("SSH host key verification failed for alias '{}'", alias)
        return host_keys.verification_error()

    if isinstance(error, asyncssh.PermissionDenied):
        if await _needs_agent(options):
            return SshAgentRequiredError(alias)
        return SshAuthenticationError(alias)

    if isinstance(error, asyncssh.KeyImportError | asyncssh.KeyEncryptionError):
        return _key_error(alias, error)

    if isinstance(error, TimeoutError):
        return SshConnectionError(alias, reason=_REASON_TIMEOUT)

    if isinstance(error, asyncssh.ConnectionLost):
        return SshConnectionError(alias, reason=_REASON_CONNECTION_LOST)

    if isinstance(error, asyncssh.ProtocolError | asyncssh.KeyExchangeFailed):
        return SshConnectionError(alias, reason=_REASON_PROTOCOL)

    if isinstance(error, OSError | socket.gaierror):
        return SshConnectionError(alias, reason=_REASON_UNREACHABLE)

    if isinstance(error, asyncssh.Error):
        return SshConnectionError(alias, reason=_REASON_PROTOCOL)

    if isinstance(error, Exception):
        logger.warning("Unexpected SSH failure for alias '{}': {}", alias, type(error).__name__)
        return SshConnectionError(alias, reason=_REASON_UNREACHABLE)

    return error
