# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AsyncSSH host-key verification and first-use persistence."""

import fcntl
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import asyncssh
from loguru import logger

from exceptions import SshHostKeyConfirmationRequiredError, SshHostKeyMismatchError, SshHostKeyUnknownError


@dataclass(slots=True)
class _HostKeyMatch:
    consulted: bool = False
    trusted_keys: int = 0
    ca_keys: int = 0
    revoked_keys: int = 0

    @property
    def has_entry(self) -> bool:
        return bool(self.trusted_keys or self.ca_keys or self.revoked_keys)


class _FingerprintValidationClient(asyncssh.SSHClient):
    def __init__(self, verifier: "AsyncSshHostKeyVerifier") -> None:
        self._verifier = verifier

    def validate_host_public_key(self, host: str, addr: str, port: int, key: asyncssh.SSHKey) -> bool:
        return self._verifier.validate_host_public_key(host, addr, port, key)


class AsyncSshHostKeyVerifier:
    """Verify AsyncSSH host keys and persist an explicitly accepted first key."""

    def __init__(self, alias: str, known_hosts_path: Path, accepted_fingerprint: str | None = None) -> None:
        self._alias = alias
        self._known_hosts_path = known_hosts_path
        self._accepted_fingerprint = accepted_fingerprint
        self.reset()

    def reset(self) -> None:
        """Clear handshake state before a connection attempt."""
        self._match = _HostKeyMatch()
        self._presented_fingerprint: str | None = None

    def create_client(self) -> asyncssh.SSHClient:
        """Create the AsyncSSH callback adapter for a connection attempt."""
        return _FingerprintValidationClient(self)

    def match_known_hosts(
        self,
        host: str,
        addr: str,
        port: int | None,
    ) -> tuple[Sequence[object], ...]:
        """Return matching known-host entries and record their categories."""
        source: str | bytes = str(self._known_hosts_path) if self._known_hosts_path.is_file() else b""
        result = asyncssh.match_known_hosts(source, host, addr, port)
        self._match = _HostKeyMatch(
            consulted=True,
            trusted_keys=len(result[0]),
            ca_keys=len(result[1]),
            revoked_keys=len(result[2]),
        )
        return result

    def validate_host_public_key(self, host: str, addr: str, port: int, key: asyncssh.SSHKey) -> bool:
        """Persist a first-seen host key only after fingerprint confirmation."""
        if not self._match.consulted or self._match.has_entry:
            return False

        fingerprint = key.get_fingerprint()
        self._presented_fingerprint = fingerprint
        if fingerprint != self._accepted_fingerprint:
            return False

        self._known_hosts_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        host_pattern = host if port == 22 else f"[{host}]:{port}"
        entry = host_pattern.encode() + b" " + key.export_public_key().strip() + b"\n"
        descriptor = os.open(self._known_hosts_path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(descriptor, "r+b") as known_hosts_file:
            fcntl.flock(known_hosts_file, fcntl.LOCK_EX)
            current = asyncssh.match_known_hosts(known_hosts_file.read(), host, addr, port)
            trusted_keys, ca_keys, revoked_keys = current[:3]
            if revoked_keys or ca_keys:
                return False
            if trusted_keys:
                return key in trusted_keys

            known_hosts_file.seek(0, os.SEEK_END)
            known_hosts_file.write(entry)
            known_hosts_file.flush()
            os.fsync(known_hosts_file.fileno())
        logger.info("Accepted first-seen SSH host key for alias '{}'", self._alias)
        return True

    def verification_error(
        self,
    ) -> SshHostKeyConfirmationRequiredError | SshHostKeyUnknownError | SshHostKeyMismatchError:
        """Classify an AsyncSSH host-key verification failure."""
        if self._match.consulted and not self._match.has_entry:
            if self._presented_fingerprint is not None:
                return SshHostKeyConfirmationRequiredError(self._alias, self._presented_fingerprint)
            return SshHostKeyUnknownError(self._alias)
        return SshHostKeyMismatchError(self._alias)
