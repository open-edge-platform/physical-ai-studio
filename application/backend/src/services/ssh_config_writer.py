# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Append-only writer for a new ``Host`` stanza in ``~/.ssh/config``.

Pairs with :mod:`services.ssh_config_reader`: the reader stays read-only, this
module is the one place that ever appends a stanza to the user's own SSH
config, so a user can add a host from the UI without hand-editing the file.

Same non-secret guarantee as the rest of the SSH feature: the only
credential-adjacent field accepted is ``identity_file``, a *path* the user
already has on disk. Studio never reads its contents, and nothing here writes
a key, password, or passphrase.

Never edits an existing stanza: adding an alias that already exists in the
config is rejected with :class:`exceptions.ResourceAlreadyExistsError` rather
than merged or overwritten, so this module never has to parse and rewrite a
stanza it did not create.

``add_verified_host_alias`` is the entry point everything above this module
should call: it never leaves a stanza in the file that Studio has not itself
confirmed it can dial. A typo'd hostname, port, user, or key path is rejected
before it ever reaches the user's real ``~/.ssh/config``, the same way a
training-server save is gated on Tier 1 preflight before it reaches the
database.
"""

import asyncio
from pathlib import Path

from exceptions import ResourceAlreadyExistsError, SshConnectionError
from schemas.remote_server import SshHostAliasCreate, SshHostAliasOption
from services.ssh.transport import open_transport
from services.ssh_config_reader import list_host_aliases
from settings import Settings, get_settings


def _render_stanza(config: SshHostAliasCreate) -> str:
    lines = [f"Host {config.alias}", f"    HostName {config.hostname}", f"    Port {config.port}"]
    if config.user:
        lines.append(f"    User {config.user}")
    if config.identity_file:
        lines.append(f"    IdentityFile {config.identity_file}")
        # Without this, ssh still offers agent keys and other default identities
        # before (or instead of) the one the user just named, so a server can
        # reject the connection on an unrelated key, or authenticate with the
        # wrong one, before the specified key is ever tried.
        lines.append("    IdentitiesOnly yes")
    return "\n".join(lines) + "\n"


def add_host_alias(config_path: Path, config: SshHostAliasCreate) -> SshHostAliasOption:
    """Append a new ``Host`` stanza for ``config.alias`` and return it as an option.

    Writes unconditionally, with no connectivity check - see
    `add_verified_host_alias` for the gated entry point every caller outside
    this module's own tests should use instead.

    Raises:
        ResourceAlreadyExistsError: ``config.alias`` already names a literal
            ``Host`` pattern in the config (or a file it ``Include``s).
    """
    if any(existing.alias == config.alias for existing in list_host_aliases(config_path)):
        raise ResourceAlreadyExistsError(
            "SSH host alias", f"A Host entry named '{config.alias}' already exists in your SSH config."
        )

    config_path = config_path.expanduser()
    config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing_text = config_path.read_text() if config_path.is_file() else ""
    separator = "\n" if existing_text and not existing_text.endswith("\n") else ""
    with config_path.open("a") as handle:
        handle.write(f"{separator}\n{_render_stanza(config)}" if existing_text else _render_stanza(config))

    return SshHostAliasOption(alias=config.alias, hostname=config.hostname, port=config.port, user=config.user)


async def add_verified_host_alias(
    config_path: Path, config: SshHostAliasCreate, settings: Settings | None = None
) -> SshHostAliasOption:
    """Append a new ``Host`` stanza and confirm Studio can actually dial it.

    Appends the stanza, then opens one bounded SSH connection to the new
    alias. If that connection fails for any reason - unreachable host, wrong
    port, rejected user, bad key path - the just-appended stanza is removed
    before the original ``Ssh*Error`` (already an actionable
    `exceptions.BaseException`) is re-raised, so a typo never lands in the
    user's real ``~/.ssh/config``.

    Raises:
        ResourceAlreadyExistsError: ``config.alias`` already exists.
        SshConnectionError | SshAuthenticationError | SshAgentRequiredError |
            SshHostKeyUnknownError | SshHostKeyMismatchError: The new host
            could not be verified. See `services.ssh.transport.SshTransport`.
    """
    settings = settings or get_settings()
    resolved_path = config_path.expanduser()
    prior_size = resolved_path.stat().st_size if resolved_path.is_file() else None

    option = add_host_alias(config_path, config)

    try:
        async with asyncio.timeout(settings.ssh_preflight_timeout_s):
            async with open_transport(config.alias, settings):
                pass
    except TimeoutError as error:
        _rollback(resolved_path, prior_size)
        raise SshConnectionError(config.alias, reason="timed_out") from error
    except BaseException:
        _rollback(resolved_path, prior_size)
        raise

    return option


def _rollback(resolved_path: Path, prior_size: int | None) -> None:
    """Undo exactly the append `add_host_alias` just made.

    ``prior_size`` is the file's exact byte length before the append, captured
    by the only caller of this helper. Truncating back to it - or deleting the
    file entirely when it did not exist before - removes precisely the bytes
    this call added, never anything a user (or a previous call) wrote.
    """
    if prior_size is None:
        resolved_path.unlink(missing_ok=True)
        return
    with resolved_path.open("r+") as handle:
        handle.truncate(prior_size)
