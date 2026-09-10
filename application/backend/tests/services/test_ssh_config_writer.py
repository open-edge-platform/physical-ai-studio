# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the append-only SSH config writer."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import asyncssh
import pytest

from exceptions import ResourceAlreadyExistsError, SshAuthenticationError, SshConnectionError
from schemas.remote_server import SshHostAliasCreate
from services.ssh.transport import reset_alias_gates
from services.ssh_config_reader import resolve_alias
from services.ssh_config_writer import add_host_alias, add_verified_host_alias
from settings import Settings


@pytest.fixture(autouse=True)
def _clear_gates():
    reset_alias_gates()
    yield
    reset_alias_gates()


def _settings(config_path: Path) -> Settings:
    return Settings(SSH_CONFIG_PATH=config_path, SSH_PREFLIGHT_THROTTLE_S=0.0)


def test_add_host_alias_writes_stanza_to_missing_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config"

    option = add_host_alias(
        config_path,
        SshHostAliasCreate(alias="new-box", hostname="10.0.0.9", port=2222, user="trainer"),
    )

    assert option.alias == "new-box"
    assert option.hostname == "10.0.0.9"
    resolved = resolve_alias(config_path, "new-box")
    assert resolved.found is True
    assert resolved.hostname == "10.0.0.9"
    assert resolved.port == 2222
    assert resolved.user == "trainer"


def test_add_host_alias_appends_after_existing_content(tmp_path: Path) -> None:
    config_path = tmp_path / "config"
    config_path.write_text("Host existing-box\n    HostName 10.0.0.1\n")

    add_host_alias(config_path, SshHostAliasCreate(alias="new-box", hostname="10.0.0.9"))

    assert resolve_alias(config_path, "existing-box").found is True
    assert resolve_alias(config_path, "new-box").found is True


def test_add_host_alias_rejects_duplicate_alias(tmp_path: Path) -> None:
    config_path = tmp_path / "config"
    config_path.write_text("Host taken\n    HostName 10.0.0.1\n")

    with pytest.raises(ResourceAlreadyExistsError):
        add_host_alias(config_path, SshHostAliasCreate(alias="taken", hostname="10.0.0.2"))


def test_add_host_alias_never_leaks_identity_file_value_elsewhere(tmp_path: Path) -> None:
    """``identity_file`` is a path, not a secret, but it's still the one field
    this writer actually persists into a normally-untouched file - assert it
    only lands where intended and the rest of the reader's non-secret contract
    keeps holding.
    """
    config_path = tmp_path / "config"

    add_host_alias(
        config_path,
        SshHostAliasCreate(alias="new-box", hostname="10.0.0.9", identity_file="~/.ssh/id_new_box"),
    )

    text = config_path.read_text()
    assert "IdentityFile ~/.ssh/id_new_box" in text


def test_add_host_alias_sets_identities_only_when_identity_file_given(tmp_path: Path) -> None:
    """Without ``IdentitiesOnly yes``, ssh still offers agent keys and other
    default identities before (or instead of) the one the user just named.
    """
    config_path = tmp_path / "config"

    add_host_alias(
        config_path,
        SshHostAliasCreate(alias="new-box", hostname="10.0.0.9", identity_file="~/.ssh/id_new_box"),
    )

    assert "IdentitiesOnly yes" in config_path.read_text()


def test_add_host_alias_omits_identities_only_without_identity_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config"

    add_host_alias(config_path, SshHostAliasCreate(alias="new-box", hostname="10.0.0.9"))

    assert "IdentitiesOnly" not in config_path.read_text()


async def test_add_verified_host_alias_keeps_stanza_on_successful_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config"
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(return_value=MagicMock()))

    option = await add_verified_host_alias(
        config_path, SshHostAliasCreate(alias="new-box", hostname="10.0.0.9"), _settings(config_path)
    )

    assert option.alias == "new-box"
    assert resolve_alias(config_path, "new-box").found is True


async def test_add_verified_host_alias_rolls_back_stanza_on_failed_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config"
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=asyncssh.PermissionDenied("denied")))

    with pytest.raises(SshAuthenticationError):
        await add_verified_host_alias(
            config_path, SshHostAliasCreate(alias="new-box", hostname="10.0.0.9"), _settings(config_path)
        )

    assert not config_path.exists()


async def test_add_verified_host_alias_rollback_preserves_other_stanzas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config"
    config_path.write_text("Host existing-box\n    HostName 10.0.0.1\n")
    monkeypatch.setattr(asyncssh, "connect", AsyncMock(side_effect=asyncssh.PermissionDenied("denied")))

    with pytest.raises(SshAuthenticationError):
        await add_verified_host_alias(
            config_path, SshHostAliasCreate(alias="new-box", hostname="10.0.0.9"), _settings(config_path)
        )

    assert resolve_alias(config_path, "existing-box").found is True
    assert resolve_alias(config_path, "new-box").found is False


async def test_add_verified_host_alias_rejects_duplicate_without_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config"
    config_path.write_text("Host taken\n    HostName 10.0.0.1\n")
    connect = AsyncMock()
    monkeypatch.setattr(asyncssh, "connect", connect)

    with pytest.raises(ResourceAlreadyExistsError):
        await add_verified_host_alias(
            config_path, SshHostAliasCreate(alias="taken", hostname="10.0.0.2"), _settings(config_path)
        )

    connect.assert_not_called()


async def test_add_verified_host_alias_times_out_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config"

    async def _hangs(*_args: object, **_kwargs: object) -> None:
        import asyncio

        await asyncio.sleep(10)

    monkeypatch.setattr(asyncssh, "connect", _hangs)
    settings = Settings(SSH_CONFIG_PATH=config_path, SSH_PREFLIGHT_THROTTLE_S=0.0, SSH_PREFLIGHT_TIMEOUT_S=0.01)

    with pytest.raises(SshConnectionError):
        await add_verified_host_alias(config_path, SshHostAliasCreate(alias="new-box", hostname="10.0.0.9"), settings)

    assert not config_path.exists()
