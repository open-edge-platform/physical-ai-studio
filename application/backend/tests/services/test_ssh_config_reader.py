# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the read-only SSH config reader.

The critical test here is ``test_resolve_never_leaks_credential_directives``:
it builds a fixture stanza where ``IdentityFile``, ``IdentityAgent``,
``CertificateFile``, and a fabricated ``Password`` directive are all present
with recognizable dummy values, then asserts none of those values appear
anywhere in the serialized response - not just that the schema has no field
for them.
"""

from pathlib import Path

from services.ssh_config_reader import list_host_aliases, resolve_alias

_DUMMY_IDENTITY_FILE = "id_dummy_test_key"
_DUMMY_IDENTITY_AGENT = "SSH_AUTH_SOCK"
_DUMMY_CERTIFICATE = "id_dummy-cert.pub"
_DUMMY_PASSWORD = "somesecretvalue"


def _write_config(tmp_path: Path, contents: str, name: str = "config") -> Path:
    config_path = tmp_path / name
    config_path.write_text(contents)
    return config_path


def test_resolve_alias_missing_config_file_returns_not_found(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist"

    result = resolve_alias(missing_path, "my-gpu-box")

    assert result.found is False
    assert result.alias == "my-gpu-box"
    assert result.hostname is None
    assert result.port is None
    assert result.user is None


def test_list_host_aliases_missing_config_file_returns_empty(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist"

    assert list_host_aliases(missing_path) == []


def test_resolve_alias_empty_config_file_returns_not_found(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "")

    result = resolve_alias(config_path, "my-gpu-box")

    assert result.found is False


def test_list_host_aliases_empty_config_file_returns_empty(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "")

    assert list_host_aliases(config_path) == []


def test_resolve_alias_finds_literal_host_stanza(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        Host my-gpu-box
            HostName 10.0.0.5
            Port 2222
            User trainer
        """,
    )

    result = resolve_alias(config_path, "my-gpu-box")

    assert result.found is True
    assert result.alias == "my-gpu-box"
    assert result.hostname == "10.0.0.5"
    assert result.port == 2222
    assert result.user == "trainer"


def test_resolve_alias_falls_back_to_alias_as_hostname(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        Host my-gpu-box
            User trainer
        """,
    )

    result = resolve_alias(config_path, "my-gpu-box")

    assert result.found is True
    assert result.hostname == "my-gpu-box"
    assert result.port is None


def test_resolve_alias_missing_alias_returns_not_found(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        Host my-gpu-box
            HostName 10.0.0.5
        """,
    )

    result = resolve_alias(config_path, "some-other-box")

    assert result.found is False


def test_resolve_alias_wildcard_only_stanza_returns_not_found(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        Host my-gpu-*
            HostName 10.0.0.5
        """,
    )

    result = resolve_alias(config_path, "my-gpu-box")

    assert result.found is False


def test_list_host_aliases_excludes_wildcard_only_stanza(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        Host *
            User default-user

        Host bastion*
            HostName 10.0.0.9

        Host my-gpu-box
            HostName 10.0.0.5
            Port 2222
            User trainer
        """,
    )

    aliases = list_host_aliases(config_path)

    assert [option.alias for option in aliases] == ["my-gpu-box"]
    assert aliases[0].hostname == "10.0.0.5"
    assert aliases[0].port == 2222
    assert aliases[0].user == "trainer"


def test_list_host_aliases_falls_back_to_alias_as_hostname(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        Host my-gpu-box
        """,
    )

    aliases = list_host_aliases(config_path)

    assert len(aliases) == 1
    assert aliases[0].alias == "my-gpu-box"
    assert aliases[0].hostname == "my-gpu-box"


def test_resolve_alias_follows_include_one_level_deep(tmp_path: Path) -> None:
    included_path = _write_config(
        tmp_path,
        """
        Host my-gpu-box
            HostName 10.0.0.5
            Port 2222
            User trainer
        """,
        name="included_config",
    )
    main_path = _write_config(
        tmp_path,
        f"""
        Include {included_path.name}
        """,
    )

    result = resolve_alias(main_path, "my-gpu-box")

    assert result.found is True
    assert result.hostname == "10.0.0.5"
    assert result.port == 2222
    assert result.user == "trainer"


def test_resolve_alias_later_stanza_overrides_earlier_one(tmp_path: Path) -> None:
    """A later ``Host`` stanza with the same alias overrides earlier fields.

    This is last-stanza-wins, deliberately the opposite of real ssh's
    first-obtained-value-wins rule, so an ``Include``d override file takes
    effect. Fields the later stanza does not set (here, ``User``) keep the
    earlier stanza's value rather than being cleared.
    """
    config_path = _write_config(
        tmp_path,
        """
        Host my-gpu-box
            HostName 10.0.0.5
            Port 2222
            User trainer

        Host my-gpu-box
            HostName 10.0.0.9
        """,
    )

    result = resolve_alias(config_path, "my-gpu-box")

    assert result.found is True
    assert result.hostname == "10.0.0.9"
    assert result.port == 2222
    assert result.user == "trainer"


def test_list_host_aliases_merges_duplicate_alias_last_stanza_wins(tmp_path: Path) -> None:
    """A duplicate ``Host`` alias must be listed once, merged like ``resolve_alias``.

    Mirrors ``test_resolve_alias_later_stanza_overrides_earlier_one``: the two
    functions must agree on the resolved fields for the same alias, and the
    alias must not appear twice just because two stanzas define it.
    """
    config_path = _write_config(
        tmp_path,
        """
        Host my-gpu-box
            HostName 10.0.0.5
            Port 2222
            User trainer

        Host my-gpu-box
            HostName 10.0.0.9
        """,
    )

    aliases = list_host_aliases(config_path)

    assert [option.alias for option in aliases] == ["my-gpu-box"]
    option = aliases[0]
    assert option.hostname == "10.0.0.9"
    assert option.port == 2222
    assert option.user == "trainer"


def test_list_host_aliases_follows_include_one_level_deep(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
        Host my-gpu-box
            HostName 10.0.0.5
        """,
        name="included_config",
    )
    main_path = _write_config(
        tmp_path,
        """
        Include included_config
        """,
    )

    aliases = list_host_aliases(main_path)

    assert [option.alias for option in aliases] == ["my-gpu-box"]


def test_resolve_never_leaks_credential_directives(tmp_path: Path) -> None:
    """The reader must never surface identity/credential directive values.

    A fabricated ``Password`` directive is included even though real
    ssh_config has no such keyword - this is a defense-in-depth check, not a
    claim that ssh_config supports it.
    """
    config_path = _write_config(
        tmp_path,
        f"""
        Host my-gpu-box
            HostName 10.0.0.5
            Port 2222
            User trainer
            IdentityFile ~/.ssh/{_DUMMY_IDENTITY_FILE}
            IdentityAgent {_DUMMY_IDENTITY_AGENT}
            CertificateFile ~/.ssh/{_DUMMY_CERTIFICATE}
            Password {_DUMMY_PASSWORD}
        """,
    )

    result = resolve_alias(config_path, "my-gpu-box")
    serialized_json = result.model_dump_json()
    serialized_dict = str(result.model_dump())

    for dummy_secret in (_DUMMY_IDENTITY_FILE, _DUMMY_IDENTITY_AGENT, _DUMMY_CERTIFICATE, _DUMMY_PASSWORD):
        assert dummy_secret not in serialized_json
        assert dummy_secret not in serialized_dict

    # Sanity: the non-secret fields still resolved correctly around the
    # credential directives, proving the parser did not just fail silently.
    assert result.found is True
    assert result.hostname == "10.0.0.5"
    assert result.port == 2222
    assert result.user == "trainer"


def test_list_host_aliases_never_leaks_credential_directives(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        f"""
        Host my-gpu-box
            HostName 10.0.0.5
            IdentityFile ~/.ssh/{_DUMMY_IDENTITY_FILE}
            IdentityAgent {_DUMMY_IDENTITY_AGENT}
            CertificateFile ~/.ssh/{_DUMMY_CERTIFICATE}
            Password {_DUMMY_PASSWORD}
        """,
    )

    aliases = list_host_aliases(config_path)
    serialized_json = "".join(option.model_dump_json() for option in aliases)

    for dummy_secret in (_DUMMY_IDENTITY_FILE, _DUMMY_IDENTITY_AGENT, _DUMMY_CERTIFICATE, _DUMMY_PASSWORD):
        assert dummy_secret not in serialized_json
