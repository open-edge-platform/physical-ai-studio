# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Security-gate and validation tests for the remote-server schemas.

The most important assertion here is the absence one: no field on any of
these models ever names or carries a credential.
"""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.hardware import DeviceType
from schemas.remote_server import RemoteServer, RemoteServerCreate, RemoteServerUpdate

# Matched as whole underscore-delimited words, not raw substrings, so a
# multi-word forbidden phrase (e.g. "host_key") only matches when a field
# spells out that exact sequence of words.
_FORBIDDEN_FIELD_WORDS = (
    "username",
    "auth_type",
    "ssh_secret_encrypted",
    "ssh_key_passphrase_encrypted",
    "host_key",
    "host",
    "port",
    "password",
    "identity_file",
    "private_key",
)

# `ssh_host_alias` is the one deliberate, documented exception: it is the name
# of an SSH config `Host` stanza, not a hostname, credential, or port, and its
# presence is the entire point of this schema
_KNOWN_SAFE_FIELDS = frozenset({"ssh_host_alias"})


def _contains_word_sequence(words: list[str], needle_words: list[str]) -> bool:
    """Return True if ``needle_words`` appears as a contiguous run in ``words``."""
    span = len(needle_words)
    return any(words[start : start + span] == needle_words for start in range(len(words) - span + 1))


def _assert_no_secret_fields(model_fields: dict) -> None:
    for field_name in model_fields:
        if field_name in _KNOWN_SAFE_FIELDS:
            continue
        words = field_name.lower().split("_")
        for forbidden in _FORBIDDEN_FIELD_WORDS:
            matched = _contains_word_sequence(words, forbidden.split("_"))
            assert not matched, f"field `{field_name}` looks like a secret field (matched `{forbidden}`)"


def test_remote_server_create_has_no_secret_fields() -> None:
    _assert_no_secret_fields(RemoteServerCreate.model_fields)


def test_remote_server_update_has_no_secret_fields() -> None:
    _assert_no_secret_fields(RemoteServerUpdate.model_fields)


def test_remote_server_has_no_secret_fields() -> None:
    _assert_no_secret_fields(RemoteServer.model_fields)


@pytest.mark.parametrize("device_type", [DeviceType.CPU, DeviceType.NPU])
def test_create_rejects_unsupported_device_type(device_type: DeviceType) -> None:
    with pytest.raises(ValidationError):
        RemoteServerCreate(name="server", ssh_host_alias="my-gpu-box", device_type=device_type)


@pytest.mark.parametrize("device_type", [DeviceType.CUDA, DeviceType.XPU])
def test_create_accepts_supported_device_type(device_type: DeviceType) -> None:
    config = RemoteServerCreate(name="server", ssh_host_alias="my-gpu-box", device_type=device_type)
    assert config.device_type == device_type


@pytest.mark.parametrize("device_type", [DeviceType.CPU, DeviceType.NPU])
def test_update_rejects_unsupported_device_type(device_type: DeviceType) -> None:
    with pytest.raises(ValidationError):
        RemoteServerUpdate(device_type=device_type)


@pytest.mark.parametrize("device_type", [DeviceType.CUDA, DeviceType.XPU])
def test_update_accepts_supported_device_type(device_type: DeviceType) -> None:
    update = RemoteServerUpdate(device_type=device_type)
    assert update.device_type == device_type


@pytest.mark.parametrize("alias", ["*", "foo*", "192.168.*.1", "host name", "host;rm -rf", "$(whoami)", "host`x`"])
def test_create_rejects_glob_and_shell_chars_in_alias(alias: str) -> None:
    with pytest.raises(ValidationError):
        RemoteServerCreate(name="server", ssh_host_alias=alias, device_type=DeviceType.CUDA)


@pytest.mark.parametrize("alias", ["*", "foo*", "host name", "host;rm -rf"])
def test_update_rejects_glob_and_shell_chars_in_alias(alias: str) -> None:
    with pytest.raises(ValidationError):
        RemoteServerUpdate(ssh_host_alias=alias)


def test_create_strips_whitespace_from_name_and_alias() -> None:
    config = RemoteServerCreate(name="  server  ", ssh_host_alias="my-gpu-box", device_type=DeviceType.CUDA)
    assert config.name == "server"
    assert config.ssh_host_alias == "my-gpu-box"


def test_create_rejects_whitespace_only_name() -> None:
    with pytest.raises(ValidationError):
        RemoteServerCreate(name="   ", ssh_host_alias="my-gpu-box", device_type=DeviceType.CUDA)


def test_update_strips_whitespace_from_name_and_alias() -> None:
    update = RemoteServerUpdate(name="  server  ", ssh_host_alias="  my-gpu-box  ")
    assert update.name == "server"
    assert update.ssh_host_alias == "my-gpu-box"


def test_remote_server_defaults_to_unknown_check_status() -> None:
    server = RemoteServer(id=uuid4(), name="server", ssh_host_alias="my-gpu-box", device_type=DeviceType.CUDA)
    assert server.last_check_status == "unknown"
    assert server.last_check_at is None
