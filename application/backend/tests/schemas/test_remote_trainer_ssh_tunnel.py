# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for the optional SSH tunnel config on a direct trainer."""

import pytest
from pydantic import ValidationError

from schemas.remote_trainer import RemoteTrainerCreate


def test_remote_trainer_without_ssh_fields_has_tunnel_disabled() -> None:
    config = RemoteTrainerCreate(name="trainer", url="http://127.0.0.1:8001")

    assert config.ssh_host_alias is None
    assert config.ssh_remote_port is None
    assert config.ssh_local_port is None


def test_remote_trainer_ssh_tunnel_defaults_remote_port_from_url() -> None:
    config = RemoteTrainerCreate(
        name="trainer", url="http://127.0.0.1:8001", ssh_host_alias="training-box", ssh_local_port=8001
    )

    assert config.ssh_remote_port == 8001


def test_remote_trainer_ssh_tunnel_requires_local_port() -> None:
    with pytest.raises(ValidationError, match="ssh_local_port is required"):
        RemoteTrainerCreate(name="trainer", url="http://127.0.0.1:8001", ssh_host_alias="training-box")


def test_remote_trainer_rejects_ssh_port_fields_without_alias() -> None:
    with pytest.raises(ValidationError, match="require ssh_host_alias"):
        RemoteTrainerCreate(name="trainer", url="http://127.0.0.1:8001", ssh_local_port=8001)


def test_remote_trainer_ssh_host_alias_rejects_wildcard() -> None:
    with pytest.raises(ValidationError):
        RemoteTrainerCreate(
            name="trainer", url="http://127.0.0.1:8001", ssh_host_alias="*.internal", ssh_local_port=8001
        )
