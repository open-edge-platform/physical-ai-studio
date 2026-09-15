# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for the optional SSH tunnel config on a direct trainer."""

import pytest
from pydantic import ValidationError

from schemas.remote_trainer import ManualSshConnection, RemoteTrainerConnectionMode, RemoteTrainerCreate


def test_remote_trainer_without_ssh_fields_has_tunnel_disabled() -> None:
    config = RemoteTrainerCreate(name="trainer", url="http://127.0.0.1:8001")

    assert config.ssh_host_alias is None
    assert config.ssh_connection is None
    assert config.ssh_remote_port is None
    assert config.ssh_local_port is None


def test_remote_trainer_ssh_tunnel_computes_local_url() -> None:
    config = RemoteTrainerCreate(
        name="trainer",
        connection_mode=RemoteTrainerConnectionMode.SSH,
        ssh_host_alias="training-box",
        ssh_remote_port=8001,
        ssh_local_port=9001,
    )

    assert config.ssh_remote_port == 8001
    assert str(config.url) == "http://127.0.0.1:9001/"


def test_remote_trainer_accepts_a_manual_ssh_connection() -> None:
    connection = ManualSshConnection(
        hostname="gpu.example.test",
        port=2222,
        user="trainer",
        identity_file="~/.ssh/trainer",
    )

    config = RemoteTrainerCreate(
        name="trainer",
        connection_mode=RemoteTrainerConnectionMode.SSH,
        ssh_connection=connection,
        ssh_remote_port=8001,
        ssh_local_port=8001,
    )

    assert config.ssh_connection == connection
    assert config.ssh_remote_port == 8001


def test_manual_ssh_connection_uses_ec2_user_by_default() -> None:
    connection = ManualSshConnection(hostname="gpu.example.test")

    assert connection.user == "ec2-user"


def test_remote_trainer_ssh_tunnel_uses_trainer_ports_by_default() -> None:
    config = RemoteTrainerCreate(
        name="trainer",
        connection_mode=RemoteTrainerConnectionMode.SSH,
        ssh_host_alias="training-box",
    )

    assert config.ssh_remote_port == 8001
    assert config.ssh_local_port == 8001
    assert str(config.url) == "http://127.0.0.1:8001/"


def test_remote_trainer_rejects_alias_and_manual_connection_together() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        RemoteTrainerCreate(
            name="trainer",
            connection_mode=RemoteTrainerConnectionMode.SSH,
            ssh_host_alias="training-box",
            ssh_connection=ManualSshConnection(hostname="gpu.example.test"),
            ssh_remote_port=8001,
            ssh_local_port=8001,
        )


def test_remote_trainer_ssh_tunnel_requires_local_port() -> None:
    with pytest.raises(ValidationError, match="ssh_local_port is required"):
        RemoteTrainerCreate(
            name="trainer",
            connection_mode=RemoteTrainerConnectionMode.SSH,
            ssh_host_alias="training-box",
            ssh_remote_port=8001,
            ssh_local_port=None,
        )


def test_remote_trainer_rejects_ssh_port_fields_without_connection() -> None:
    with pytest.raises(ValidationError, match="requires either"):
        RemoteTrainerCreate(
            name="trainer",
            connection_mode=RemoteTrainerConnectionMode.SSH,
            ssh_remote_port=8001,
            ssh_local_port=8001,
        )


def test_remote_trainer_ssh_host_alias_rejects_wildcard() -> None:
    with pytest.raises(ValidationError):
        RemoteTrainerCreate(
            name="trainer",
            connection_mode=RemoteTrainerConnectionMode.SSH,
            ssh_host_alias="*.internal",
            ssh_remote_port=8001,
            ssh_local_port=8001,
        )
