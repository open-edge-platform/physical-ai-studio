# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from db.schema import RemoteTrainerDB
from repositories.mappers.remote_trainer_mapper import RemoteTrainerMapper


def test_from_schema_uses_alias_instead_of_manual_connection() -> None:
    db_row = RemoteTrainerDB(
        id=str(uuid4()),
        name="trainer",
        connection_mode="ssh",
        url="http://127.0.0.1:8001",
        ssh_host_alias="training-box",
        ssh_hostname="ignored.example.test",
        ssh_remote_port=8001,
        ssh_local_port=8001,
    )

    trainer = RemoteTrainerMapper.from_schema(db_row)

    assert trainer.ssh_host_alias == "training-box"
    assert trainer.ssh_connection is None


def test_from_schema_uses_manual_connection_when_alias_is_absent() -> None:
    db_row = RemoteTrainerDB(
        id=str(uuid4()),
        name="trainer",
        connection_mode="ssh",
        url="http://127.0.0.1:8001",
        ssh_host_alias=None,
        ssh_hostname="gpu.example.test",
        ssh_port=22,
        ssh_username="ec2-user",
        ssh_remote_port=8001,
        ssh_local_port=8001,
    )

    trainer = RemoteTrainerMapper.from_schema(db_row)

    assert trainer.ssh_connection is not None
    assert trainer.ssh_connection.hostname == "gpu.example.test"
    assert trainer.ssh_connection.user == "ec2-user"
