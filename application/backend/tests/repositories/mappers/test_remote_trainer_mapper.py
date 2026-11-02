# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from db.schema import RemoteTrainerDB
from repositories.mappers.remote_trainer_mapper import RemoteTrainerMapper
from schemas.remote_trainer import (
    AwsBatchConnection,
    AwsBatchTarget,
    DirectConnection,
    ManualSshConnection,
    RemoteTrainer,
    SshConnection,
)


@pytest.mark.parametrize(
    "connection",
    [
        DirectConnection(url="https://trainer.test"),
        SshConnection(ssh_host_alias="training-box", ssh_remote_port=8001, ssh_local_port=9001),
        SshConnection(ssh_connection=ManualSshConnection(hostname="gpu.example.test", user="ec2-user")),
        AwsBatchConnection(
            region="eu-west-1",
            studio_role_arn="arn:aws:iam::123456789012:role/studio",
            bucket="jobs-bucket",
            targets={
                "g4dn.xlarge": AwsBatchTarget(
                    queue="arn:aws:batch:eu-west-1:123456789012:job-queue/studio",
                    job_definition="arn:aws:batch:eu-west-1:123456789012:job-definition/studio:1",
                )
            },
        ),
    ],
    ids=["direct", "ssh-alias", "ssh-manual", "aws-batch"],
)
def test_round_trip_preserves_connection(connection) -> None:
    trainer = RemoteTrainer(id=uuid4(), name="trainer", connection=connection)

    row = RemoteTrainerMapper.to_schema(trainer)
    restored = RemoteTrainerMapper.from_schema(row)

    assert row.connection_mode == connection.connection_mode.value
    assert restored.connection == connection
    assert restored.connection_mode is connection.connection_mode


@pytest.mark.parametrize(
    ("port_fields", "expected_url"),
    [
        ({}, None),
        ({"ssh_local_port": None}, None),
        ({"ssh_local_port": 8001}, "http://127.0.0.1:8001/"),
        ({"ssh_local_port": 9001}, "http://127.0.0.1:9001/"),
    ],
    ids=["omitted-local-port", "unassigned-local-port", "explicit-local-port", "allocated-local-port"],
)
def test_from_schema_reads_json_connection(port_fields: dict[str, int | None], expected_url: str | None) -> None:
    db_row = RemoteTrainerDB(
        id=str(uuid4()),
        name="trainer",
        connection_mode="ssh",
        connection={"connection_mode": "ssh", "ssh_host_alias": "training-box", **port_fields},
    )

    trainer = RemoteTrainerMapper.from_schema(db_row)

    assert isinstance(trainer.connection, SshConnection)
    assert trainer.connection.ssh_host_alias == "training-box"
    assert trainer.connection.ssh_local_port == port_fields.get("ssh_local_port")
    assert (str(trainer.url) if trainer.url is not None else None) == expected_url
