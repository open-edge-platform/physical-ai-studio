# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for the remote trainer connection union."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.remote_trainer import (
    AwsBatchConnection,
    DirectConnection,
    ManualSshConnection,
    RemoteTrainer,
    RemoteTrainerConnectionMode,
    RemoteTrainerCreate,
    SshConnection,
)


def test_direct_connection_from_dict_discriminates_on_mode() -> None:
    config = RemoteTrainerCreate.model_validate(
        {"name": "trainer", "connection": {"connection_mode": "direct", "url": "http://127.0.0.1:8001"}}
    )

    assert isinstance(config.connection, DirectConnection)
    assert str(config.connection.url) == "http://127.0.0.1:8001/"


def test_ssh_connection_computes_local_url() -> None:
    connection = SshConnection(ssh_host_alias="training-box", ssh_remote_port=8001, ssh_local_port=9001)

    assert str(connection.url) == "http://127.0.0.1:9001/"
    assert connection.host_name == "training-box"


def test_ssh_connection_accepts_a_manual_host() -> None:
    manual = ManualSshConnection(hostname="gpu.example.test", port=2222, user="trainer", identity_file="~/.ssh/t")

    connection = SshConnection(ssh_connection=manual)

    assert connection.ssh_connection == manual
    assert connection.host_name == "gpu.example.test"


def test_manual_ssh_connection_uses_ec2_user_by_default() -> None:
    assert ManualSshConnection(hostname="gpu.example.test").user == "ec2-user"


def test_ssh_connection_uses_trainer_ports_by_default() -> None:
    connection = SshConnection(ssh_host_alias="training-box")

    assert connection.ssh_remote_port == 8001
    assert connection.ssh_local_port == 8001
    assert str(connection.url) == "http://127.0.0.1:8001/"


def test_ssh_connection_rejects_alias_and_manual_host_together() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        SshConnection(ssh_host_alias="training-box", ssh_connection=ManualSshConnection(hostname="gpu.example.test"))


def test_ssh_connection_requires_a_host() -> None:
    with pytest.raises(ValidationError, match="requires either"):
        SshConnection(ssh_remote_port=8001, ssh_local_port=8001)


def test_ssh_host_alias_rejects_wildcard() -> None:
    with pytest.raises(ValidationError):
        SshConnection(ssh_host_alias="*.internal")


def test_aws_batch_connection_parses_cloudformation_output() -> None:
    output = {
        "schema_version": 1,
        "region": "eu-west-1",
        "studio_role_arn": "arn:aws:iam::123456789012:role/studio",
        "bucket": "jobs-bucket",
        "targets": {"g4dn.xlarge": {"queue": "arn:aws:batch:q", "job_definition": "arn:aws:batch:jd"}},
    }

    config = RemoteTrainerCreate.model_validate(
        {"name": "batch", "connection": {"connection_mode": "aws_batch", **output}}
    )

    assert isinstance(config.connection, AwsBatchConnection)
    assert config.connection.targets["g4dn.xlarge"].queue == "arn:aws:batch:q"


def test_aws_batch_connection_requires_at_least_one_target() -> None:
    with pytest.raises(ValidationError):
        AwsBatchConnection(region="eu-west-1", studio_role_arn="arn", bucket="jobs-bucket", targets={})


def test_remote_trainer_exposes_mode_and_url_as_computed_fields() -> None:
    aws = RemoteTrainer(
        id=uuid4(),
        name="batch",
        connection=AwsBatchConnection(
            region="eu-west-1",
            studio_role_arn="arn",
            bucket="jobs-bucket",
            targets={"g4dn.xlarge": {"queue": "q", "job_definition": "jd"}},
        ),
    )
    direct = RemoteTrainer(id=uuid4(), name="direct", connection=DirectConnection(url="https://trainer.test"))

    assert aws.connection_mode is RemoteTrainerConnectionMode.AWS_BATCH
    assert aws.url is None
    assert direct.model_dump(mode="json")["url"] == "https://trainer.test/"
    assert direct.model_dump(mode="json")["connection_mode"] == "direct"


def test_unknown_connection_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RemoteTrainerCreate.model_validate({"name": "x", "connection": {"connection_mode": "gcp"}})
