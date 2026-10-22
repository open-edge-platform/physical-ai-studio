# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for the remote trainer connection union."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.remote_trainer import (
    AwsBatchConnection,
    AwsBatchResourceConfiguration,
    AwsBatchTarget,
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


def test_ssh_connection_leaves_local_port_for_service_to_assign() -> None:
    connection = SshConnection(ssh_host_alias="training-box")

    assert connection.ssh_remote_port == 8001
    assert connection.ssh_local_port is None
    assert connection.url is None


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
    target = {
        "queue": "arn:aws:batch:eu-west-1:123456789012:job-queue/studio",
        "job_definition": "arn:aws:batch:eu-west-1:123456789012:job-definition/studio:1",
    }
    output = {
        "schema_version": 1,
        "region": "eu-west-1",
        "studio_role_arn": "arn:aws:iam::123456789012:role/studio",
        "bucket": "jobs-bucket",
        "targets": {"g4dn.xlarge": target},
    }

    config = RemoteTrainerCreate.model_validate(
        {"name": "batch", "connection": {"connection_mode": "aws_batch", **output}}
    )

    assert isinstance(config.connection, AwsBatchConnection)
    assert config.connection.targets["g4dn.xlarge"].queue == target["queue"]
    assert config.connection.targets["g4dn.xlarge"].max_trainers is None


@pytest.mark.parametrize("capacity", [1, 4, 16])
def test_batch_target_accepts_configured_capacity(capacity: int) -> None:
    target = AwsBatchTarget(
        queue="arn:aws:batch:eu-west-1:123456789012:job-queue/studio",
        job_definition="arn:aws:batch:eu-west-1:123456789012:job-definition/studio:1",
        max_trainers=capacity,
    )
    assert target.max_trainers == capacity


@pytest.mark.parametrize("capacity", [0, -1, 1.5])
def test_batch_target_rejects_invalid_capacity(capacity: float) -> None:
    with pytest.raises(ValidationError, match="max_trainers"):
        AwsBatchTarget.model_validate(
            {
                "queue": "arn:aws:batch:eu-west-1:123456789012:job-queue/studio",
                "job_definition": "arn:aws:batch:eu-west-1:123456789012:job-definition/studio:1",
                "max_trainers": capacity,
            }
        )


def test_aws_batch_connection_requires_at_least_one_target() -> None:
    with pytest.raises(ValidationError, match="targets"):
        AwsBatchConnection(
            region="eu-west-1",
            studio_role_arn="arn:aws:iam::123456789012:role/studio",
            bucket="jobs-bucket",
            targets={},
        )


def test_remote_trainer_exposes_mode_and_url_as_computed_fields() -> None:
    aws = RemoteTrainer(
        id=uuid4(),
        name="batch",
        connection=AwsBatchConnection(
            region="eu-west-1",
            studio_role_arn="arn:aws:iam::123456789012:role/studio",
            bucket="jobs-bucket",
            targets={
                "g4dn.xlarge": {
                    "queue": "arn:aws:batch:eu-west-1:123456789012:job-queue/studio",
                    "job_definition": "arn:aws:batch:eu-west-1:123456789012:job-definition/studio:1",
                }
            },
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


@pytest.fixture(params=[AwsBatchResourceConfiguration, AwsBatchConnection])
def aws_resources(request):
    return request.param, {
        "schema_version": 1,
        "region": "eu-west-1",
        "studio_role_arn": "arn:aws:iam::123456789012:role/studio",
        "bucket": "jobs-bucket",
        "targets": {
            "gpu": {
                "queue": "arn:aws:batch:eu-west-1:123456789012:job-queue/studio",
                "job_definition": "arn:aws:batch:eu-west-1:123456789012:job-definition/studio:1",
            }
        },
    }


@pytest.mark.parametrize(
    "partition,region",
    [("aws", "eu-west-1"), ("aws-cn", "cn-north-1"), ("aws-us-gov", "us-gov-west-1"), ("aws-eusc", "eusc-de-east-1")],
)
@pytest.mark.parametrize("revision", ["", ":1", ":123"])
def test_aws_resources_accept_valid_arns_and_strip_whitespace(aws_resources, partition, region, revision) -> None:
    model, resources = aws_resources
    resources.update(
        region=f" {region} ",
        studio_role_arn=f" arn:{partition}:iam::123456789012:role/service-role/studio ",
    )
    resources["targets"]["gpu"] = {
        "queue": f" arn:{partition}:batch:{region}:123456789012:job-queue/studio-queue_1 ",
        "job_definition": f" arn:{partition}:batch:{region}:123456789012:job-definition/studio_1{revision} ",
    }

    result = model.model_validate(resources)

    assert result.region == region
    assert result.studio_role_arn.endswith("/studio")
    assert result.targets["gpu"].job_definition.endswith(f"studio_1{revision}")


@pytest.mark.parametrize("region", ["", "us", "US-east-1", "u-east-1", "us-east", "us-east-12", "us--east-1"])
def test_aws_resources_reject_invalid_regions(aws_resources, region) -> None:
    model, resources = aws_resources
    resources["region"] = region
    with pytest.raises(ValidationError, match="region"):
        model.model_validate(resources)


@pytest.mark.parametrize(
    "bucket",
    ["abc", "a" * 63, "jobs.example.com", "123-jobs", "jobs-123456789012-eu-west-1-an"],
)
def test_aws_resources_accept_valid_buckets(aws_resources, bucket) -> None:
    model, resources = aws_resources
    resources["bucket"] = bucket
    assert model.model_validate(resources).bucket == bucket


@pytest.mark.parametrize(
    "bucket",
    [
        "ab",
        "a" * 64,
        "Jobs-bucket",
        "jobs_bucket",
        "-jobs",
        "jobs-",
        ".jobs",
        "jobs.",
        "jobs..bucket",
        "192.168.0.1",
        "001.002.003.004",
        "xn--jobs",
        "sthree-jobs",
        "amzn-s3-demo-jobs",
        "jobs-s3alias",
        "jobs--ol-s3",
        "jobs.mrap",
        "jobs--x-s3",
        "jobs--table-s3",
        "jobs-an",
    ],
)
def test_aws_resources_reject_invalid_buckets(aws_resources, bucket) -> None:
    model, resources = aws_resources
    resources["bucket"] = bucket
    with pytest.raises(ValidationError, match="bucket"):
        model.model_validate(resources)


@pytest.mark.parametrize(
    "arn",
    [
        "arn",
        "arn:aws:iam::123:role/studio",
        "arn:aws:iam::123456789012:user/studio",
        "arn:aws:sts::123456789012:assumed-role/studio/session",
        "arn:aws:iam:eu-west-1:123456789012:role/studio",
        "arn:aws:iam::123456789012:role/",
        "arn:aws:iam::123456789012:role/has space",
        "arn:aws:iam::123456789012:role/stüdio",
        "arn:aws:iam::123456789012:role/" + "a" * 65,
    ],
)
def test_aws_resources_reject_invalid_role_arns(aws_resources, arn) -> None:
    model, resources = aws_resources
    resources["studio_role_arn"] = arn
    with pytest.raises(ValidationError, match="studio_role_arn"):
        model.model_validate(resources)


@pytest.mark.parametrize("field,resource", [("queue", "job-queue"), ("job_definition", "job-definition")])
@pytest.mark.parametrize(
    "arn",
    [
        "plain-name",
        "arn:aws:batch:q",
        "arn:aws:batch:eu-west-1:123:{resource}/studio",
        "arn:aws:ecs:eu-west-1:123456789012:{resource}/studio",
        "arn:aws:batch:eu-west-1:123456789012:job/studio",
        "arn:aws:batch:EU-west-1:123456789012:{resource}/studio",
        "arn:aws:batch:eu-west-1:123456789012:{resource}/",
        "arn:aws:batch:eu-west-1:123456789012:{resource}/has space",
        "arn:aws:batch:eu-west-1:123456789012:{resource}/" + "a" * 129,
    ],
)
def test_aws_resources_reject_invalid_batch_arns(aws_resources, field, resource, arn) -> None:
    model, resources = aws_resources
    resources["targets"]["gpu"][field] = arn.format(resource=resource)
    with pytest.raises(ValidationError, match=field):
        model.model_validate(resources)


@pytest.mark.parametrize("revision", [":0", ":-1", ":latest", ":1:2", ":"])
def test_aws_resources_reject_invalid_job_definition_revisions(aws_resources, revision) -> None:
    model, resources = aws_resources
    resources["targets"]["gpu"]["job_definition"] = (
        f"arn:aws:batch:eu-west-1:123456789012:job-definition/studio{revision}"
    )
    with pytest.raises(ValidationError, match="job_definition"):
        model.model_validate(resources)


@pytest.mark.parametrize("target", ["gpu", "second-gpu"])
@pytest.mark.parametrize("field", ["queue", "job_definition"])
def test_aws_resources_reject_batch_arns_in_another_region(aws_resources, target, field) -> None:
    model, resources = aws_resources
    resources["targets"][target] = resources["targets"]["gpu"].copy()
    resources["targets"][target][field] = resources["targets"][target][field].replace("eu-west-1", "us-east-1")
    with pytest.raises(ValidationError, match=rf"targets\.{target}\.{field} must be in region eu-west-1"):
        model.model_validate(resources)
