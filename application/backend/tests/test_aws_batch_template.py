import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from schemas.remote_trainer import AwsBatchResourceConfiguration
from trainer.batch.settings import BatchSettings


def _template() -> dict:
    path = Path(__file__).resolve().parents[2] / "cloudformation" / "aws-batch-trainer.yaml"
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def _resolved_configuration(template: dict) -> dict:
    references = {
        "AWS::Region": "us-east-1",
        "StudioRole.Arn": "arn:aws:iam::123456789012:role/studio",
        "JobsBucket": "jobs-bucket",
        "ParallelTrainersPerPool": template["Parameters"]["ParallelTrainersPerPool"]["Default"],
    }
    for name, resource in template["Resources"].items():
        if resource["Type"] == "AWS::Batch::JobQueue":
            references[name] = f"arn:aws:batch:us-east-1:123456789012:job-queue/{name}"
        elif resource["Type"] == "AWS::Batch::JobDefinition":
            references[name] = f"arn:aws:batch:us-east-1:123456789012:job-definition/{name}:1"
    configuration = template["Resources"]["EmptyJobsBucket"]["Properties"]["Configuration"]
    return {
        "schema_version": int(configuration["schema_version"]),
        **{field: references[configuration[field]] for field in ("region", "studio_role_arn", "bucket")},
        "targets": {
            instance_type: {field: references[value] for field, value in target.items()}
            for instance_type, target in configuration["targets"].items()
        },
    }


def test_template_configuration_matches_resource_schema() -> None:
    template = _template()
    for target in template["Resources"]["EmptyJobsBucket"]["Properties"]["Configuration"]["targets"].values():
        assert target["max_trainers"] == "ParallelTrainersPerPool"
    parsed = AwsBatchResourceConfiguration.model_validate(_resolved_configuration(template))
    assert set(parsed.targets) == {"g4dn.2xlarge", "g6e.2xlarge", "g7e.8xlarge"}
    assert parsed.region == "us-east-1"
    assert all(target.max_trainers == 1 for target in parsed.targets.values())
    assert template["Rules"]["SupportedRegion"]["Assertions"][0]["Assert"] == {
        "Fn::Equals": [{"Ref": "AWS::Region"}, "us-east-1"]
    }
    assert parsed.studio_role_arn == "arn:aws:iam::123456789012:role/studio"
    assert set(template["Outputs"]) == {"ConfigurationUri"}


def test_configuration_object_is_excluded_from_data_expiration() -> None:
    rules = _template()["Resources"]["JobsBucket"]["Properties"]["LifecycleConfiguration"]["Rules"]
    assert {rule["Prefix"] for rule in rules} == {"jobs/", "validation/"}
    assert all(rule["ExpirationInDays"] == "30" for rule in rules)


def test_studio_can_read_only_the_trainer_log_group() -> None:
    policies = _template()["Resources"]["StudioRole"]["Properties"]["Policies"]
    statements = [statement for policy in policies for statement in policy["PolicyDocument"]["Statement"]]
    logs = next(statement for statement in statements if statement["Action"] == "logs:GetLogEvents")
    assert logs["Effect"] == "Allow"
    assert logs["Resource"] == "TrainerLogGroup.Arn"


def test_studio_can_inspect_ecs_startup_without_modifying_tasks() -> None:
    policies = _template()["Resources"]["StudioRole"]["Properties"]["Policies"]
    policy = next(policy for policy in policies if policy["PolicyName"] == "StudioBatchStartupDiagnostics")
    assert policy["PolicyDocument"]["Statement"] == [
        {
            "Effect": "Allow",
            "Action": "ecs:DescribeTasks",
            "Resource": "arn:${AWS::Partition}:ecs:${AWS::Region}:${AWS::AccountId}:task/*",
        }
    ]


def test_batch_uses_consistent_cuda_trainer_image_without_image_parameter() -> None:
    template = _template()
    definitions = [
        resource for resource in template["Resources"].values() if resource["Type"] == "AWS::Batch::JobDefinition"
    ]
    assert len(definitions) == 3
    image = definitions[0]["Properties"]["ContainerProperties"]["Image"]
    assert image.startswith("ghcr.io/")
    assert image.endswith("/physicalai-trainer-cuda:main")
    assert all(definition["Properties"]["ContainerProperties"]["Image"] == image for definition in definitions)
    assert "TrainerImage" not in json.dumps(template)


@pytest.mark.parametrize("definition", ["G4dnJobDefinition", "G6eJobDefinition", "G7eJobDefinition"])
@pytest.mark.parametrize("prefix", ["jobs/job-id", "jobs/job with 'quotes'; $(exit 1)"])
def test_job_prefix_parameter_reaches_trainer(definition: str, prefix: str, tmp_path: Path) -> None:
    properties = _template()["Resources"][definition]["Properties"]
    container = properties["ContainerProperties"]
    assert "job_prefix" in properties["Parameters"]
    assert "Ref::job_prefix" in container["Command"]
    command = [prefix if argument == "Ref::job_prefix" else argument for argument in container["Command"]]
    executable = tmp_path / "physicalai-trainer-batch"
    executable.write_text('#!/bin/sh\nprintf "%s" "$PHYSICALAI_BATCH_JOB_PREFIX"\n', encoding="utf-8")
    executable.chmod(0o755)
    environment = {entry["Name"]: entry["Value"] for entry in container["Environment"]}
    environment["PATH"] = f"{tmp_path}{os.pathsep}{os.defpath}"

    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True, timeout=5)

    assert result.stdout == prefix
    settings = BatchSettings(**environment, PHYSICALAI_BATCH_JOB_PREFIX=result.stdout)
    assert (settings.spec_key, settings.dataset_key, settings.status_key, settings.artifact_key) == (
        "spec.json",
        "dataset.zip",
        "status.json",
        "artifact.zip",
    )


def test_job_definitions_keep_resource_limits_and_use_defaults() -> None:
    template = _template()
    assert set(template["Parameters"]) == {"JobTimeoutSeconds", "ParallelTrainersPerPool"}
    assert template["Resources"]["TrainerLogGroup"]["Properties"]["RetentionInDays"] == "14"
    for definition, vcpus, memory in [
        ("G4dnJobDefinition", "4", "14000"),
        ("G6eJobDefinition", "8", "30000"),
        ("G7eJobDefinition", "16", "240000"),
    ]:
        properties = template["Resources"][definition]["Properties"]
        assert properties["Timeout"] == {"AttemptDurationSeconds": "JobTimeoutSeconds"}
        assert "PlatformCapabilities" not in properties
        assert "RetryStrategy" not in properties
        container = properties["ContainerProperties"]
        assert {entry["Name"] for entry in container["Environment"]} == {
            "PHYSICALAI_BATCH_PROVIDER",
            "PHYSICALAI_BATCH_BUCKET",
        }
        assert {entry["Type"]: entry["Value"] for entry in container["ResourceRequirements"]} == {
            "VCPU": vcpus,
            "MEMORY": memory,
            "GPU": "1",
        }


def test_parallel_trainers_sets_each_pool_capacity() -> None:
    template = _template()
    parameter = template["Parameters"]["ParallelTrainersPerPool"]
    assert parameter["Default"] == "1"
    assert parameter["AllowedValues"] == [str(count) for count in range(1, 17)]
    for pool, size, vcpus in [("G4dn", "Small", 8), ("G6e", "Small", 8), ("G7e", "Large", 32)]:
        resources = template["Resources"][f"{pool}ComputeEnvironment"]["Properties"]["ComputeResources"]
        assert resources["MaxvCpus"] == ["PoolCapacity", "ParallelTrainersPerPool", size]
        assert resources["MinvCpus"] == resources["DesiredvCpus"] == "0"
        assert resources["AllocationStrategy"] == "BEST_FIT"
        for count in parameter["AllowedValues"]:
            assert int(template["Mappings"]["PoolCapacity"][count][size]) == int(count) * vcpus


@pytest.mark.parametrize(
    ("role", "prefixes"),
    [("StudioRole", ["jobs/*", "validation/*"]), ("BatchJobRole", "jobs/*")],
)
def test_bucket_location_permission_has_no_prefix_condition(role: str, prefixes: list[str] | str) -> None:
    policies = _template()["Resources"][role]["Properties"]["Policies"]
    statements = [statement for policy in policies for statement in policy["PolicyDocument"]["Statement"]]
    location = next(statement for statement in statements if statement["Action"] == "s3:GetBucketLocation")
    assert location["Effect"] == "Allow"
    assert location["Resource"] == "JobsBucket.Arn"
    assert "Condition" not in location
    listing = next(statement for statement in statements if statement["Action"] == "s3:ListBucket")
    assert listing["Resource"] == "JobsBucket.Arn"
    assert listing["Condition"] == {"StringLike": {"s3:prefix": prefixes}}


@pytest.mark.parametrize("request_type", ["Create", "Update", "Delete"])
def test_template_publishes_configuration_and_cleans_bucket(request_type: str) -> None:
    template = _template()
    properties = template["Resources"]["EmptyJobsBucket"]["Properties"]
    properties["Configuration"] = _resolved_configuration(template)
    for target in properties["Configuration"]["targets"].values():
        target["max_trainers"] = "4"
    code = template["Resources"]["BucketCleanupFunction"]["Properties"]["Code"]["ZipFile"]
    boto = MagicMock()
    response = MagicMock()
    namespace: dict[str, Any] = {}
    event = {
        "RequestType": request_type,
        "PhysicalResourceId": "stable-resource",
        "ResourceProperties": properties,
    }
    context = SimpleNamespace(log_stream_name="log-stream")
    with patch.dict("sys.modules", {"boto3": boto, "cfnresponse": response}):
        exec(compile(code, "cloudformation-handler", "exec"), namespace)
        namespace["handler"](event, context)
    response.send.assert_called_once_with(event, context, response.SUCCESS, {}, "stable-resource")
    if request_type == "Delete":
        boto.resource.return_value.Bucket.return_value.object_versions.delete.assert_called_once()
        boto.client.assert_not_called()
    else:
        arguments = boto.client.return_value.put_object.call_args.kwargs
        assert arguments["Key"] == "studio-config.json"
        payload = json.loads(arguments["Body"])
        assert payload["schema_version"] == 1
        assert all(target["max_trainers"] == 4 for target in payload["targets"].values())
        AwsBatchResourceConfiguration.model_validate(payload)
