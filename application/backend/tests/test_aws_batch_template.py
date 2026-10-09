import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from schemas.remote_trainer import AwsBatchResourceConfiguration


def _template() -> dict:
    path = Path(__file__).resolve().parents[2] / "cloudformation" / "aws-batch-trainer.yaml"
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def test_template_configuration_matches_resource_schema() -> None:
    template = _template()
    configuration = template["Resources"]["EmptyJobsBucket"]["Properties"]["Configuration"]
    configuration["schema_version"] = int(configuration["schema_version"])
    parsed = AwsBatchResourceConfiguration.model_validate(configuration)
    assert set(parsed.targets) == {"g4dn.xlarge", "g6e.2xlarge", "p5.4xlarge"}
    assert parsed.region == "AWS::Region"
    assert parsed.studio_role_arn == "StudioRole.Arn"
    assert set(template["Outputs"]) == {"ConfigurationUri"}


def test_configuration_object_is_excluded_from_data_expiration() -> None:
    rules = _template()["Resources"]["JobsBucket"]["Properties"]["LifecycleConfiguration"]["Rules"]
    assert {rule["Prefix"] for rule in rules} == {"jobs/", "validation/"}
    assert all(rule["ExpirationInDays"] == "30" for rule in rules)


@pytest.mark.parametrize("request_type", ["Create", "Update", "Delete"])
def test_template_publishes_configuration_and_cleans_bucket(request_type: str) -> None:
    template = _template()
    properties = template["Resources"]["EmptyJobsBucket"]["Properties"]
    code = template["Resources"]["BucketCleanupFunction"]["Properties"]["Code"]["ZipFile"]
    boto = MagicMock()
    response = MagicMock()
    namespace = {}
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
        AwsBatchResourceConfiguration.model_validate(payload)
