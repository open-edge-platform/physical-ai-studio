# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from core.security import require_aws_feature_active
from exceptions import AwsFeatureDisabledError
from schemas.aws_auth import AwsCredentialSelection
from schemas.remote_trainer import AwsBatchConnection, AwsBatchProviderConfiguration
from services.aws_auth_service import check_credentials, get_capabilities
from services.aws_batch import _make_session, resolve_aws_batch_configuration


@pytest.fixture
def disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ENABLED", "false")


@pytest.mark.parametrize("enabled", [True, False])
def test_aws_provider_depends_only_on_setting(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ENABLED", str(enabled))
    if enabled:
        require_aws_feature_active()
    else:
        with pytest.raises(AwsFeatureDisabledError) as error:
            require_aws_feature_active()
        assert error.value.http_status == 503
        assert error.value.error_code == "aws_feature_unavailable"


def test_aws_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_ENABLED", raising=False)
    with pytest.raises(AwsFeatureDisabledError):
        require_aws_feature_active()


@pytest.mark.usefixtures("disabled")
@pytest.mark.parametrize("operation", ["credentials", "configuration", "session"])
def test_aws_access_rejected_before_reading_credentials(operation: str) -> None:
    with patch("boto3.Session") as session:
        with pytest.raises(AwsFeatureDisabledError):
            if operation == "credentials":
                check_credentials(AwsCredentialSelection(profile_name="studio"))
            elif operation == "configuration":
                resolve_aws_batch_configuration(
                    AwsBatchProviderConfiguration(
                        profile_name="studio", configuration_uri="s3://config-bucket/studio-config.json"
                    )
                )
            else:
                _make_session(
                    AwsBatchConnection(
                        region="us-east-1",
                        studio_role_arn="arn:aws:iam::123456789012:role/studio",
                        bucket="jobs-bucket",
                        targets={
                            "g4dn.2xlarge": {
                                "queue": "arn:aws:batch:us-east-1:123456789012:job-queue/queue",
                                "job_definition": "arn:aws:batch:us-east-1:123456789012:job-definition/definition:1",
                            }
                        },
                    )
                )
        session.assert_not_called()


@pytest.mark.usefixtures("disabled")
async def test_profile_discovery_rejected_before_reading_profiles() -> None:
    with patch("services.aws_auth_service._profiles") as profiles:
        with pytest.raises(AwsFeatureDisabledError):
            await get_capabilities()
        profiles.assert_not_called()
