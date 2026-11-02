import shlex
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound, SSOTokenLoadError

from schemas.aws_auth import AwsCredentialSelection
from schemas.remote_trainer import AwsBatchConnection
from services import aws_auth_service as auth
from services.aws_batch import _make_session


@pytest.mark.parametrize("profile_name", ["default", "studio"])
def test_credential_check_uses_selected_profile(profile_name: str) -> None:
    with patch("boto3.Session") as session:
        session.return_value.region_name = None
        session.return_value.client.return_value.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:iam::123456789012:role/developer",
        }
        identity = auth.check_credentials(AwsCredentialSelection(profile_name=profile_name))
    session.assert_called_once_with(profile_name=profile_name)
    assert session.return_value.client.call_args.kwargs["region_name"] == "us-east-1"
    assert identity.account == "123456789012"


def test_assume_role_uses_selected_source_profile() -> None:
    connection = AwsBatchConnection(
        region="us-east-1",
        profile_name="studio",
        studio_role_arn="arn:aws:iam::123456789012:role/studio",
        bucket="jobs-bucket",
        targets={
            "gpu": {
                "queue": "arn:aws:batch:us-east-1:123456789012:job-queue/queue",
                "job_definition": "arn:aws:batch:us-east-1:123456789012:job-definition/definition:1",
            }
        },
    )
    source, assumed = MagicMock(), MagicMock()
    source.client.return_value.assume_role.return_value = {
        "Credentials": {"AccessKeyId": "key", "SecretAccessKey": "secret", "SessionToken": "token"}
    }
    with patch("boto3.Session", side_effect=[source, assumed]) as session:
        assert _make_session(connection) is assumed
    assert session.call_args_list[0].kwargs == {"profile_name": "studio"}
    assert source.client.call_args.args == ("sts",)
    assert source.client.call_args.kwargs["region_name"] == "us-east-1"
    source.client.return_value.assume_role.assert_called_once_with(
        RoleArn=connection.studio_role_arn, RoleSessionName="physicalai-studio"
    )


async def test_lists_backend_profiles_without_authenticating() -> None:
    with patch("boto3.Session") as session:
        session.return_value.available_profiles = ["studio", "default"]
        result = await auth.get_capabilities()
    assert result.profiles == ["default", "studio"]
    session.return_value.client.assert_not_called()


@pytest.mark.parametrize(
    "error,expected",
    [
        (NoCredentialsError(), "No AWS credentials"),
        (ProfileNotFound(profile="private-profile"), "profile not found"),
        (SSOTokenLoadError(error_msg="private-cache"), "aws sso login"),
        (ClientError({"Error": {"Code": "ExpiredToken", "Message": "private"}}, "GetObject"), "expired"),
    ],
)
def test_credential_errors_are_actionable_and_sanitized(error: Exception, expected: str) -> None:
    detail = auth.aws_auth_error_detail(error)
    assert expected in detail
    assert "private" not in detail


@pytest.mark.parametrize("profile", ["studio", "team profile", "team's profile", "--debug;$(whoami)"])
def test_sso_recovery_command_quotes_selected_profile(profile: str) -> None:
    result = auth.aws_auth_error(SSOTokenLoadError(error_msg="private"), profile)
    assert result.code == "aws_credentials_expired"
    assert result.title == "AWS SSO session missing or expired"
    assert profile in result.detail
    assert shlex.split(result.command) == ["aws", "sso", "login", f"--profile={profile}"]
    assert "private" not in result.model_dump_json()


def test_generic_expiration_does_not_suggest_sso_for_non_sso_profile() -> None:
    result = auth.aws_auth_error(ClientError({"Error": {"Code": "ExpiredToken"}}, "GetObject"), "studio")
    assert result.code == "aws_credentials_expired"
    assert result.command is None
