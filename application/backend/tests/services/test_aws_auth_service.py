import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
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


def test_credential_provider_factory_returns_profile_provider() -> None:
    provider = auth.get_credential_provider(AwsCredentialSelection(profile_name="studio"))
    assert isinstance(provider, auth.AwsCredentialProvider)
    assert isinstance(provider, auth.ProfileCredentialProvider)
    issue = provider.diagnose_error(SSOTokenLoadError(error_msg="expired"))
    assert issue is not None
    assert issue.profile_name == "studio"
    op_error = provider.to_operational_error(SSOTokenLoadError(error_msg="expired"))
    assert op_error.copy_text == "aws sso login --profile=studio"


def _write_greengrass_bundle(tmp_path: Path) -> Path:
    cert = tmp_path / "device.pem.crt"
    key = tmp_path / "private.pem.key"
    ca = tmp_path / "AmazonRootCA1.pem"
    for p in (cert, key, ca):
        p.write_text("-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "system": {
                    "certificateFilePath": str(cert),
                    "privateKeyPath": str(key),
                    "rootCaPath": str(ca),
                    "thingName": "robot-01",
                },
                "services": {
                    "aws.greengrass.Nucleus": {
                        "configuration": {
                            "awsRegion": "eu-central-1",
                            "iotRoleAlias": "pai-studio-default",
                            "iotCredEndpoint": "abc123def.credentials.iot.eu-central-1.amazonaws.com",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return config


async def test_iot_certificate_provider_exchanges_cert_for_session_without_role_chaining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_greengrass_bundle(tmp_path)
    monkeypatch.setenv("GREENGRASS_CONFIG_PATH", str(config_path))

    capabilities = await auth.get_capabilities()
    assert capabilities.use_device_certificate is True

    connection = AwsBatchConnection.model_validate(
        {
            "stack_id": "123456789012-eu-central-1-f4a9b2c0",
            "profile_name": None,
        }
    )
    provider = auth.get_credential_provider(connection)
    assert isinstance(provider, auth.IotCertificateCredentialProvider)

    http_response = MagicMock(status_code=200)
    http_response.json.return_value = {
        "credentials": {
            "accessKeyId": "ASIAIOTKEY",
            "secretAccessKey": "iot-secret",
            "sessionToken": "iot-token",
            "expiration": "2099-01-01T00:00:00Z",
        }
    }
    http_client = MagicMock()
    http_client.__enter__.return_value.get.return_value = http_response
    ssl_ctx = MagicMock()

    with (
        patch("ssl.create_default_context", return_value=ssl_ctx) as create_ssl,
        patch("httpx.Client", return_value=http_client),
    ):
        session = _make_session(connection)

    create_ssl.assert_called_once()
    ssl_ctx.load_cert_chain.assert_called_once_with(
        certfile=str(tmp_path / "device.pem.crt"),
        keyfile=str(tmp_path / "private.pem.key"),
    )
    http_client.__enter__.return_value.get.assert_called_once_with(
        "https://abc123def.credentials.iot.eu-central-1.amazonaws.com/role-aliases/pai-studio-f4a9b2c0/credentials",
        headers={"x-amzn-iot-thingname": "robot-01"},
    )
    assert session.region_name == "eu-central-1"
    creds = session.get_credentials()
    assert creds is not None
    assert creds.method == "iot-certificate"
    frozen = creds.get_frozen_credentials()
    assert (frozen.access_key, frozen.secret_key, frozen.token) == ("ASIAIOTKEY", "iot-secret", "iot-token")


def test_iot_certificate_provider_refreshes_expired_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_greengrass_bundle(tmp_path)
    monkeypatch.setenv("GREENGRASS_CONFIG_PATH", str(config_path))
    provider = auth.IotCertificateCredentialProvider(config_path)

    first_response = MagicMock(status_code=200)
    first_response.json.return_value = {
        "credentials": {
            "accessKeyId": "ASIAFIRST",
            "secretAccessKey": "secret-1",
            "sessionToken": "token-1",
            "expiration": "2000-01-01T00:00:00Z",
        }
    }
    second_response = MagicMock(status_code=200)
    second_response.json.return_value = {
        "credentials": {
            "accessKeyId": "ASIASECOND",
            "secretAccessKey": "secret-2",
            "sessionToken": "token-2",
            "expiration": "2099-01-01T00:00:00Z",
        }
    }
    http_client = MagicMock()
    http_client.__enter__.return_value.get.side_effect = [first_response, second_response]

    with (
        patch("ssl.create_default_context", return_value=MagicMock()),
        patch("httpx.Client", return_value=http_client),
    ):
        session = provider.create_session(
            region="eu-central-1",
            role_arn="arn:aws:iam::123456789012:role/pai-studio-f4a9b2c0",
        )
        frozen = session.get_credentials().get_frozen_credentials()

    assert http_client.__enter__.return_value.get.call_count == 2
    assert (frozen.access_key, frozen.secret_key, frozen.token) == ("ASIASECOND", "secret-2", "token-2")


def test_iot_certificate_provider_diagnoses_rejected_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_greengrass_bundle(tmp_path)
    monkeypatch.setenv("GREENGRASS_CONFIG_PATH", str(config_path))
    provider = auth.IotCertificateCredentialProvider(config_path)

    http_response = MagicMock(status_code=403)
    http_client = MagicMock()
    http_client.__enter__.return_value.get.return_value = http_response

    with (
        patch("ssl.create_default_context", return_value=MagicMock()),
        patch("httpx.Client", return_value=http_client),
        pytest.raises(auth.IotCertificateError) as exc_info,
    ):
        provider.create_session(region="eu-central-1", role_arn="arn:aws:iam::123456789012:role/pai-studio-f4a9b2c0")

    issue = provider.diagnose_error(exc_info.value)
    assert issue is not None
    assert issue.code == "aws_credentials_expired"
    assert "robot-01" in issue.detail
    assert issue.command is None
