from unittest.mock import AsyncMock, patch

import pytest
from botocore.exceptions import NoCredentialsError, ProfileNotFound, SSOTokenLoadError
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.remote_trainers import router
from schemas.aws_auth import AwsAuthCapabilities, AwsIdentity

app = FastAPI()
app.include_router(router)
client = TestClient(app)
BASE = "/api/remote-trainers/providers/aws"


def test_lists_backend_profiles() -> None:
    with patch(
        "api.aws_auth.aws_auth_service.get_capabilities",
        AsyncMock(return_value=AwsAuthCapabilities(profiles=["studio"])),
    ):
        response = client.get(f"{BASE}/auth")
    assert response.status_code == 200
    assert response.json() == {"profiles": ["studio"]}


def test_credential_check_uses_selected_profile() -> None:
    identity = AwsIdentity(account="123", arn="arn:aws:iam::123:role/developer")
    with patch("api.aws_auth.aws_auth_service.check_credentials", return_value=identity) as check:
        response = client.post(f"{BASE}/credentials/check", json={"profile_name": "studio"})
    assert response.status_code == 200
    assert response.json() == identity.model_dump()
    assert check.call_args.args[0].profile_name == "studio"


@pytest.mark.parametrize("body", [{}, {"profile_name": None}, {"profile_name": ""}, {"profile_name": "  "}])
@pytest.mark.parametrize("endpoint", ["credentials/check", "configuration"])
def test_requires_explicit_named_profile(body: dict, endpoint: str) -> None:
    with patch("api.aws_auth.aws_auth_service.check_credentials") as check:
        response = client.post(
            f"{BASE}/{endpoint}",
            json={"configuration_uri": "s3://config-bucket/studio-config.json", **body}
            if endpoint == "configuration"
            else body,
        )
    assert response.status_code == 422
    check.assert_not_called()


@pytest.mark.parametrize("error", [NoCredentialsError(), ProfileNotFound(profile="private")])
def test_credential_errors_are_actionable(error: Exception) -> None:
    with patch("api.aws_auth.aws_auth_service.check_credentials", side_effect=error):
        response = client.post(f"{BASE}/credentials/check", json={"profile_name": "studio"})
    assert response.status_code == 400
    assert "backend" in response.json()["detail"]
    assert "private" not in response.text


@pytest.mark.parametrize("endpoint", ["credentials/check", "configuration"])
def test_sso_error_contains_structured_recovery_in_both_flows(endpoint: str) -> None:
    target = (
        "api.aws_auth.aws_auth_service.check_credentials"
        if endpoint == "credentials/check"
        else "api.remote_trainers.resolve_aws_batch_configuration"
    )
    body = {"profile_name": "team studio"}
    if endpoint == "configuration":
        body["configuration_uri"] = "s3://config-bucket/studio-config.json"
    with patch(target, side_effect=SSOTokenLoadError(error_msg="private")):
        response = client.post(f"{BASE}/{endpoint}", json=body)
    assert response.status_code == 400
    assert response.json()["id"]
    assert response.json()["copy_text"] == "aws sso login '--profile=team studio'"
    assert "team studio" in response.json()["detail"]
    assert "private" not in response.text
