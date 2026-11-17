from unittest.mock import patch

from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.remote_trainers import router
from schemas.remote_trainer import AwsBatchConnection

app = FastAPI()
app.include_router(router)
client = TestClient(app)

CONFIGURATION = {
    "profile_name": "studio",
    "stack_id": "123456789012-eu-west-1-f4a9b2c0",
}


def test_provider_schema_exposes_only_connection_fields() -> None:
    response = client.get("/api/remote-trainers/providers")
    assert response.status_code == 200
    provider = response.json()[0]
    assert provider["id"] == "aws"
    assert [field["name"] for field in provider["fields"]] == ["stack_id"]
    assert provider["fields"][0]["pattern"].startswith("^[0-9]{12}-")


def test_resolve_provider_configuration() -> None:
    resolved = AwsBatchConnection.model_validate(CONFIGURATION)
    with patch("api.remote_trainers.resolve_aws_batch_configuration", return_value=resolved):
        response = client.post("/api/remote-trainers/providers/aws/configuration", json=CONFIGURATION)
    assert response.status_code == 200
    assert response.json() == resolved.model_dump(mode="json")


def test_resolver_returns_sanitized_aws_error() -> None:
    error = ClientError({"Error": {"Code": "AccessDenied", "Message": "private details"}}, "DescribeJobQueues")
    with patch("api.remote_trainers.resolve_aws_batch_configuration", side_effect=error):
        response = client.post("/api/remote-trainers/providers/aws/configuration", json=CONFIGURATION)
    assert response.status_code == 400
    assert "private details" not in response.text
    assert "permissions" in response.json()["detail"]


def test_resolver_rejects_invalid_document() -> None:
    with patch("api.remote_trainers.resolve_aws_batch_configuration", side_effect=ValueError):
        response = client.post("/api/remote-trainers/providers/aws/configuration", json=CONFIGURATION)
    assert response.status_code == 400


def test_resolver_rejects_invalid_stack_id_before_aws_request() -> None:
    with patch("api.remote_trainers.resolve_aws_batch_configuration") as resolve:
        response = client.post(
            "/api/remote-trainers/providers/aws/configuration",
            json={**CONFIGURATION, "stack_id": "invalid-stack-id"},
        )
    assert response.status_code == 422
    resolve.assert_not_called()
