# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""API tests for the remote-server router.

Everything here is exercised exclusively through ``app.dependency_overrides``
with ``MagicMock``/``AsyncMock`` stand-ins - no real ``RemoteServerService``,
repository, or SSH transport is ever instantiated, so these tests do not
depend on whether PR2/PR3 have actually landed.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_remote_server_service
from exceptions import ResourceNotFoundError, ResourceType
from main import app
from schemas.hardware import DeviceType
from schemas.remote_server import RemoteServer, SshHostAliasOption
from schemas.ssh_preflight import CheckKey, CheckOutcome, PreflightCheck, PreflightResult

CHECKED_AT = datetime.now(UTC)


def _make_server(**overrides: object) -> RemoteServer:
    fields: dict[str, object] = {
        "id": uuid4(),
        "name": "gpu-box",
        "ssh_host_alias": "gpu-box",
        "device_type": DeviceType.CUDA,
    }
    fields.update(overrides)
    return RemoteServer(**fields)  # type: ignore[arg-type]


def _passing_check(key: CheckKey, *, tier: int = 1, blocking: bool = True) -> PreflightCheck:
    return PreflightCheck(
        key=key,
        tier=tier,  # type: ignore[arg-type]
        outcome=CheckOutcome.PASSED,
        blocking=blocking,
        checked_at=CHECKED_AT,
    )


def _passing_tier1_result() -> PreflightResult:
    keys = [
        CheckKey.ALIAS_RESOLVED,
        CheckKey.REACHABLE,
        CheckKey.AUTHENTICATED,
        CheckKey.HOST_KEY_VERIFIED,
        CheckKey.DOCKER_USABLE,
        CheckKey.DISK_SPACE,
        CheckKey.DRIVER_PRESENT,
        CheckKey.REGISTRY_REACHABLE,
        CheckKey.GPU_FREE,
    ]
    return PreflightResult(
        remote_server_id=None,
        tiers_run=[1],  # type: ignore[list-item]
        checks=[_passing_check(key) for key in keys],
        checked_at=CHECKED_AT,
        latency_ms=42,
    )


def _failing_tier1_result(key: CheckKey, reason_code: str) -> PreflightResult:
    failing = PreflightCheck(
        key=key,
        tier=1,  # type: ignore[arg-type]
        outcome=CheckOutcome.FAILED,
        blocking=True,
        checked_at=CHECKED_AT,
        reason_code=reason_code,
    )
    return PreflightResult(
        remote_server_id=None,
        tiers_run=[1],  # type: ignore[list-item]
        checks=[failing],
        checked_at=CHECKED_AT,
        latency_ms=10,
    )


def _gpu_busy_warning_result() -> PreflightResult:
    result = _passing_tier1_result()
    checks = [check for check in result.checks if check.key is not CheckKey.GPU_FREE]
    checks.append(
        PreflightCheck(
            key=CheckKey.GPU_FREE,
            tier=1,  # type: ignore[arg-type]
            outcome=CheckOutcome.WARNING,
            blocking=False,
            checked_at=CHECKED_AT,
            reason_code="gpu_busy",
        )
    )
    return PreflightResult(
        remote_server_id=None,
        tiers_run=[1],  # type: ignore[list-item]
        checks=checks,
        checked_at=CHECKED_AT,
        latency_ms=42,
    )


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_list_remote_servers_empty():
    service = AsyncMock()
    service.list_remote_servers.return_value = []
    app.dependency_overrides[get_remote_server_service] = lambda: service

    response = TestClient(app).get("/api/remote-servers")

    assert response.status_code == 200
    assert response.json() == []


def test_list_remote_servers_non_empty():
    server = _make_server()
    service = AsyncMock()
    service.list_remote_servers.return_value = [server]
    app.dependency_overrides[get_remote_server_service] = lambda: service

    response = TestClient(app).get("/api/remote-servers")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == str(server.id)


def test_list_ssh_host_aliases_returns_mocked_list(monkeypatch: pytest.MonkeyPatch):
    options = [SshHostAliasOption(alias="gpu-box", hostname="10.0.0.5", port=22, user="trainer")]
    monkeypatch.setattr("api.remote_servers.ssh_config_reader.list_host_aliases", lambda config_path: options)

    response = TestClient(app).get("/api/remote-servers/aliases")

    assert response.status_code == 200
    assert response.json() == [
        {"alias": "gpu-box", "hostname": "10.0.0.5", "port": 22, "user": "trainer"},
    ]


def test_list_ssh_host_aliases_response_never_leaks_credential_fields(monkeypatch: pytest.MonkeyPatch):
    """Regression guard: the schema already guarantees no secret fields, but the
    acceptance criterion wants this asserted at the HTTP response level too.
    """
    options = [
        SshHostAliasOption(alias="gpu-box", hostname="10.0.0.5", port=22, user="trainer"),
        SshHostAliasOption(alias="another-box", hostname="10.0.0.6", port=2222, user="root"),
    ]
    monkeypatch.setattr("api.remote_servers.ssh_config_reader.list_host_aliases", lambda config_path: options)

    response = TestClient(app).get("/api/remote-servers/aliases")

    assert response.status_code == 200
    body_text = response.text.lower()
    for forbidden in ("identityfile", "identityagent", "certificatefile", "password"):
        assert forbidden not in body_text


def test_create_remote_server_success(monkeypatch: pytest.MonkeyPatch):
    run_tier1 = AsyncMock(return_value=_passing_tier1_result())
    monkeypatch.setattr("api.remote_servers.run_tier1_preflight", run_tier1)

    created = _make_server()
    service = AsyncMock()
    service.create_remote_server.return_value = created
    app.dependency_overrides[get_remote_server_service] = lambda: service

    payload = {"name": "gpu-box", "ssh_host_alias": "gpu-box", "device_type": "cuda"}
    response = TestClient(app).post("/api/remote-servers", json=payload)

    assert response.status_code == 201
    assert response.json()["id"] == str(created.id)
    service.create_remote_server.assert_awaited_once()
    run_tier1.assert_awaited_once()


def test_create_remote_server_blocking_failure_returns_400_and_never_persists(monkeypatch: pytest.MonkeyPatch):
    run_tier1 = AsyncMock(return_value=_failing_tier1_result(CheckKey.AUTHENTICATED, "ssh_authentication_failed"))
    monkeypatch.setattr("api.remote_servers.run_tier1_preflight", run_tier1)

    service = AsyncMock()
    app.dependency_overrides[get_remote_server_service] = lambda: service

    payload = {"name": "gpu-box", "ssh_host_alias": "gpu-box", "device_type": "cuda"}
    response = TestClient(app).post("/api/remote-servers", json=payload)

    assert response.status_code == 400
    assert response.json()["error_code"] == "remote_server_preflight_failed"
    service.create_remote_server.assert_not_called()
    service.create_remote_server.assert_not_awaited()


def test_create_remote_server_succeeds_when_only_gpu_free_warns(monkeypatch: pytest.MonkeyPatch):
    """Named acceptance criterion: a save must succeed even when GPU_FREE is a
    non-blocking WARNING.
    """
    run_tier1 = AsyncMock(return_value=_gpu_busy_warning_result())
    monkeypatch.setattr("api.remote_servers.run_tier1_preflight", run_tier1)

    created = _make_server()
    service = AsyncMock()
    service.create_remote_server.return_value = created
    app.dependency_overrides[get_remote_server_service] = lambda: service

    payload = {"name": "gpu-box", "ssh_host_alias": "gpu-box", "device_type": "cuda"}
    response = TestClient(app).post("/api/remote-servers", json=payload)

    assert response.status_code == 201
    service.create_remote_server.assert_awaited_once()


def test_create_remote_server_never_triggers_tier2_pull(monkeypatch: pytest.MonkeyPatch):
    """Most important test in this file: a save request never performs an image
    pull. Asserted on the transport (Tier 2 is never invoked), not on timing.
    """
    run_tier1 = AsyncMock(return_value=_passing_tier1_result())
    run_tier2 = AsyncMock(return_value=_passing_tier1_result())
    monkeypatch.setattr("api.remote_servers.run_tier1_preflight", run_tier1)
    monkeypatch.setattr("api.remote_servers.run_tier2_preflight", run_tier2)

    created = _make_server()
    service = AsyncMock()
    service.create_remote_server.return_value = created
    app.dependency_overrides[get_remote_server_service] = lambda: service

    payload = {"name": "gpu-box", "ssh_host_alias": "gpu-box", "device_type": "cuda"}
    response = TestClient(app).post("/api/remote-servers", json=payload)

    assert response.status_code == 201
    run_tier2.assert_not_called()
    run_tier2.assert_not_awaited()


def test_update_remote_server_success(monkeypatch: pytest.MonkeyPatch):
    run_tier1 = AsyncMock(return_value=_passing_tier1_result())
    monkeypatch.setattr("api.remote_servers.run_tier1_preflight", run_tier1)

    existing = _make_server()
    updated = _make_server(id=existing.id, name="renamed-box")
    service = AsyncMock()
    service.get_remote_server.return_value = existing
    service.update_remote_server.return_value = updated
    app.dependency_overrides[get_remote_server_service] = lambda: service

    response = TestClient(app).patch(f"/api/remote-servers/{existing.id}", json={"name": "renamed-box"})

    assert response.status_code == 200
    assert response.json()["name"] == "renamed-box"
    service.update_remote_server.assert_awaited_once()


def test_update_remote_server_not_found(monkeypatch: pytest.MonkeyPatch):
    run_tier1 = AsyncMock(return_value=_passing_tier1_result())
    monkeypatch.setattr("api.remote_servers.run_tier1_preflight", run_tier1)

    missing_id = uuid4()
    service = AsyncMock()
    service.get_remote_server.side_effect = ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(missing_id))
    app.dependency_overrides[get_remote_server_service] = lambda: service

    response = TestClient(app).patch(f"/api/remote-servers/{missing_id}", json={"name": "renamed-box"})

    assert response.status_code == 404
    service.update_remote_server.assert_not_called()


def test_update_remote_server_blocking_failure(monkeypatch: pytest.MonkeyPatch):
    run_tier1 = AsyncMock(return_value=_failing_tier1_result(CheckKey.DISK_SPACE, "insufficient_disk"))
    monkeypatch.setattr("api.remote_servers.run_tier1_preflight", run_tier1)

    existing = _make_server()
    service = AsyncMock()
    service.get_remote_server.return_value = existing
    app.dependency_overrides[get_remote_server_service] = lambda: service

    response = TestClient(app).patch(f"/api/remote-servers/{existing.id}", json={"name": "renamed-box"})

    assert response.status_code == 400
    assert response.json()["error_code"] == "remote_server_preflight_failed"
    service.update_remote_server.assert_not_called()


def test_delete_remote_server_success():
    service = AsyncMock()
    service.delete_remote_server.return_value = None
    app.dependency_overrides[get_remote_server_service] = lambda: service

    server_id = uuid4()
    response = TestClient(app).delete(f"/api/remote-servers/{server_id}")

    assert response.status_code == 204
    service.delete_remote_server.assert_awaited_once_with(server_id)


def test_delete_remote_server_not_found():
    server_id = uuid4()
    service = AsyncMock()
    service.delete_remote_server.side_effect = ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(server_id))
    app.dependency_overrides[get_remote_server_service] = lambda: service

    response = TestClient(app).delete(f"/api/remote-servers/{server_id}")

    assert response.status_code == 404


def test_check_remote_server_calls_tier2(monkeypatch: pytest.MonkeyPatch):
    """Proves the /check action - and only this action - triggers Tier 2."""
    server = _make_server()
    tier2_result = PreflightResult(
        remote_server_id=server.id,
        tiers_run=[2],  # type: ignore[list-item]
        checks=[_passing_check(CheckKey.IMAGE_RESOLVED, tier=2, blocking=True)],
        checked_at=CHECKED_AT,
        latency_ms=5000,
    )
    run_tier2 = AsyncMock(return_value=tier2_result)
    monkeypatch.setattr("api.remote_servers.run_tier2_preflight", run_tier2)

    service = AsyncMock()
    service.get_remote_server.return_value = server
    app.dependency_overrides[get_remote_server_service] = lambda: service

    response = TestClient(app).post(f"/api/remote-servers/{server.id}/check")

    assert response.status_code == 200
    run_tier2.assert_awaited_once_with(server)
    assert response.json()["tiers_run"] == [2]


def test_check_remote_server_persists_healthy_status_on_pass(monkeypatch: pytest.MonkeyPatch):
    """A passing Tier 2 check must persist `last_check_status="healthy"`.

    Regression guard: before this, neither `/status` (Tier 1) nor `/check`
    (Tier 2) ever wrote back to the DB, so `last_check_status` stayed
    "unknown" forever and job submission could never succeed.
    """
    server = _make_server()
    tier2_result = PreflightResult(
        remote_server_id=server.id,
        tiers_run=[2],  # type: ignore[list-item]
        checks=[_passing_check(CheckKey.IMAGE_RESOLVED, tier=2, blocking=True)],
        checked_at=CHECKED_AT,
        latency_ms=5000,
    )
    monkeypatch.setattr("api.remote_servers.run_tier2_preflight", AsyncMock(return_value=tier2_result))

    service = AsyncMock()
    service.get_remote_server.return_value = server
    app.dependency_overrides[get_remote_server_service] = lambda: service

    response = TestClient(app).post(f"/api/remote-servers/{server.id}/check")

    assert response.status_code == 200
    service.record_check_result.assert_awaited_once_with(server.id, tier2_result)


def test_check_remote_server_persists_degraded_status_on_blocking_failure(monkeypatch: pytest.MonkeyPatch):
    server = _make_server()
    failing_check = PreflightCheck(
        key=CheckKey.IMAGE_RESOLVED,
        tier=2,  # type: ignore[arg-type]
        outcome=CheckOutcome.FAILED,
        blocking=True,
        checked_at=CHECKED_AT,
        reason_code="image_pull_failed",
    )
    tier2_result = PreflightResult(
        remote_server_id=server.id,
        tiers_run=[2],  # type: ignore[list-item]
        checks=[failing_check],
        checked_at=CHECKED_AT,
        latency_ms=1000,
    )
    monkeypatch.setattr("api.remote_servers.run_tier2_preflight", AsyncMock(return_value=tier2_result))

    service = AsyncMock()
    service.get_remote_server.return_value = server
    app.dependency_overrides[get_remote_server_service] = lambda: service

    response = TestClient(app).post(f"/api/remote-servers/{server.id}/check")

    assert response.status_code == 200
    service.record_check_result.assert_awaited_once_with(server.id, tier2_result)
    assert tier2_result.passed is False


def test_status_endpoint_assembles_from_mocked_tier1(monkeypatch: pytest.MonkeyPatch):
    server = _make_server()
    run_tier1 = AsyncMock(return_value=_passing_tier1_result())
    monkeypatch.setattr("api.remote_servers.run_tier1_preflight", run_tier1)

    service = AsyncMock()
    service.get_remote_server.return_value = server
    app.dependency_overrides[get_remote_server_service] = lambda: service

    response = TestClient(app).get(f"/api/remote-servers/{server.id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["remote_server_id"] == str(server.id)
    assert body["status"] == "healthy"
    assert body["device_type"] == "cuda"
    assert len(body["checks"]) == 9
    assert body["in_use_by_job_id"] is None
    assert body["waiting_for_gpu"] is False


@pytest.mark.parametrize(
    ("check_key", "reason_code", "expected_code"),
    [
        (CheckKey.ALIAS_RESOLVED, "alias_not_found", "ssh_host_alias_not_found"),
        (CheckKey.HOST_KEY_VERIFIED, "host_key_unknown", "ssh_host_key_unknown"),
        (CheckKey.AUTHENTICATED, "ssh_agent_required", "ssh_agent_required"),
    ],
)
def test_actionable_ssh_errors_surface_with_own_error_code(
    monkeypatch: pytest.MonkeyPatch, check_key: CheckKey, reason_code: str, expected_code: str
):
    """Each of the three actionable credential errors is returned with its own
    distinct error_code.

    `run_tier1_preflight` never raises (see `services/ssh/preflight.py`) - it
    always encodes an SSH failure as a FAILED check carrying a `reason_code`.
    The router is responsible for translating specific reason codes back into
    their dedicated exception so this response's `error_code` differs per
    cause instead of every failure collapsing onto
    `remote_server_preflight_failed`.
    """
    run_tier1 = AsyncMock(return_value=_failing_tier1_result(check_key, reason_code))
    monkeypatch.setattr("api.remote_servers.run_tier1_preflight", run_tier1)

    service = AsyncMock()
    app.dependency_overrides[get_remote_server_service] = lambda: service

    payload = {"name": "gpu-box", "ssh_host_alias": "gpu-box", "device_type": "cuda"}
    response = TestClient(app).post("/api/remote-servers", json=payload)

    assert response.status_code == 400
    assert response.json()["error_code"] == expected_code
    service.create_remote_server.assert_not_called()
