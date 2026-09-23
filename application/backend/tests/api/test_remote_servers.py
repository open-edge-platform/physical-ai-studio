# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""API tests for SSH host aliases and SSH feature availability."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.dependencies import require_ssh_feature_active
from core.security import SshFeatureAvailability, get_ssh_feature_availability
from exceptions import ResourceAlreadyExistsError, SshConnectionError
from main import app
from schemas.remote_server import SshHostAliasOption


def _active() -> SshFeatureAvailability:
    return SshFeatureAvailability(network_exposed=False)


@pytest.fixture(autouse=True)
def _clear_overrides():
    # Every route but `/feature-status` fails closed behind `require_ssh_feature_active`;
    # default it to "active" so existing tests exercise their own behavior rather
    # than the gate, and let the dedicated fail-closed test override it back off.
    app.dependency_overrides[require_ssh_feature_active] = _active
    yield
    app.dependency_overrides.clear()


def test_list_ssh_host_aliases_fails_closed_when_network_exposed(monkeypatch: pytest.MonkeyPatch):
    """No override for `require_ssh_feature_active`: the real check runs."""
    del app.dependency_overrides[require_ssh_feature_active]
    monkeypatch.setenv("HOST", "0.0.0.0")
    get_ssh_feature_availability.cache_clear()

    try:
        response = TestClient(app).get("/api/remote-servers/aliases")
    finally:
        get_ssh_feature_availability.cache_clear()

    assert response.status_code == 503


def test_feature_status_endpoint_reports_active_on_loopback(monkeypatch: pytest.MonkeyPatch):
    """Reachable with no dependency override and no auth - it explains the 503 above."""
    del app.dependency_overrides[require_ssh_feature_active]
    monkeypatch.setenv("HOST", "127.0.0.1")
    get_ssh_feature_availability.cache_clear()

    try:
        response = TestClient(app).get("/api/remote-servers/feature-status")
    finally:
        get_ssh_feature_availability.cache_clear()

    assert response.status_code == 200
    body = response.json()
    assert body["network_exposed"] is False


def test_feature_status_endpoint_reports_network_exposed(monkeypatch: pytest.MonkeyPatch):
    """The unauthenticated status endpoint reflects the fail-closed check too."""
    del app.dependency_overrides[require_ssh_feature_active]
    monkeypatch.setenv("HOST", "0.0.0.0")
    get_ssh_feature_availability.cache_clear()

    try:
        response = TestClient(app).get("/api/remote-servers/feature-status")
    finally:
        get_ssh_feature_availability.cache_clear()

    assert response.status_code == 200
    body = response.json()
    assert body["network_exposed"] is True
    assert body["reason"] is not None


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


def test_create_ssh_host_alias_appends_and_returns_option(monkeypatch: pytest.MonkeyPatch):
    option = SshHostAliasOption(alias="new-box", hostname="10.0.0.9", port=22, user=None)
    add_verified_host_alias = AsyncMock(return_value=option)
    monkeypatch.setattr("api.remote_servers.ssh_config_writer.add_verified_host_alias", add_verified_host_alias)

    response = TestClient(app).post(
        "/api/remote-servers/aliases",
        json={"alias": "new-box", "hostname": "10.0.0.9"},
    )

    assert response.status_code == 201
    assert response.json() == {"alias": "new-box", "hostname": "10.0.0.9", "port": 22, "user": None}
    add_verified_host_alias.assert_awaited_once()


def test_create_ssh_host_alias_conflict_returns_409(monkeypatch: pytest.MonkeyPatch):
    async def _raise(_config_path, _config, _settings, *, accepted_host_key_fingerprint=None):
        raise ResourceAlreadyExistsError("SSH host alias", "A Host entry named 'taken' already exists.")

    monkeypatch.setattr("api.remote_servers.ssh_config_writer.add_verified_host_alias", _raise)

    response = TestClient(app).post(
        "/api/remote-servers/aliases",
        json={"alias": "taken", "hostname": "10.0.0.9"},
    )

    assert response.status_code == 409


def test_create_ssh_host_alias_passes_accepted_host_key_fingerprint_header(monkeypatch: pytest.MonkeyPatch):
    add_verified_host_alias = AsyncMock(
        return_value=SshHostAliasOption(alias="new-box", hostname="10.0.0.9", port=22, user=None)
    )
    monkeypatch.setattr("api.remote_servers.ssh_config_writer.add_verified_host_alias", add_verified_host_alias)

    response = TestClient(app).post(
        "/api/remote-servers/aliases",
        json={"alias": "new-box", "hostname": "10.0.0.9"},
        headers={"accepted-host-key-fingerprint": "SHA256:confirmed"},
    )

    assert response.status_code == 201
    _, kwargs = add_verified_host_alias.call_args
    assert kwargs["accepted_host_key_fingerprint"] == "SHA256:confirmed"


def test_create_ssh_host_alias_unreachable_returns_502(monkeypatch: pytest.MonkeyPatch):
    async def _raise(_config_path, _config, _settings, *, accepted_host_key_fingerprint=None):
        raise SshConnectionError("new-box", reason="unreachable")

    monkeypatch.setattr("api.remote_servers.ssh_config_writer.add_verified_host_alias", _raise)

    response = TestClient(app).post(
        "/api/remote-servers/aliases",
        json={"alias": "new-box", "hostname": "10.0.0.9"},
    )

    assert response.status_code == 502
