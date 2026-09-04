# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the user-configurable settings API."""

from pathlib import Path

from fastapi.testclient import TestClient

from api.settings import SettingsUpdate, update_user_settings
from main import app
from settings import SshProvisioningSettings, get_settings, write_user_settings


def test_get_settings_masks_huggingface_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    write_user_settings({"huggingface": {"hf_token": "super-secret"}})

    with TestClient(app) as client:
        response = client.get("/api/settings")

    assert response.status_code == 200
    assert "super-secret" not in response.text
    assert response.json()["huggingface"]["hf_token"] is not None


def test_patch_settings_preserves_omitted_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    write_user_settings({"huggingface": {"hf_token": "super-secret"}})

    with TestClient(app) as client:
        response = client.patch("/api/settings", json={"trainer": {"request_timeout_s": 45.0}})

    assert response.status_code == 200
    assert response.json()["trainer"]["request_timeout_s"] == 45.0
    assert "super-secret" not in response.text
    assert get_settings().huggingface.hf_token is not None
    assert get_settings().huggingface.hf_token.get_secret_value() == "super-secret"


def test_patch_settings_clears_huggingface_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    write_user_settings({"huggingface": {"hf_token": "super-secret"}})

    with TestClient(app) as client:
        response = client.patch("/api/settings", json={"huggingface": {"hf_token": None}})

    assert response.status_code == 200
    assert get_settings().huggingface.hf_token is None


def test_patch_empty_huggingface_token_clears_setting(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    write_user_settings({"huggingface": {"hf_token": "super-secret"}})

    with TestClient(app) as client:
        response = client.patch("/api/settings", json={"huggingface": {"hf_token": ""}})

    assert response.status_code == 200
    assert get_settings().huggingface.hf_token is None


def test_get_settings_reports_ssh_provisioning_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with TestClient(app) as client:
        response = client.get("/api/settings")

    assert response.status_code == 200
    assert response.json()["ssh"] == get_settings().ssh.model_dump()


def test_patch_ssh_settings_updates_flat_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with TestClient(app) as client:
        response = client.patch("/api/settings", json={"ssh": {"connect_timeout_s": 42.0}})

    assert response.status_code == 200
    assert response.json()["ssh"]["connect_timeout_s"] == 42.0
    # The grouped API field and the flat field the SSH services read are the
    # same setting, so a save has to be visible through both.
    assert get_settings().ssh_connect_timeout_s == 42.0


def test_patch_ssh_settings_preserves_other_groups(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    write_user_settings({"huggingface": {"hf_token": "super-secret"}})

    with TestClient(app) as client:
        response = client.patch("/api/settings", json={"ssh": {"preflight_timeout_s": 12.0}})

    assert response.status_code == 200
    assert get_settings().ssh_preflight_timeout_s == 12.0
    assert get_settings().huggingface.hf_token is not None


def test_patch_ssh_settings_preserves_omitted_ssh_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with TestClient(app) as client:
        client.patch("/api/settings", json={"ssh": {"connect_timeout_s": 42.0}})
        response = client.patch("/api/settings", json={"ssh": {"command_timeout_s": 7.0}})

    assert response.status_code == 200
    assert get_settings().ssh_connect_timeout_s == 42.0
    assert get_settings().ssh_command_timeout_s == 7.0


def test_patch_null_ssh_group_clears_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    default_connect_timeout = get_settings().ssh_connect_timeout_s

    with TestClient(app) as client:
        client.patch("/api/settings", json={"ssh": {"connect_timeout_s": 42.0}})
        response = client.patch("/api/settings", json={"ssh": None})

    assert response.status_code == 200
    assert get_settings().ssh_connect_timeout_s == default_connect_timeout


def test_patch_rejects_non_positive_ssh_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with TestClient(app) as client:
        response = client.patch("/api/settings", json={"ssh": {"connect_timeout_s": 0}})

    # `exception_handlers.validation_exception_handler` remaps FastAPI's
    # default 422 for a `RequestValidationError` to 400 across this API.
    assert response.status_code == 400


def test_patch_ssh_settings_ignores_environment_only_fields(monkeypatch, tmp_path: Path) -> None:
    """Config the settings page must never be able to override.

    `ssh_config_path`, `ssh_known_hosts_path`, `trainer_image_registry`, and
    the cosign policy configure *how* Studio trusts a host or an image, not a
    bounded timeout - they stay environment-only even after this migration,
    and the (extra) keys below are silently dropped by `SshProvisioningSettings`
    rather than accepted.
    """
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    identity_regexp_before = get_settings().cosign_certificate_identity_regexp

    with TestClient(app) as client:
        response = client.patch(
            "/api/settings",
            json={
                "ssh": {
                    "connect_timeout_s": 42.0,
                    "cosign_certificate_identity_regexp": ".*",
                    "ssh_known_hosts_path": "/tmp/attacker-known-hosts",
                    "trainer_image_registry": "attacker.example/registry",
                }
            },
        )

    assert response.status_code == 200
    settings_file = (tmp_path / "settings.json").read_text(encoding="utf-8")
    assert "cosign" not in settings_file.lower()
    assert "known_hosts" not in settings_file.lower()
    assert "attacker" not in settings_file.lower()
    assert get_settings().cosign_certificate_identity_regexp == identity_regexp_before


def test_ssh_settings_no_longer_read_from_environment(monkeypatch, tmp_path: Path) -> None:
    """After this migration the SSH settings are settings-file only.

    An operator's `.env`/environment variable for any of these must be a
    no-op: the settings page and its JSON file are the single source of
    truth, so there is exactly one place to look for the effective value.
    """
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setenv("SSH_CONNECT_TIMEOUT_S", "999")

    with TestClient(app) as client:
        response = client.get("/api/settings")

    assert response.status_code == 200
    assert response.json()["ssh"]["connect_timeout_s"] == 10.0


async def test_patch_ssh_settings_updates_a_timeout(monkeypatch, tmp_path: Path) -> None:
    """There is no master switch anymore - patching SSH settings is a plain save."""
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    response = await update_user_settings(SettingsUpdate(ssh=SshProvisioningSettings(connect_timeout_s=42.0)))

    assert response.ssh.connect_timeout_s == 42.0
    assert get_settings().ssh_connect_timeout_s == 42.0
