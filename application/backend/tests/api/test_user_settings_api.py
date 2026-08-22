# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the user-configurable settings API."""

from pathlib import Path

from fastapi.testclient import TestClient

from main import app
from settings import get_settings, write_user_settings


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
