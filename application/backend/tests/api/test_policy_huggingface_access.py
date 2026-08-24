from pathlib import Path

from fastapi.testclient import TestClient

from main import app
from settings import write_user_settings


def test_huggingface_access_reports_missing_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with TestClient(app) as client:
        response = client.get("/api/policies/pi05/huggingface-access")

    assert response.json() == {
        "requirements": [
            {
                "repository": "lerobot/pi05_base",
                "status": "missing_token",
                "access_url": "https://huggingface.co/lerobot/pi05_base",
                "required": True,
            },
            {
                "repository": "google/paligemma-3b-pt-224",
                "status": "missing_token",
                "access_url": "https://huggingface.co/google/paligemma-3b-pt-224",
                "required": True,
            },
        ],
    }


def test_huggingface_access_reports_gated_access_denied(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    write_user_settings({"huggingface": {"hf_token": "hf-secret"}})

    class AccessDenied(Exception):
        pass

    def deny_access(*_args, **_kwargs) -> None:
        raise AccessDenied

    monkeypatch.setattr("api.policies.GatedRepoError", AccessDenied)
    monkeypatch.setattr("api.policies.HfApi.auth_check", deny_access)
    with TestClient(app) as client:
        response = client.get("/api/policies/pi05/huggingface-access")

    assert [requirement["status"] for requirement in response.json()["requirements"]] == ["denied", "denied"]
