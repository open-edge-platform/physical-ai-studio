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


def test_huggingface_access_lists_xr0_requirements(monkeypatch, tmp_path: Path) -> None:
    """XR0 is selectable in the UI, so its Hub dependencies must be registered."""
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with TestClient(app) as client:
        response = client.get("/api/policies/xr0/huggingface-access")

    assert response.status_code == 200
    assert [requirement["repository"] for requirement in response.json()["requirements"]] == [
        "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",
        "Qwen/Qwen3-VL-4B-Instruct",
    ]


def test_huggingface_access_unknown_policy_is_404(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with TestClient(app) as client:
        response = client.get("/api/policies/unknown/huggingface-access")

    assert response.status_code == 404


def test_backends_cover_every_selectable_policy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with TestClient(app) as client:
        response = client.get("/api/policies/backends")

    backends = response.json()
    assert set(backends) == {"act", "molmoact2", "pi05", "rldx1", "smolvla", "xr0"}
    assert backends["xr0"]
