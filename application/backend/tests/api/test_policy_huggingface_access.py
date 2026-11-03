import subprocess
import sys
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from settings import write_user_settings


def test_importing_policy_api_does_not_import_policy_classes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import api.policies; assert 'physicalai.policies' not in sys.modules",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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


def test_backends_skip_policy_with_missing_optional_dependency(monkeypatch, tmp_path: Path) -> None:
    from physicalai.policies import get_physicalai_policy_class

    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    def load_policy(name: str):
        if name == "pi05":
            raise ModuleNotFoundError("No module named 'diffusers'", name="diffusers")
        return get_physicalai_policy_class(name)

    with (
        patch("physicalai.policies.get_physicalai_policy_class", side_effect=load_policy),
        patch("api.policies.logger") as logger,
        TestClient(app) as client,
    ):
        response = client.get("/api/policies/backends")

    assert response.status_code == 200
    assert "pi05" not in response.json()
    assert "act" in response.json()
    logger.warning.assert_called_once()


def test_backends_skip_policy_with_deferred_missing_dependency(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    installed_version = metadata.version

    def version(name: str) -> str:
        if name == "num2words":
            raise metadata.PackageNotFoundError(name)
        return installed_version(name)

    with (
        patch("training.policy_catalog.metadata.version", side_effect=version),
        patch("api.policies.logger") as logger,
        TestClient(app) as client,
    ):
        response = client.get("/api/policies/backends")
        available = client.get("/api/policies/available")

    assert response.status_code == 200
    assert "smolvla" not in response.json()
    assert "smolvla" not in {policy["id"] for policy in available.json()}
    assert "act" in response.json()
    assert sum(call.args[1] == "smolvla" for call in logger.warning.call_args_list) == 2


def test_available_policies_include_display_and_vram_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with TestClient(app) as client:
        response = client.get("/api/policies/available")

    assert response.status_code == 200
    policies = {policy["id"]: policy for policy in response.json()}
    assert policies["act"] == {
        "id": "act",
        "name": "ACT",
        "description": "Action Chunking with Transformers, lightweight and fast to train (BSD 3-Clause license)",
        "min_vram": 2 * 1024**3,
    }


def test_unsupported_library_policies_are_not_selectable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with (
        patch("physicalai.policies.list_physicalai_policies", return_value=["act", "cosmos3", "new"]),
        TestClient(app) as client,
    ):
        available = client.get("/api/policies/available")
        backends = client.get("/api/policies/backends")

    assert [policy["id"] for policy in available.json()] == ["act"]
    assert set(backends.json()) == {"act"}


def test_policy_names_come_from_library(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with patch("physicalai.policies.list_physicalai_policies", return_value=["act"]), TestClient(app) as client:
        backends = client.get("/api/policies/backends")
        unknown = client.get("/api/policies/pi05/huggingface-access")

    assert set(backends.json()) == {"act"}
    assert unknown.status_code == 404


def test_backends_cover_every_selectable_policy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    with TestClient(app) as client:
        response = client.get("/api/policies/backends")

    backends = response.json()
    assert set(backends) == {"act", "molmoact2", "pi05", "rldx1", "smolvla", "xr0"}
    assert backends["xr0"]
