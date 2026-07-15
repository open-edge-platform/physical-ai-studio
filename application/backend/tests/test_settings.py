from pathlib import Path

import pytest
from pydantic import ValidationError

import settings as settings_module
from settings import Settings, get_default_storage_dir


def test_default_storage_dir_uses_xdg_data_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings_module.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))

    assert get_default_storage_dir() == tmp_path / "xdg-data" / "physicalai"


def test_default_storage_dir_ignores_relative_xdg_data_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings_module.sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", "relative/path")

    assert get_default_storage_dir() == tmp_path / ".local" / "share" / "physicalai"


def test_default_storage_dir_uses_macos_application_support(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings_module.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert get_default_storage_dir() == tmp_path / "Library" / "Application Support" / "physicalai"


def test_storage_dir_override_expands_user(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    settings = Settings(STORAGE_DIR="~/custom-storage")

    assert settings.storage_dir == tmp_path / "custom-storage"


def test_data_dir_is_storage_backed_even_with_data_dir_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    custom_data_dir = tmp_path / "custom-data"
    monkeypatch.setenv("DATA_DIR", str(custom_data_dir))

    settings = Settings(STORAGE_DIR="~/custom-storage")

    assert settings.data_dir == tmp_path / "custom-storage" / "data"


def _clear_trainer_env(monkeypatch) -> None:
    for var in ("TRAINING_MODE", "TRAINER_URL"):
        monkeypatch.delenv(var, raising=False)


def test_training_mode_defaults_to_local(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _clear_trainer_env(monkeypatch)

    assert Settings(STORAGE_DIR="~/s").training_mode == "local"


def test_remote_training_requires_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _clear_trainer_env(monkeypatch)

    with pytest.raises(ValidationError, match="TRAINER_URL"):
        Settings(TRAINING_MODE="remote", STORAGE_DIR="~/s")


def test_remote_training_accepts_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _clear_trainer_env(monkeypatch)

    settings = Settings(
        TRAINING_MODE="remote",
        TRAINER_URL="https://trainer.test",
        STORAGE_DIR="~/s",
    )

    assert settings.training_mode == "remote"
    assert settings.trainer_url == "https://trainer.test"


@pytest.mark.parametrize(
    "bad_url",
    ["ftp://trainer.test", "trainer.test", "https://", "not a url"],
)
def test_trainer_url_rejects_invalid_scheme_or_host_in_remote_mode(monkeypatch, tmp_path: Path, bad_url: str) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _clear_trainer_env(monkeypatch)

    with pytest.raises(ValidationError, match="TRAINER_URL"):
        Settings(TRAINING_MODE="remote", TRAINER_URL=bad_url, STORAGE_DIR="~/s")


def test_trainer_url_not_validated_in_local_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _clear_trainer_env(monkeypatch)

    settings = Settings(TRAINING_MODE="local", TRAINER_URL="not a url", STORAGE_DIR="~/s")

    assert settings.trainer_url == "not a url"
