from pathlib import Path

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
