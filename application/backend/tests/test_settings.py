from pathlib import Path

import settings as settings_module
from settings import Settings, get_default_storage_dir, load_user_settings_file, merge_user_settings


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


def test_trainer_settings_are_loaded_from_json(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))

    merge_user_settings({"trainer": {"request_timeout_s": 5.0}})

    settings = Settings()
    assert settings.trainer.request_timeout_s == 5.0
    assert settings.trainer.download_read_timeout_s == 120.0


def test_trainer_settings_patch_keeps_omitted_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    merge_user_settings({"trainer": {"request_timeout_s": 5.0}})
    merge_user_settings({"trainer": {"download_read_timeout_s": 10.0}})

    settings = Settings()
    assert settings.trainer.request_timeout_s == 5.0
    assert settings.trainer.download_read_timeout_s == 10.0


def test_huggingface_token_is_loaded_from_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    merge_user_settings({"huggingface": {"hf_token": "hf_example"}})

    settings = Settings()
    assert settings.huggingface.hf_token is not None
    assert settings.huggingface.hf_token.get_secret_value() == "hf_example"
    assert load_user_settings_file() == {"huggingface": {"hf_token": "hf_example"}}
