from pathlib import Path

import pytest

import build_info as build_info_module
import settings as settings_module
from build_info import get_build_info
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


def test_application_version_comes_from_distribution_metadata(monkeypatch) -> None:
    monkeypatch.setenv("VERSION", "9.9.9")
    monkeypatch.setattr(settings_module, "distribution_version", lambda name: "0.3.0")

    assert Settings().version == "0.3.0"


def test_build_info_uses_injected_metadata(monkeypatch) -> None:
    monkeypatch.setenv("STUDIO_GIT_SHA", "a" * 40)
    monkeypatch.setenv("STUDIO_GIT_COMMIT_DATE", "2026-10-09T09:46:34Z")
    monkeypatch.setenv("STUDIO_GIT_TAG", "app/v0.3.0")
    monkeypatch.setenv("STUDIO_GIT_DIRTY", "true")

    assert get_build_info().git_sha == "a" * 40
    assert get_build_info().git_commit_date == "2026-10-09T09:46:34Z"
    assert get_build_info().git_tag == "app/v0.3.0"
    assert get_build_info().git_dirty is True


def test_build_info_reads_checkout_metadata_in_two_git_calls(monkeypatch) -> None:
    for name in ("STUDIO_GIT_SHA", "STUDIO_GIT_COMMIT_DATE", "STUDIO_GIT_TAG", "STUDIO_GIT_DIRTY"):
        monkeypatch.delenv(name, raising=False)

    calls: list[tuple[str, ...]] = []

    def run_git(*args: str) -> str | None:
        calls.append(args)
        if args[0] == "show":
            return "a" * 40 + "\n2026-10-09T09:46:34Z\ntag: app/v0.3.0"
        if args[0] == "status":
            return None
        return None

    monkeypatch.setattr(build_info_module, "_run_git", run_git)

    assert get_build_info() == build_info_module.BuildInfo(
        git_sha="a" * 40,
        git_commit_date="2026-10-09T09:46:34Z",
        git_tag="app/v0.3.0",
    )
    assert calls == [
        ("show", "-s", "--format=%H%n%cI%n%D", "HEAD"),
        ("status", "--porcelain"),
    ]


def test_build_info_does_not_probe_git_when_build_metadata_is_injected(monkeypatch) -> None:
    monkeypatch.setenv("STUDIO_GIT_SHA", "a" * 40)
    monkeypatch.delenv("STUDIO_GIT_COMMIT_DATE", raising=False)
    monkeypatch.delenv("STUDIO_GIT_TAG", raising=False)
    monkeypatch.setattr(build_info_module, "_run_git", lambda *_: pytest.fail("Git should not be called"))

    assert get_build_info() == build_info_module.BuildInfo(git_sha="a" * 40)


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


def test_hotkey_bindings_are_loaded_from_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    merge_user_settings({"hotkeys": {"bindings": {"recording.discard_episode": "Shift+ArrowLeft"}}})

    settings = Settings()
    assert settings.hotkeys.bindings == {"recording.discard_episode": "Shift+ArrowLeft"}


def test_hotkey_bindings_patch_replaces_whole_map(monkeypatch, tmp_path: Path) -> None:
    """The `bindings` dict is a single top-level key of the `hotkeys` group, so
    merge_user_settings's shallow merge replaces it wholesale rather than deep-merging
    individual action ids. Callers must always PATCH the complete bindings map."""
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    merge_user_settings({"hotkeys": {"bindings": {"a": "X", "b": "Y"}}})
    merge_user_settings({"hotkeys": {"bindings": {"a": "Z"}}})

    settings = Settings()
    assert settings.hotkeys.bindings == {"a": "Z"}


def test_video_recording_settings_are_loaded_from_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    merge_user_settings({"video_recording": {"preset": "cpu-h264"}})

    settings = Settings()
    assert settings.video_recording.preset == "cpu-h264"
