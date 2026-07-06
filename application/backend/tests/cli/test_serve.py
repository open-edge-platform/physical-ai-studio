import importlib
from pathlib import Path

import pytest

serve_module = importlib.import_module("cli.serve")


def test_sync_missing_robot_assets_skips_when_available(monkeypatch) -> None:
    sync_called = False

    monkeypatch.setattr(serve_module, "builtin_robot_assets_are_available", lambda: True)

    def fake_sync_robot_assets() -> None:
        nonlocal sync_called
        sync_called = True

    monkeypatch.setattr(serve_module, "sync_robot_assets", fake_sync_robot_assets)

    serve_module._sync_missing_robot_assets()

    assert not sync_called


def test_sync_missing_robot_assets_syncs_when_unavailable(monkeypatch) -> None:
    sync_called = False

    monkeypatch.setattr(serve_module, "builtin_robot_assets_are_available", lambda: False)

    def fake_sync_robot_assets() -> None:
        nonlocal sync_called
        sync_called = True

    monkeypatch.setattr(serve_module, "sync_robot_assets", fake_sync_robot_assets)

    serve_module._sync_missing_robot_assets()

    assert sync_called


def test_sync_missing_robot_assets_exits_when_sync_fails(monkeypatch) -> None:
    monkeypatch.setattr(serve_module, "builtin_robot_assets_are_available", lambda: False)

    def fake_sync_robot_assets() -> None:
        raise OSError("no network")

    monkeypatch.setattr(serve_module, "sync_robot_assets", fake_sync_robot_assets)

    with pytest.raises(SystemExit):
        serve_module._sync_missing_robot_assets()


def test_configure_packaged_runtime_refreshes_cached_settings(monkeypatch) -> None:
    settings_module = importlib.import_module("settings")
    settings_module.get_settings.cache_clear()

    stale_settings = settings_module.get_settings()
    assert stale_settings.alembic_script_location == "src/alembic"

    fake_package_root = Path("/tmp/packaged-root")
    monkeypatch.setattr(serve_module, "_package_root", lambda: fake_package_root)

    monkeypatch.delenv("ALEMBIC_CONFIG_PATH", raising=False)
    monkeypatch.delenv("ALEMBIC_SCRIPT_LOCATION", raising=False)
    monkeypatch.delenv("STATIC_FILES_DIR", raising=False)

    serve_module._configure_packaged_runtime()

    refreshed_settings = settings_module.get_settings()
    assert refreshed_settings.alembic_config_path == str(fake_package_root / "alembic.ini")
    assert refreshed_settings.alembic_script_location == str(fake_package_root / "alembic")


def test_serve_click_defaults_are_lazy_and_use_settings() -> None:
    host_option = next(param for param in serve_module.serve.params if param.name == "host")
    port_option = next(param for param in serve_module.serve.params if param.name == "port")

    assert callable(host_option.default)
    assert callable(port_option.default)
