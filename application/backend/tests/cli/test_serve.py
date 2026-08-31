import importlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

serve_module = importlib.import_module("cli.serve")


@pytest.fixture(autouse=True)
def _clear_packaged_runtime_environment() -> Iterator[None]:
    security_module = importlib.import_module("core.security")
    os.environ.pop("ALEMBIC_CONFIG_PATH", None)
    os.environ.pop("ALEMBIC_SCRIPT_LOCATION", None)
    os.environ.pop("STATIC_FILES_DIR", None)
    os.environ.pop("HOST", None)
    os.environ.pop("PORT", None)
    security_module.get_ssh_feature_availability.cache_clear()
    yield
    os.environ.pop("ALEMBIC_CONFIG_PATH", None)
    os.environ.pop("ALEMBIC_SCRIPT_LOCATION", None)
    os.environ.pop("STATIC_FILES_DIR", None)
    os.environ.pop("HOST", None)
    os.environ.pop("PORT", None)
    security_module.get_ssh_feature_availability.cache_clear()


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


def test_configure_packaged_runtime_updates_settings(monkeypatch) -> None:
    settings_module = importlib.import_module("settings")

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


def test_start_server_reconciles_settings_host_with_the_actual_bind_argument(monkeypatch) -> None:
    """A `--host` override must be visible to anything reading `Settings.host` afterwards.

    Otherwise the SSH feature's loopback check (which only sees `Settings`)
    could evaluate a stale default instead of the address uvicorn actually
    binds to.
    """
    settings_module = importlib.import_module("settings")

    monkeypatch.setattr(serve_module, "_configure_packaged_runtime", lambda: None)
    monkeypatch.setattr(serve_module, "_sync_missing_robot_assets", lambda: None)
    monkeypatch.setattr(serve_module, "_run_migrations", lambda: None)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)

    serve_module.start_server("0.0.0.0", 9999)

    assert settings_module.get_settings().host == "0.0.0.0"
    assert settings_module.get_settings().port == 9999
