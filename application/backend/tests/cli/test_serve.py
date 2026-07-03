import importlib

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
