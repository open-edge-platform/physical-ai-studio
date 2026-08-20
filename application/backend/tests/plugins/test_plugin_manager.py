from __future__ import annotations

from importlib import metadata
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from exceptions import BaseException as AppBaseException
from exceptions import PluginOperationError
from plugins.manifest import PluginManifestEntry
from plugins.plugin_manager import PluginManager


class _FakeDefinition:
    def __init__(self, type: str, display_name: str, role: str) -> None:
        self.type = type
        self.display_name = display_name
        self.role = role


class _FakeRegistry:
    """Minimal registry stub exposing the attribution API used by the manager."""

    def __init__(self, definitions: list[_FakeDefinition], dist_robots: dict[str, list[str]]) -> None:
        self._definitions = definitions
        self._dist_robots = dist_robots

    def list_definitions(self) -> list[_FakeDefinition]:
        return self._definitions

    def get_definition(self, robot_type: str) -> _FakeDefinition | None:
        return next((definition for definition in self._definitions if definition.type == robot_type), None)

    def robot_types_for_distribution(self, distribution: str) -> list[str]:
        return self._dist_robots.get(distribution, [])


def _manifest_entry(plugin_id: str = "demo-plugin") -> PluginManifestEntry:
    return PluginManifestEntry(
        id=plugin_id,
        name="Demo Plugin",
        description="A demo plugin.",
        category="Demo",
        source="first_party",
        repo_url="https://example.com/demo",
        install_source="demo-plugin",
        robots=[
            {"type": "Demo_Follower", "display_name": "Demo Follower", "role": "follower"},
            {"type": "Demo_Leader", "display_name": "Demo Leader", "role": "leader"},
        ],
        extensions=[
            {
                "id": "demo-extension",
                "name": "Demo Extension",
                "description": "An optional add-on.",
                "install_source": "demo-extension",
            }
        ],
    )


def _manager(
    manifest: list[PluginManifestEntry] | None = None,
    registry: _FakeRegistry | None = None,
) -> PluginManager:
    return PluginManager(
        manifest=manifest if manifest is not None else [_manifest_entry()],
        registry=registry if registry is not None else _FakeRegistry([], {}),
    )


def _fake_distribution(installed: set[str]):
    def _distribution(name: str) -> SimpleNamespace:
        if name in installed:
            return SimpleNamespace(version="1.2.3")
        raise PackageNotFoundError(name)

    return _distribution


def test_list_plugins_reports_available_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution(set()))
    manager = _manager()

    plugins = manager.list_plugins()

    assert len(plugins) == 1
    plugin = plugins[0]
    assert plugin.installed is False
    assert plugin.installed_version is None
    assert [robot.type for robot in plugin.robots] == ["Demo_Follower", "Demo_Leader"]
    assert all(robot.installed is False for robot in plugin.robots)


def test_list_plugins_reports_installed_with_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution({"demo-plugin"}))
    registry = _FakeRegistry(
        [_FakeDefinition("Demo_Follower", "Demo Follower", "follower")],
        {"demo-plugin": ["Demo_Follower", "Demo_Leader"]},
    )
    manager = _manager(registry=registry)

    plugin = manager.list_plugins()[0]

    assert plugin.installed is True
    assert plugin.installed_version == "1.2.3"
    followers = [robot for robot in plugin.robots if robot.type == "Demo_Follower"]
    assert followers and followers[0].installed is True


def test_list_plugins_merges_catalog_types_from_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution({"demo-plugin"}))
    registry = _FakeRegistry(
        [
            _FakeDefinition("Demo_Follower", "Demo Follower", "follower"),
            _FakeDefinition("Demo_Leader", "Demo Leader", "leader"),
        ],
        {"demo-plugin": ["Demo_Follower", "Demo_Leader"]},
    )
    manager = _manager(registry=registry)

    plugin = manager.list_plugins()[0]

    assert {robot.type for robot in plugin.robots} == {"Demo_Follower", "Demo_Leader"}
    assert all(robot.installed for robot in plugin.robots)


def test_get_unknown_plugin_raises() -> None:
    manager = _manager()

    with pytest.raises(AppBaseException) as excinfo:
        manager.get("missing-plugin")
    assert excinfo.value.http_status == 404


def test_install_runs_uv_pip_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution(set()))
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr="", stdout=""))
    monkeypatch.setattr("plugins.plugin_manager.subprocess.run", run)
    manager = _manager()

    manager.install("demo-plugin")

    command = run.call_args.args[0]
    assert command[:2] == ["uv", "pip"]
    assert command[2] == "install"
    assert command[-1] == "demo-plugin"


def test_install_already_installed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution({"demo-plugin"}))
    manager = _manager()

    with pytest.raises(PluginOperationError):
        manager.install("demo-plugin")


def test_install_failure_raises_user_facing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution(set()))
    run = Mock(return_value=SimpleNamespace(returncode=1, stderr="boom", stdout=""))
    monkeypatch.setattr("plugins.plugin_manager.subprocess.run", run)
    manager = _manager()

    with pytest.raises(PluginOperationError, match="boom"):
        manager.install("demo-plugin")


def test_uninstall_runs_uv_pip_uninstall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution({"demo-plugin"}))
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr="", stdout=""))
    monkeypatch.setattr("plugins.plugin_manager.subprocess.run", run)
    manager = _manager()

    manager.uninstall("demo-plugin")

    command = run.call_args.args[0]
    assert command[:2] == ["uv", "pip"]
    assert command[2] == "uninstall"
    assert command[-1] == "demo-plugin"


def test_uninstall_not_installed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution(set()))
    manager = _manager()

    with pytest.raises(PluginOperationError):
        manager.uninstall("demo-plugin")


def test_robot_types_combines_manifest_and_installed_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution({"demo-plugin"}))
    registry = _FakeRegistry(
        [_FakeDefinition("Extra_Follower", "Extra Follower", "follower")],
        {"demo-plugin": ["Extra_Follower"]},
    )
    manager = _manager(registry=registry)

    assert set(manager.robot_types("demo-plugin")) == {"Demo_Follower", "Demo_Leader", "Extra_Follower"}


def test_list_plugins_reports_extensions_with_install_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution({"demo-plugin", "demo-extension"}))
    manager = _manager()

    plugin = manager.list_plugins()[0]

    assert len(plugin.extensions) == 1
    extension = plugin.extensions[0]
    assert extension.id == "demo-extension"
    assert extension.installed is True
    assert extension.installed_version == "1.2.3"


def test_get_extension_returns_extension_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution(set()))
    manager = _manager()

    extension = manager.get("demo-extension")

    assert extension.id == "demo-extension"
    assert extension.installed is False


def test_install_extension_requires_parent_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution(set()))
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr="", stdout=""))
    monkeypatch.setattr("plugins.plugin_manager.subprocess.run", run)
    manager = _manager()

    with pytest.raises(PluginOperationError, match="Install 'Demo Plugin' first"):
        manager.install("demo-extension")

    run.assert_not_called()


def test_install_extension_allowed_when_parent_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution({"demo-plugin"}))
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr="", stdout=""))
    monkeypatch.setattr("plugins.plugin_manager.subprocess.run", run)
    manager = _manager()

    manager.install("demo-extension")

    command = run.call_args.args[0]
    assert command[-1] == "demo-extension"


def test_uninstall_extension_runs_uv_pip_uninstall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "distribution", _fake_distribution({"demo-plugin", "demo-extension"}))
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr="", stdout=""))
    monkeypatch.setattr("plugins.plugin_manager.subprocess.run", run)
    manager = _manager()

    manager.uninstall("demo-extension")

    command = run.call_args.args[0]
    assert command[2] == "uninstall"
    assert command[-1] == "demo-extension"


def test_load_manifest_roundtrip() -> None:
    from plugins.manifest import load_plugin_manifest

    manifest = load_plugin_manifest()

    assert manifest
    assert all(entry.id and entry.install_source for entry in manifest)
    assert any(entry.id == "physicalai-rebot-b601-plugin" for entry in manifest)
