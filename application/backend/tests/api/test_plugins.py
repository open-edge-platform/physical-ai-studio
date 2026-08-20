from __future__ import annotations

from unittest.mock import Mock

from fastapi.testclient import TestClient

import api.plugins as plugins_api
import api.system as system_api
from api.dependencies import get_health_service
from api.plugins import PluginExtensionInfo, PluginInfo, PluginManager, PluginRobot, get_plugin_manager
from main import app
from services.health_service import HealthService


def _plugin_info(
    *,
    plugin_id: str = "demo-plugin",
    installed: bool = False,
    robots: list[PluginRobot] | None = None,
    extensions: list[PluginExtensionInfo] | None = None,
) -> PluginInfo:
    return PluginInfo(
        id=plugin_id,
        name="Demo Plugin",
        description="A demo plugin.",
        category="Demo",
        source="first_party",
        repo_url="https://example.com/demo",
        installed=installed,
        installed_version="1.2.3" if installed else None,
        robots=robots if robots is not None else [],
        extensions=extensions if extensions is not None else [],
    )


def _stub_manager(*, info: PluginInfo, robot_types: list[str] | None = None) -> Mock:
    manager = Mock(spec=PluginManager)
    manager.list_plugins.return_value = [info]
    manager.get.return_value = info
    manager.robot_types.return_value = robot_types if robot_types is not None else [r.type for r in info.robots]
    manager.install.return_value = None
    manager.uninstall.return_value = None
    return manager


def _override(manager: Mock, in_use: list[str]) -> None:
    async def _fake_in_use(_session, robot_types: list[str]) -> list[str]:
        return [type_ for type_ in in_use if type_ in robot_types]

    app.dependency_overrides[get_plugin_manager] = lambda: manager
    app.dependency_overrides[get_health_service] = lambda: HealthService()
    plugins_api.find_robot_types_in_use_async = _fake_in_use


def test_health_reports_server_instance_and_restart_state() -> None:
    health_service = HealthService()
    health_service.mark_plugin_restart_required()
    app.dependency_overrides[get_health_service] = lambda: health_service

    try:
        client = TestClient(app)
        response = client.get("/api/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "instance_id": health_service.instance_id,
        "restart_required": True,
    }
    assert response.headers["cache-control"] == "no-store"


def test_list_plugins_returns_manifest_plugins() -> None:
    manager = _stub_manager(
        info=_plugin_info(robots=[PluginRobot("Demo_Follower", "Demo Follower", "follower", False)]),
    )
    _override(manager, in_use=[])

    try:
        client = TestClient(app)
        response = client.get("/api/plugins")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    plugin = body[0]
    assert plugin["id"] == "demo-plugin"
    assert plugin["installed"] is False
    assert plugin["installed_version"] is None
    assert plugin["in_use_robot_count"] == 0
    assert plugin["robots"][0]["type"] == "Demo_Follower"
    assert plugin["robots"][0]["installed"] is False


def test_list_plugins_reports_installed_and_in_use() -> None:
    manager = _stub_manager(
        info=_plugin_info(
            installed=True,
            robots=[PluginRobot("Demo_Follower", "Demo Follower", "follower", True)],
        ),
        robot_types=["Demo_Follower"],
    )
    _override(manager, in_use=["Demo_Follower"])

    try:
        client = TestClient(app)
        response = client.get("/api/plugins")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    plugin = response.json()[0]
    assert plugin["installed"] is True
    assert plugin["installed_version"] == "1.2.3"
    assert plugin["in_use_robot_count"] == 1


def test_install_plugin_returns_restart_required() -> None:
    manager = _stub_manager(info=_plugin_info())
    _override(manager, in_use=[])

    try:
        client = TestClient(app)
        response = client.post("/api/plugins/demo-plugin/install")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"restart_required": True}
    manager.install.assert_called_once_with("demo-plugin")


def test_uninstall_plugin_returns_restart_required() -> None:
    manager = _stub_manager(info=_plugin_info(installed=True))
    _override(manager, in_use=[])

    try:
        client = TestClient(app)
        response = client.post("/api/plugins/demo-plugin/uninstall")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"restart_required": True}
    manager.uninstall.assert_called_once_with("demo-plugin")


def test_uninstall_plugin_blocked_when_robots_in_use() -> None:
    manager = _stub_manager(
        info=_plugin_info(installed=True, robots=[PluginRobot("Demo_Follower", "Demo Follower", "follower", True)]),
        robot_types=["Demo_Follower"],
    )
    _override(manager, in_use=["Demo_Follower"])

    try:
        client = TestClient(app)
        response = client.post("/api/plugins/demo-plugin/uninstall")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert "Demo_Follower" in response.json()["message"]
    manager.uninstall.assert_not_called()


def test_restart_endpoint_reexecs_process(monkeypatch) -> None:
    execv = Mock(side_effect=SystemExit)
    execvp = Mock()
    kill = Mock()
    monkeypatch.setattr(system_api.os, "execv", execv)
    monkeypatch.setattr(system_api.os, "execvp", execvp)
    monkeypatch.setattr(system_api.os, "kill", kill)
    monkeypatch.setattr(system_api.time, "sleep", lambda _seconds: None)

    class _FakeThread:
        def __init__(self, target, *args, **kwargs):
            self.target = target

        def start(self) -> None:
            try:
                self.target()
            except SystemExit:
                pass

    monkeypatch.setattr(system_api.threading, "Thread", _FakeThread)
    monkeypatch.setattr(system_api.sys, "orig_argv", [system_api.sys.executable, "-m", "pytest"])

    client = TestClient(app)
    response = client.post("/api/system/restart")

    assert response.status_code == 202
    assert response.json() == {"status": "restarting"}
    execv.assert_called_once()
    execvp.assert_not_called()
    kill.assert_not_called()


def test_restart_endpoint_falls_back_to_sigterm(monkeypatch) -> None:
    execv = Mock(side_effect=OSError("boom"))
    execvp = Mock(side_effect=OSError("boom"))
    kill = Mock()
    monkeypatch.setattr(system_api.os, "execv", execv)
    monkeypatch.setattr(system_api.os, "execvp", execvp)
    monkeypatch.setattr(system_api.os, "kill", kill)
    monkeypatch.setattr(system_api.time, "sleep", lambda _seconds: None)

    class _FakeThread:
        def __init__(self, target, *args, **kwargs):
            self.target = target

        def start(self) -> None:
            self.target()

    monkeypatch.setattr(system_api.threading, "Thread", _FakeThread)
    monkeypatch.setattr(system_api.sys, "orig_argv", ["python-not-found", "-m", "pytest"])

    client = TestClient(app)
    response = client.post("/api/system/restart")

    assert response.status_code == 202
    assert response.json() == {"status": "restarting"}
    execvp.assert_called_once_with("python-not-found", ["python-not-found", "-m", "pytest"])
    execv.assert_called_once()
    kill.assert_called_once()
    pid, signal = kill.call_args.args
    assert signal == system_api.signal.SIGTERM
