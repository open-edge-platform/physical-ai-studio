from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.dependencies import get_health_service
from api.plugins import get_plugin_manager
from main import app
from plugins.plugin_manager import PluginInfo, PluginManager, PluginRobot
from services.health_service import HealthService


def _plugin_info(
    *,
    plugin_id: str = "demo-plugin",
    installed: bool = False,
    robots: list[PluginRobot] | None = None,
) -> PluginInfo:
    return PluginInfo(
        id=plugin_id,
        name="Demo Plugin",
        description="A demo plugin.",
        repo_url="https://example.com/demo",
        installed=installed,
        installed_version="1.2.3" if installed else None,
        robots=robots if robots is not None else [],
    )


def _stub_manager(*, info: PluginInfo) -> AsyncMock:
    manager = AsyncMock(spec=PluginManager)
    manager.list_plugins.return_value = [info]
    manager.install.return_value = None
    manager.uninstall.return_value = None
    return manager


def _override(manager: AsyncMock) -> None:
    app.dependency_overrides[get_plugin_manager] = lambda: manager
    app.dependency_overrides[get_health_service] = lambda: HealthService()


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
    _override(manager)

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
    assert plugin["robot_count"] == 1


def test_list_plugins_reports_installed_and_robot_count() -> None:
    manager = _stub_manager(
        info=_plugin_info(
            installed=True,
            robots=[PluginRobot("Demo_Follower", "Demo Follower", "follower", True)],
        ),
    )
    _override(manager)

    try:
        client = TestClient(app)
        response = client.get("/api/plugins")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    plugin = response.json()[0]
    assert plugin["installed"] is True
    assert plugin["installed_version"] == "1.2.3"
    assert plugin["robot_count"] == 1


def test_install_plugin_returns_restart_required() -> None:
    manager = _stub_manager(info=_plugin_info())
    _override(manager)

    try:
        client = TestClient(app)
        response = client.post("/api/plugins", json={"plugin_id": "demo-plugin"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"restart_required": True}
    manager.install.assert_called_once_with("demo-plugin")


def test_uninstall_plugin_returns_restart_required() -> None:
    manager = _stub_manager(info=_plugin_info(installed=True))
    _override(manager)

    try:
        client = TestClient(app)
        response = client.delete("/api/plugins/demo-plugin")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"restart_required": True}
    manager.uninstall.assert_called_once_with("demo-plugin")


def test_uninstall_plugin_allows_robots_in_use() -> None:
    manager = _stub_manager(
        info=_plugin_info(installed=True, robots=[PluginRobot("Demo_Follower", "Demo Follower", "follower", True)]),
    )
    _override(manager)

    try:
        client = TestClient(app)
        response = client.delete("/api/plugins/demo-plugin")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"restart_required": True}
    manager.uninstall.assert_called_once_with("demo-plugin")
