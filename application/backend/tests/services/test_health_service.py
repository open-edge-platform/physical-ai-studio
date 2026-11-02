from services import health_service
from services.health_service import HealthService


def test_health_service_marks_restart_when_catalog_plugins_change(monkeypatch) -> None:
    initial_plugins = {("plugin", "package:register", "1.0.0")}
    updated_plugins = {("plugin", "package:register", "2.0.0")}
    service = HealthService(_installed_catalog_plugins=initial_plugins)

    monkeypatch.setattr(health_service, "_installed_catalog_plugins", lambda: updated_plugins)
    service.refresh_plugin_restart_required()

    assert service.plugin_restart_required is True


def test_health_service_ignores_unchanged_catalog_plugins(monkeypatch) -> None:
    installed_plugins = {("plugin", "package:register", "1.0.0")}
    service = HealthService(_installed_catalog_plugins=installed_plugins)
    monkeypatch.setattr(health_service, "_installed_catalog_plugins", lambda: installed_plugins)

    service.refresh_plugin_restart_required()

    assert service.plugin_restart_required is False
