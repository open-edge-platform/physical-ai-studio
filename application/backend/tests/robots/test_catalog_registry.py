from __future__ import annotations

from unittest.mock import patch

from robots.catalog.registry import RobotCatalogRegistry


class _FakeEntryPoint:
    def __init__(self, name: str, value: str, loader):
        self.name = name
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader()


def test_registry_continues_when_plugin_discovery_fails() -> None:
    with patch("robots.catalog.registry.entry_points", side_effect=RuntimeError("boom")):
        registry = RobotCatalogRegistry()

    definitions = registry.list_definitions()
    assert definitions
    assert any(definition.type == "SO101_Follower" for definition in definitions)


def test_registry_skips_bad_plugins_and_continues_loading_remaining_plugins() -> None:
    called: list[str] = []

    def _raises_on_load():
        raise RuntimeError("cannot import")

    def _returns_non_callable():
        return 123

    def _raises_on_register():
        def _plugin(_registry):
            raise RuntimeError("register failed")

        return _plugin

    def _good_plugin():
        def _plugin(_registry):
            called.append("ok")

        return _plugin

    entry_points = [
        _FakeEntryPoint("broken-load", "plugin.module:bad_load", _raises_on_load),
        _FakeEntryPoint("non-callable", "plugin.module:not_callable", _returns_non_callable),
        _FakeEntryPoint("broken-register", "plugin.module:broken_register", _raises_on_register),
        _FakeEntryPoint("good", "plugin.module:register", _good_plugin),
    ]

    with patch("robots.catalog.registry.entry_points", return_value=entry_points):
        registry = RobotCatalogRegistry()

    assert called == ["ok"]
    definitions = registry.list_definitions()
    assert definitions
    assert any(definition.type == "SO101_Follower" for definition in definitions)
