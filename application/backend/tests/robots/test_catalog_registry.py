from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from physicalai_studio_plugin import RobotCatalogDefinition

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
    assert definitions == []


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
    assert registry.list_definitions() == []


def test_so101_types_are_discovered_from_the_installed_plugin() -> None:
    registry = RobotCatalogRegistry()

    expected_types = {"SO101_Follower", "SO101_Leader"}
    discovered_types = [definition.type for definition in registry.list_definitions()]

    assert {robot_type for robot_type in discovered_types if robot_type in expected_types} == expected_types
    assert all(discovered_types.count(robot_type) == 1 for robot_type in expected_types)
    distributions = {registry.get_plugin_distribution(robot_type) for robot_type in expected_types}
    assert distributions == {"physicalai-so101-plugin"}


@pytest.mark.skipif(sys.platform == "win32", reason="The Trossen plugin is not supported on Windows.")
def test_trossen_types_are_discovered_from_the_installed_plugin() -> None:
    registry = RobotCatalogRegistry()

    expected_types = {
        "Trossen_WidowXAI_Follower",
        "Trossen_WidowXAI_Leader",
        "Trossen_Bimanual_WidowXAI_Follower",
        "Trossen_Bimanual_WidowXAI_Leader",
    }

    assert expected_types <= {definition.type for definition in registry.list_definitions()}
    assert {registry.get_plugin_distribution(robot_type) for robot_type in expected_types} == {
        "physicalai-trossen-plugin"
    }


def test_coerce_definition_keeps_every_catalog_field() -> None:
    # A duck-typed (legacy) definition must keep optional fields such as
    # zero_calibration instead of falling back to their defaults.
    values = {field.name: object() for field in dataclasses.fields(RobotCatalogDefinition)}
    values |= {"type": "Legacy_Follower", "display_name": "Legacy", "role": "follower", "asset": None}

    definition = RobotCatalogRegistry._coerce_definition(SimpleNamespace(**values))

    assert isinstance(definition, RobotCatalogDefinition)
    for name, value in values.items():
        assert getattr(definition, name) is value, name
