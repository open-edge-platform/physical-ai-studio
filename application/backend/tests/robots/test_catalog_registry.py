from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from robots.catalog.registry import RobotCatalogRegistry


class _FakeEntryPoint:
    def __init__(self, name: str, value: str, loader, dist: str | None = None):
        self.name = name
        self.value = value
        self._loader = loader
        self.dist = SimpleNamespace(name=dist) if dist is not None else None

    def load(self):
        return self._loader()


def _register_fake_robot(robot_type: str):
    def _plugin(registry):
        registry.register_robot(
            MockRobotCatalogDefinition(robot_type=robot_type),
        )

    def _loader():
        return _plugin

    return _loader


class MockRobotCatalogDefinition:
    """Stand-in RobotCatalogDefinition for registry tests."""

    def __init__(self, robot_type: str) -> None:
        self.type = robot_type
        self.display_name = robot_type
        self.role = "follower"
        self.category = "Other"
        self.source = "external"
        self.robot_builder = None
        self.robot_payload = None
        self.asset = None
        self.adapter_options = MockAdapterOptions()
        self.probe = None


class MockAdapterOptions:
    include_velocities = False
    goal_time_scale = 1.0
    external_effort_gain = None


def test_registry_installs_lerobot_types_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "lerobot.lerobot_types", raising=False)

    with patch("robots.catalog.registry.entry_points", return_value=[]):
        RobotCatalogRegistry()

    alias = sys.modules.get("lerobot.lerobot_types")
    assert alias is not None
    assert getattr(alias, "RobotAction", None) is not None


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


def test_registry_attributes_robot_types_to_owning_distribution() -> None:
    entry_points = [
        _FakeEntryPoint(
            "plugin-a",
            "plugin_a.module:register",
            _register_fake_robot("Demo_A_Follower"),
            dist="physicalai-plugin-a",
        ),
        _FakeEntryPoint(
            "plugin-b",
            "plugin_b.module:register",
            _register_fake_robot("Demo_B_Follower"),
            dist="physicalai-plugin-b",
        ),
    ]

    with patch("robots.catalog.registry.entry_points", return_value=entry_points):
        registry = RobotCatalogRegistry()

    assert registry.get_plugin_distribution("Demo_A_Follower") == "physicalai-plugin-a"
    assert registry.get_plugin_distribution("Demo_B_Follower") == "physicalai-plugin-b"
    assert registry.get_plugin_distribution("SO101_Follower") is None
    assert registry.robot_types_for_distribution("physicalai-plugin-a") == ["Demo_A_Follower"]
    assert registry.robot_types_for_distribution("physicalai-plugin-b") == ["Demo_B_Follower"]


# include_velocities decides whether ".vel" keys join the robot's feature list,
# which sets the recorded dataset schema and the observation vector fed to a
# policy. Changing it silently makes existing checkpoints incompatible with the
# robot, so the built-in values are pinned here.
@pytest.mark.parametrize(
    ("robot_type", "include_velocities", "external_effort_gain"),
    [
        ("SO101_Follower", False, None),
        ("SO101_Leader", False, None),
        ("Trossen_WidowXAI_Follower", True, 0.1),
        ("Trossen_WidowXAI_Leader", True, 0.1),
        ("Trossen_Bimanual_WidowXAI_Follower", True, 0.1),
        ("Trossen_Bimanual_WidowXAI_Leader", True, 0.1),
    ],
)
def test_builtin_adapter_options_are_pinned(
    robot_type: str, include_velocities: bool, external_effort_gain: float | None
) -> None:
    definition = RobotCatalogRegistry().get_definition(robot_type)

    assert definition is not None
    assert definition.adapter_options.include_velocities is include_velocities
    assert definition.adapter_options.external_effort_gain == external_effort_gain
