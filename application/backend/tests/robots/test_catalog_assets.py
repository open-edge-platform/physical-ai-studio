from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import HTTPException

import robots.catalog.assets as assets
from robots.catalog.registry import RobotCatalogRegistry


def test_resolve_so101_plugin_assets_uses_plugin_root_resolver(monkeypatch) -> None:
    builtin_root = Path("/builtin-assets")
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", builtin_root)
    definition = RobotCatalogRegistry().get_definition("SO101_Follower")

    assert definition is not None
    assert definition.asset is not None
    assert definition.asset.root_resolver is not None

    urdf_path = assets.resolve_robot_urdf_path(definition)

    assert urdf_path == definition.asset.root_resolver() / definition.asset.urdf_relative_path
    assert not urdf_path.is_relative_to(builtin_root)

    mesh_path = assets.resolve_robot_relative_asset_path(definition, Path("assets/base_so101_v2.stl"))
    package_root = next(iter(definition.asset.packages.values()))
    assert mesh_path == definition.asset.root_resolver() / package_root / "assets/base_so101_v2.stl"
    assert not mesh_path.is_relative_to(builtin_root)


def test_resolve_relative_asset_rejects_ambiguous_package_roots() -> None:
    definition = RobotCatalogRegistry().get_definition("SO101_Follower")

    assert definition is not None
    assert definition.asset is not None
    definition = replace(
        definition,
        asset=replace(definition.asset, packages={"one": Path("one"), "two": Path("two")}),
    )

    with pytest.raises(HTTPException, match="multiple package roots") as error:
        assets.resolve_robot_relative_asset_path(definition, Path("assets/base.stl"))

    assert error.value.status_code == 500
