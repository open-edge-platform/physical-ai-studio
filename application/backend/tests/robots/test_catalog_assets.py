import robots.catalog.assets as assets
from robots.catalog import so101, widowxai


def test_get_builtin_robot_assets_root_uses_shared_constant() -> None:
    assert assets.get_builtin_robot_assets_root() == assets.BUILTIN_ROBOT_ASSETS_ROOT


def test_builtin_robot_assets_are_available_requires_all_urdfs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)
    definitions = so101.get_definitions() + widowxai.get_definitions()
    urdf_relative_paths = {
        definition.asset.urdf_relative_path for definition in definitions if definition.asset is not None
    }

    for relative_path in urdf_relative_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("urdf", encoding="utf-8")

    assert assets.builtin_robot_assets_are_available()

    (tmp_path / next(iter(urdf_relative_paths))).unlink()

    assert not assets.builtin_robot_assets_are_available()


def test_resolve_robot_urdf_path_uses_builtin_assets_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)
    definition = next(definition for definition in so101.get_definitions() if definition.type == "SO101_Follower")

    urdf_path = tmp_path / "SO101" / "so101_new_calib.urdf"
    urdf_path.parent.mkdir(parents=True)
    urdf_path.write_text("urdf", encoding="utf-8")

    assert assets.resolve_robot_urdf_path(definition) == urdf_path


def test_resolve_robot_relative_asset_path_uses_declared_package_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)
    definition = next(
        definition for definition in widowxai.get_definitions() if definition.type == "Trossen_WidowXAI_Follower"
    )

    asset_path = tmp_path / "widowx" / "meshes" / "base.stl"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text("mesh", encoding="utf-8")

    assert (
        assets.resolve_robot_relative_asset_path(
            definition,
            asset_path=asset_path.relative_to(tmp_path / "widowx"),
        )
        == asset_path
    )


def test_resolve_robot_relative_asset_path_uses_model_asset_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)
    definition = next(definition for definition in so101.get_definitions() if definition.type == "SO101_Follower")

    asset_path = tmp_path / "SO101" / "meshes" / "base.stl"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text("mesh", encoding="utf-8")

    assert (
        assets.resolve_robot_relative_asset_path(
            definition,
            asset_path=asset_path.relative_to(tmp_path / "SO101"),
        )
        == asset_path
    )
