import robots.catalog.assets as assets
from schemas.robot import RobotType


def test_get_builtin_robot_assets_root_uses_shared_constant() -> None:
    assert assets.get_builtin_robot_assets_root() == assets.BUILTIN_ROBOT_ASSETS_ROOT


def test_builtin_robot_assets_are_available_requires_all_urdfs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)

    for relative_path in set(assets._ROBOT_URDF_PATHS.values()):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("urdf", encoding="utf-8")

    assert assets.builtin_robot_assets_are_available()

    (tmp_path / assets._ROBOT_URDF_PATHS[RobotType.SO101_FOLLOWER]).unlink()

    assert not assets.builtin_robot_assets_are_available()


def test_resolve_robot_urdf_path_uses_builtin_assets_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)

    urdf_path = tmp_path / "SO101" / "so101_new_calib.urdf"
    urdf_path.parent.mkdir(parents=True)
    urdf_path.write_text("urdf", encoding="utf-8")

    assert assets.resolve_robot_urdf_path(RobotType.SO101_FOLLOWER) == urdf_path
