from fastapi.testclient import TestClient

import robots.catalog.assets as assets
from main import app


def test_get_robot_catalog_urdf_returns_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)

    urdf_path = tmp_path / "SO101" / "so101_new_calib.urdf"
    urdf_path.parent.mkdir(parents=True)
    urdf_path.write_text("<robot />", encoding="utf-8")

    client = TestClient(app)

    response = client.get("/api/robots/catalog/SO101_Follower/urdf")

    assert response.status_code == 200
    assert response.text == "<robot />"


def test_get_robot_catalog_asset_returns_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)

    asset_path = tmp_path / "SO101" / "meshes" / "base.stl"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text("mesh", encoding="utf-8")

    client = TestClient(app)

    response = client.get("/api/robots/catalog/SO101_Follower/meshes/base.stl")

    assert response.status_code == 200
    assert response.text == "mesh"


def test_get_robot_catalog_asset_rejects_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)

    client = TestClient(app)

    response = client.get("/api/robots/catalog/SO101_Follower/%2E%2E/secret.txt")

    assert response.status_code == 403


def test_get_robot_catalog_asset_returns_404_for_missing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)

    client = TestClient(app)

    response = client.get("/api/robots/catalog/SO101_Follower/missing.stl")

    assert response.status_code == 404
