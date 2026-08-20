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


def test_list_robot_catalog_returns_definitions_without_internal_fields() -> None:
    client = TestClient(app)

    response = client.get("/api/robots/catalog")

    assert response.status_code == 200
    payload = response.json()

    assert payload
    first = payload[0]
    assert set(first.keys()) == {"type", "display_name", "role", "urdf_path", "package_map", "joint_map"}
    assert "urdf_relative_path" not in first

    so101 = next(definition for definition in payload if definition["type"] == "SO101_Follower")
    assert so101["urdf_path"] == "/api/robots/catalog/SO101_Follower/urdf"
    assert so101["package_map"] == {"SO101": "/api/robots/catalog/SO101_Follower"}


def test_get_robot_catalog_schema_returns_pydantic_schema() -> None:
    client = TestClient(app)

    response = client.get("/api/robots/catalog/SO101_Follower/schema")

    assert response.status_code == 200
    schema = response.json()
    assert "serial_number" in schema["properties"]
    assert "connection_string" in schema["properties"]
    assert schema["example"] == {
        "connection_string": "",
        "serial_number": "SO101-2024-001",
        "calibration": None,
    }


def test_get_external_robot_catalog_schema_returns_pydantic_schema() -> None:
    client = TestClient(app)

    response = client.get("/api/robots/catalog/LeKiwi_Follower/schema")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "Robot_not_found"


def test_check_robot_online_invalid_payload_returns_validation_error() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/robots/catalog/LeKiwi_Follower/is-online",
        json={"connection_string": "/dev/ttyUSB0"},
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "Robot_not_found"


def test_identify_robot_invalid_payload_returns_validation_error() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/robots/catalog/LeKiwi_Follower/identify",
        json={"connection_string": "/dev/ttyUSB0"},
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "Robot_not_found"


def test_discover_unknown_robot_type_returns_not_found() -> None:
    client = TestClient(app)

    response = client.get("/api/robots/catalog/UnknownRobot/discover")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "Robot_not_found"


def test_is_online_unknown_robot_type_returns_not_found() -> None:
    client = TestClient(app)

    response = client.post("/api/robots/catalog/UnknownRobot/is-online", json={})

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "Robot_not_found"


def test_identify_unknown_robot_type_returns_not_found() -> None:
    client = TestClient(app)

    response = client.post("/api/robots/catalog/UnknownRobot/identify", json={})

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "Robot_not_found"


def test_get_robot_catalog_urdf_unknown_robot_type_returns_not_found() -> None:
    client = TestClient(app)

    response = client.get("/api/robots/catalog/UnknownRobot/urdf")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "Robot_not_found"


def test_get_robot_catalog_schema_unknown_robot_type_returns_not_found() -> None:
    client = TestClient(app)

    response = client.get("/api/robots/catalog/UnknownRobot/schema")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "Robot_not_found"


def test_get_robot_catalog_asset_unknown_robot_type_returns_not_found() -> None:
    client = TestClient(app)

    response = client.get("/api/robots/catalog/UnknownRobot/packages/SO101/meshes/base.stl")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "Robot_not_found"


def test_get_robot_catalog_asset_returns_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)

    asset_path = tmp_path / "SO101" / "meshes" / "base.stl"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text("mesh", encoding="utf-8")

    client = TestClient(app)

    response = client.get("/api/robots/catalog/SO101_Follower/meshes/base.stl")

    assert response.status_code == 200
    assert response.text == "mesh"


def test_get_robot_catalog_relative_asset_returns_file(tmp_path, monkeypatch) -> None:
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
