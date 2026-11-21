"""Registry construction preserves the existing camera recipes and ownership."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from physicalai.config import Config, export_config, instantiate

from cameras.catalog import CameraCatalogDefinition, CameraClaimPolicy, CatalogCamera
from cameras.registry import CameraCatalogRegistry
from exceptions import CameraExclusiveUseError
from schemas.project_camera import CameraAdapter, USBCameraPayload
from services.camera_claims import CameraClaim, CameraClaimRegistry


def camera(driver="usb_camera", fingerprint=None, payload=None):
    return CameraAdapter.validate_python(
        {
            "driver": driver,
            "name": "Camera",
            "hardware_name": "Camera",
            "fingerprint": fingerprint or {"serial": "test-camera"},
            "payload": payload or {"width": 640, "height": 480, "fps": 30},
        }
    )


@pytest.mark.parametrize(
    "driver,class_path,identity",
    [
        ("usb_camera", "physicalai.capture.UVCCamera", {"device": {"serial": "test-camera"}}),
        ("realsense", "physicalai.capture.RealSenseCamera", {"serial_number": "test-camera"}),
        ("basler", "physicalai.capture.BaslerCamera", {"serial_number": "test-camera"}),
        ("ipcam", "physicalai.capture.IPCamera", {"url": "http://127.0.0.1:5000/mjpeg"}),
    ],
)
def test_config_matches_existing_constructor_recipe(driver, class_path, identity):
    payload = {"width": 640, "height": 480, "fps": 30}
    if driver == "ipcam":
        payload.update(identity)
    expected = Config(class_path, {"width": 640, "height": 480, "fps": 30, **identity})
    registry = CameraCatalogRegistry()
    actual = registry.build_camera_config(camera(driver, payload=payload))
    assert actual.to_dict() == Config.from_instance(instantiate(expected)).to_dict()
    assert not instantiate(actual).is_connected


def test_registration_rejects_duplicates_and_does_not_open_hardware():
    registry = CameraCatalogRegistry()
    assert [definition.type for definition in registry.list_definitions()] == [
        "usb_camera",
        "realsense",
        "basler",
        "ipcam",
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register_camera(registry.get_definition("usb_camera"))
    assert not registry.supports("genicam")
    assert not registry.supports("simulation")


@export_config
class FakeCamera:
    def __init__(self, width: int):
        self.width = width


@pytest.mark.parametrize("in_event_loop", [False, True])
async def test_builder_receives_typed_payload_and_sync_bridge_works(in_event_loop):
    import asyncio

    registry = CameraCatalogRegistry()
    received = []

    async def build(descriptor, factory):
        received.append(descriptor)
        assert factory is registry
        return FakeCamera(descriptor.payload.width)

    registry.register_camera(
        CameraCatalogDefinition(
            type="custom",
            display_name="Custom",
            camera_payload=USBCameraPayload,
            camera_builder=build,
        )
    )
    row = SimpleNamespace(
        driver="custom", fingerprint={"id": "custom"}, payload=USBCameraPayload(width=640, height=480, fps=30)
    )
    if in_event_loop:
        result = registry.build_camera_config(row)
    else:
        result = await asyncio.to_thread(registry.build_camera_config, row)
    assert instantiate(result).width == 640
    assert received == [CatalogCamera(type="custom", fingerprint=row.fingerprint, payload=row.payload)]


def test_builder_errors_propagate():
    registry = CameraCatalogRegistry()
    with pytest.raises(ValueError, match="reselected"):
        registry.build_camera_config(camera().model_copy(update={"fingerprint": None}))
    with pytest.raises(ValueError, match="serial"):
        registry.build_camera_config(camera("basler", fingerprint={"id": "not-a-serial"}))
    with pytest.raises(ValueError, match="Unsupported"):
        registry.build_camera_config(camera().model_copy(update={"driver": "unknown"}))


def test_exclusive_policy_applies_to_current_session_claims(monkeypatch):
    from runtime_session.session import RuntimeSession

    registry = CameraCatalogRegistry()
    definition = registry.get_definition("usb_camera")
    registry._definitions["usb_camera"] = replace(definition, claim_policy=CameraClaimPolicy.EXCLUSIVE)
    monkeypatch.setattr("runtime_session.session.get_camera_catalog_registry", lambda: registry)
    claims = CameraClaimRegistry()
    row = camera()
    claims.claim([CameraClaim(row.fingerprint, (640, 480, 30), "other", uuid4(), "Other")])
    sessions = Mock()
    session = RuntimeSession(
        project_id=uuid4(),
        project_name="Test",
        registry=sessions,
        camera_claims=claims,
        robot_factory=Mock(),
        resolve=Mock(),
    )
    with pytest.raises(CameraExclusiveUseError), session._claim(SimpleNamespace(cameras={"cam": row}, robot_names={})):
        pytest.fail("Exclusive camera was shared")
    sessions.release_robots.assert_called_once_with(session)
