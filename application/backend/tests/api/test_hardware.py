import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from physicalai.capture import DeviceInfo

from api.dependencies import get_robot_manager_service
from api.hardware import _fingerprint_from_device_info, get_cameras
from main import app
from schemas import SerialPortInfo


def _make_device(
    device_id="/dev/video0",
    name="Test Camera",
    hardware_fingerprint=None,
):
    return DeviceInfo(
        device_id=device_id,
        index=0,
        name=name,
        driver="uvc",
        hardware_payload=hardware_fingerprint,
    )


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestFingerprintFromDeviceInfo:
    def test_returns_hardware_payload(self):
        info = _make_device(device_id="/dev/video0", hardware_fingerprint={"id": "usb-cam"})
        assert _fingerprint_from_device_info(info) == {"id": "usb-cam"}

    def test_returns_none_without_hardware_payload(self):
        info = _make_device(device_id="/dev/video0", hardware_fingerprint=None)
        assert _fingerprint_from_device_info(info) is None


class TestGetCameras:
    def test_maps_uvc_to_usb_camera(self, event_loop):
        devices = {"uvc": [_make_device(name="Logitech C920", hardware_fingerprint={"serial": "abc"})]}
        with patch("api.hardware.discover_all", return_value=devices):
            cameras = event_loop.run_until_complete(get_cameras())
        assert len(cameras) == 1
        assert cameras[0].driver == "usb_camera"
        assert cameras[0].name == "Logitech C920"

    def test_maps_realsense_driver(self, event_loop):
        rs_device = DeviceInfo(
            device_id="123456789",
            index=0,
            name="Intel RealSense D435",
            driver="realsense",
            hardware_payload={"serial": "123456789"},
        )
        devices = {"realsense": [rs_device]}
        with patch("api.hardware.discover_all", return_value=devices):
            cameras = event_loop.run_until_complete(get_cameras())
        assert len(cameras) == 1
        assert cameras[0].driver == "realsense"
        assert cameras[0].fingerprint == {"serial": "123456789"}

    def test_skips_devices_without_hardware_payload(self, event_loop):
        devices = {"uvc": [_make_device()]}
        with patch("api.hardware.discover_all", return_value=devices):
            cameras = event_loop.run_until_complete(get_cameras())
        assert cameras == []

    def test_skips_unknown_drivers(self, event_loop):
        devices = {"ip": [_make_device(name="IP cam")], "genicam": [_make_device(name="GenICam cam")]}
        with patch("api.hardware.discover_all", return_value=devices):
            cameras = event_loop.run_until_complete(get_cameras())
        assert len(cameras) == 0

    def test_empty_discovery(self, event_loop):
        with patch("api.hardware.discover_all", return_value={}):
            cameras = event_loop.run_until_complete(get_cameras())
        assert cameras == []

    def test_all_false_uses_only_usable(self, event_loop):
        def fake_discover(*, only_usable: bool = True):
            assert only_usable is True
            return {"uvc": [_make_device(name="Cam A", hardware_fingerprint={"serial": "abc"})]}

        with patch("api.hardware.discover_all", side_effect=fake_discover):
            cameras = event_loop.run_until_complete(get_cameras(all=False))
        assert len(cameras) == 1


class _StubRobotManager:
    def __init__(self, robots: list[SerialPortInfo]):
        self.robots = robots
        self.find_robots = AsyncMock()


class TestHardwareApi:
    def test_serial_devices_returns_devices_without_serial_numbers(self):
        robot_manager = _StubRobotManager(
            [
                SerialPortInfo(connection_string="/dev/ttyUSB0", serial_number="ABC123"),
                SerialPortInfo(connection_string="/dev/ttyUSB1", serial_number=None),
            ]
        )
        app.dependency_overrides[get_robot_manager_service] = lambda: robot_manager

        try:
            client = TestClient(app)
            response = client.get("/api/robots/catalog/SO101_Follower/discover")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200, response.text
        assert response.json() == [
            {"connection_string": "/dev/ttyUSB0", "serial_number": "ABC123"},
            {"connection_string": "/dev/ttyUSB1", "serial_number": None},
        ]
        robot_manager.find_robots.assert_awaited_once()

    def test_identify_so101_uses_robot_manager_dependency(self):
        robot_manager = _StubRobotManager([SerialPortInfo(connection_string="/dev/ttyUSB0", serial_number=None)])
        app.dependency_overrides[get_robot_manager_service] = lambda: robot_manager

        try:
            client = TestClient(app)
            with patch("robots.catalog.so101.identify_so101_robot_visually", new_callable=Mock) as identify:
                response = client.post(
                    "/api/robots/catalog/SO101_Follower/identify",
                    json={"connection_string": "/dev/ttyUSB0", "serial_number": ""},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200, response.text
        identify.assert_called_once()
        (connection_string,) = identify.call_args.args
        assert connection_string == "/dev/ttyUSB0"

    def test_identify_trossen_calls_trossen_identifier(self):
        robot_manager = _StubRobotManager([])
        app.dependency_overrides[get_robot_manager_service] = lambda: robot_manager

        try:
            client = TestClient(app)
            with patch("robots.catalog.widowxai.identify_trossen_robot_visually", new_callable=AsyncMock) as identify:
                response = client.post(
                    "/api/robots/catalog/Trossen_WidowXAI_Follower/identify",
                    json={"connection_string": "192.168.1.100"},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200, response.text
        identify.assert_awaited_once()
        (robot_arg,) = identify.await_args.args
        assert robot_arg.type == "Trossen_WidowXAI_Follower"
        assert robot_arg.payload.connection_string == "192.168.1.100"
