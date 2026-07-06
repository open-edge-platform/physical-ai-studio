import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

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
    hardware_id=None,
    id_stable=False,
):
    return DeviceInfo(
        device_id=device_id,
        index=0,
        name=name,
        driver="uvc",
        hardware_id=hardware_id,
        id_stable=id_stable,
    )


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestFingerprintFromDeviceInfo:
    def test_stable_prefers_hardware_id(self):
        info = _make_device(device_id="/dev/video0", hardware_id="/dev/v4l/by-id/usb-cam", id_stable=True)
        assert _fingerprint_from_device_info(info) == "/dev/v4l/by-id/usb-cam"

    def test_unstable_falls_back_to_device_id(self):
        info = _make_device(device_id="/dev/video0", hardware_id=None, id_stable=False)
        assert _fingerprint_from_device_info(info) == "/dev/video0"

    def test_stable_but_no_hardware_id_falls_back(self):
        info = _make_device(device_id="/dev/video0", hardware_id=None, id_stable=True)
        assert _fingerprint_from_device_info(info) == "/dev/video0"

    def test_unstable_ignores_hardware_id(self):
        info = _make_device(device_id="/dev/video0", hardware_id="/dev/v4l/by-id/usb-cam", id_stable=False)
        assert _fingerprint_from_device_info(info) == "/dev/video0"


class TestGetCameras:
    def test_maps_uvc_to_usb_camera(self, event_loop):
        devices = {"uvc": [_make_device(name="Logitech C920")]}
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
            hardware_id="123456789",
            id_stable=True,
        )
        devices = {"realsense": [rs_device]}
        with patch("api.hardware.discover_all", return_value=devices):
            cameras = event_loop.run_until_complete(get_cameras())
        assert len(cameras) == 1
        assert cameras[0].driver == "realsense"
        assert cameras[0].fingerprint == "123456789"

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
            return {"uvc": [_make_device(name="Cam A")]}

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
            response = client.get("/api/hardware/serial_devices")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200, response.text
        assert response.json() == [
            {"connection_string": "/dev/ttyUSB0", "serial_number": "ABC123"},
            {"connection_string": "/dev/ttyUSB1", "serial_number": None},
        ]
        robot_manager.find_robots.assert_awaited_once()

    def test_identify_so101_uses_robot_manager_dependency(self):
        robot_manager = _StubRobotManager([])
        app.dependency_overrides[get_robot_manager_service] = lambda: robot_manager

        try:
            client = TestClient(app)
            with patch("api.hardware.identify_so101_robot_visually", new_callable=AsyncMock) as identify:
                response = client.post(
                    "/api/hardware/identify",
                    params={"joint": "gripper"},
                    json={
                        "id": str(uuid4()),
                        "name": "Test SO101",
                        "type": "SO101_Follower",
                        "payload": {
                            "connection_string": "/dev/ttyUSB0",
                            "serial_number": "",
                        },
                        "active_calibration_id": None,
                    },
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200, response.text
        identify.assert_awaited_once()
        manager_arg, robot_arg, joint_arg = identify.await_args.args
        assert manager_arg is robot_manager
        assert robot_arg.type == "SO101_Follower"
        assert robot_arg.payload.connection_string == "/dev/ttyUSB0"
        assert joint_arg == "gripper"

    def test_identify_trossen_calls_trossen_identifier(self):
        robot_manager = _StubRobotManager([])
        app.dependency_overrides[get_robot_manager_service] = lambda: robot_manager

        try:
            client = TestClient(app)
            with patch("api.hardware.identify_trossen_robot_visually", new_callable=AsyncMock) as identify:
                response = client.post(
                    "/api/hardware/identify",
                    json={
                        "id": str(uuid4()),
                        "name": "Test Trossen",
                        "type": "Trossen_WidowXAI_Follower",
                        "payload": {
                            "connection_string": "192.168.1.100",
                            "serial_number": "",
                        },
                        "active_calibration_id": None,
                    },
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200, response.text
        identify.assert_awaited_once()
        (robot_arg,) = identify.await_args.args
        assert robot_arg.type == "Trossen_WidowXAI_Follower"
        assert robot_arg.payload.connection_string == "192.168.1.100"
