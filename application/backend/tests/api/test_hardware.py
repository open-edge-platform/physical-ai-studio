import asyncio
from unittest.mock import patch

import pytest
from physicalai.capture import DeviceInfo

from api.hardware import _fingerprint_from_device_info, get_cameras


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
