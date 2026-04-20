# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, PLR2004

"""Tests for BaslerCamera."""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.discovery import DeviceInfo
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame


@pytest.fixture
def basler_cls():  # noqa: ANN201
    """Inject mocked pypylon and reload BaslerCamera.

    Yields:
        tuple[type, MagicMock, MagicMock]: BaslerCamera class, mocked pylon, mocked genicam.
    """
    mock_pylon = mock.MagicMock()
    mock_genicam = mock.MagicMock()

    # TimeoutException must be a real exception class so except clauses work
    class _TimeoutException(Exception):
        pass

    mock_genicam.TimeoutException = _TimeoutException

    mock_pypylon = mock.MagicMock()
    mock_pypylon.pylon = mock_pylon
    mock_pypylon.genicam = mock_genicam

    # Default grab result — succeeds, returns 480×640×3 uint8
    raw_array = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_converted = mock.MagicMock()
    mock_converted.GetArray.return_value = raw_array

    mock_grab_result = mock.MagicMock()
    mock_grab_result.GrabSucceeded.return_value = True
    mock_grab_result.GetErrorDescription.return_value = ""

    mock_converter = mock.MagicMock()
    mock_converter.Convert.return_value = mock_converted
    mock_pylon.ImageFormatConverter.return_value = mock_converter

    # InstantCamera
    mock_camera = mock.MagicMock()
    mock_camera.RetrieveResult.return_value = mock_grab_result
    mock_pylon.InstantCamera.return_value = mock_camera

    # Discovery device
    mock_dev = mock.MagicMock()
    mock_dev.GetSerialNumber.return_value = "12345"
    mock_dev.GetModelName.return_value = "acA640-90uc"
    mock_dev.GetVendorName.return_value = "Basler"
    mock_pylon.TlFactory.GetInstance.return_value.EnumerateDevices.return_value = [mock_dev]

    sys.modules["pypylon"] = mock_pypylon
    sys.modules["pypylon.pylon"] = mock_pylon
    sys.modules["pypylon.genicam"] = mock_genicam
    sys.modules.pop("physicalai.capture.cameras.basler._camera", None)
    sys.modules.pop("physicalai.capture.cameras.basler._discover", None)

    module = importlib.import_module("physicalai.capture.cameras.basler._camera")
    camera_cls = module.BaslerCamera

    yield camera_cls, mock_pylon, mock_genicam

    sys.modules.pop("pypylon", None)
    sys.modules.pop("pypylon.pylon", None)
    sys.modules.pop("pypylon.genicam", None)
    sys.modules.pop("physicalai.capture.cameras.basler._camera", None)
    sys.modules.pop("physicalai.capture.cameras.basler._discover", None)


def test_connect_opens_camera_and_starts_grabbing(basler_cls: tuple) -> None:
    """connect() opens the camera and starts continuous grabbing."""
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_pylon.InstantCamera.return_value.Open.assert_called_once()
    mock_pylon.InstantCamera.return_value.StartGrabbing.assert_called_once_with(
        mock_pylon.GrabStrategy_LatestImageOnly,
    )


def test_connect_uses_serial_number(basler_cls: tuple) -> None:
    """connect() configures DeviceInfo with the requested serial number."""
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="test-serial")
    camera.connect()
    mock_pylon.DeviceInfo.return_value.SetSerialNumber.assert_called_once_with("test-serial")


def test_connect_verifies_first_frame(basler_cls: tuple) -> None:
    """connect() retrieves a first frame to verify camera operation."""
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_pylon.InstantCamera.return_value.RetrieveResult.assert_called()


def test_connect_timeout_raises_capture_timeout(basler_cls: tuple) -> None:
    """connect() raises CaptureTimeoutError when first frame times out."""
    camera_cls, mock_pylon, mock_genicam = basler_cls
    mock_pylon.InstantCamera.return_value.RetrieveResult.side_effect = mock_genicam.TimeoutException(
        "timeout",
    )
    camera = camera_cls(serial_number="123")
    with pytest.raises(CaptureTimeoutError):
        camera.connect()


def test_disconnect_stops_grabbing_and_closes(basler_cls: tuple) -> None:
    """disconnect() stops grabbing, closes the camera, and clears state."""
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    camera.disconnect()
    mock_pylon.InstantCamera.return_value.StopGrabbing.assert_called()
    mock_pylon.InstantCamera.return_value.Close.assert_called()
    assert not camera.is_connected


def test_disconnect_without_connect_is_safe(basler_cls: tuple) -> None:
    """disconnect() on an unconnected camera does not raise."""
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.disconnect()


def test_read_returns_frame_with_correct_shape_and_dtype(basler_cls: tuple) -> None:
    """read() returns uint8 RGB frame with expected shape."""
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    frame = camera.read()
    assert isinstance(frame, Frame)
    assert frame.data.shape == (480, 640, 3)
    assert frame.data.dtype == np.uint8


def test_read_increments_sequence(basler_cls: tuple) -> None:
    """read() increments frame sequence numbers from 1."""
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    f1 = camera.read()
    f2 = camera.read()
    f3 = camera.read()
    assert f1.sequence == 1
    assert f2.sequence == 2
    assert f3.sequence == 3


def test_read_timeout_raises_capture_timeout(basler_cls: tuple) -> None:
    """read() raises CaptureTimeoutError on grab timeout."""
    camera_cls, mock_pylon, mock_genicam = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_pylon.InstantCamera.return_value.RetrieveResult.side_effect = mock_genicam.TimeoutException(
        "timeout",
    )
    with pytest.raises(CaptureTimeoutError):
        camera.read()


def test_read_grab_failed_raises_capture_error(basler_cls: tuple) -> None:
    """read() raises CaptureError when GrabSucceeded() is False."""
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_pylon.InstantCamera.return_value.RetrieveResult.return_value.GrabSucceeded.return_value = False
    with pytest.raises(CaptureError):
        camera.read()


def test_read_latest_returns_frame(basler_cls: tuple) -> None:
    """read_latest() returns a frame when grab data is available."""
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    frame = camera.read_latest()
    assert isinstance(frame, Frame)


def test_read_latest_returns_cached_when_no_new_frame(basler_cls: tuple) -> None:
    """read_latest() returns the cached frame when no new grab is available."""
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    first = camera.read()
    mock_pylon.InstantCamera.return_value.RetrieveResult.return_value = None
    latest = camera.read_latest()
    assert latest.sequence == first.sequence


def test_discover_returns_device_info_list(basler_cls: tuple) -> None:
    """discover_basler() returns DeviceInfo list for mocked devices."""
    _, _, _ = basler_cls
    sys.modules.pop("physicalai.capture.cameras.basler._discover", None)
    discover_module = importlib.import_module("physicalai.capture.cameras.basler._discover")
    devices = discover_module.discover_basler()
    assert len(devices) == 1
    assert isinstance(devices[0], DeviceInfo)
    assert devices[0].device_id == "basler:12345"
    assert devices[0].driver == "basler"
    assert devices[0].manufacturer == "Basler"


def test_discover_returns_empty_when_no_sdk() -> None:
    """discover_basler() returns empty list when pypylon is not installed."""
    sys.modules.pop("pypylon", None)
    sys.modules.pop("pypylon.pylon", None)
    sys.modules.pop("pypylon.genicam", None)
    sys.modules.pop("physicalai.capture.cameras.basler._discover", None)
    module = importlib.import_module("physicalai.capture.cameras.basler._discover")
    result = module.discover_basler()
    assert result == []


def test_color_mode_bgr(basler_cls: tuple) -> None:
    """BGR mode configures converter with BGR8packed pixel type."""
    camera_cls, mock_pylon, _ = basler_cls
    camera = camera_cls(serial_number="123", color_mode=ColorMode.BGR)
    camera.connect()
    converter = mock_pylon.ImageFormatConverter.return_value
    assert converter.OutputPixelFormat == mock_pylon.PixelType_BGR8packed


def test_color_mode_gray(basler_cls: tuple) -> None:
    """GRAY mode produces a 2D uint8 frame."""
    camera_cls, mock_pylon, _ = basler_cls
    gray_array = np.zeros((480, 640), dtype=np.uint8)
    mock_pylon.ImageFormatConverter.return_value.Convert.return_value.GetArray.return_value = gray_array
    camera = camera_cls(serial_number="123", color_mode=ColorMode.GRAY)
    camera.connect()
    frame = camera.read()
    assert frame.data.shape == (480, 640)
    assert frame.data.dtype == np.uint8


def test_read_not_connected_raises(basler_cls: tuple) -> None:
    """read() before connect() raises NotConnectedError."""
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    with pytest.raises(NotConnectedError):
        camera.read()


def test_read_latest_not_connected_raises(basler_cls: tuple) -> None:
    """read_latest() before connect() raises NotConnectedError."""
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="123")
    with pytest.raises(NotConnectedError):
        camera.read_latest()


def test_context_manager_lifecycle(basler_cls: tuple) -> None:
    """Context manager connects on enter and disconnects on exit."""
    camera_cls, mock_pylon, _ = basler_cls
    with camera_cls(serial_number="123") as camera:
        assert camera.is_connected
    mock_pylon.InstantCamera.return_value.StopGrabbing.assert_called()
    assert not camera.is_connected


def test_device_id_format(basler_cls: tuple) -> None:
    """device_id formats as basler:<serial>."""
    camera_cls, _, _ = basler_cls
    camera = camera_cls(serial_number="ABC123")
    assert camera.device_id == "basler:ABC123"
