# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, PLR2004

"""Tests for RealSenseCamera."""

from __future__ import annotations

import builtins
import importlib
import sys
from unittest import mock

import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.discovery import DeviceInfo
from physicalai.capture.errors import CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame


def _make_falsy_sentinel() -> mock.MagicMock:
    sentinel = mock.MagicMock()
    sentinel.__bool__ = mock.Mock(return_value=False)
    return sentinel


@pytest.fixture
def realsense_cls():  # noqa: ANN201
    """Inject mocked pyrealsense2 and reload RealSenseCamera.

    Yields:
        tuple[type, mock.MagicMock]: RealSenseCamera class and mocked pyrealsense2 module.
    """
    mock_rs = mock.MagicMock()

    mock_pipeline = mock.MagicMock()
    mock_config = mock.MagicMock()
    mock_align = mock.MagicMock()
    mock_context = mock.MagicMock()

    mock_rs.pipeline.return_value = mock_pipeline
    mock_rs.config.return_value = mock_config
    mock_rs.align.return_value = mock_align
    mock_rs.context.return_value = mock_context

    mock_rs.stream.color = "color-stream"
    mock_rs.stream.depth = "depth-stream"
    mock_rs.format.rgb8 = "rgb8"
    mock_rs.format.z16 = "z16"
    mock_rs.camera_info.serial_number = "serial-number"
    mock_rs.camera_info.name = "name"

    color_frame = mock.MagicMock()
    color_frame.get_data.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_frame = mock.MagicMock()
    depth_frame.get_data.return_value = np.zeros((480, 640), dtype=np.uint16)

    frameset = mock.MagicMock()
    frameset.get_color_frame.return_value = color_frame
    frameset.get_depth_frame.return_value = depth_frame

    mock_pipeline.wait_for_frames.return_value = frameset
    mock_pipeline.poll_for_frames.side_effect = [frameset, _make_falsy_sentinel()]
    mock_align.process.side_effect = lambda f: f

    mock_device = mock.MagicMock()

    def _get_info(key: object) -> str:
        if key == mock_rs.camera_info.serial_number:
            return "test-serial"
        if key == mock_rs.camera_info.name:
            return "Test RealSense"
        return ""

    mock_device.get_info.side_effect = _get_info
    mock_context.query_devices.return_value = [mock_device]

    sys.modules["pyrealsense2"] = mock_rs
    sys.modules.pop("physicalai.capture.cameras.realsense._camera", None)
    sys.modules.pop("physicalai.capture.cameras.realsense._discover", None)

    module = importlib.import_module("physicalai.capture.cameras.realsense._camera")
    camera_cls = module.RealSenseCamera

    yield camera_cls, mock_rs

    sys.modules.pop("pyrealsense2", None)
    sys.modules.pop("physicalai.capture.cameras.realsense._camera", None)
    sys.modules.pop("physicalai.capture.cameras.realsense._discover", None)


def test_connect_starts_pipeline(realsense_cls: tuple) -> None:
    """connect() starts pipeline with config object."""
    camera_cls, mock_rs = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_rs.pipeline.return_value.start.assert_called_once_with(mock_rs.config.return_value)


def test_connect_enables_device_with_serial(realsense_cls: tuple) -> None:
    """connect() enables the requested serial-number device."""
    camera_cls, mock_rs = realsense_cls
    camera = camera_cls(serial_number="test-serial")
    camera.connect()
    mock_rs.config.return_value.enable_device.assert_called_once_with("test-serial")


def test_connect_enables_color_and_depth_streams(realsense_cls: tuple) -> None:
    """connect() enables both color and depth streams."""
    camera_cls, mock_rs = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    assert mock_rs.config.return_value.enable_stream.call_count == 2
    mock_rs.config.return_value.enable_stream.assert_any_call(
        mock_rs.stream.color,
        640,
        480,
        mock_rs.format.rgb8,
        30,
    )
    mock_rs.config.return_value.enable_stream.assert_any_call(
        mock_rs.stream.depth,
        640,
        480,
        mock_rs.format.z16,
        30,
    )


def test_connect_timeout_raises(realsense_cls: tuple) -> None:
    """connect() converts wait timeout RuntimeError to CaptureTimeoutError."""
    camera_cls, mock_rs = realsense_cls
    mock_rs.pipeline.return_value.wait_for_frames.side_effect = RuntimeError("timeout")
    camera = camera_cls(serial_number="123")
    with pytest.raises(CaptureTimeoutError):
        camera.connect()


def test_disconnect_stops_pipeline(realsense_cls: tuple) -> None:
    """disconnect() stops pipeline and clears connection state."""
    camera_cls, mock_rs = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    camera.disconnect()
    mock_rs.pipeline.return_value.stop.assert_called_once()
    assert not camera.is_connected


def test_read_returns_frame_with_correct_shape_and_dtype(realsense_cls: tuple) -> None:
    """read() returns uint8 RGB frame with expected shape."""
    camera_cls, _ = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    frame = camera.read()
    assert isinstance(frame, Frame)
    assert frame.data.shape == (480, 640, 3)
    assert frame.data.dtype == np.uint8


def test_read_increments_sequence(realsense_cls: tuple) -> None:
    """read() increments frame sequence numbers from 1."""
    camera_cls, _ = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    f1 = camera.read()
    f2 = camera.read()
    f3 = camera.read()
    assert f1.sequence == 1
    assert f2.sequence == 2
    assert f3.sequence == 3


def test_read_timeout_raises_capture_timeout(realsense_cls: tuple) -> None:
    """read() raises CaptureTimeoutError on wait failure."""
    camera_cls, mock_rs = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_rs.pipeline.return_value.wait_for_frames.side_effect = RuntimeError("timeout")
    with pytest.raises(CaptureTimeoutError):
        camera.read()


def test_read_latest_returns_new_frame_when_available(realsense_cls: tuple) -> None:
    """read_latest() returns a new frame when poll has data."""
    camera_cls, mock_rs = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    latest_frameset = mock.MagicMock()
    latest_frameset.get_color_frame.return_value = (
        mock_rs.pipeline.return_value.wait_for_frames.return_value.get_color_frame.return_value
    )
    latest_frameset.get_depth_frame.return_value = (
        mock_rs.pipeline.return_value.wait_for_frames.return_value.get_depth_frame.return_value
    )
    mock_rs.pipeline.return_value.poll_for_frames.side_effect = [latest_frameset, _make_falsy_sentinel()]
    frame = camera.read_latest()
    assert isinstance(frame, Frame)
    assert frame.sequence == 1


def test_read_latest_returns_cached_on_no_new_frame(realsense_cls: tuple) -> None:
    """read_latest() returns cached frame when poll is empty."""
    camera_cls, mock_rs = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    first = camera.read()
    mock_rs.pipeline.return_value.poll_for_frames.side_effect = [_make_falsy_sentinel()]
    latest = camera.read_latest()
    assert latest.sequence == first.sequence


def test_read_depth_returns_uint16(realsense_cls: tuple) -> None:
    """read_depth() returns uint16 depth frame with expected shape."""
    camera_cls, _ = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    frame = camera.read_depth()
    assert isinstance(frame, Frame)
    assert frame.data.shape == (480, 640)
    assert frame.data.dtype == np.uint16


def test_read_rgbd_single_pipeline_read(realsense_cls: tuple) -> None:
    """read_rgbd() performs exactly one pipeline wait call."""
    camera_cls, mock_rs = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    mock_rs.pipeline.return_value.wait_for_frames.reset_mock()
    camera.read_rgbd()
    mock_rs.pipeline.return_value.wait_for_frames.assert_called_once_with(5000)


def test_read_rgbd_returns_aligned_pair(realsense_cls: tuple) -> None:
    """read_rgbd() returns aligned color/depth Frame tuple."""
    camera_cls, _ = realsense_cls
    camera = camera_cls(serial_number="123")
    camera.connect()
    color, depth = camera.read_rgbd()
    assert isinstance(color, Frame)
    assert isinstance(depth, Frame)
    assert color.data.shape == (480, 640, 3)
    assert color.data.dtype == np.uint8
    assert depth.data.shape == (480, 640)
    assert depth.data.dtype == np.uint16


def test_discover_returns_device_info_list(realsense_cls: tuple) -> None:
    """discover() returns DeviceInfo metadata for mocked device."""
    camera_cls, _ = realsense_cls
    devices = camera_cls.discover()
    assert len(devices) == 1
    assert isinstance(devices[0], DeviceInfo)
    assert devices[0].device_id == "test-serial"
    assert devices[0].driver == "realsense"
    assert devices[0].manufacturer == "RealSense"


def test_discover_returns_empty_when_no_sdk() -> None:
    """discover_realsense() returns empty list when SDK import fails."""
    sys.modules.pop("pyrealsense2", None)
    sys.modules.pop("physicalai.capture.cameras.realsense._discover", None)
    real_import = builtins.__import__

    def _import(  # noqa: ANN202
        name: str,
        globalns: dict[str, object] | None = None,
        localns: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        if name == "pyrealsense2":
            msg = "missing"
            raise ImportError(msg)
        return real_import(name, globalns, localns, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=_import):
        module = importlib.import_module("physicalai.capture.cameras.realsense._discover")
        assert module.discover_realsense() == []


def test_color_mode_bgr_conversion(realsense_cls: tuple) -> None:
    """BGR mode swaps channels so red appears in channel 2."""
    camera_cls, mock_rs = realsense_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    raw[:, :, 0] = 100
    mock_rs.pipeline.return_value.wait_for_frames.return_value.get_color_frame.return_value.get_data.return_value = raw
    camera = camera_cls(serial_number="123", color_mode=ColorMode.BGR)
    camera.connect()
    frame = camera.read()
    assert frame.data[:, :, 2][0, 0] == 100


def test_color_mode_gray_conversion(realsense_cls: tuple) -> None:
    """GRAY mode returns a uint8 2D frame."""
    camera_cls, _ = realsense_cls
    camera = camera_cls(serial_number="123", color_mode=ColorMode.GRAY)
    camera.connect()
    frame = camera.read()
    assert frame.data.shape == (480, 640)
    assert frame.data.dtype == np.uint8


def test_read_not_connected_raises(realsense_cls: tuple) -> None:
    """read() before connect() raises NotConnectedError."""
    camera_cls, _ = realsense_cls
    camera = camera_cls(serial_number="123")
    with pytest.raises(NotConnectedError):
        camera.read()


def test_context_manager_lifecycle(realsense_cls: tuple) -> None:
    """Context manager connects on enter and stops on exit."""
    camera_cls, mock_rs = realsense_cls
    with camera_cls(serial_number="123") as camera:
        assert camera.is_connected
    mock_rs.pipeline.return_value.stop.assert_called_once()


def test_device_id_format(realsense_cls: tuple) -> None:
    """device_id formats as realsense:<serial>."""
    camera_cls, _ = realsense_cls
    camera = camera_cls(serial_number="ABC123")
    assert camera.device_id == "realsense:ABC123"
