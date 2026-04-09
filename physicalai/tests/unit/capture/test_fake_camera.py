# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for FakeCamera — validates the Camera ABC contract."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.errors import NotConnectedError
from physicalai.capture.frame import Frame
from tests.unit.capture.fake import FakeCamera


class TestFakeCameraLifecycle:
    """Connect / disconnect state transitions."""

    def test_starts_disconnected(self) -> None:
        cam = FakeCamera()
        assert not cam.is_connected

    def test_connect_sets_connected(self) -> None:
        cam = FakeCamera()
        cam.connect()
        assert cam.is_connected
        cam.disconnect()

    def test_disconnect_clears_connected(self) -> None:
        cam = FakeCamera()
        cam.connect()
        cam.disconnect()
        assert not cam.is_connected

    def test_context_manager(self) -> None:
        with FakeCamera() as cam:
            assert cam.is_connected
        assert not cam.is_connected

    def test_device_id(self) -> None:
        cam = FakeCamera(device_name="test-cam")
        assert cam.device_id == "test-cam"


class TestFakeCameraRead:
    """read() and read_latest() behaviour."""

    def test_read_returns_frame(self) -> None:
        with FakeCamera(width=320, height=240) as cam:
            frame = cam.read()
            assert isinstance(frame, Frame)
            assert frame.data.shape == (240, 320, 3)
            assert frame.data.dtype == np.uint8

    def test_read_increments_sequence(self) -> None:
        with FakeCamera() as cam:
            f0 = cam.read()
            f1 = cam.read()
            assert f0.sequence == 0
            assert f1.sequence == 1

    def test_read_timestamp_is_monotonic(self) -> None:
        with FakeCamera() as cam:
            f0 = cam.read()
            f1 = cam.read()
            assert f1.timestamp >= f0.timestamp

    def test_read_latest_returns_frame(self) -> None:
        with FakeCamera() as cam:
            frame = cam.read_latest()
            assert isinstance(frame, Frame)

    def test_read_before_connect_raises(self) -> None:
        cam = FakeCamera()
        with pytest.raises(NotConnectedError):
            cam.read()

    def test_read_latest_before_connect_raises(self) -> None:
        cam = FakeCamera()
        with pytest.raises(NotConnectedError):
            cam.read_latest()

    def test_grayscale_mode(self) -> None:
        with FakeCamera(color_mode=ColorMode.GRAY) as cam:
            frame = cam.read()
            assert frame.data.shape == (480, 640)

    def test_sequence_resets_on_reconnect(self) -> None:
        cam = FakeCamera()
        cam.connect()
        cam.read()
        cam.read()
        cam.disconnect()
        cam.connect()
        frame = cam.read()
        assert frame.sequence == 0
        cam.disconnect()


class TestFakeCameraAsync:
    """async_read() via the ABC's default executor path."""

    def test_async_read_returns_frame(self) -> None:
        async def _run() -> Frame:
            async with FakeCamera() as cam:
                return await cam.async_read()

        frame = asyncio.run(_run())
        assert isinstance(frame, Frame)

    def test_async_context_manager(self) -> None:
        async def _run() -> bool:
            async with FakeCamera() as cam:
                return cam.is_connected

        assert asyncio.run(_run())
