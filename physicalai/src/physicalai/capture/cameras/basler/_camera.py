# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame

if TYPE_CHECKING:
    import numpy as np

    from physicalai.capture.discovery import DeviceInfo


class BaslerCamera(Camera):
    """Basler camera using pypylon SDK."""

    def __init__(
        self,
        *,
        serial_number: str,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        super().__init__(color_mode=color_mode)
        self._serial_number = serial_number
        self._fps = fps
        self._width = width
        self._height = height
        self._connected = False
        self._sequence = 0
        self._last_timestamp: float = 0.0
        self._camera: Any | None = None
        self._converter: Any | None = None
        self._last_frame_data: np.ndarray | None = None

    def connect(self, timeout: float = 5.0) -> None:
        from pypylon import genicam, pylon  # type: ignore[import-not-found]  # noqa: PLC0415

        info = pylon.DeviceInfo()
        info.SetSerialNumber(self._serial_number)
        self._camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))
        self._camera.Open()
        self._camera.Width.Value = self._width
        self._camera.Height.Value = self._height
        try:
            self._camera.AcquisitionFrameRateEnable.Value = True
            self._camera.AcquisitionFrameRate.Value = float(self._fps)
        except Exception as err:  # noqa: BLE001
            _ = err

        self._converter = pylon.ImageFormatConverter()
        if self._color_mode == ColorMode.RGB:
            self._converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        elif self._color_mode == ColorMode.BGR:
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        else:
            self._converter.OutputPixelFormat = pylon.PixelType_Mono8

        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        try:
            grab_result = self._camera.RetrieveResult(int(timeout * 1000), pylon.TimeoutHandling_ThrowException)
        except genicam.TimeoutException as err:
            self._do_disconnect()
            msg = f"Timed out waiting for first frame after {timeout}s"
            raise CaptureTimeoutError(msg) from err
        except Exception as err:
            self._do_disconnect()
            msg = "Failed to start Basler camera"
            raise CaptureError(msg) from err

        if not grab_result.GrabSucceeded():
            desc = grab_result.GetErrorDescription()
            grab_result.Release()
            self._do_disconnect()
            msg = f"First grab failed: {desc}"
            raise CaptureError(msg)

        converted = self._converter.Convert(grab_result)
        self._last_frame_data = converted.GetArray().copy()
        grab_result.Release()
        self._connected = True
        self._sequence = 0

    def _do_disconnect(self) -> None:
        if self._camera is not None:
            with contextlib.suppress(Exception):
                self._camera.StopGrabbing()
            with contextlib.suppress(Exception):
                self._camera.Close()
        self._camera = None
        self._converter = None
        self._last_frame_data = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_id(self) -> str:
        return f"basler:{self._serial_number}"

    def read(self, timeout: float | None = None) -> Frame:
        if not self._connected or self._camera is None or self._converter is None:
            raise NotConnectedError

        timeout_ms = int(timeout * 1000) if timeout is not None else 15000
        from pypylon import genicam, pylon  # type: ignore[import-not-found]  # noqa: PLC0415

        try:
            grab_result = self._camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        except genicam.TimeoutException as err:
            msg = f"Timed out waiting for frame after {timeout_ms}ms"
            raise CaptureTimeoutError(msg) from err

        if not grab_result.GrabSucceeded():
            desc = grab_result.GetErrorDescription()
            grab_result.Release()
            msg = f"Grab failed: {desc}"
            raise CaptureError(msg)

        converted = self._converter.Convert(grab_result)
        data = converted.GetArray().copy()
        grab_result.Release()

        self._last_frame_data = data
        self._sequence += 1
        self._last_timestamp = time.monotonic()
        return Frame(data=data, timestamp=self._last_timestamp, sequence=self._sequence)

    def read_latest(self) -> Frame:
        if not self._connected or self._camera is None or self._converter is None:
            raise NotConnectedError

        from pypylon import pylon  # type: ignore[import-not-found]  # noqa: PLC0415

        grab_result = self._camera.RetrieveResult(0, pylon.TimeoutHandling_Return)

        if grab_result is not None and grab_result.GrabSucceeded():
            converted = self._converter.Convert(grab_result)
            data = converted.GetArray().copy()
            grab_result.Release()
            self._last_frame_data = data
            self._sequence += 1
            self._last_timestamp = time.monotonic()
            return Frame(data=data, timestamp=self._last_timestamp, sequence=self._sequence)
        if grab_result is not None:
            grab_result.Release()

        if self._last_frame_data is not None:
            return Frame(data=self._last_frame_data, timestamp=self._last_timestamp, sequence=self._sequence)
        msg = "No frame available"
        raise CaptureError(msg)

    @classmethod
    def discover(cls) -> list[DeviceInfo]:
        from ._discover import discover_basler  # noqa: PLC0415

        return discover_basler()
