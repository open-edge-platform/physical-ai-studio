# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
from loguru import logger
from pypylon import genicam, pylon

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
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        super().__init__(color_mode=color_mode)
        self._serial_number = serial_number
        self._fps = fps
        # User-requested dimensions (None = full sensor). Resolved to actual
        # output dimensions in _configure_camera once the sensor size is known.
        self._requested_width = width
        self._requested_height = height
        self._width: int = 0
        self._height: int = 0
        self._connected = False
        self._sequence = 0
        self._last_timestamp: float = 0.0
        self._camera: pylon.InstantCamera | None = None
        self._converter: pylon.ImageFormatConverter | None = None
        self._last_frame_data: np.ndarray | None = None
        self._needs_resize = False

    def connect(self, timeout: float = 5.0) -> None:
        dev_info = self._find_device()
        try:
            camera, converter = self._configure_camera(dev_info)
        except Exception as err:
            self._do_disconnect()
            msg = f"Failed to open Basler camera {self._serial_number}"
            raise CaptureError(msg) from err
        self._wait_first_frame(timeout, camera, converter)

    def _find_device(self) -> pylon.DeviceInfo:
        factory = pylon.TlFactory.GetInstance()
        for di in factory.EnumerateDevices():
            if di.GetSerialNumber() == self._serial_number:
                return di
        msg = f"Basler camera with serial {self._serial_number} not found"
        raise CaptureError(msg)

    def _configure_camera(self, dev_info: pylon.DeviceInfo) -> tuple[pylon.InstantCamera, pylon.ImageFormatConverter]:
        factory = pylon.TlFactory.GetInstance()
        self._camera = pylon.InstantCamera(factory.CreateDevice(dev_info))
        self._camera.Open()

        # Grab at full sensor resolution; resize to requested dimensions in software.
        # On color GigE models (e.g. a2A1920-51gcBAS) binning is not available,
        # and setting Width/Height directly would crop the sensor ROI.
        self._camera.Width.Value = self._camera.Width.Max
        self._camera.Height.Value = self._camera.Height.Max
        sensor_w = self._camera.Width.Value
        sensor_h = self._camera.Height.Value

        # Resolve requested dimensions: None → full sensor, clamp to sensor max.
        req_w = self._requested_width
        req_h = self._requested_height
        self._width = sensor_w if req_w is None else min(req_w, sensor_w)
        self._height = sensor_h if req_h is None else min(req_h, sensor_h)
        if (req_w is not None and req_w > sensor_w) or (req_h is not None and req_h > sensor_h):
            logger.warning(
                f"Basler {self._serial_number}: requested {req_w}x{req_h} exceeds sensor "
                f"{sensor_w}x{sensor_h}. Upscaling is not supported; using {self._width}x{self._height}."
            )

        self._needs_resize = sensor_w != self._width or sensor_h != self._height
        if self._needs_resize:
            logger.info(
                f"Basler {self._serial_number}: sensor {sensor_w}x{sensor_h}, "
                f"will resize to {self._width}x{self._height}"
            )

        try:
            self._camera.AcquisitionFrameRateEnable.Value = True
            self._camera.AcquisitionFrameRate.Value = float(self._fps)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error setting FPS for basler camera {self._serial_number}: {exc}")

        self._converter = pylon.ImageFormatConverter()
        if self._color_mode == ColorMode.RGB:
            self._converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        elif self._color_mode == ColorMode.BGR:
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        else:
            self._converter.OutputPixelFormat = pylon.PixelType_Mono8

        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        return self._camera, self._converter

    def _wait_first_frame(
        self, timeout: float, camera: pylon.InstantCamera, converter: pylon.ImageFormatConverter
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error = ""

        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                grab_result = camera.RetrieveResult(remaining_ms, pylon.TimeoutHandling_ThrowException)
            except genicam.TimeoutException as err:
                self._do_disconnect()
                msg = f"Timed out waiting for first frame after {timeout}s"
                raise CaptureTimeoutError(msg) from err
            except Exception as err:
                self._do_disconnect()
                msg = "Failed to start Basler camera"
                raise CaptureError(msg) from err

            if grab_result.GrabSucceeded():
                self._last_frame_data = self._convert_result(grab_result, converter)
                self._connected = True
                self._sequence = 0
                logger.info(
                    f"Basler camera {self._serial_number} connected ({self._width}x{self._height} @ {self._fps}fps)"
                )
                return

            last_error = grab_result.GetErrorDescription()
            grab_result.Release()

        self._do_disconnect()
        msg = f"First grab failed: {last_error}"
        raise CaptureError(msg)

    def _ensure_connected(self) -> tuple[pylon.InstantCamera, pylon.ImageFormatConverter]:
        camera = self._camera
        converter = self._converter
        if not self._connected or camera is None or converter is None:
            raise NotConnectedError
        return camera, converter

    def _do_disconnect(self) -> None:
        if self._camera is not None:
            try:
                self._camera.StopGrabbing()
            except Exception:  # noqa: BLE001
                logger.debug(f"Error stopping grab for basler camera {self._serial_number}")
            try:
                self._camera.Close()
            except Exception:  # noqa: BLE001
                logger.debug(f"Error closing basler camera {self._serial_number}")
        self._camera = None
        self._converter = None
        self._last_frame_data = None
        self._connected = False

    def _resize(self, data: np.ndarray) -> np.ndarray:
        """Downsample using area-based interpolation (anti-aliased).

        Returns:
            Resized image data.
        """
        if self._needs_resize:
            return cv2.resize(data, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return data

    def _convert_result(
        self,
        grab_result: pylon.GrabResult,
        converter: pylon.ImageFormatConverter,
    ) -> np.ndarray:
        """Convert a grab result to an owned numpy array and release the buffer.

        Returns:
            Converted frame data as a numpy array.
        """
        converted = converter.Convert(grab_result)
        # GetArray() returns a view into the PylonImage's C++ buffer; copy
        # so the numpy array survives after converted is garbage-collected.
        data = self._resize(converted.GetArray().copy())
        grab_result.Release()
        return data

    @property
    def is_connected(self) -> bool:
        return self._connected and self._camera is not None and self._converter is not None

    @property
    def device_id(self) -> str:
        return f"basler:{self._serial_number}"

    def read(self, timeout: float | None = None) -> Frame:
        camera, converter = self._ensure_connected()

        timeout_s = timeout if timeout is not None else 5.0
        deadline = time.monotonic() + timeout_s
        last_error = ""

        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                grab_result = camera.RetrieveResult(remaining_ms, pylon.TimeoutHandling_ThrowException)
            except genicam.TimeoutException as err:
                msg = f"Timed out waiting for frame after {timeout_s}s"
                raise CaptureTimeoutError(msg) from err

            if grab_result.GrabSucceeded():
                data = self._convert_result(grab_result, converter)

                self._last_frame_data = data
                self._sequence += 1
                self._last_timestamp = time.monotonic()
                return Frame(data=data, timestamp=self._last_timestamp, sequence=self._sequence)

            last_error = grab_result.GetErrorDescription()
            grab_result.Release()

        msg = f"Grab failed: {last_error}"
        raise CaptureError(msg)

    def read_latest(self) -> Frame:
        camera, converter = self._ensure_connected()

        grab_result = camera.RetrieveResult(0, pylon.TimeoutHandling_Return)

        if grab_result is not None and grab_result.IsValid():
            if grab_result.GrabSucceeded():
                data = self._convert_result(grab_result, converter)
                self._last_frame_data = data
                self._sequence += 1
                self._last_timestamp = time.monotonic()
                return Frame(data=data, timestamp=self._last_timestamp, sequence=self._sequence)
            grab_result.Release()

        if self._last_frame_data is not None:
            return Frame(data=self._last_frame_data, timestamp=self._last_timestamp, sequence=self._sequence)
        msg = "No frame available"
        raise CaptureError(msg)

    @classmethod
    def discover(cls) -> list[DeviceInfo]:
        from ._discover import discover_basler  # noqa: PLC0415

        return discover_basler()
