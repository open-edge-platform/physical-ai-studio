# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyrealsense2 as rs
from loguru import logger

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame
from physicalai.capture.mixins.depth import DepthMixin

if TYPE_CHECKING:
    from physicalai.capture.discovery import DeviceInfo


class RealSenseCamera(DepthMixin, Camera):
    """RealSense color and depth camera."""

    def __init__(
        self,
        *,
        serial_number: str,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        color_mode: ColorMode = ColorMode.RGB,
        depth_width: int | None = None,
        depth_height: int | None = None,
    ) -> None:
        super().__init__(color_mode=color_mode)
        self._serial_number = serial_number
        self._fps = fps
        self._width = width
        self._height = height
        self._color_mode = color_mode
        self._depth_width = width if depth_width is None else depth_width
        self._depth_height = height if depth_height is None else depth_height
        self._connected = False
        self._sequence = 0
        self._depth_sequence = 0
        self._pipeline: Any | None = None
        self._config: Any | None = None
        self._align: Any | None = None
        self._last_frameset: Any | None = None

    def connect(self, timeout: float = 5.0) -> None:
        """Open the RealSense pipeline and confirm first frame.

        Raises:
            CaptureError: If the pipeline fails to start.
            CaptureTimeoutError: If no first frame arrives in time.
        """
        rs_any = cast("Any", rs)  # cast to Any to avoid false positive "missing-attribute"
        pipeline_factory = rs_any.pipeline
        config_factory = rs_any.config
        stream = rs_any.stream
        frame_format = rs_any.format
        align_factory = rs_any.align

        self._pipeline = pipeline_factory()
        self._config = config_factory()
        self._config.enable_device(self._serial_number)
        self._config.enable_stream(stream.color, self._width, self._height, frame_format.rgb8, self._fps)
        self._config.enable_stream(
            stream.depth,
            self._depth_width,
            self._depth_height,
            frame_format.z16,
            self._fps,
        )

        try:
            self._pipeline.start(self._config)
        except RuntimeError as err:
            self._pipeline = None
            self._config = None
            msg = "Failed to start RealSense pipeline"
            raise CaptureError(msg) from err

        self._align = align_factory(stream.color)
        timeout_ms = int(timeout * 1000)
        try:
            first_frameset = self._pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as err:
            self._do_disconnect()
            msg = f"Timed out waiting for first frame after {timeout}s"
            raise CaptureTimeoutError(msg) from err

        self._last_frameset = self._align.process(first_frameset)
        self._connected = True
        self._sequence = 0
        self._depth_sequence = 0
        logger.info(f"RealSense camera {self._serial_number} connected ({self._width}x{self._height} @ {self._fps}fps)")

    def _do_disconnect(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001
                logger.debug(f"Error stopping RealSense pipeline {self._serial_number}")
        self._pipeline = None
        self._config = None
        self._align = None
        self._last_frameset = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_id(self) -> str:
        """Backend-specific identifier for this camera.

        Returns:
            Stable ``realsense:<serial>`` device identifier.
        """
        return f"realsense:{self._serial_number}"

    def read(self, timeout: float | None = None) -> Frame:
        """Read the next aligned color frame.

        Returns:
            Captured color frame.

        Raises:
            NotConnectedError: If camera is not connected.
            CaptureTimeoutError: If no frame arrives before timeout.
            CaptureError: If color frame extraction fails.
        """
        if not self._connected or self._pipeline is None or self._align is None:
            raise NotConnectedError

        timeout_ms = int(timeout * 1000) if timeout is not None else 15000
        try:
            frameset = self._pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as err:
            msg = f"Timed out waiting for frame after {timeout_ms}ms"
            raise CaptureTimeoutError(msg) from err

        frameset = self._align.process(frameset)
        color_frame = frameset.get_color_frame()
        if not color_frame:
            msg = "No color frame available"
            raise CaptureError(msg)

        color_data = np.asanyarray(color_frame.get_data())
        converted = self._convert_color(color_data)
        self._last_frameset = frameset
        self._sequence += 1
        return Frame(data=converted, timestamp=time.monotonic(), sequence=self._sequence)

    def read_latest(self) -> Frame:
        """Read the freshest aligned color frame without blocking.

        Returns:
            Newest available color frame.

        Raises:
            NotConnectedError: If camera is not connected.
            CaptureError: If no frame data is available.
        """
        if not self._connected or self._pipeline is None or self._align is None:
            raise NotConnectedError

        latest_frameset = None
        while True:
            frameset = self._pipeline.poll_for_frames()
            if not frameset:
                break
            latest_frameset = frameset

        if latest_frameset is not None:
            latest_frameset = self._align.process(latest_frameset)
            color_frame = latest_frameset.get_color_frame()
            if not color_frame:
                msg = "No color frame available"
                raise CaptureError(msg)
            color_data = np.asanyarray(color_frame.get_data())
            converted = self._convert_color(color_data)
            self._last_frameset = latest_frameset
            self._sequence += 1
            return Frame(data=converted, timestamp=time.monotonic(), sequence=self._sequence)

        if self._last_frameset is None:
            msg = "No frame available"
            raise CaptureError(msg)

        color_frame = self._last_frameset.get_color_frame()
        if not color_frame:
            msg = "No color frame available"
            raise CaptureError(msg)
        color_data = np.asanyarray(color_frame.get_data())
        converted = self._convert_color(color_data)
        return Frame(data=converted, timestamp=time.monotonic(), sequence=self._sequence)

    def read_depth(self) -> Frame:
        """Read the next aligned depth frame.

        Returns:
            Captured depth frame.

        Raises:
            NotConnectedError: If camera is not connected.
            CaptureTimeoutError: If no frame arrives before timeout.
            CaptureError: If depth frame extraction fails.
        """
        if not self._connected or self._pipeline is None or self._align is None:
            raise NotConnectedError

        try:
            frameset = self._pipeline.wait_for_frames(5000)
        except RuntimeError as err:
            msg = "Timed out waiting for depth frame"
            raise CaptureTimeoutError(msg) from err

        frameset = self._align.process(frameset)
        depth_frame = frameset.get_depth_frame()
        if not depth_frame:
            msg = "No depth frame available"
            raise CaptureError(msg)

        depth_data = np.asanyarray(depth_frame.get_data())
        self._last_frameset = frameset
        self._depth_sequence += 1
        return Frame(data=depth_data, timestamp=time.monotonic(), sequence=self._depth_sequence)

    def read_rgbd(self) -> tuple[Frame, Frame]:
        """Read aligned RGB and depth frames from one capture.

        Returns:
            Tuple of ``(color_frame, depth_frame)``.

        Raises:
            NotConnectedError: If camera is not connected.
            CaptureTimeoutError: If no frame arrives before timeout.
            CaptureError: If color or depth extraction fails.
        """
        if not self._connected or self._pipeline is None or self._align is None:
            raise NotConnectedError

        try:
            frameset = self._pipeline.wait_for_frames(5000)
        except RuntimeError as err:
            msg = "Timed out waiting for RGBD frame"
            raise CaptureTimeoutError(msg) from err

        frameset = self._align.process(frameset)
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        if not color_frame:
            msg = "No color frame available"
            raise CaptureError(msg)
        if not depth_frame:
            msg = "No depth frame available"
            raise CaptureError(msg)

        color_data = np.asanyarray(color_frame.get_data())
        depth_data = np.asanyarray(depth_frame.get_data())
        color_converted = self._convert_color(color_data)
        self._sequence += 1
        self._depth_sequence += 1
        self._last_frameset = frameset
        timestamp = time.monotonic()
        return (
            Frame(data=color_converted, timestamp=timestamp, sequence=self._sequence),
            Frame(data=depth_data, timestamp=timestamp, sequence=self._depth_sequence),
        )

    def _convert_color(self, frame: np.ndarray) -> np.ndarray:
        if self._color_mode == ColorMode.RGB:
            return frame
        if self._color_mode == ColorMode.BGR:
            return frame[:, :, ::-1]
        if self._color_mode == ColorMode.GRAY:
            return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return frame

    @classmethod
    def discover(cls) -> list[DeviceInfo]:
        """Discover available RealSense devices.

        Returns:
            List of discovered RealSense devices.
        """
        from ._discover import discover_realsense  # noqa: PLC0415

        return discover_realsense()
