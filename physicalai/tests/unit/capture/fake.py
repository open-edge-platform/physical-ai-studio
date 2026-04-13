# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FakeCamera implementation for testing.

Provides a concrete :class:`~physicalai.capture.camera.Camera` subclass that
returns synthetic frames without touching any hardware.  Used by all capture
unit tests.
"""

from __future__ import annotations

import time

import numpy as np

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.errors import NotConnectedError
from physicalai.capture.frame import Frame


class FakeCamera(Camera):
    """In-memory camera that produces synthetic frames.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        color_mode: Pixel format for image reads.
        device_name: Identifier returned by :attr:`device_id`.
    """

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        color_mode: ColorMode = ColorMode.RGB,
        device_name: str = "fake-0",
    ) -> None:
        super().__init__(color_mode=color_mode)
        self._width = width
        self._height = height
        self._device_name = device_name
        self._connected = False
        self._sequence = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, timeout: float = 5.0) -> None:  # noqa: ARG002
        """Mark the camera as connected."""
        self._connected = True
        self._sequence = 0

    def _do_disconnect(self) -> None:
        """Mark the camera as disconnected."""
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Whether the fake camera is connected."""
        return self._connected

    @property
    def device_id(self) -> str:
        """Return the configured device name."""
        return self._device_name

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _make_frame(self) -> Frame:
        """Build a synthetic frame with the current sequence number."""
        if self._color_mode == ColorMode.GRAY:
            shape = (self._height, self._width)
        else:
            shape = (self._height, self._width, 3)
        data = np.zeros(shape, dtype=np.uint8)
        frame = Frame(data=data, timestamp=time.monotonic(), sequence=self._sequence)
        self._sequence += 1
        return frame

    def read(self, timeout: float | None = None) -> Frame:  # noqa: ARG002
        """Return the next synthetic frame.

        Raises:
            NotConnectedError: If not connected.
        """
        if not self._connected:
            msg = "Cannot read: camera is not connected. Call connect() first."
            raise NotConnectedError(msg)
        return self._make_frame()

    def read_latest(self) -> Frame:
        """Return the latest synthetic frame (same as read for fake).

        Raises:
            NotConnectedError: If not connected.
        """
        if not self._connected:
            msg = "Cannot read: camera is not connected. Call connect() first."
            raise NotConnectedError(msg)
        return self._make_frame()
