# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Camera abstract base class and ColorMode enum."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

    from physicalai.capture.discovery import DeviceInfo
    from physicalai.capture.frame import Frame


class ColorMode(StrEnum):
    """Pixel format for colour image reads."""

    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"


class CameraType(StrEnum):
    """Known camera type names for :func:`create_camera`."""

    UVC = "uvc"
    IP = "ip"
    REALSENSE = "realsense"
    GENICAM = "genicam"
    BASLER = "basler"


class Camera(ABC):
    """Abstract interface for live camera hardware.

    Subclasses must implement:
        - :meth:`connect`
        - :meth:`_do_disconnect`
        - :meth:`read`
        - :meth:`read_latest`
        - :attr:`is_connected`
        - :attr:`device_id`

    The base class provides :meth:`disconnect` (executor cleanup),
    :meth:`async_read`, context managers, and :meth:`discover` /
    :meth:`from_config` convenience hooks.

    Thread Safety:
        Camera instances are **not** thread-safe.  Concurrent calls to
        ``read()`` or ``read_latest()`` on the same instance from multiple
        threads may raise exceptions or produce undefined behaviour depending
        on the underlying SDK.  Use one thread per camera instance, or add
        external synchronisation.  The multi-camera helpers in
        :mod:`physicalai.capture.multi` already respect this contract.
    """

    def __init__(
        self,
        *,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        """Store requested capture parameters.

        Args:
            color_mode: Pixel format for colour image reads.
        """
        self._color_mode = color_mode
        self.__executor: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self, timeout: float = 5.0) -> None:
        """Open the camera and verify it produces frames.

        Blocks until the first frame is successfully captured, confirming
        the hardware is operational. If no frame arrives within *timeout*
        seconds, raises ``CaptureTimeoutError``.

        After ``connect()`` returns, ``read()`` and ``read_latest()`` are
        guaranteed to succeed (barring subsequent hardware failures).

        Args:
            timeout: Maximum seconds to wait for the first frame.

        Raises:
            CaptureTimeoutError: Camera opened but no frame within timeout.
            CaptureError: Hardware-level connection failure.
        """
        ...

    @abstractmethod
    def _do_disconnect(self) -> None:
        """Release hardware resources.  Called by :meth:`disconnect`.

        Subclasses implement this to release SDK handles, close devices,
        and stop background capture loops.  Do **not** override
        :meth:`disconnect` directly — the base class handles executor
        cleanup.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from camera hardware and release all resources.

        Calls :meth:`_do_disconnect` to release hardware, then shuts
        down the async executor if it was created.  Subclasses override
        :meth:`_do_disconnect`, not this method.
        """
        self._do_disconnect()
        if self.__executor is not None:
            self.__executor.shutdown(wait=False, cancel_futures=True)
            self.__executor = None

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the camera is currently open."""
        ...

    @property
    @abstractmethod
    def device_id(self) -> str:
        """Identifier for the physical device this instance targets.

        Stable for the lifetime of the connection.  May change across
        reconnects for OS-assigned paths (e.g. ``/dev/video0``).

        Should match the corresponding ``DeviceInfo.device_id`` returned
        by :meth:`discover` for the same device.
        """
        ...

    @property
    def color_mode(self) -> ColorMode:
        """Pixel format for colour image reads."""
        return self._color_mode

    @property
    def _executor(self) -> ThreadPoolExecutor:
        """Lazy-initialised per-camera executor for async reads."""
        if self.__executor is None:
            self.__executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"capture-{self.device_id}",
            )
        return self.__executor

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @abstractmethod
    def read(self, timeout: float | None = None) -> Frame:
        """Read the next frame.  Blocks until available.

        Frames are returned in sequence; no frames are skipped.  Use for
        recording, sequential processing, or any case where every frame
        matters.

        Args:
            timeout: Maximum seconds to wait.  ``None`` waits
                indefinitely.

        Raises:
            NotConnectedError: If not connected.
            CaptureTimeoutError: If no frame arrives within *timeout*.
            CaptureError: If frame acquisition fails.
        """
        ...

    @abstractmethod
    def read_latest(self) -> Frame:
        """Read the most recent frame.  Non-blocking.

        Returns immediately with the latest captured frame.  May skip
        intermediate frames.  Use for real-time control, teleoperation,
        or any case where freshness matters more than completeness.

        Raises:
            NotConnectedError: If not connected.
            CaptureError: If frame acquisition fails.
        """
        ...

    async def async_read(self, timeout: float | None = None) -> Frame:  # noqa: ASYNC109
        """Read the next frame, yielding to the event loop while waiting.

        Default implementation offloads :meth:`read` to a dedicated
        per-camera ``ThreadPoolExecutor(max_workers=1)``, lazily created
        on first call and cleaned up by :meth:`disconnect`.  Sync-only
        usage incurs no thread overhead.

        Subclasses with native async support may override.

        Args:
            timeout: Maximum seconds to wait.  ``None`` waits
                indefinitely.

        Returns:
            The next captured frame.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.read, timeout)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls) -> list[DeviceInfo]:
        """List available devices of this camera type.

        Returns:
            Discovered devices, or an empty list if discovery is
            not supported for this camera type.
        """
        return []

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Create an instance from a configuration dictionary.

        Keys are forwarded as keyword arguments to the constructor.

        Returns:
            A new camera instance.
        """
        return cls(**config)

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        """Connect and return self.

        Returns:
            The connected camera instance.
        """
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        """Disconnect on exit."""
        self.disconnect()

    async def __aenter__(self) -> Self:
        """Async connect and return self.

        Returns:
            The connected camera instance.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.connect)
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async disconnect on exit."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.disconnect)
