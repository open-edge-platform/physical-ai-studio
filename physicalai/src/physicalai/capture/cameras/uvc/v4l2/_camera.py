# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""V4L2 camera backend."""

from __future__ import annotations

import contextlib
import ctypes
import mmap
import os
import select
from typing import TYPE_CHECKING, cast

import numpy as np

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.cameras.uvc._camera_setting import CameraSetting  # noqa: PLC2701
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame

from ._controls import V4L2CameraControls
from ._ioctl import (
    V4L2_BUF_TYPE_VIDEO_CAPTURE,
    V4L2_CAP_STREAMING,
    V4L2_CAP_VIDEO_CAPTURE,
    V4L2_FIELD_NONE,
    V4L2_MEMORY_MMAP,
    V4L2_PIX_FMT_MJPEG,
    V4L2_PIX_FMT_YUYV,
    VIDIOC_DQBUF,
    VIDIOC_QBUF,
    VIDIOC_QUERYBUF,
    VIDIOC_QUERYCAP,
    VIDIOC_REQBUFS,
    VIDIOC_S_FMT,
    VIDIOC_S_PARM,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    v4l2_buffer,
    v4l2_capability,
    v4l2_format,
    v4l2_requestbuffers,
    v4l2_streamparm,
    xioctl,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from turbojpeg import TurboJPEG  # type: ignore[import-not-found]

    from physicalai.capture.discovery import DeviceInfo


class V4L2Camera(Camera):
    def __init__(
        self,
        *,
        device_path: str = "/dev/video0",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        num_buffers: int = 4,
        pixel_format: str = "mjpeg",
        color_mode: ColorMode = ColorMode.RGB,
        controls: dict[int, int] | None = None,
    ) -> None:
        if pixel_format not in {"mjpeg", "yuyv"}:
            msg = f"Unsupported pixel_format {pixel_format!r}; use 'mjpeg' or 'yuyv'"
            raise ValueError(msg)

        super().__init__(color_mode=color_mode)
        self._device_path = device_path
        self._width = width
        self._height = height
        self._fps = fps
        self._num_buffers = num_buffers
        self._pixel_format = pixel_format
        self._initial_controls = controls or {}

        self._connected: bool = False
        self._sequence: int = 0
        self._fd: int | None = None
        self._buffers: list[tuple[mmap.mmap, int]] = []
        self._jpeg: TurboJPEG | None = None
        self._controls: V4L2CameraControls | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_id(self) -> str:
        return self._device_path

    def connect(self, timeout: float = 5.0) -> None:
        fd = -1
        try:
            fd = os.open(self._device_path, os.O_RDWR | os.O_NONBLOCK)
            self._fd = fd

            cap = v4l2_capability()
            xioctl(fd, VIDIOC_QUERYCAP, cap)
            self._validate_capabilities(cap.capabilities)

            fmt = v4l2_format()
            fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            fmt.fmt.pix.width = self._width
            fmt.fmt.pix.height = self._height
            fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG if self._pixel_format == "mjpeg" else V4L2_PIX_FMT_YUYV
            fmt.fmt.pix.field = V4L2_FIELD_NONE
            xioctl(fd, VIDIOC_S_FMT, fmt)

            streamparm = v4l2_streamparm()
            streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            streamparm.parm.capture.timeperframe.numerator = 1
            streamparm.parm.capture.timeperframe.denominator = self._fps
            xioctl(fd, VIDIOC_S_PARM, streamparm)

            req = v4l2_requestbuffers()
            req.count = self._num_buffers
            req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            req.memory = V4L2_MEMORY_MMAP
            xioctl(fd, VIDIOC_REQBUFS, req)

            for index in range(req.count):
                buf = v4l2_buffer()
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
                buf.memory = V4L2_MEMORY_MMAP
                buf.index = index
                xioctl(fd, VIDIOC_QUERYBUF, buf)

                mm = mmap.mmap(fd, buf.length, offset=buf.m.offset)
                self._buffers.append((mm, buf.length))

                xioctl(fd, VIDIOC_QBUF, buf)

            stream_type = cast("ctypes.Structure", ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))
            xioctl(fd, VIDIOC_STREAMON, stream_type)
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                self._raise_connect_timeout()

            self._connected = True
            self._sequence = 0
            self._controls = V4L2CameraControls(self._device_path, fd=self._fd)

            for ctrl_id, ctrl_value in self._initial_controls.items():
                self._controls.set_control(ctrl_id, ctrl_value)
        except Exception as exc:
            self._cleanup_connection()
            if isinstance(exc, CaptureTimeoutError):
                raise
            if isinstance(exc, CaptureError):
                raise
            msg = f"Failed to connect V4L2 device {self._device_path}: {exc}"
            raise CaptureError(msg) from exc

    def _do_disconnect(self) -> None:
        fd = self._fd
        if fd is not None:
            with contextlib.suppress(OSError):
                stream_type = cast("ctypes.Structure", ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))
                xioctl(fd, VIDIOC_STREAMOFF, stream_type)

        for mm, _ in self._buffers:
            mm.close()
        self._buffers = []

        if fd is not None:
            os.close(fd)
            self._fd = None

        self._connected = False

    def _cleanup_connection(self) -> None:
        try:
            self._do_disconnect()
        except OSError:
            pass
        finally:
            self._fd = None
            self._buffers = []
            self._connected = False

    @staticmethod
    def _validate_capabilities(capabilities: int) -> None:
        required = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING
        if (capabilities & required) != required:
            msg = "Device lacks required V4L2 capture/streaming capabilities"
            raise CaptureError(msg)

    def _raise_connect_timeout(self) -> None:
        msg = f"Timed out waiting for first frame from {self._device_path}"
        raise CaptureTimeoutError(msg)

    def _raise_read_timeout(self) -> None:
        msg = f"Timed out waiting for frame from {self._device_path}"
        raise CaptureTimeoutError(msg)

    def read(self, timeout: float | None = None) -> Frame:
        if not self._connected or self._fd is None:
            msg = "Cannot read: camera is not connected. Call connect() first."
            raise NotConnectedError(msg)

        decoded: NDArray[np.uint8]
        try:
            ready, _, _ = select.select([self._fd], [], [], timeout)
            if not ready:
                self._raise_read_timeout()

            buf = v4l2_buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            xioctl(self._fd, VIDIOC_DQBUF, buf)

            acquired_at = buf.timestamp.tv_sec + buf.timestamp.tv_usec / 1_000_000

            mm, _ = self._buffers[buf.index]
            raw = bytes(mm[: buf.bytesused])
            xioctl(self._fd, VIDIOC_QBUF, buf)

            decoded = self._decode(raw)
        except Exception as exc:
            if isinstance(exc, (CaptureTimeoutError, NotConnectedError)):
                raise
            msg = f"Failed to read frame from {self._device_path}: {exc}"
            raise CaptureError(msg) from exc
        else:
            frame = Frame(
                data=decoded,
                timestamp=acquired_at,
                sequence=self._sequence,
            )
            self._sequence += 1
            return frame

    def read_latest(self) -> Frame:
        if not self._connected or self._fd is None:
            msg = "Cannot read: camera is not connected. Call connect() first."
            raise NotConnectedError(msg)

        latest_raw: bytes | None = None
        acquired_at: float = 0.0
        decoded: NDArray[np.uint8]

        try:
            while True:
                ready, _, _ = select.select([self._fd], [], [], 0)
                if not ready:
                    break

                buf = v4l2_buffer()
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
                buf.memory = V4L2_MEMORY_MMAP
                xioctl(self._fd, VIDIOC_DQBUF, buf)

                acquired_at = buf.timestamp.tv_sec + buf.timestamp.tv_usec / 1_000_000

                mm, _ = self._buffers[buf.index]
                latest_raw = bytes(mm[: buf.bytesused])

                xioctl(self._fd, VIDIOC_QBUF, buf)

            if latest_raw is None:
                return self.read()

            decoded = self._decode(latest_raw)
        except Exception as exc:
            if isinstance(exc, NotConnectedError):
                raise
            msg = f"Failed to read latest frame from {self._device_path}: {exc}"
            raise CaptureError(msg) from exc
        else:
            frame = Frame(
                data=decoded,
                timestamp=acquired_at,
                sequence=self._sequence,
            )
            self._sequence += 1
            return frame

    def _decode(self, raw: bytes) -> NDArray[np.uint8]:
        if self._pixel_format == "mjpeg":
            from turbojpeg import (  # type: ignore[import-not-found]  # noqa: PLC0415
                TJPF_BGR,
                TJPF_GRAY,
                TJPF_RGB,
                TurboJPEG,
            )

            if self._jpeg is None:
                self._jpeg = TurboJPEG()

            if self._color_mode == ColorMode.RGB:
                tjpf = TJPF_RGB
            elif self._color_mode == ColorMode.BGR:
                tjpf = TJPF_BGR
            else:
                tjpf = TJPF_GRAY

            decoded = self._jpeg.decode(raw, pixel_format=tjpf)
            if self._color_mode == ColorMode.GRAY:
                return decoded.squeeze(axis=2)
            return decoded

        yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(self._height, self._width // 2, 4)
        y0 = yuyv[:, :, 0].astype(np.float32)
        u = yuyv[:, :, 1].astype(np.float32) - 128.0
        y1 = yuyv[:, :, 2].astype(np.float32)
        v = yuyv[:, :, 3].astype(np.float32) - 128.0

        y = np.empty((self._height, self._width), dtype=np.float32)
        y[:, 0::2] = y0
        y[:, 1::2] = y1
        u_full = np.repeat(u, 2, axis=1)
        v_full = np.repeat(v, 2, axis=1)

        if self._color_mode == ColorMode.GRAY:
            return np.clip(y, 0, 255).astype(np.uint8)

        r = np.clip(y + 1.402 * v_full, 0, 255).astype(np.uint8)
        g = np.clip(y - 0.344136 * u_full - 0.714136 * v_full, 0, 255).astype(np.uint8)
        b = np.clip(y + 1.772 * u_full, 0, 255).astype(np.uint8)

        if self._color_mode == ColorMode.BGR:
            return np.stack([b, g, r], axis=2)
        return np.stack([r, g, b], axis=2)

    # ------------------------------------------------------------------
    # V4L2 Controls (delegated to V4L2CameraControls)
    # ------------------------------------------------------------------

    def _ensure_controls(self) -> V4L2CameraControls:
        if self._controls is None:
            msg = "Cannot access controls: camera is not connected."
            raise NotConnectedError(msg)
        return self._controls

    def get_settings(self) -> list[CameraSetting]:
        return self._ensure_controls().list_controls()

    def apply_settings(self, settings: CameraSetting | list[CameraSetting]) -> None:
        """Apply one or more camera settings.

        Read-only, inactive, and valueless settings are silently skipped.
        """
        items = [settings] if isinstance(settings, CameraSetting) else settings
        controls = self._ensure_controls()
        for setting in items:
            if setting.value is None or setting.read_only or setting.inactive:
                continue
            controls.set_control(int(setting.id), setting.value)

    @classmethod
    def discover(cls) -> list[DeviceInfo]:
        from ._discover import discover_v4l2  # noqa: PLC0415

        return discover_v4l2()


__all__ = ["V4L2Camera"]
