# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pynokhwa as omni_camera  # rename omni_camera references

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.cameras.uvc._camera_setting import CameraSetting  # noqa: PLC2701
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame

if TYPE_CHECKING:
    from physicalai.capture.discovery import DeviceInfo


_MISSING_DEP_PKG = "omni_camera"
_MISSING_DEP_EXTRA = "capture"


class OmniCamera(Camera):
    _POLL_INTERVAL_S = 0.001

    def __init__(
        self,
        *,
        device_id: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        super().__init__(color_mode=color_mode)
        self._device_id_raw = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._color_mode = color_mode
        self._connected = False
        self._sequence = 0
        self._cam: omni_camera.Camera | None = None
        self._last_frame: np.ndarray | None = None

    @staticmethod
    def _resolve_device_info(infos: list[omni_camera.CameraInfo], device_id: int | str) -> omni_camera.CameraInfo:
        normalized_device_id: int
        if isinstance(device_id, str):
            if device_id.isdecimal():
                normalized_device_id = int(device_id)
            elif device_id.startswith("/dev/video"):
                suffix = device_id.removeprefix("/dev/video")
                if not suffix.isdecimal():
                    msg = f"Invalid device path: {device_id}"
                    raise ValueError(msg)
                normalized_device_id = int(suffix)
            else:
                msg = (
                    "OmniCamera backend does not support device path strings on this platform. "
                    "Use an integer camera index instead."
                )
                raise ValueError(msg)
        else:
            normalized_device_id = device_id

        info = next((candidate for candidate in infos if candidate.index == normalized_device_id), None)
        if info is None:
            msg = f"No camera found at index {normalized_device_id}"
            raise CaptureError(msg)
        if not info.can_open():
            msg = "Camera cannot be opened"
            raise CaptureError(msg)
        return info

    def _resolve_format(self) -> omni_camera.CameraFormat:
        if self._cam is None:
            msg = "Camera cannot be opened"
            raise CaptureError(msg)
        try:
            opts = self._cam.get_format_options()
            # noqa: TD002, TD003, FIX002 # TODO: Switch back to keyword args when omni_camera type stubs support them.
            opts = opts.prefer_width_range(self._width, self._width)
            opts = opts.prefer_height_range(self._height, self._height)
            opts = opts.prefer_fps_range(self._fps, self._fps)
            return opts.resolve(key=lambda x: x.width)
        except (RuntimeError, ValueError, TypeError):
            try:
                opts2 = self._cam.get_format_options()
                opts2 = opts2.prefer_width_range(self._width)
                opts2 = opts2.prefer_height_range(self._height)
                return opts2.resolve()
            except (RuntimeError, ValueError, TypeError):
                try:
                    return self._cam.get_format_options().resolve_default()
                except (RuntimeError, ValueError, TypeError) as err:
                    msg = "No compatible camera format found"
                    raise CaptureError(msg) from err

    def connect(self, timeout: float = 5.0) -> None:
        # On macOS, nokhwa_initialize() fires an async AVFoundation permission
        # request at module import. If we query before that callback resolves,
        # only_usable=True filters out the camera, yielding an empty list.
        # Retry briefly to give the TCC callback time to deliver.
        query_deadline = time.monotonic() + 2.0
        infos = omni_camera.query(only_usable=True)
        while not infos and time.monotonic() < query_deadline:
            time.sleep(0.1)
            infos = omni_camera.query(only_usable=True)
        info = self._resolve_device_info(infos, self._device_id_raw)

        self._cam = omni_camera.Camera(info)
        fmt = self._resolve_format()

        if fmt.width != self._width or fmt.height != self._height or fmt.frame_rate != self._fps:
            from loguru import logger  # noqa: PLC0415

            logger.warning(
                f"Requested {self._width}x{self._height}@{self._fps}fps, "
                f"using {fmt.width}x{fmt.height}@{fmt.frame_rate}fps",
            )

        self._cam.open(fmt)

        frame_data = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame_data = self._cam.poll_frame_np()
            if frame_data is not None:
                break
            time.sleep(self._POLL_INTERVAL_S)

        if frame_data is None:
            msg = f"Timed out waiting for first frame after {timeout}s"
            raise CaptureTimeoutError(msg)

        self._last_frame = frame_data
        self._connected = True
        self._sequence = 0

    def _do_disconnect(self) -> None:
        if self._cam is not None:
            self._cam.close()
        self._cam = None
        self._connected = False
        self._last_frame = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_id(self) -> str:
        return str(self._device_id_raw)

    def read(self, timeout: float | None = None) -> Frame:
        if not self._connected or self._cam is None:
            err = NotConnectedError()
            raise err

        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            frame_data, seq = self._cam.poll_frame_np_with_seq()
            if frame_data is not None:
                converted = self._convert_color(frame_data)
                self._sequence = seq
                self._last_frame = frame_data
                return Frame(data=converted, timestamp=time.monotonic(), sequence=self._sequence)

            if deadline is not None and time.monotonic() >= deadline:
                msg = f"Timed out waiting for frame after {timeout}s"
                raise CaptureTimeoutError(msg)

            time.sleep(self._POLL_INTERVAL_S)

    def read_latest(self) -> Frame:
        if not self._connected or self._cam is None:
            err = NotConnectedError()
            raise err

        frame_data, seq = self._cam.poll_frame_np_with_seq()
        if frame_data is not None:
            converted = self._convert_color(frame_data)
            self._sequence = seq
            self._last_frame = frame_data
            return Frame(data=converted, timestamp=time.monotonic(), sequence=self._sequence)

        if self._last_frame is not None:
            return Frame(
                data=self._convert_color(self._last_frame),
                timestamp=time.monotonic(),
                sequence=self._sequence,
            )

        msg = "No frame available"
        raise CaptureError(msg)

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
        from physicalai.capture.discovery import DeviceInfo  # noqa: PLC0415

        # NOTE: We intentionally use ``only_usable=False`` and do NOT call
        # ``info.can_open()`` here. On macOS those probes briefly open the
        # camera through AVFoundation, which holds a process-wide lock that
        # prevents any subprocess (e.g. the publisher worker) from opening
        # the same device until the parent exits.
        infos = omni_camera.query(only_usable=False)
        return [
            DeviceInfo(
                device_id=str(info.index),
                index=info.index,
                name=info.name,
                driver="uvc",
                hardware_id="",
                manufacturer="",
                model=info.name,
                metadata={
                    "description": info.description,
                    "misc": info.misc,
                    "backend": "omnicamera",
                },
            )
            for info in infos
        ]

    def get_settings(self) -> list[CameraSetting]:
        if not self._connected or self._cam is None:
            raise NotConnectedError

        get_controls = getattr(self._cam, "get_controls", None)
        if not callable(get_controls):
            msg = "get_settings is not available for this OmniCamera build."
            raise NotImplementedError(msg)

        raw_controls = get_controls()
        if not isinstance(raw_controls, dict):
            raw_controls = dict(cast("Any", raw_controls))

        controls: list[CameraSetting] = []
        for name, ctrl in raw_controls.items():
            vr = ctrl.value_range
            has_range = len(vr) > 0

            controls.append(
                CameraSetting(
                    id=name,
                    name=name,
                    setting_type="integer",
                    min=vr.start if has_range else None,
                    max=vr[-1] if has_range else None,
                    step=vr.step if has_range else None,
                    default=None,
                    value=None,
                    inactive=not ctrl.is_active,
                    read_only=False,
                )
            )
        return controls

    def apply_settings(self, settings: CameraSetting | list[CameraSetting]) -> None:
        """Apply one or more camera settings.

        Read-only, inactive, and valueless settings are silently skipped.
        """
        raise NotImplementedError
