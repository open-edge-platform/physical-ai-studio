# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""UVC camera facade.

This module exposes :class:`~physicalai.capture.cameras.uvc.UVCCamera` as the
user-facing entry point for "standard USB video cameras" (UVC devices).

Internally it delegates to one of:
  - :class:`~physicalai.capture.cameras.uvc.v4l2.V4L2Camera` on Linux
  - :class:`~physicalai.capture.cameras.uvc._omnicamera.OmniCamera` elsewhere
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.cameras.uvc._camera_setting import CameraSetting  # noqa: PLC2701

if TYPE_CHECKING:
    from physicalai.capture.cameras.uvc.v4l2 import V4L2Camera
    from physicalai.capture.cameras.uvc.v4l2._controls import V4L2CameraControls
    from physicalai.capture.frame import Frame


class UVCCamera(Camera):
    """Camera facade for UVC devices (USB Video Class).

    Args:
        device: Unified device selector.
            - On Linux (V4L2): ``0`` maps to ``/dev/video0``.
            - On macOS/Windows (OmniCamera): ``0`` maps to OmniCamera index ``0``.
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        fps: Requested frames per second.
        color_mode: Pixel format for returned frames.
        backend: ``"v4l2"``, or ``"omnicamera"``.
        backend_options: Backend-specific overrides forwarded to the selected
            backend constructor.
    """

    def __init__(
        self,
        *,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        color_mode: ColorMode = ColorMode.RGB,
        backend: Literal["v4l2", "omnicamera"] = "omnicamera",
        backend_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(color_mode=color_mode)

        self._device = device
        self._backend = backend
        opts = dict(backend_options or {})

        # Resolve device path for V4L2 controls (used on Linux regardless of backend).
        if isinstance(device, int) or (isinstance(device, str) and device.isdecimal()):
            self._device_path: str = f"/dev/video{device}"
        elif isinstance(device, str):
            self._device_path = device
        else:
            self._device_path = f"/dev/video{device}"

        self._inner: V4L2Camera | OmniCamera

        if backend == "v4l2":
            from .v4l2 import V4L2Camera  # noqa: PLC0415

            device_path: str
            if isinstance(device, int) or (isinstance(device, str) and device.isdecimal()):
                device_path = f"/dev/video{device}"
            else:
                device_path = device

            # Forward V4L2-specific overrides (e.g. num_buffers, pixel_format).
            # The facade's ``device`` maps to V4L2's ``device_path``.
            opts.setdefault("device_path", device_path)
            self._inner = V4L2Camera(
                width=width,
                height=height,
                fps=fps,
                color_mode=color_mode,
                **opts,
            )
        elif backend == "omnicamera":
            from ._omnicamera import OmniCamera  # noqa: PLC0415

            # Forward OmniCamera-specific overrides while mapping facade
            # ``device`` to OmniCamera.device_id.
            opts.setdefault("device_id", device)
            self._inner = OmniCamera(
                width=width,
                height=height,
                fps=fps,
                color_mode=color_mode,
                **opts,
            )
        elif backend == "opencv":
            msg = "The 'opencv' backend has been removed. Use backend='omnicamera' or backend='auto' instead."
            raise ValueError(msg)
        else:
            msg = f"Unknown backend {backend!r}. Use 'auto', 'v4l2', or 'omnicamera'."
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, timeout: float = 5.0) -> None:
        self._inner.connect(timeout=timeout)

    def _do_disconnect(self) -> None:
        self._inner.disconnect()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self, timeout: float | None = None) -> Frame:
        return self._inner.read(timeout=timeout)

    def read_latest(self) -> Frame:
        return self._inner.read_latest()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._inner.is_connected

    @property
    def device_id(self) -> str:
        return self._inner.device_id

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls) -> list[Any]:  # pragma: no cover - wrapper uses typed DeviceInfo
        from ._discover import discover_uvc  # noqa: PLC0415

        return discover_uvc()

    # ------------------------------------------------------------------
    # Camera Settings
    # ------------------------------------------------------------------

    def _get_v4l2_controls(self) -> V4L2CameraControls | None:
        """Return a V4L2CameraControls instance on Linux, None otherwise.

        If the inner backend is V4L2Camera, reuses its shared-fd controls.
        If the inner backend is OmniCamera on Linux, returns a standalone
        controls instance that opens/closes a fd per call.
        """
        if not sys.platform.startswith("linux"):
            return None

        from .v4l2 import V4L2CameraControls  # noqa: PLC0415

        return V4L2CameraControls(self._device_path)

    def get_settings(self) -> list[CameraSetting]:
        """List all available camera settings.

        Returns:
            The available settings reported by the active backend.
        """
        if (v4l2 := self._get_v4l2_controls()) is not None:
            return v4l2.list_controls()
        return self._inner.get_settings()

    def apply_settings(self, settings: CameraSetting | list[CameraSetting]) -> None:
        """Apply one or more camera settings.

        Read-only, inactive, and valueless settings are silently skipped.
        """
        items = [settings] if isinstance(settings, CameraSetting) else settings
        if (v4l2 := self._get_v4l2_controls()) is not None:
            for s in items:
                if s.value is None or s.read_only or s.inactive:
                    continue
                v4l2.set_control(int(s.id), s.value)
            return
        self._inner.apply_settings(settings)


__all__ = ["UVCCamera"]
