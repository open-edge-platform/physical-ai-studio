# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Camera capture interfaces.

Public API::

    from physicalai.capture import Camera, ColorMode, Frame
    from physicalai.capture import read_cameras, async_read_cameras, SyncedFrames
    from physicalai.capture import create_camera, discover_all
    from physicalai.capture import DeviceInfo, DepthMixin
    from physicalai.capture import UVCCamera   # requires: opencv-python on macos/win32 or PyTurboJPEG on linux
    from physicalai.capture import IPCamera    # stub — not yet implemented
"""

from physicalai.capture.camera import Camera, CameraType, ColorMode
from physicalai.capture.discovery import DeviceInfo, discover_all
from physicalai.capture.errors import (
    CaptureError,
    CaptureTimeoutError,
    MissingDependencyError,
    NotConnectedError,
)
from physicalai.capture.factory import create_camera
from physicalai.capture.frame import Frame
from physicalai.capture.mixins import DepthMixin
from physicalai.capture.multi import SyncedFrames, async_read_cameras, read_cameras

__all__ = [  # noqa: F822, RUF022
    # ABC & types
    "Camera",
    "CameraType",
    "ColorMode",
    "Frame",
    "DeviceInfo",
    "SyncedFrames",
    # Mixins
    "DepthMixin",
    # Errors
    "CaptureError",
    "CaptureTimeoutError",
    "MissingDependencyError",
    "NotConnectedError",
    # Functions
    "async_read_cameras",
    "create_camera",
    "discover_all",
    "read_cameras",
    # Concrete cameras (lazy-loaded)
    "IPCamera",
    "RealSenseCamera",
    "UVCCamera",
]


def __getattr__(name: CameraType) -> object:
    """Lazy-load concrete camera implementations.

    This avoids pulling in hardware SDKs (e.g. ``opencv-python``,
    ``pyrealsense2``) at package import time.

    Args:
        name: The attribute name being looked up.

    Returns:
        The requested camera class.

    Raises:
        AttributeError: If *name* does not match a known lazy-loaded symbol.
    """
    if name == CameraType.UVC:
        from physicalai.capture.cameras.uvc import UVCCamera  # noqa: PLC0415

        return UVCCamera

    if name == CameraType.IP:
        from physicalai.capture.cameras.ip import IPCamera  # noqa: PLC0415

        return IPCamera

    if name == CameraType.REALSENSE:
        from physicalai.capture.cameras.realsense import RealSenseCamera  # noqa: PLC0415

        return RealSenseCamera

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
