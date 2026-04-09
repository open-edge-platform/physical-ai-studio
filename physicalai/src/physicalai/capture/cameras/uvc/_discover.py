# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""V4L2 device enumeration via sysfs and QUERYCAP ioctl.

Scans ``/sys/class/video4linux/`` for ``video*`` entries, opens each
``/dev/videoN`` device, and queries capabilities via ``VIDIOC_QUERYCAP``.
Only devices whose **per-node** ``device_caps`` advertise video capture
are included — this correctly filters out UVC metadata nodes that share
the same physical device.

Returns an empty list on non-Linux hosts (no sysfs present) and silently
skips devices that cannot be opened due to permission or I/O errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.capture.cameras.uvc.v4l2 import discover_v4l2

if TYPE_CHECKING:
    from physicalai.capture.discovery import DeviceInfo

__all__ = ["discover_uvc"]


def discover_uvc() -> list[DeviceInfo]:
    """Discover UVC devices for the current platform.

    On Linux this uses native V4L2/sysfs discovery.
    On other platforms it falls back to OmniCamera discovery.

    Returns:
        List of discovered UVC devices for the current platform.
    """
    import sys  # noqa: PLC0415

    if sys.platform == "linux":
        return discover_v4l2()

    from physicalai.capture.errors import MissingDependencyError  # noqa: PLC0415

    from ._omnicamera import OmniCamera  # noqa: PLC0415

    try:
        return OmniCamera.discover()
    except MissingDependencyError:
        return []
