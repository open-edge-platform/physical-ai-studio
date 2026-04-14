# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory convenience function for config-driven camera creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.capture.camera import CameraType

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera


def create_camera(camera_type: str, **kwargs) -> Camera:  # noqa: ANN003
    """Create a camera by type name.

    Convenience function for config-driven instantiation.  Prefer
    dedicated camera classes for direct usage.

    Args:
        camera_type: Camera type — one of ``"uvc"``, ``"ip"``,
            ``"realsense"``, ``"basler"``, ``"genicam"``.
            Case-insensitive.
        **kwargs: Forwarded to the camera constructor.

    Returns:
        A new camera instance.

    Raises:
        ValueError: If *camera_type* is not a recognised name.
    """
    camera_type = camera_type.lower()

    if camera_type == CameraType.UVC:
        from physicalai.capture.cameras.uvc import UVCCamera  # noqa: PLC0415

        return UVCCamera(**kwargs)

    if camera_type == CameraType.IP:
        from physicalai.capture.cameras.ip import IPCamera  # noqa: PLC0415

        return IPCamera(**kwargs)

    if camera_type == CameraType.REALSENSE:
        from physicalai.capture.cameras.realsense import RealSenseCamera  # noqa: PLC0415

        return RealSenseCamera(**kwargs)

    if camera_type == CameraType.BASLER:
        from physicalai.capture.cameras.basler import BaslerCamera  # noqa: PLC0415

        return BaslerCamera(**kwargs)

    if camera_type == CameraType.GENICAM:
        from physicalai.capture.cameras.genicam import GenicamCamera  # noqa: PLC0415

        return GenicamCamera(**kwargs)

    msg = f"Unknown camera type {camera_type!r}. Expected one of: {', '.join(CameraType)}"
    raise ValueError(msg)
