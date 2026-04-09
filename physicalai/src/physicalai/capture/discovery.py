# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Device discovery types and utilities."""

from __future__ import annotations

import contextlib
from typing import Any

from pydantic import BaseModel, Field


class DeviceInfo(BaseModel):
    """Metadata about a discovered camera device.

    Returned by :meth:`~physicalai.capture.camera.Camera.discover` and
    :func:`discover_all`.
    """

    device_id: str = Field(description="Backend-specific identifier (e.g. '/dev/video0', index, IP address).")
    index: int = Field(description="Numeric device index when one exists for the backend.")
    name: str = Field(default="", description="Human-readable name (e.g. 'Logitech C920', 'D435').")
    driver: str = Field(default="", description="Backend that found the device (e.g. 'v4l2', 'realsense').")
    hardware_id: str | None = Field(
        default=None, description="Stable cross-backend identifier such as a serial number or USB bus path."
    )
    manufacturer: str | None = Field(default=None, description="Device manufacturer (e.g. 'Intel', 'Basler').")
    model: str | None = Field(default=None, description="Device model when available.")
    metadata: dict[str, Any] | None = Field(default=None, description="Backend-specific extras.")


def discover_all() -> dict[str, list[DeviceInfo]]:
    """Discover available cameras across all supported camera types.

    Each camera type is tried independently; failures are silently
    skipped so that a missing SDK does not prevent discovery of other
    camera types.

    Returns:
        Dict mapping camera type name to list of discovered devices.
        Types that are not installed or find no devices return an
        empty list.
    """
    results: dict[str, list[DeviceInfo]] = {}

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.uvc import discover_uvc  # noqa: PLC0415

        results["uvc"] = discover_uvc()

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.ip import IPCamera  # noqa: PLC0415

        results["ip"] = IPCamera.discover()

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.realsense import RealSenseCamera  # noqa: PLC0415

        results["realsense"] = RealSenseCamera.discover()

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.basler import BaslerCamera  # noqa: PLC0415

        results["basler"] = BaslerCamera.discover()

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.genicam import GenicamCamera  # noqa: PLC0415

        results["genicam"] = GenicamCamera.discover()

    return results
