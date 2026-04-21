# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

from physicalai.capture.discovery import DeviceInfo

__all__ = ["discover_realsense"]


def discover_realsense() -> list[DeviceInfo]:
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError:
        return []

    rs_any = cast("Any", rs)  # cast to Any to avoid false positive "missing-attribute"
    ctx = rs_any.context()
    results: list[DeviceInfo] = []

    for i, dev in enumerate(ctx.query_devices()):
        try:
            serial = dev.get_info(rs_any.camera_info.serial_number)
            name = dev.get_info(rs_any.camera_info.name)
        except RuntimeError:
            continue

        results.append(
            DeviceInfo(
                device_id=serial,
                index=i,
                name=name,
                driver="realsense",
                hardware_id=serial,
                manufacturer="Intel",
                model=name,
            )
        )

    return results
