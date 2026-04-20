# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

from physicalai.capture.discovery import DeviceInfo

__all__ = ["discover_basler"]


def discover_basler() -> list[DeviceInfo]:
    try:
        pylon = importlib.import_module("pypylon").pylon
    except ImportError:
        return []

    factory = pylon.TlFactory.GetInstance()
    results: list[DeviceInfo] = []

    for i, dev in enumerate(factory.EnumerateDevices()):
        try:
            serial = dev.GetSerialNumber()
            model = dev.GetModelName()
            vendor = dev.GetVendorName()
        except (AttributeError, RuntimeError, ValueError):
            continue

        results.append(
            DeviceInfo(
                device_id=f"basler:{serial}",
                index=i,
                name=model,
                driver="basler",
                hardware_id=serial,
                manufacturer=vendor,
                model=model,
            )
        )

    return results
