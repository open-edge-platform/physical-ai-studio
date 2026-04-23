# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from loguru import logger

from physicalai.capture.discovery import DeviceInfo

__all__ = ["discover_basler"]

try:
    from pypylon import pylon
except ImportError:
    pylon = None  # type: ignore[assignment]


def discover_basler() -> list[DeviceInfo]:
    if pylon is None:
        return []

    factory = pylon.TlFactory.GetInstance()
    results: list[DeviceInfo] = []

    for i, dev in enumerate(factory.EnumerateDevices()):
        try:
            serial = dev.GetSerialNumber()
            results.append(
                DeviceInfo(
                    device_id=f"{serial}",
                    index=i,
                    name=dev.GetUserDefinedName(),
                    driver="basler",
                    hardware_id=serial,
                    manufacturer=dev.GetVendorName(),
                    model=dev.GetModelName(),
                    metadata={"address": dev.GetAddress()},
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug(f"Skipping Basler device at index {i}: failed to query info")

    return results
