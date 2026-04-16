# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: DOC201, PLR0914

"""V4L2 device enumeration via sysfs and QUERYCAP ioctl.

Scans ``/sys/class/video4linux/`` for ``video*`` entries, opens each
``/dev/videoN`` device, and queries capabilities via ``VIDIOC_QUERYCAP``.
Only devices whose **per-node** ``device_caps`` advertise video capture
are included — this correctly filters out UVC metadata nodes that share
the same physical device.
"""

from __future__ import annotations

import os
import pathlib

from physicalai.capture.discovery import DeviceInfo

from ._ioctl import (
    V4L2_CAP_DEVICE_CAPS,
    V4L2_CAP_VIDEO_CAPTURE,
    VIDIOC_QUERYCAP,
    v4l2_capability,
    xioctl,
)

__all__ = ["discover_v4l2"]

_SYSFS_V4L2 = pathlib.Path("/sys/class/video4linux")
_V4L_BY_ID = pathlib.Path("/dev/v4l/by-id")
_V4L_BY_PATH = pathlib.Path("/dev/v4l/by-path")


def _read_sysfs_attr(entry: pathlib.Path, attr: str) -> str:
    """Read a single-line sysfs attribute, returning '' on any error."""
    try:
        return (entry / attr).read_text().strip()
    except (FileNotFoundError, OSError, PermissionError):
        return ""


def _find_usb_parent(entry: pathlib.Path) -> pathlib.Path | None:
    """Walk up the sysfs device tree to find the USB device directory.

    The USB device directory contains ``idVendor``, ``idProduct``, and
    optionally ``serial``, ``manufacturer``, ``product``.
    """
    device_link = entry / "device"
    if not device_link.exists():
        return None
    try:
        real = device_link.resolve()
    except OSError:
        return None
    # Walk up from the interface dir (e.g. 1-6.2:1.0) to USB device (e.g. 1-6.2)
    for parent in [real, *list(real.parents)]:
        if (parent / "idVendor").exists():
            return parent
    return None


def _find_symlink(symlink_dir: pathlib.Path, dev_path: str) -> str:
    """Find the symlink in *symlink_dir* that resolves to *dev_path*."""
    if not symlink_dir.is_dir():
        return ""
    try:
        for link in symlink_dir.iterdir():
            try:
                if link.resolve() == pathlib.Path(dev_path).resolve():
                    return link.name
            except OSError:
                continue
    except OSError:
        pass
    return ""


def discover_v4l2() -> list[DeviceInfo]:
    """Enumerate V4L2 capture devices via sysfs and QUERYCAP.

    Uses ``device_caps`` (per-node capabilities) to filter out metadata
    and non-capture nodes.  Enriches results with USB metadata from
    sysfs and stable ``by-id`` / ``by-path`` symlinks.

    Returns:
        Sorted list of :class:`~physicalai.capture.discovery.DeviceInfo`
        for every accessible V4L2 video capture device.
    """
    if not _SYSFS_V4L2.exists():
        return []

    try:
        entries = sorted(_SYSFS_V4L2.iterdir())
    except FileNotFoundError:
        return []

    devices: list[DeviceInfo] = []

    for entry in entries:
        if not entry.name.startswith("video"):
            continue

        device_path = f"/dev/{entry.name}"
        suffix = entry.name.removeprefix("video")
        video_index = int(suffix) if suffix.isdecimal() else -1

        sysfs_name = _read_sysfs_attr(entry, "name")

        fd = -1
        try:
            fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
            cap = v4l2_capability()
            xioctl(fd, VIDIOC_QUERYCAP, cap)

            # Use per-node device_caps when available (kernel >= 3.4),
            # falling back to union capabilities for ancient kernels.
            effective_caps = cap.device_caps if cap.capabilities & V4L2_CAP_DEVICE_CAPS else cap.capabilities

            if not (effective_caps & V4L2_CAP_VIDEO_CAPTURE):
                continue

            card_name = cap.card.decode().rstrip("\x00") or sysfs_name
            bus_info = cap.bus_info.decode().rstrip("\x00")
            driver_name = cap.driver.decode().rstrip("\x00")

            # Enrich from USB sysfs tree
            vid = ""
            pid = ""
            serial = ""
            manufacturer = ""
            product = ""
            usb_parent = _find_usb_parent(entry)
            if usb_parent is not None:
                vid = _read_sysfs_attr(usb_parent, "idVendor")
                pid = _read_sysfs_attr(usb_parent, "idProduct")
                serial = _read_sysfs_attr(usb_parent, "serial")
                manufacturer = _read_sysfs_attr(usb_parent, "manufacturer")
                product = _read_sysfs_attr(usb_parent, "product")

            # Stable symlinks
            by_id = _find_symlink(_V4L_BY_ID, device_path)
            by_path = _find_symlink(_V4L_BY_PATH, device_path)

            # Prefer by-id (includes serial) for hardware_id, fall back to bus_info
            hardware_id = by_id or bus_info
            devices.append(
                DeviceInfo(
                    device_id=device_path,
                    index=video_index,
                    name=card_name,
                    driver="v4l2",
                    hardware_id=hardware_id,
                    manufacturer=manufacturer,
                    model=product or card_name,
                    metadata={
                        "device_caps": effective_caps,
                        "capabilities": cap.capabilities,
                        "bus_info": bus_info,
                        "kernel_driver": driver_name,
                        "product": product,
                        "vid": vid,
                        "pid": pid,
                        "serial": serial,
                        "by_id": by_id,
                        "by_path": by_path,
                    },
                ),
            )
        except PermissionError:
            continue
        except OSError:
            continue
        finally:
            if fd >= 0:
                os.close(fd)

    return devices
