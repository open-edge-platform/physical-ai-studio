# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Discover a UVC camera and print a few live frame summaries."""

# ruff: noqa: D103, INP001

import sys

from physicalai.capture.cameras.uvc import UVCCamera


def main() -> None:
    devices = UVCCamera.discover()
    for i, device in enumerate(devices):
        print(f"[{device.index}] {device.name} ({device.device_id})")
    if not devices:
        print("No cameras found.")
        return

    device_index = input(f"Select camera (index or path): ")

    with UVCCamera(device=device_index) as cam:
        controls = cam.get_settings()
        print(f"Camera settings:")
        for ctrl in controls:
            print(f"{ctrl}")
        print("\nFrames:")
        for _ in range(10):
            frame = cam.read_latest()
            print(
                f"shape={frame.data.shape} timestamp={frame.timestamp} sequence={frame.sequence}",
            )


if __name__ == "__main__":
    main()
