# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test V4L2 control changes on a live omni_camera stream.

Run on Linux with a USB webcam and v4l2-ctl installed.
Keyboard controls adjust camera parameters via v4l2-ctl subprocess calls
while omni_camera captures frames.
"""

# ruff: noqa: D103, INP001, T201

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import omni_camera

DEVICE = "/dev/video0"

CONTROL_KEYS: dict[str, tuple[str, str]] = {
    # key_lower -> (control_name, display_label)
    "b": ("brightness", "Brightness"),
    "e": ("exposure_absolute", "Exposure"),
    "g": ("gain", "Gain"),
    "c": ("contrast", "Contrast"),
    "s": ("saturation", "Saturation"),
}


@dataclass
class CtrlInfo:
    name: str
    current: int
    min: int
    max: int
    step: int


def check_v4l2_ctl() -> bool:
    return shutil.which("v4l2-ctl") is not None


def list_controls(device: str) -> dict[str, CtrlInfo]:
    """Parse v4l2-ctl --list-ctrls output into a dict keyed by control name."""
    result = subprocess.run(
        ["v4l2-ctl", "-d", device, "--list-ctrls"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"v4l2-ctl --list-ctrls failed: {result.stderr.strip()}")
        return {}

    controls: dict[str, CtrlInfo] = {}
    # Lines look like:
    #   brightness 0x00980900 (int)    : min=0 max=255 step=1 default=128 value=128
    pattern = re.compile(
        r"^\s*(\w+)\s+0x[\da-fA-F]+\s+\(\w+\)\s*:\s*(.*)", re.MULTILINE
    )
    for match in pattern.finditer(result.stdout):
        name = match.group(1)
        attrs_str = match.group(2)
        attrs: dict[str, int] = {}
        for kv in re.finditer(r"(\w+)=(-?\d+)", attrs_str):
            attrs[kv.group(1)] = int(kv.group(2))
        if "value" in attrs:
            controls[name] = CtrlInfo(
                name=name,
                current=attrs.get("value", 0),
                min=attrs.get("min", 0),
                max=attrs.get("max", 0),
                step=attrs.get("step", 1),
            )
    return controls


def get_control_value(device: str, name: str) -> int | None:
    result = subprocess.run(
        ["v4l2-ctl", "-d", device, "--get-ctrl", name],
        capture_output=True,
        text=True,
        check=False,
    )
    # Output like: brightness: 128
    m = re.search(r":\s*(-?\d+)", result.stdout)
    return int(m.group(1)) if m else None


def set_control_value(device: str, name: str, value: int) -> bool:
    result = subprocess.run(
        ["v4l2-ctl", "-d", device, "--set-ctrl", f"{name}={value}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"Failed to set {name}={value}: {result.stderr.strip()}")
        return False
    return True


def draw_overlay(
    frame: np.ndarray,
    controls: dict[str, CtrlInfo],
    tracked_names: list[str],
) -> np.ndarray:
    overlay = frame.copy()
    y = 30
    line_h = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (0, 255, 0)
    bg_color = (0, 0, 0)

    lines = ["Keys: b/B e/E g/G c/C s/S  q=quit"]
    for key_lower, (ctrl_name, label) in CONTROL_KEYS.items():
        info = controls.get(ctrl_name)
        if info:
            lines.append(
                f"  {key_lower}/{key_lower.upper()}: {label} = {info.current} "
                f"[{info.min}..{info.max}]"
            )
        else:
            lines.append(f"  {key_lower}/{key_lower.upper()}: {label} (not available)")

    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, 1)
        cv2.rectangle(overlay, (8, y - th - 4), (12 + tw, y + 4), bg_color, -1)
        cv2.putText(overlay, line, (10, y), font, scale, color, 1, cv2.LINE_AA)
        y += line_h

    return overlay


def main() -> None:
    if not check_v4l2_ctl():
        print("Error: v4l2-ctl is not installed. Install v4l2-utils and retry.")
        sys.exit(1)

    # List and print available controls.
    controls = list_controls(DEVICE)
    print(f"Available V4L2 controls on {DEVICE}:")
    for info in controls.values():
        print(
            f"  {info.name}: value={info.current} "
            f"min={info.min} max={info.max} step={info.step}"
        )
    if not controls:
        print("  (none found)")

    # Open camera via omni_camera.
    cameras = omni_camera.query(only_usable=True)
    if not cameras:
        print("No usable cameras found via omni_camera.")
        sys.exit(1)

    cam_info = cameras[0]
    print(f"Opening camera: index={cam_info.index} name={cam_info.name}")
    cam = omni_camera.Camera(cam_info)

    fmt = cam.get_format_options().resolve_default()
    print(f"Format: {fmt.width}x{fmt.height} @ {fmt.frame_rate}fps")
    cam.open(fmt)

    tracked = [name for _, (name, _) in CONTROL_KEYS.items()]

    window = "V4L2 Controls Test"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    print("\nStreaming. Use keyboard to adjust controls. Press 'q' to quit.")

    try:
        while True:
            frame = cam.poll_frame_np()
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                bgr = draw_overlay(bgr, controls, tracked)
                cv2.imshow(window, bgr)
            else:
                time.sleep(0.001)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            ch = chr(key)

            if ch == "q":
                break

            key_lower = ch.lower()
            if key_lower not in CONTROL_KEYS:
                continue

            ctrl_name, label = CONTROL_KEYS[key_lower]
            info = controls.get(ctrl_name)
            if info is None:
                print(f"{label} ({ctrl_name}) not available on this camera.")
                continue

            step = info.step if info.step > 0 else 1
            direction = step if ch.isupper() else -step
            new_val = max(info.min, min(info.max, info.current + direction))

            if set_control_value(DEVICE, ctrl_name, new_val):
                actual = get_control_value(DEVICE, ctrl_name)
                if actual is not None:
                    info.current = actual
                else:
                    info.current = new_val
                print(f"{label}: {info.current}")
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
