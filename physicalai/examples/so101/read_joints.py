# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hardware check script for the SO-101 robot arm.

Connects to the robot, reads joint states in a loop, and prints them to the
terminal so the user can verify communication and manually move the arm.

Usage::

    python examples/so101/read_joints.py --port /dev/ttyUSB0
    python examples/so101/read_joints.py --port /dev/ttyUSB0 --calibration cal.json
"""

from __future__ import annotations

import argparse
import sys
import time

from physicalai.robot.so101.so101 import SO101


def main(argv: list[str] | None = None) -> None:
    """Run the SO-101 hardware check.

    Connects to the robot, prints live joint positions on a single
    updating line, and cleanly disconnects on Ctrl+C.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description="SO-101 hardware check: connect, read state, and let the user move the arm.",
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Serial baudrate (default: 1000000)")
    parser.add_argument("--hz", type=float, default=100.0, help="Read frequency in Hz (default: 100)")
    parser.add_argument("--calibration", default=None, help="Path to calibration JSON file")
    args = parser.parse_args(argv)

    if args.calibration:
        robot = SO101(
            port=args.port,
            baudrate=args.baudrate,
            role="leader",
            calibration=args.calibration,
        )
    else:
        print(
            "WARNING: No calibration provided. Running in raw ticks mode; values are not radians. "
            "Do not use uncalibrated mode for policy inference/deployment.",
        )  # noqa: T201
        robot = SO101.uncalibrated(
            port=args.port,
            baudrate=args.baudrate,
            role="leader",
        )

    unit = robot.unit
    period = 1.0 / args.hz

    print(f"Connecting to SO-101 on {args.port} ...")  # noqa: T201
    robot.connect()
    print(f"Connected. Reading at {args.hz} Hz. Press Ctrl+C to stop.\n")  # noqa: T201

    # Print header
    header = "  ".join(f"{name:>14s}" for name in SO101.JOINT_ORDER)
    print(f"  {'timestamp':>10s}  {header}  {'fps':>6s}  [{unit}]")  # noqa: T201
    print("-" * (2 + 10 + 2 + len(header) + 2 + 6 + 2 + len(unit) + 2))  # noqa: T201

    fps = 0.0
    last_time = time.monotonic()
    frame_count = 0

    try:
        while True:
            t0 = time.monotonic()
            obs = robot.get_observation()
            state = obs.joint_positions
            ts = obs.timestamp

            # Update FPS every 0.5 seconds
            frame_count += 1
            now = time.monotonic()
            dt = now - last_time
            if dt >= 0.5:  # noqa: PLR2004
                fps = frame_count / dt
                frame_count = 0
                last_time = now

            if robot.calibrated:
                values = "  ".join(f"{v:>14.4f}" for v in state)
            else:
                values = "  ".join(f"{v:>14.0f}" for v in state)

            line = f"\r  {ts:>10.3f}  {values}  {fps:>6.1f}"
            sys.stdout.write(line)
            sys.stdout.flush()

            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")  # noqa: T201
    finally:
        robot.disconnect()
        print("Disconnected.")  # noqa: T201


if __name__ == "__main__":
    main()
