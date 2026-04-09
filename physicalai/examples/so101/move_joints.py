# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Actuation smoke test for the SO-101 robot arm.

Connects as a follower, reads the current pose, then moves each joint
individually by a small offset and back. Verifies that ``send_action()``
works and that joint ordering matches the physical wiring.

Usage::

    python examples/so101/move_joints.py --port /dev/ttyUSB0
    python examples/so101/move_joints.py --port /dev/ttyUSB0 --calibration cal.json
    python examples/so101/move_joints.py --port /dev/ttyUSB0 --calibration cal.json --offset 0.15
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from physicalai.robot.so101.so101 import SO101

_DEFAULT_OFFSET_RAD = 0.08
"""Default movement offset in radians (~4.6 degrees) when calibrated."""

_DEFAULT_OFFSET_TICKS = 100.0
"""Default movement offset in ticks when uncalibrated."""

_STEP_DELAY = 0.5
"""Seconds to wait after each movement so you can visually confirm it."""


def _test_joint(
    robot: SO101,
    joint_idx: int,
    name: str,
    start_pose: np.ndarray,
    offset: float,
    delay: float,
    calibrated: bool,
) -> bool:
    """Move a single joint by +offset and -offset, return True if either works.

    Tries both directions because one side may be blocked at a joint limit.

    Returns:
        True if the joint responded within tolerance in at least one direction.
    """
    tolerance = 0.035 if calibrated else 20.0
    unit = "rad" if calibrated else "ticks"
    any_ok = False

    for sign, label in [(+1, "+"), (-1, "-")]:
        target = start_pose.copy()
        target[joint_idx] += sign * offset
        robot.send_action(target)
        time.sleep(delay)

        obs = robot.get_observation()
        actual = obs["state"][joint_idx]
        expected = start_pose[joint_idx] + sign * offset
        delta = abs(actual - expected)
        if calibrated:
            print(  # noqa: T201
                f"  {label}{offset:.3f} {unit} -> read: {actual:.4f} "
                f"(expected: {expected:.4f}, delta: {delta:.4f})"
            )
        else:
            print(  # noqa: T201
                f"  {label}{offset:.0f} {unit} -> read: {actual:.0f} "
                f"(expected: {expected:.0f}, delta: {delta:.0f})"
            )

        # Return to start
        robot.send_action(start_pose)
        time.sleep(delay)

        if delta < tolerance:
            any_ok = True

    if any_ok:
        print(f"  OK {name}")  # noqa: T201
    else:
        print(f"  FAIL {name} (delta too large in both directions - check wiring or servo ID)")  # noqa: T201
    return any_ok


def main(argv: list[str] | None = None) -> None:
    """Run the SO-101 joint movement smoke test.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description="SO-101 actuation smoke test: move each joint one at a time to verify wiring.",
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Serial baudrate (default: 1000000)")
    parser.add_argument("--calibration", default=None, help="Path to LeRobot calibration JSON file")
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help=(
            "Movement offset. With --calibration: radians (default: "
            f"{_DEFAULT_OFFSET_RAD}). Without --calibration: ticks "
            f"(default: {_DEFAULT_OFFSET_TICKS:.0f})."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=_STEP_DELAY,
        help=f"Seconds to pause after each movement (default: {_STEP_DELAY})",
    )
    args = parser.parse_args(argv)

    if args.calibration:
        robot = SO101(
            port=args.port,
            calibration=args.calibration,
            baudrate=args.baudrate,
            role="follower",
        )
    else:
        print(
            "WARNING: No calibration provided. Running in raw ticks mode; actions are not radians. "
            "Do not use uncalibrated mode for policy inference/deployment.",
        )  # noqa: T201
        robot = SO101.uncalibrated(
            port=args.port,
            baudrate=args.baudrate,
            role="follower",
        )
    calibrated = robot.calibrated
    offset = args.offset if args.offset is not None else (_DEFAULT_OFFSET_RAD if calibrated else _DEFAULT_OFFSET_TICKS)
    unit = "rad" if calibrated else "ticks"

    print(f"Connecting to SO-101 on {args.port} (role=follower)...")  # noqa: T201
    robot.connect()

    # Read starting pose
    obs = robot.get_observation()
    start_pose = obs["state"].copy()
    if calibrated:
        start_pose_text = ", ".join(f"{v:.4f}" for v in start_pose)
    else:
        start_pose_text = ", ".join(f"{v:.0f}" for v in start_pose)
    print(f"Connected. Starting pose ({unit}): {start_pose_text}\n")  # noqa: T201

    passed = 0
    failed = 0

    try:
        for i, name in enumerate(SO101.JOINT_ORDER):
            print(f"Testing joint {i} ({name})...")  # noqa: T201
            if _test_joint(robot, i, name, start_pose, offset, args.delay, calibrated):
                passed += 1
            else:
                failed += 1
            print()  # noqa: T201

    except KeyboardInterrupt:
        print("\nInterrupted by user.")  # noqa: T201
    finally:
        # Return to starting pose, then release torque on disconnect
        print("Returning to starting pose...")  # noqa: T201
        robot.send_action(start_pose)
        time.sleep(args.delay)
        print("Releasing torque (motors off)...")  # noqa: T201
        robot.torque_on_disconnect = False
        robot.disconnect()
        print("Disconnected.")  # noqa: T201

    print(f"\nResults: {passed} passed, {failed} failed out of {SO101.NUM_JOINTS} joints.")  # noqa: T201
    if failed > 0:
        print("Check servo IDs and wiring for failed joints.")  # noqa: T201


if __name__ == "__main__":
    main()
