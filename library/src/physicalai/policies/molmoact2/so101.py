# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO-101 joint frame for the released MolmoAct2 SO-100/101 checkpoint.

The checkpoint was trained on legacy LeRobot joints: body joints in degrees,
``shoulder_lift`` flipped, and ``shoulder_lift``/``elbow_flex`` offset by 90°.
The PhysicalAI SO-101 driver reports each body joint in ``[-100, 100]`` over
its calibrated tick range, and the gripper in ``[0, 100]``.

The runtime-to-checkpoint transform is::

    checkpoint = sign * scale * runtime + offset
    runtime = sign * (checkpoint - offset) / scale

where ``scale`` is the number of degrees in one runtime unit, derived from the
arm calibration. Without a calibration the scale is 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from physicalai.policies.utils import JointFrameTransform

from .constants import SO101_JOINT_OFFSETS, SO101_JOINT_SIGNS

SO101_BODY_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# LeRobot degrees mode: degrees = (ticks - mid) * 360 / 4095.
SO101_DEGREES_PER_TICK = 360.0 / 4095.0

# PhysicalAI normalized mode: runtime = (ticks - mid) * 200 / range_width.
SO101_RUNTIME_UNITS_PER_RANGE = 200.0


def load_so101_calibration(calibration: dict[str, Any] | str | Path | None) -> dict[str, Any] | None:
    """Load an SO-101 calibration from a dictionary or JSON file path.

    Args:
        calibration: Calibration dictionary, path to a calibration JSON file, or ``None``.

    Returns:
        The calibration dictionary, or ``None`` when no calibration is given.
    """
    if calibration is None or isinstance(calibration, dict):
        return calibration
    with Path(calibration).expanduser().open(encoding="utf-8") as file:
        return json.load(file)


def so101_degrees_per_runtime_unit(calibration: dict[str, Any]) -> tuple[float, ...]:
    """Compute the degrees represented by one PhysicalAI runtime unit per joint.

    Body joints use ``range_width / 200 * 360 / 4095``. The gripper keeps
    ``[0, 100]`` in both conventions, so its scale is 1.

    Args:
        calibration: SO-101 calibration dictionary.

    Returns:
        One scale per joint, in ``shoulder_pan ... wrist_roll, gripper`` order.

    Raises:
        ValueError: If a body joint is missing or has an invalid range.
    """
    scales: list[float] = []
    for joint in SO101_BODY_JOINTS:
        if joint not in calibration:
            msg = f"SO-101 calibration is missing required joint '{joint}'."
            raise ValueError(msg)

        try:
            range_min = float(calibration[joint]["range_min"])
            range_max = float(calibration[joint]["range_max"])
        except KeyError as exc:
            msg = f"SO-101 calibration for '{joint}' is missing {exc.args[0]!r}."
            raise ValueError(msg) from exc

        range_width = range_max - range_min
        if range_width <= 0:
            msg = f"Invalid SO-101 calibration range for '{joint}': range_min={range_min}, range_max={range_max}."
            raise ValueError(msg)

        scales.append(range_width / SO101_RUNTIME_UNITS_PER_RANGE * SO101_DEGREES_PER_TICK)

    gripper_scale = 1.0
    return (*scales, gripper_scale)


def so101_joint_scales(calibration: dict[str, Any] | None) -> tuple[float, ...]:
    """Resolve the per-joint runtime-to-degrees scales.

    Args:
        calibration: SO-101 calibration dictionary, or ``None`` for unit scales.

    Returns:
        Calibration-derived scales, or 1 for every joint without a calibration.
    """
    if calibration is None:
        return (1.0,) * len(SO101_JOINT_SIGNS)
    return so101_degrees_per_runtime_unit(calibration)


def make_so101_joint_transform(calibration: dict[str, Any] | None) -> JointFrameTransform:
    """Build the runtime-to-checkpoint SO-101 joint transform.

    Args:
        calibration: SO-101 calibration dictionary. When ``None``, scales are 1
            and only the legacy signs and offsets are applied.

    Returns:
        The joint transform used by the processors and feature statistics.
    """
    return JointFrameTransform(
        signs=SO101_JOINT_SIGNS,
        offsets=SO101_JOINT_OFFSETS,
        scales=so101_joint_scales(calibration),
    )
