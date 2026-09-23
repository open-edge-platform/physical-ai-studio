# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 constants."""

from typing import Any

MOLMOACT2_TOKENIZER_REPO_ID = "allenai/MolmoAct2"
MOLMOACT2_TOKENIZER_REVISION = "e432d85f6e039edca44afb93c262f3084ab72a9c"

SO101_JOINT_SIGNS = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
SO101_JOINT_OFFSETS = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)

SO101_BODY_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

SO101_MAX_POSITION_TICKS = 4095.0


def get_so101_degrees_per_normalized_unit_from_config(
    calibration: dict[str, Any],
) -> tuple[float, ...]:
    """Calculate LeRobot degrees per PhysicalAI normalized SO-101 unit.

    The PhysicalAI SO-101 driver maps each calibrated body-joint range to
    [-100, 100]. MolmoAct2's released SO-101 checkpoint expects LeRobot
    degrees.

    Args:
        calibration: Loaded SO-101 calibration dictionary.

    Returns:
        Degrees represented by one normalized unit for each body joint.

    Raises:
        ValueError: If a required joint is missing or has an invalid range.
    """
    widths: list[float] = []

    for joint in SO101_BODY_JOINTS:
        if joint not in calibration:
            msg = f"SO-101 calibration is missing required joint '{joint}'."
            raise ValueError(msg)

        joint_calibration = calibration[joint]

        try:
            range_min = float(joint_calibration["range_min"])
            range_max = float(joint_calibration["range_max"])
        except KeyError as exc:
            msg = f"SO-101 calibration for '{joint}' is missing {exc.args[0]!r}."
            raise ValueError(msg) from exc

        width = range_max - range_min
        if width <= 0:
            msg = f"Invalid SO-101 calibration range for '{joint}': range_min={range_min}, range_max={range_max}."
            raise ValueError(msg)

        widths.append(width)

    return tuple(width * 360.0 / (200.0 * SO101_MAX_POSITION_TICKS) for width in widths)
