# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 constants."""

SO101_JOINT_SIGNS = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
SO101_JOINT_OFFSETS = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)

# Released SO-101 statistics use LeRobot degrees, while the PhysicalAI driver
# maps calibrated body-joint ranges to [-100, 100].
SO101_EXPECTED_BODY_RANGE_WIDTHS = (2666.0, 2313.0, 2196.0, 2302.0, 3829.0)
SO101_MAX_POSITION_TICKS = 4095.0
SO101_DEGREES_PER_NORMALIZED_UNIT = tuple(
    range_width * 360.0 / (200.0 * SO101_MAX_POSITION_TICKS) for range_width in SO101_EXPECTED_BODY_RANGE_WIDTHS
)
