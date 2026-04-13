# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO-101 hardware constants (Feetech STS3215 servos)."""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Encoder / conversion
# ---------------------------------------------------------------------------

TICKS_PER_REVOLUTION: Final = 4096
"""STS3215 encoder resolution: 4096 ticks per full 360° revolution."""

RADIANS_PER_TICK: Final = 2.0 * np.pi / TICKS_PER_REVOLUTION

# ---------------------------------------------------------------------------
# Joint ordering
# ---------------------------------------------------------------------------

SO101_JOINT_ORDER: Final = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

VALID_ROLES: Final = frozenset({"leader", "follower"})

# ---------------------------------------------------------------------------
# Feetech STS3215 control table
# ---------------------------------------------------------------------------

PROTOCOL_VERSION: Final = 0


class STS3215Addr(IntEnum):
    """Feetech STS3215 control table addresses."""

    TORQUE_ENABLE = 40
    GOAL_POSITION = 42
    PRESENT_POSITION = 56


class STS3215Len(IntEnum):
    """Byte widths for STS3215 sync read/write fields."""

    GOAL_POSITION = 2
    PRESENT_POSITION = 2
