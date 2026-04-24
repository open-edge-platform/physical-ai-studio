# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trossen WidowX AI hardware constants."""

from __future__ import annotations

from typing import Final

WIDOWXAI_JOINT_ORDER: Final = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)

VALID_ROLES: Final = frozenset({"leader", "follower"})

HOME_POSITION: Final = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
