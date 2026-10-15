# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standalone tests for the LeRobot engine plugin."""

from __future__ import annotations

from physicalai.engine import Engine

from physicalai_lerobot_policy_plugin.engine import LeRobotEngine


def test_engine_is_a_lightweight_framework_specific_engine_class() -> None:
    assert issubclass(LeRobotEngine, Engine)
    assert LeRobotEngine.framework == "lerobot"
    assert LeRobotEngine.env is not None
    assert LeRobotEngine.env.framework == "lerobot"
