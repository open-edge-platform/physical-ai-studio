# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot plugin lock and supported-version consistency tests."""

from __future__ import annotations

from physicalai.engine._environment import locked_version, supported_versions
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine


def test_worker_lock_version_is_inside_the_plugin_supported_range() -> None:
    env = LeRobotEngine.env

    assert locked_version(env) in supported_versions(env)
