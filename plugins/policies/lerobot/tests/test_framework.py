# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standalone tests for the LeRobot framework plugin entry point."""

from __future__ import annotations

from physicalai.engine import FrameworkRegistry

from physicalai_lerobot_policy_plugin.framework import register


def test_entrypoint_registers_le_robot_framework_without_importing_its_config() -> None:
    registry = FrameworkRegistry()
    register(registry)

    assert registry.names() == ["lerobot"]
    config_class = type("TrainPipelineConfig", (), {"__module__": "lerobot.configs.train"})
    assert registry.get("lerobot").accepts(config_class())
