# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy Runtime adapter registration for native LeRobot and exported LeRobot bundle policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.inference.adapters.registry import RuntimeAdapterRegistry


_NATIVE_ADAPTER_MODULE = "physicalai_lerobot_policy_plugin.runtime"
_BUNDLE_ADAPTER_MODULE = "physicalai_lerobot_policy_plugin.bundle_runtime"


def register(registry: RuntimeAdapterRegistry) -> None:
    """Register LeRobot native and bundle Runtime adapters without importing heavy dependencies.

    Args:
        registry: Runtime's inference-adapter registry.
    """
    registry.register_lazy_module(
        "lerobot",
        _NATIVE_ADAPTER_MODULE,
        extensions=(".lerobot",),
    )
    registry.register_lazy_module(
        "lerobot_bundle",
        _BUNDLE_ADAPTER_MODULE,
        extensions=(".lerobot-bundle",),
    )
